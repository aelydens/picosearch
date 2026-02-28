"""
Picosearch LangGraph agent.

Modes:
  - agent    — ReAct loop: LLM decides between search_internal (keyword),
               search_clip (CLIP kNN), search_external (Pexels).
  - clip     — CLIP kNN search directly, no LLM
  - hybrid   — keyword + CLIP merged with score fusion, no LLM
  - external — Pexels only, no LLM
"""

import logging
import os
import random
from functools import lru_cache
from typing import Annotated, Any, Literal

import httpx
import torch
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from openai import OpenAI
from opensearchpy import OpenSearch
from pydantic import BaseModel, Field
from transformers import CLIPModel, CLIPProcessor
from typing_extensions import TypedDict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("picosearch.agent")

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

SearchMode = Literal["agent", "clip", "hybrid", "external"]


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=200)
    mode: SearchMode = "agent"
    limit: int = Field(default=12, ge=1, le=50)
    rerank: bool = Field(default=True, description="Apply Cohere reranking")


class ImageResult(BaseModel):
    id: str
    url: str
    thumbnail_url: str
    description: str
    score: float
    source: Literal["internal", "external"]
    tags: list[str] = []


class SearchResponse(BaseModel):
    results: list[ImageResult]
    mode_used: SearchMode
    mode_explanation: str | None = None
    confidence: float


# ---------------------------------------------------------------------------
# OpenSearch client
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _os_client() -> OpenSearch:
    raw = os.getenv("OPENSEARCH_HOST", "")
    host = raw.replace("https://", "").replace("http://", "").rstrip("/")
    logger.info(f"Connecting to OpenSearch: {host}")
    return OpenSearch(
        hosts=[{"host": host, "port": 443}],
        http_auth=(os.getenv("OPENSEARCH_USER"), os.getenv("OPENSEARCH_PASSWORD")),
        use_ssl=True,
        verify_certs=True,
        ssl_show_warn=False,
    )


OS_INDEX = os.getenv("OPENSEARCH_INDEX", "multimodal-images")

# ---------------------------------------------------------------------------
# CLIP model — lazy singleton
# ---------------------------------------------------------------------------

_clip_model: CLIPModel | None = None
_clip_processor: CLIPProcessor | None = None
_clip_device: str | None = None
_openai_embed_client: OpenAI | None = None


def _get_clip() -> tuple[CLIPModel, CLIPProcessor, str]:
    global _clip_model, _clip_processor, _clip_device
    if _clip_model is None:
        model_name = "openai/clip-vit-base-patch32"
        _clip_processor = CLIPProcessor.from_pretrained(model_name, local_files_only=True)
        _clip_model = CLIPModel.from_pretrained(model_name, local_files_only=True)
        _clip_device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        _clip_model = _clip_model.to(_clip_device)
        logger.info(f"[CLIP] loaded on {_clip_device}")
    return _clip_model, _clip_processor, _clip_device


# ---------------------------------------------------------------------------
# Search primitives
# ---------------------------------------------------------------------------

def _hit_to_dict(h: dict, score: float) -> dict:
    s = h["_source"]
    image_url = s.get("photo_image_url", s.get("photo_url", ""))
    description = (
        s.get("llm_description")
        or s.get("ai_description")
        or s.get("original_description")
        or s.get("photo_description", "")
    )
    return {
        "id": s.get("photo_id", h["_id"]),
        "url": image_url,
        "thumbnail_url": f"{image_url}?w=400&q=80",
        "description": description,
        "score": round(score, 4),
        "source": "internal",
        "tags": s.get("tags", []),
    }


def _keyword_search(query: str, size: int = 20) -> list[dict]:
    logger.info(f"[OS] keyword search — query={query!r}")
    resp = _os_client().search(
        index=OS_INDEX,
        body={
            "size": size,
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": [
                        "original_description^2", "photo_description^2",
                        "llm_description^1.5", "ai_description^1.5",
                        "tags",
                    ],
                    "type": "best_fields",
                    "fuzziness": "AUTO",
                }
            },
        },
    )
    hits = resp["hits"]["hits"]
    logger.info(f"[OS] keyword returned {len(hits)} hits")
    max_score = max((h["_score"] for h in hits), default=1.0)
    return [_hit_to_dict(h, h["_score"] / max_score) for h in hits]


def _embed_clip(text: str) -> list[float]:
    model, processor, device = _get_clip()
    inputs = processor(text=[text], return_tensors="pt", padding=True)
    text_inputs = {k: v.to(device) for k, v in inputs.items() if k in ("input_ids", "attention_mask")}
    with torch.no_grad():
        pooled = model.text_model(**text_inputs).pooler_output  # (1, hidden)
        features = model.text_projection(pooled)                # (1, projection_dim)
    features = features / features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy().flatten().tolist()


def _embed_text(text: str) -> list[float]:
    global _openai_embed_client
    if _openai_embed_client is None:
        _openai_embed_client = OpenAI()
    resp = _openai_embed_client.embeddings.create(input=text, model="text-embedding-3-small")
    return resp.data[0].embedding


def _description_embedding_search(query: str, size: int = 20) -> list[dict]:
    logger.info(f"[OS] description embedding search — query={query!r}")
    vector = _embed_text(query)
    resp = _os_client().search(
        index=OS_INDEX,
        body={"size": size, "query": {"knn": {"description_embedding": {"vector": vector, "k": size}}}},
    )
    hits = resp["hits"]["hits"]
    logger.info(f"[OS] description kNN returned {len(hits)} hits")
    max_score = max((h["_score"] for h in hits), default=1.0)
    return [_hit_to_dict(h, h["_score"] / max_score) for h in hits]


def _clip_search(query: str, size: int = 20) -> list[dict]:
    logger.info(f"[OS] CLIP kNN search — query={query!r}")
    vector = _embed_clip(query)
    resp = _os_client().search(
        index=OS_INDEX,
        body={"size": size, "query": {"knn": {"clip_embedding": {"vector": vector, "k": size}}}},
    )
    hits = resp["hits"]["hits"]
    logger.info(f"[OS] CLIP returned {len(hits)} hits")
    max_score = max((h["_score"] for h in hits), default=1.0)
    return [_hit_to_dict(h, h["_score"] / max_score) for h in hits]


def _pexels_search(query: str, limit: int = 12) -> list[dict]:
    logger.info(f"[Pexels] search — query={query!r}")
    api_key = os.getenv("PEXELS_API_KEY", "")
    if not api_key or api_key == "your_pexels_api_key_here":
        return []
    try:
        resp = httpx.get(
            "https://api.pexels.com/v1/search",
            headers={"Authorization": api_key},
            params={"query": query, "per_page": limit},
            timeout=10,
        )
        resp.raise_for_status()
        photos = resp.json().get("photos", [])
        logger.info(f"[Pexels] returned {len(photos)} photos")
        return [
            {
                "id": f"pex-{p['id']}",
                "url": p["src"]["large"],
                "thumbnail_url": p["src"]["medium"],
                "description": p.get("alt", query),
                "score": round(random.uniform(0.50, 0.70), 3),
                "source": "external",
                "tags": ["pexels"],
            }
            for p in photos
        ]
    except Exception as e:
        logger.warning(f"[Pexels] error: {e}")
        return []


def _rrf_merge(lists: list[list[dict]], k: int = 60) -> list[dict]:
    """Reciprocal Rank Fusion — rank-based fusion that's robust to score scale differences."""
    rrf_scores: dict[str, float] = {}
    best: dict[str, dict] = {}
    for ranked in lists:
        for rank, doc in enumerate(ranked):
            doc_id = doc["id"]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
            if doc_id not in best or doc["score"] > best[doc_id]["score"]:
                best[doc_id] = doc

    merged = sorted(best.values(), key=lambda d: rrf_scores[d["id"]], reverse=True)
    top = rrf_scores[merged[0]["id"]] if merged else 1.0
    for doc in merged:
        doc["score"] = round(rrf_scores[doc["id"]] / top, 4)
    return merged


def _confidence(results: list[dict]) -> float:
    top = results[:6]
    return round(sum(r["score"] for r in top) / len(top), 4) if top else 0.0


def _rerank(query: str, results: list[dict], limit: int) -> list[dict]:
    """Rerank results using Cohere's cross-encoder. Falls back to score-sorted truncation."""
    api_key = os.getenv("COHERE_API_KEY", "")
    if not api_key or not results:
        return results[:limit]
    try:
        import cohere
        co = cohere.ClientV2(api_key)
        docs = [r["description"] or r["id"] for r in results]
        response = co.rerank(model="rerank-v3.5", query=query, documents=docs, top_n=limit)
        return [
            {**results[item.index], "score": round(item.relevance_score, 4)}
            for item in response.results
        ]
    except Exception as e:
        logger.warning(f"[Cohere] rerank failed: {e} — falling back to score sort")
        return results[:limit]


def _to_image_results(results: list[dict]) -> list[ImageResult]:
    return [ImageResult(**{k: r[k] for k in ImageResult.model_fields}) for r in results]


# ---------------------------------------------------------------------------
# Direct mode runners (no LLM)
# ---------------------------------------------------------------------------

async def _run_clip(request: SearchRequest) -> SearchResponse:
    candidates = _clip_search(request.query, size=request.limit * 2)
    results = _rerank(request.query, candidates, request.limit)
    return SearchResponse(
        results=_to_image_results(results),
        mode_used="clip",
        confidence=_confidence(results),
    )


async def _run_external(request: SearchRequest) -> SearchResponse:
    results = _pexels_search(request.query, limit=request.limit)
    return SearchResponse(
        results=_to_image_results(results),
        mode_used="external",
        confidence=_confidence(results),
    )


async def _run_hybrid(request: SearchRequest) -> SearchResponse:
    size = request.limit * 2
    kw = _keyword_search(request.query, size=size)
    desc = _description_embedding_search(request.query, size=size)
    candidates = _rrf_merge([kw, desc])
    if request.rerank:
        results = _rerank(request.query, candidates, request.limit)
    else:
        results = candidates[:request.limit]
    return SearchResponse(
        results=_to_image_results(results),
        mode_used="hybrid",
        confidence=_confidence(results),
    )


# ---------------------------------------------------------------------------
# Agent mode — ReAct loop
# ---------------------------------------------------------------------------

async def _run_react_agent(request: SearchRequest) -> SearchResponse:
    collected: list[dict] = []

    @tool
    def search_hybrid(query: str) -> str:
        """Search the internal library using keyword matching and semantic description embeddings fused with RRF. Best for descriptive, factual, or subject-based queries: people, places, objects, activities, or natural language descriptions."""
        size = request.limit * 2
        kw = _keyword_search(query, size=size)
        desc = _description_embedding_search(query, size=size)
        results = _rrf_merge([kw, desc])
        collected.extend(results)
        confidence = _confidence(results)
        logger.info(f"[AGENT TOOL] search_hybrid — {len(results)} results, confidence={confidence}")
        return f"Found {len(results)} images (confidence {confidence:.0%}). Top: {results[0]['description'][:80] if results else 'none'}"

    @tool
    def search_clip(query: str) -> str:
        """Search using CLIP visual embeddings. Best for visual moods, aesthetics, abstract feelings, color palettes, lighting, or atmospheric scenes where the vibe matters more than the subject."""
        try:
            results = _clip_search(query, size=request.limit * 2)
            collected.extend(results)
            logger.info(f"[AGENT TOOL] search_clip — {len(results)} results")
            return f"Found {len(results)} visually similar images. Top: {results[0]['description'][:80] if results else 'none'}"
        except Exception as e:
            logger.warning(f"[AGENT TOOL] search_clip failed: {e}")
            return f"CLIP search unavailable ({e}). Use search_hybrid instead."

    @tool
    def search_external(query: str) -> str:
        """Search Pexels for external stock images. Use ONLY when internal results are empty or fewer than 3."""
        results = _pexels_search(query, limit=6)
        collected.extend(results)
        logger.info(f"[AGENT TOOL] search_external — {len(results)} results")
        return f"Found {len(results)} external images from Pexels."

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    react = create_react_agent(
        llm,
        [search_hybrid, search_clip, search_external],
        prompt=(
            "You are a visual asset search agent for a marketing team.\n\n"
            "Tool selection guide:\n"
            "- search_hybrid: keyword + semantic search. Use for subjects, objects, people, places, activities, or any descriptive query.\n"
            "- search_clip: CLIP visual embeddings. Use for moods, aesthetics, color palettes, lighting, or atmosphere — when the visual feel matters more than the subject.\n"
            "- search_external: Pexels fallback. Use ONLY when internal results are empty or fewer than 3.\n\n"
            "Pick the single best tool. Only call both search_hybrid and search_clip if the query has a strong descriptive component AND a distinct visual mood.\n\n"
            "After calling tools, write one concise sentence explaining which tool you chose and why."
        ),
    )

    logger.info(f"[AGENT] ReAct starting — query={request.query!r}")
    result = await react.ainvoke({"messages": [HumanMessage(content=request.query)]})

    # Build debug trace
    tool_results_by_id = {m.tool_call_id: m.content for m in result["messages"] if isinstance(m, ToolMessage)}
    tool_call_lines = []
    for m in result["messages"]:
        if isinstance(m, AIMessage) and m.tool_calls:
            for tc in m.tool_calls:
                args_str = ", ".join(f'{k}="{v}"' for k, v in tc["args"].items())
                outcome = tool_results_by_id.get(tc["id"], "")
                tool_call_lines.append(f'▸ {tc["name"]}({args_str}) → {outcome}')

    final_text = next(
        (m.content for m in reversed(result["messages"]) if isinstance(m, AIMessage) and m.content),
        None,
    )
    debug_block = "\n".join(tool_call_lines)
    mode_explanation = f"{debug_block}\n\n{final_text}" if debug_block and final_text else (debug_block or final_text)

    logger.info(f"[AGENT] done — {len(collected)} results collected")

    seen: set[str] = set()
    deduped = []
    for r in collected:
        if r["id"] not in seen:
            seen.add(r["id"])
            deduped.append(r)
    deduped.sort(key=lambda r: r["score"], reverse=True)

    results = _rerank(request.query, deduped, request.limit)

    return SearchResponse(
        results=_to_image_results(results),
        mode_used="agent",
        mode_explanation=mode_explanation,
        confidence=_confidence(results),
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def run_search_agent(request: SearchRequest) -> SearchResponse:
    logger.info(f"[AGENT] run_search_agent — query={request.query!r} mode={request.mode}")
    match request.mode:
        case "agent":    return await _run_react_agent(request)
        case "clip":     return await _run_clip(request)
        case "hybrid":   return await _run_hybrid(request)
        case "external": return await _run_external(request)
