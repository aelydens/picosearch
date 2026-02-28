"""
Picosearch LangGraph agent.

All agent logic lives here. Two tools:
  - search_media_tool   — composite internal search (keyword / semantic / hybrid / clip)
  - pexels_fallback_tool — external Pexels API, fires when internal confidence is low

Stub implementation: returns mock data. Replace with real OpenSearch calls later.
"""

import os
import random
from typing import Annotated, Any, Literal

import httpx
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel
from typing_extensions import TypedDict

# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

SearchMode = Literal["auto", "keyword", "semantic", "hybrid", "clip"]


class SearchRequest(BaseModel):
    query: str
    mode: SearchMode = "auto"
    limit: int = 12


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
# Mock data helpers
# ---------------------------------------------------------------------------

MOCK_IMAGES = [
    {
        "id": "mock-001",
        "url": "https://images.pexels.com/photos/3184291/pexels-photo-3184291.jpeg",
        "thumbnail_url": "https://images.pexels.com/photos/3184291/pexels-photo-3184291.jpeg?w=400",
        "description": "Marketing team collaborating around a table",
        "tags": ["teamwork", "collaboration", "marketing", "office"],
    },
    {
        "id": "mock-002",
        "url": "https://images.pexels.com/photos/3182812/pexels-photo-3182812.jpeg",
        "thumbnail_url": "https://images.pexels.com/photos/3182812/pexels-photo-3182812.jpeg?w=400",
        "description": "Business presentation with charts and graphs",
        "tags": ["presentation", "data", "business", "charts"],
    },
    {
        "id": "mock-003",
        "url": "https://images.pexels.com/photos/1181316/pexels-photo-1181316.jpeg",
        "thumbnail_url": "https://images.pexels.com/photos/1181316/pexels-photo-1181316.jpeg?w=400",
        "description": "Creative brainstorming session with sticky notes",
        "tags": ["brainstorming", "creative", "ideas", "sticky-notes"],
    },
    {
        "id": "mock-004",
        "url": "https://images.pexels.com/photos/3184418/pexels-photo-3184418.jpeg",
        "thumbnail_url": "https://images.pexels.com/photos/3184418/pexels-photo-3184418.jpeg?w=400",
        "description": "Modern laptop and coffee on minimalist desk",
        "tags": ["workspace", "minimal", "laptop", "productivity"],
    },
    {
        "id": "mock-005",
        "url": "https://images.pexels.com/photos/3182773/pexels-photo-3182773.jpeg",
        "thumbnail_url": "https://images.pexels.com/photos/3182773/pexels-photo-3182773.jpeg?w=400",
        "description": "Diverse group of professionals in a meeting",
        "tags": ["diversity", "meeting", "professionals", "teamwork"],
    },
    {
        "id": "mock-006",
        "url": "https://images.pexels.com/photos/1181467/pexels-photo-1181467.jpeg",
        "thumbnail_url": "https://images.pexels.com/photos/1181467/pexels-photo-1181467.jpeg?w=400",
        "description": "Person typing on keyboard — remote work lifestyle",
        "tags": ["remote-work", "typing", "digital", "lifestyle"],
    },
]

PEXELS_MOCK = [
    {
        "id": "pex-001",
        "url": "https://images.pexels.com/photos/3183150/pexels-photo-3183150.jpeg",
        "thumbnail_url": "https://images.pexels.com/photos/3183150/pexels-photo-3183150.jpeg?w=400",
        "description": "Pexels: startup team in a bright open office",
        "tags": ["startup", "office", "team", "pexels"],
    },
    {
        "id": "pex-002",
        "url": "https://images.pexels.com/photos/3184360/pexels-photo-3184360.jpeg",
        "thumbnail_url": "https://images.pexels.com/photos/3184360/pexels-photo-3184360.jpeg?w=400",
        "description": "Pexels: woman presenting strategy to colleagues",
        "tags": ["strategy", "presentation", "woman", "leadership", "pexels"],
    },
]


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@tool
def search_media_tool(
    query: str,
    mode: str = "hybrid",
) -> dict[str, Any]:
    """
    Search the internal media library for images matching the query.

    Composite router: dispatches to keyword, semantic, hybrid, or clip search
    depending on `mode`. Returns a list of results with confidence score.

    Args:
        query: Natural-language or keyword search query.
        mode:  One of 'keyword', 'semantic', 'hybrid', 'clip'.

    Returns:
        dict with 'results' (list of image dicts) and 'confidence' (float 0-1).
    """
    # Stub: shuffle mock images and assign fake scores
    results = random.sample(MOCK_IMAGES, min(6, len(MOCK_IMAGES)))
    scored = []
    for i, img in enumerate(results):
        score = round(random.uniform(0.55, 0.98) - i * 0.04, 3)
        scored.append({**img, "score": max(score, 0.3), "source": "internal"})

    confidence = round(sum(r["score"] for r in scored) / len(scored), 3) if scored else 0.0
    return {"results": scored, "confidence": confidence, "mode_used": mode}


@tool
def pexels_fallback_tool(query: str, limit: int = 6) -> dict[str, Any]:
    """
    Fetch images from the Pexels API as a fallback when internal results
    have low confidence.

    Args:
        query: Search query to send to Pexels.
        limit: Maximum number of results to return.

    Returns:
        dict with 'results' (list of image dicts) and 'source' = 'external'.
    """
    api_key = os.getenv("PEXELS_API_KEY", "")

    if api_key and api_key != "your_pexels_api_key_here":
        try:
            with httpx.Client(timeout=10) as client:
                resp = client.get(
                    "https://api.pexels.com/v1/search",
                    headers={"Authorization": api_key},
                    params={"query": query, "per_page": limit},
                )
                resp.raise_for_status()
                data = resp.json()
                results = [
                    {
                        "id": f"pex-{p['id']}",
                        "url": p["src"]["large"],
                        "thumbnail_url": p["src"]["medium"],
                        "description": p.get("alt", query),
                        "tags": ["pexels"],
                        "score": round(random.uniform(0.50, 0.75), 3),
                        "source": "external",
                    }
                    for p in data.get("photos", [])
                ]
                return {"results": results}
        except Exception:
            pass  # fall through to mock

    # Stub fallback
    results = [
        {**img, "score": round(random.uniform(0.45, 0.70), 3), "source": "external"}
        for img in PEXELS_MOCK[:limit]
    ]
    return {"results": results}


# ---------------------------------------------------------------------------
# LangGraph agent state
# ---------------------------------------------------------------------------

CONFIDENCE_THRESHOLD = 0.60

TOOLS = [search_media_tool, pexels_fallback_tool]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}


class AgentState(TypedDict):
    query: str
    mode: SearchMode
    limit: int
    messages: Annotated[list, add_messages]
    # accumulated results
    internal_results: list[dict]
    external_results: list[dict]
    confidence: float
    mode_used: SearchMode
    mode_explanation: str | None


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------


def route_mode(state: AgentState) -> AgentState:
    """Decide which internal search mode to use (only matters for 'auto')."""
    mode = state["mode"]
    query = state["query"]

    if mode != "auto":
        return {**state, "mode_used": mode, "mode_explanation": None}

    # Simple heuristic auto-routing (replace with LLM call when ready)
    words = query.lower().split()
    if any(w in words for w in ("photo", "image", "picture", "clip", "visual")):
        chosen: SearchMode = "clip"
        explanation = "Auto selected CLIP because query references visual/image concepts."
    elif len(words) <= 2:
        chosen = "keyword"
        explanation = "Auto selected keyword search because query is short and specific."
    elif len(words) >= 8:
        chosen = "semantic"
        explanation = "Auto selected semantic search because query is long and descriptive."
    else:
        chosen = "hybrid"
        explanation = "Auto selected hybrid (keyword + semantic) for balanced coverage."

    return {**state, "mode_used": chosen, "mode_explanation": explanation}


def run_internal_search(state: AgentState) -> AgentState:
    """Invoke the search_media_tool."""
    result = search_media_tool.invoke({"query": state["query"], "mode": state["mode_used"]})
    return {
        **state,
        "internal_results": result["results"],
        "confidence": result["confidence"],
    }


def maybe_run_pexels(state: AgentState) -> AgentState:
    """Invoke pexels_fallback_tool only when confidence is below threshold."""
    if state["confidence"] >= CONFIDENCE_THRESHOLD:
        return {**state, "external_results": []}

    result = pexels_fallback_tool.invoke(
        {"query": state["query"], "limit": state["limit"] // 2}
    )
    return {**state, "external_results": result["results"]}


def compile_results(state: AgentState) -> AgentState:
    """Merge internal and external results, sort by score, trim to limit."""
    combined = state["internal_results"] + state["external_results"]
    combined.sort(key=lambda r: r["score"], reverse=True)
    return {**state, "internal_results": combined[: state["limit"]]}


# ---------------------------------------------------------------------------
# Graph wiring
# ---------------------------------------------------------------------------

_graph = (
    StateGraph(AgentState)
    .add_node("route_mode", route_mode)
    .add_node("internal_search", run_internal_search)
    .add_node("pexels_fallback", maybe_run_pexels)
    .add_node("compile", compile_results)
    .add_edge(START, "route_mode")
    .add_edge("route_mode", "internal_search")
    .add_edge("internal_search", "pexels_fallback")
    .add_edge("pexels_fallback", "compile")
    .add_edge("compile", END)
    .compile()
)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


async def run_search_agent(request: SearchRequest) -> SearchResponse:
    initial: AgentState = {
        "query": request.query,
        "mode": request.mode,
        "limit": request.limit,
        "messages": [HumanMessage(content=request.query)],
        "internal_results": [],
        "external_results": [],
        "confidence": 0.0,
        "mode_used": request.mode if request.mode != "auto" else "hybrid",
        "mode_explanation": None,
    }

    final = await _graph.ainvoke(initial)

    results = [
        ImageResult(
            id=r["id"],
            url=r["url"],
            thumbnail_url=r["thumbnail_url"],
            description=r["description"],
            score=r["score"],
            source=r["source"],
            tags=r.get("tags", []),
        )
        for r in final["internal_results"]
    ]

    return SearchResponse(
        results=results,
        mode_used=final["mode_used"],
        mode_explanation=final.get("mode_explanation"),
        confidence=final["confidence"],
    )
