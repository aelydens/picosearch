# Picosearch

**Demo application** — Multimodal asset search for marketing teams. Search your media library using natural language, visual concepts, or let an AI agent pick the best approach for your query.

> This is a proof-of-concept demonstrating multimodal search patterns. The local CLIP model is a stand-in for a production inference endpoint — in a real deployment, you'd swap it for a hosted model service (e.g., AWS SageMaker, Replicate, or a self-hosted CLIP API) to handle scale and GPU requirements.

![Picosearch search example](search-example.png)

## Overview

Picosearch is a full-stack image search application that combines multiple search strategies to help marketing teams find the right visual assets. It supports:

- **Keyword search** — Traditional text matching against image descriptions and tags
- **Semantic search** — OpenAI embeddings for meaning-based description matching
- **CLIP visual search** — Find images by visual mood, aesthetics, and atmosphere
- **External search** — Pexels API fallback for stock imagery
- **Agent mode** — A LangGraph ReAct agent that intelligently selects the best search strategy based on your query

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Next.js       │────▶│   FastAPI       │────▶│   OpenSearch    │
│   Frontend      │     │   Backend       │     │   (kNN + BM25)  │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                              │
                              ├──▶ OpenAI (GPT-4o-mini for agent, embeddings)
                              ├──▶ CLIP (local, swap for inference endpoint)
                              ├──▶ Cohere (reranking)
                              └──▶ Pexels (external images)
```

## Search Modes

| Mode | Description |
|------|-------------|
| **Agent** | ReAct agent decides between search strategies based on query intent. Best for varied queries. |
| **CLIP** | Direct CLIP kNN search. Best for visual moods, aesthetics, lighting, and atmosphere. |
| **Hybrid** | Keyword + semantic embeddings fused with RRF. Best for descriptive, subject-based queries. |
| **External** | Pexels stock photo search. Fallback when internal library lacks coverage. |

## Tech Stack

**Frontend:**
- Next.js 16 with React 19
- Tailwind CSS 4
- Framer Motion for animations
- shadcn/ui components

**Backend:**
- FastAPI with Pydantic
- LangGraph / LangChain for the ReAct agent
- OpenSearch for vector + keyword search
- CLIP (openai/clip-vit-base-patch32) for visual embeddings — stand-in for a production inference endpoint
- Cohere rerank-v3.5 for result reranking

## Setup

### Prerequisites

- Python 3.12+
- Node.js 20+
- OpenSearch instance with kNN enabled
- API keys for OpenAI, Cohere (optional), and Pexels (optional)

### Environment Variables

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

Required variables:
- `OPENAI_API_KEY` — For embeddings and agent LLM
- `OPENSEARCH_HOST`, `OPENSEARCH_USER`, `OPENSEARCH_PASSWORD` — OpenSearch connection
- `OPENSEARCH_INDEX` — Index name for image documents

Optional:
- `PEXELS_API_KEY` — For external image search
- `COHERE_API_KEY` — For result reranking

### Backend

```bash
cd backend
uv sync
uv run uvicorn main:app --reload
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

### Data Ingestion

See the `unsplash-os/` directory for a Jupyter notebook that:
1. Downloads sample images from the Unsplash Lite dataset
2. Generates rich descriptions using Claude
3. Creates CLIP embeddings
4. Indexes everything into OpenSearch

## Production Considerations

This demo runs CLIP locally for simplicity. For production:

- **Inference endpoint** — Replace local CLIP with a hosted service (SageMaker, Replicate, Modal, etc.)
- **Caching** — Add Redis/Memcached for embedding and search result caching
- **Rate limiting** — The current SlowAPI limiter is basic; consider a distributed rate limiter
- **Auth** — Add authentication for the API and frontend

## License

MIT
