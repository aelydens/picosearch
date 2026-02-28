# Certification Challenge Writeup

## Task 1: Problem, Audience, and Scope

### 1-sentence problem description

Marketers waste time manually searching asset libraries with keyword search and filters, and can miss relevant assets that are described differently, are missing metadata, or have qualities that the text search does not capture.

### Why this is a problem for your specific user

This is a problem because marketing teams can have huge libraries, with thousands or tens of thousands of images accumulated over time. Often, these images don't have great metadata, rich descriptions or meaningful tags. With traditional search and filters, good matches can be difficult to find. Additionally, traditional keyword search fails to find images that match visual / aesthetic ideas like "minimalist" or "vibrant backdrop". Poor search tools can lead marketers to waste hours searching for good media matches, or spend money on supplementing their media when they might otherwise be able to find a good existing match in their library.

### Evaluation questions / input-output pairs

I created 10 input-output pairs to assess the application:

- "sunset over water" - images with sunsets, water, warm colors
- "portrait of woman" - portrait photos of women
- "running dog" - images with dogs in action
- "bowl of fruit" - images with various fruit in bowls
- "mountain landscapes" - images containing a mountain
- "cityscape" - images containing cities
- "vibrant background" - images with bright, colorful backgrounds
- "cold minimal" - cold, icy, greyscale
- "cozy autumn" - warm colors, fall imagery, comfortable atmosphere
- "strong contrast" - images with strong contrast

## Task 2: Propose a Solution

### Solution proposal

My proposed solution is to build a multimodal search tool that combines 3 retrieval strategies: keyword, semantic, and visual. I will add an agent layer than can intelligently select the best retrieval tool based on the search query. I plan to have a tool that also allows the agent to search via an external stock image API (Pexels) to augment the search results in case the user is searching for something not adequately represented in their media library.

### Infrastructure diagram

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Next.js       │────▶│   FastAPI       │────▶│   OpenSearch    │
│   Frontend      │     │   Backend       │     │   (kNN + BM25)  │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                              │
                              ├──▶ OpenAI (GPT-4o-mini for agent, text-embedding-3-small for description and query embeddings)
                              ├──▶ CLIP (currently local for image and query embeddings. For production, we'd need to swap this for an inference endpoint or another option)
                              ├──▶ Cohere (reranking after initial retrieval)
                              └──▶ Pexels (external images)
```

### Tooling choices

- **OpenSearch**: Supports both BM25 keyword search and kNN vector search in one index, avoiding separate vector DB.
- **CLIP**: Multimodal embeddings that understand visual concepts. I used this because it was easy to bootstrap, but would swap this out for another option when productionizing.
- **OpenAI text-embedding-3-small**: Semantic embeddings for description-based search. I chose this embedding model because it was cheap, I had used it before, and performance is good enough for my purposes.
- **LangGraph ReAct agent**: My agent is implemented with Langgraph's create_react_agent. This is a convenient, quick way to add a reasoning loop to do the search tool selection.
- **Cohere rerank**: I added cohere reranking to improve precision and results.
- **NextJS / FastAPI**: I used a standard combination of FastAPI and NextJS. Nothing fancy here, just choosing familiar tools.

### RAG and agent components

The "RAG" component of my project is not traditional document search, instead this is retrieval-augmented search where retrieved images are the output, not context for generation. I believe this solves my user's problems best, so this is why I chose to go this route rather than strictly adhere to the rubric. However, the retrieved images are output to users. For RAGAS assessment, I added a "reasoning" layer in order to assess the pipeline, which could be added if it was desired by users. The agent component of my application is a Langgraph agent with tools (search_hybrid, search_clip, search_external). The agent decides which tool to call based on the query analysis. This works well to solve my user's problem.

---

## Task 3: Dealing with the Data

### Data sources and external APIs

I used a subset of 2000 images from the Unsplash Lite open-source image dataset. This dataset includes descriptions and tags as well. I used the Pexels API as the source for the "external search" tool, so that the agent could decide to fall back to a broader stock image search if good matches weren't found in the media library itself. I embedded the images with the CLIP embedding model, and the image descriptions with the text-embedding-3-small model from OpenAI. This allows me to have richer semantic / visual search across the media library, as well as supporting keyword search on image text metadata.

### Chunking strategy

I did not need a chunking strategy, since each image is a single document and my descriptions are short. Each document stored in OpenSearch includes image URL, description, CLIP embedding (512d), text embedding (1536d), and some other fields from Unsplash. The embeddings are pre-computed at index time (except for query embeddings, obviously), where I just iterated over the Unsplash dataset I had downloaded, generated the embeddings, then used the `opensearch-py` library to batch insert documents into my OpenSearch index.

---

## Task 4: Build End-to-End Prototype

[Loom walkthrough](https://www.loom.com/share/4d1f3f778e594c0b9de792dbc0af1692)

## Task 5: Evals and Task 6: Improving Your Prototype

### RAGAS evaluation

For my answers to the Evals and Improving Your Prototype, please see my [RAGAS Evals Jupyter notebook](backend/notebooks/ragas_eval.ipynb).

## Task 7: Next Steps

My original plan was to work on a different project for demo day, but if I chose to extend this project, there are several things I'd need to do to get it ready for production, but I would certainly keep the dense vector retrieval component in this project. That's because dense retrieval is essential for my use case. Keyword search alone doesn't satisfy my user's needs. The hybrid approach (keyword + dense) outperforms either alone, and the agent routing layer adds flexibility without sacrificing any single strategy.
