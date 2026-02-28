from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from agent import SearchRequest, run_search_agent

app = FastAPI(title="Picosearch API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://*.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/search")
async def search(request: SearchRequest):
    return await run_search_agent(request)
