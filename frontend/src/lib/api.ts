import type { SearchMode, SearchResponse } from "./types";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export async function searchImages(
  query: string,
  mode: SearchMode,
  limit = 12,
): Promise<SearchResponse> {
  const res = await fetch(`${API_URL}/search`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, mode, limit }),
  });

  if (!res.ok) {
    throw new Error("Search failed. Please try again.");
  }

  const data = (await res.json()) as SearchResponse;
  console.log(
    `[Picosearch] mode=${data.mode_used} confidence=${(data.confidence * 100).toFixed(0)}% results=${data.results.length} pexels=${data.results.some((r) => r.source === "external")}`,
    data,
  );
  return data;
}
