export type SearchMode = "auto" | "keyword" | "semantic" | "hybrid" | "clip";

export interface ImageResult {
  id: string;
  url: string;
  thumbnail_url: string;
  description: string;
  score: number;
  source: "internal" | "external";
  tags: string[];
}

export interface SearchResponse {
  results: ImageResult[];
  mode_used: SearchMode;
  mode_explanation: string | null;
  confidence: number;
}
