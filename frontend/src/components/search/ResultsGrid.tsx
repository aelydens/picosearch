"use client";

import type { ImageResult, SearchMode } from "@/lib/types";
import { ImageCard } from "./ImageCard";
import { MorphingLoader } from "./MorphingLoader";

interface Props {
  results: ImageResult[];
  isLoading: boolean;
  mode?: SearchMode;
}

export function ResultsGrid({ results, isLoading, mode }: Props) {
  if (isLoading) {
    const label = mode === "agent" ? "Agent is thinking…" : "Searching your library…";
    return <MorphingLoader label={label} size="lg" />;
  }

  if (!results.length) return null;

  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
      {results.map((r, i) => (
        <ImageCard key={r.id} result={r} index={i} />
      ))}
    </div>
  );
}
