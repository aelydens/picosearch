"use client";

import { Skeleton } from "@/components/ui/skeleton";
import type { ImageResult } from "@/lib/types";
import { ImageCard } from "./ImageCard";

interface Props {
  results: ImageResult[];
  isLoading: boolean;
}

function SkeletonCard() {
  return (
    <div className="overflow-hidden rounded-xl border border-neutral-200 bg-white">
      <Skeleton className="aspect-[4/3] w-full rounded-none" />
      <div className="px-3 py-2.5 space-y-2">
        <Skeleton className="h-3.5 w-full" />
        <Skeleton className="h-3.5 w-2/3" />
        <div className="flex gap-1">
          <Skeleton className="h-4 w-14 rounded-full" />
          <Skeleton className="h-4 w-14 rounded-full" />
        </div>
      </div>
    </div>
  );
}

export function ResultsGrid({ results, isLoading }: Props) {
  if (isLoading) {
    return (
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {Array.from({ length: 6 }).map((_, i) => (
          <SkeletonCard key={i} />
        ))}
      </div>
    );
  }

  if (!results.length) return null;

  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
      {results.map((r) => (
        <ImageCard key={r.id} result={r} />
      ))}
    </div>
  );
}
