"use client";

import { Badge } from "@/components/ui/badge";
import type { ImageResult } from "@/lib/types";
import Image from "next/image";

interface Props {
  result: ImageResult;
}

export function ImageCard({ result }: Props) {
  const scorePercent = (result.score * 100).toFixed(0);
  const scoreColor =
    result.score >= 0.8
      ? "text-emerald-600"
      : result.score >= 0.6
        ? "text-amber-600"
        : "text-rose-500";

  return (
    <div className="group relative overflow-hidden rounded-xl border border-neutral-200 bg-white shadow-sm transition-shadow hover:shadow-md">
      {/* Image */}
      <div className="relative aspect-[4/3] w-full overflow-hidden bg-neutral-100">
        <Image
          src={result.thumbnail_url}
          alt={result.description}
          fill
          sizes="(max-width: 640px) 100vw, (max-width: 1024px) 50vw, 33vw"
          className="object-cover transition-transform duration-300 group-hover:scale-105"
        />
        {/* Source badge overlay */}
        <div className="absolute left-2 top-2">
          <Badge
            className={
              result.source === "external"
                ? "bg-orange-500/90 text-white hover:bg-orange-500"
                : "bg-neutral-900/80 text-white hover:bg-neutral-900"
            }
          >
            {result.source === "external" ? "Pexels" : "Internal"}
          </Badge>
        </div>
      </div>

      {/* Info */}
      <div className="px-3 py-2.5 space-y-1.5">
        <p className="text-sm text-neutral-700 line-clamp-2 leading-snug">
          {result.description}
        </p>

        <div className="flex items-center justify-between">
          <div className="flex flex-wrap gap-1">
            {result.tags.slice(0, 3).map((tag) => (
              <span
                key={tag}
                className="rounded-full bg-neutral-100 px-2 py-0.5 text-[10px] text-neutral-500"
              >
                {tag}
              </span>
            ))}
          </div>
          <span className={`text-xs font-semibold tabular-nums ${scoreColor}`}>
            {scorePercent}%
          </span>
        </div>
      </div>
    </div>
  );
}
