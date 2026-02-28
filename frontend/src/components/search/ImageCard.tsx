"use client";

import { Badge } from "@/components/ui/badge";
import type { ImageResult } from "@/lib/types";
import { motion } from "framer-motion";
import Image from "next/image";

interface Props {
  result: ImageResult;
  index?: number;
}

export function ImageCard({ result, index = 0 }: Props) {
  const scorePercent = (result.score * 100).toFixed(0);
  const scoreColor =
    result.score >= 0.8
      ? "text-violet-600"
      : result.score >= 0.6
        ? "text-amber-500"
        : "text-rose-500";

  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.35, delay: index * 0.05, ease: "easeOut" }}
      className="group relative overflow-hidden rounded-xl border border-violet-100 bg-white shadow-sm transition-shadow hover:shadow-md hover:shadow-violet-100"
    >
      <div className="relative aspect-[4/3] w-full overflow-hidden bg-violet-50">
        <Image
          src={result.thumbnail_url}
          alt={result.description}
          fill
          sizes="(max-width: 640px) 100vw, (max-width: 1024px) 50vw, 33vw"
          className="object-cover transition-transform duration-300 group-hover:scale-105"
        />
        <div className="absolute left-2 top-2">
          <Badge
            className={
              result.source === "external"
                ? "bg-amber-400 text-amber-950 hover:bg-amber-400"
                : "bg-violet-600/90 text-white hover:bg-violet-600"
            }
          >
            {result.source === "external" ? "Pexels" : "Internal"}
          </Badge>
        </div>
      </div>

      <div className="px-3 py-2.5 space-y-1.5">
        <p className="text-sm text-neutral-700 line-clamp-2 leading-snug">
          {result.description}
        </p>

        <div className="flex items-center justify-between">
          <div className="flex flex-wrap gap-1">
            {result.tags.slice(0, 3).map((tag) => (
              <span
                key={tag}
                className="rounded-full bg-violet-50 px-2 py-0.5 text-[10px] text-violet-500"
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
    </motion.div>
  );
}
