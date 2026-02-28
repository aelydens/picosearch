"use client";

import { Badge } from "@/components/ui/badge";
import type { SearchMode, SearchResponse } from "@/lib/types";
import { Info } from "lucide-react";

const MODE_COLORS: Record<SearchMode, string> = {
  auto: "bg-purple-100 text-purple-800",
  keyword: "bg-blue-100 text-blue-800",
  semantic: "bg-emerald-100 text-emerald-800",
  hybrid: "bg-amber-100 text-amber-800",
  clip: "bg-rose-100 text-rose-800",
};

interface Props {
  response: SearchResponse;
  requestedMode: SearchMode;
}

export function AutoModeInfo({ response, requestedMode }: Props) {
  if (requestedMode !== "auto" && !response.mode_explanation) return null;

  return (
    <div className="flex flex-wrap items-start gap-3 rounded-lg border border-neutral-200 bg-neutral-50 px-4 py-3 text-sm">
      <Info className="mt-0.5 h-4 w-4 shrink-0 text-neutral-400" />
      <div className="flex-1 space-y-1">
        <div className="flex flex-wrap items-center gap-2">
          {requestedMode === "auto" && (
            <span className="text-neutral-500">Auto routed to</span>
          )}
          <span
            className={`inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-semibold ${MODE_COLORS[response.mode_used]}`}
          >
            {response.mode_used.toUpperCase()}
          </span>
          <span className="text-neutral-400">·</span>
          <span className="text-neutral-500">
            Confidence{" "}
            <span className="font-medium text-neutral-700">
              {(response.confidence * 100).toFixed(0)}%
            </span>
          </span>
          {response.results.some((r) => r.source === "external") && (
            <>
              <span className="text-neutral-400">·</span>
              <Badge variant="outline" className="text-xs">
                Pexels fallback active
              </Badge>
            </>
          )}
        </div>
        {response.mode_explanation && (
          <p className="text-neutral-500">{response.mode_explanation}</p>
        )}
      </div>
    </div>
  );
}
