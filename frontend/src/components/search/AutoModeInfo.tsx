"use client";

import { Badge } from "@/components/ui/badge";
import type { SearchMode, SearchResponse } from "@/lib/types";
import { Sparkles } from "lucide-react";

const MODE_COLORS: Record<SearchMode, string> = {
  agent: "bg-violet-100 text-violet-800",
  clip: "bg-rose-100 text-rose-800",
  hybrid: "bg-amber-100 text-amber-800",
  external: "bg-blue-100 text-blue-800",
};

const MODE_LABELS: Record<SearchMode, string> = {
  agent: "Agent",
  clip: "CLIP",
  hybrid: "Hybrid",
  external: "External",
};

interface Props {
  response: SearchResponse;
  requestedMode: SearchMode;
}

export function AgentModeInfo({ response, requestedMode }: Props) {
  const pexelsActive = response.results.some((r) => r.source === "external");
  const isAgentMode = requestedMode === "agent";

  if (!isAgentMode && !pexelsActive) return null;

  return (
    <div className="rounded-lg border border-violet-100 bg-violet-50/60 px-4 py-3 space-y-2.5 text-sm">
      {/* Row: mode badge + stats */}
      <div className="flex flex-wrap items-center gap-2">
        <Sparkles className="h-3.5 w-3.5 text-violet-400" />
        <span
          className={`inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-semibold ${MODE_COLORS[response.mode_used]}`}
        >
          {MODE_LABELS[response.mode_used]}
        </span>
        <span className="text-neutral-400">·</span>
        <span className="text-neutral-500">
          Confidence{" "}
          <span className="font-medium text-neutral-700">
            {(response.confidence * 100).toFixed(0)}%
          </span>
        </span>
        {pexelsActive && (
          <>
            <span className="text-neutral-400">·</span>
            <Badge className="bg-amber-100 text-amber-800 hover:bg-amber-100 border-0 text-xs font-medium">
              Pexels fallback active
            </Badge>
          </>
        )}
      </div>

      {/* Agent explanation */}
      {response.mode_explanation && (() => {
        const [toolCalls, summary] = response.mode_explanation!.split("\n\n");
        const hasToolCalls = toolCalls?.startsWith("▸");
        return (
          <div className="border-l-2 border-violet-300 pl-3 space-y-2">
            {hasToolCalls && (
              <pre className="text-[11px] font-mono text-violet-600 whitespace-pre-wrap leading-relaxed">
                {toolCalls}
              </pre>
            )}
            {summary && (
              <p className="text-neutral-600 italic leading-relaxed text-xs">
                {summary}
              </p>
            )}
          </div>
        );
      })()}
    </div>
  );
}
