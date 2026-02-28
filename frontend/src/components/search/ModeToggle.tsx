"use client";

import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import type { SearchMode } from "@/lib/types";

const MODES: { value: SearchMode; label: string; title: string }[] = [
  { value: "auto", label: "Auto", title: "Let the agent pick the best mode" },
  { value: "keyword", label: "Keyword", title: "Exact keyword matching (BM25)" },
  { value: "semantic", label: "Semantic", title: "Dense vector similarity" },
  { value: "hybrid", label: "Hybrid", title: "Keyword + semantic fusion" },
  { value: "clip", label: "CLIP", title: "Vision-language embedding search" },
];

interface Props {
  value: SearchMode;
  onChange: (mode: SearchMode) => void;
}

export function ModeToggle({ value, onChange }: Props) {
  return (
    <ToggleGroup
      type="single"
      value={value}
      onValueChange={(v) => v && onChange(v as SearchMode)}
      className="gap-1"
    >
      {MODES.map((m) => (
        <ToggleGroupItem
          key={m.value}
          value={m.value}
          title={m.title}
          className="h-8 px-3 text-xs font-medium data-[state=on]:bg-neutral-900 data-[state=on]:text-white"
        >
          {m.label}
        </ToggleGroupItem>
      ))}
    </ToggleGroup>
  );
}
