"use client";

import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import type { SearchMode } from "@/lib/types";

const MODES: { value: SearchMode; label: string; title: string }[] = [
  { value: "agent", label: "Agent", title: "LLM agent picks the best tool(s) for your query" },
  { value: "clip", label: "CLIP", title: "Vision-language embedding search" },
  { value: "hybrid", label: "Hybrid", title: "Keyword + CLIP fusion" },
  { value: "external", label: "External", title: "Pexels stock photos only" },
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
          className="h-8 px-3 text-xs font-medium data-[state=on]:bg-amber-400 data-[state=on]:text-amber-950 data-[state=on]:shadow-sm"
        >
          {m.label}
        </ToggleGroupItem>
      ))}
    </ToggleGroup>
  );
}
