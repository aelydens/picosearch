"use client";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import type { SearchMode } from "@/lib/types";
import { Search } from "lucide-react";
import { type FormEvent, useState } from "react";
import { ModeToggle } from "./ModeToggle";

interface Props {
  onSearch: (query: string, mode: SearchMode) => void;
  isLoading: boolean;
}

export function SearchBar({ onSearch, isLoading }: Props) {
  const [query, setQuery] = useState("");
  const [mode, setMode] = useState<SearchMode>("auto");

  function handleSubmit(e: FormEvent) {
    e.preventDefault();
    const trimmed = query.trim();
    if (trimmed) onSearch(trimmed, mode);
  }

  return (
    <form onSubmit={handleSubmit} className="w-full space-y-3">
      <div className="flex gap-2">
        <Input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search for images by concept, mood, or subject…"
          className="h-11 text-base"
          disabled={isLoading}
        />
        <Button
          type="submit"
          disabled={isLoading || !query.trim()}
          className="h-11 px-5"
        >
          <Search className="mr-2 h-4 w-4" />
          Search
        </Button>
      </div>
      <div className="flex items-center gap-2">
        <span className="text-xs text-neutral-500 font-medium">Mode:</span>
        <ModeToggle value={mode} onChange={setMode} />
      </div>
    </form>
  );
}
