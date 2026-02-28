"use client";

import { AutoModeInfo } from "@/components/search/AutoModeInfo";
import { ResultsGrid } from "@/components/search/ResultsGrid";
import { SearchBar } from "@/components/search/SearchBar";
import { searchImages } from "@/lib/api";
import type { SearchMode, SearchResponse } from "@/lib/types";
import { useState } from "react";

export default function Home() {
  const [isLoading, setIsLoading] = useState(false);
  const [response, setResponse] = useState<SearchResponse | null>(null);
  const [requestedMode, setRequestedMode] = useState<SearchMode>("auto");
  const [error, setError] = useState<string | null>(null);

  async function handleSearch(query: string, mode: SearchMode) {
    setIsLoading(true);
    setError(null);
    setRequestedMode(mode);
    try {
      const data = await searchImages(query, mode);
      setResponse(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Search failed");
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <div className="min-h-screen bg-neutral-50">
      {/* Header */}
      <header className="border-b border-neutral-200 bg-white px-6 py-4">
        <div className="mx-auto flex max-w-5xl items-center gap-3">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-neutral-900">
            <span className="text-sm font-bold text-white">P</span>
          </div>
          <div>
            <h1 className="text-lg font-semibold text-neutral-900 leading-none">
              Picosearch
            </h1>
            <p className="text-xs text-neutral-400 mt-0.5">
              Multimodal asset search for marketing teams
            </p>
          </div>
        </div>
      </header>

      {/* Main */}
      <main className="mx-auto max-w-5xl px-6 py-10 space-y-8">
        {/* Hero + search */}
        <section className="space-y-6">
          {!response && !isLoading && (
            <div className="text-center space-y-2 pb-2">
              <h2 className="text-3xl font-bold tracking-tight text-neutral-900">
                Find assets by concept
              </h2>
              <p className="text-neutral-500 max-w-lg mx-auto">
                Search your media library using natural language. Switch modes
                or let Auto pick the best approach for your query.
              </p>
            </div>
          )}
          <SearchBar onSearch={handleSearch} isLoading={isLoading} />
        </section>

        {/* Error */}
        {error && (
          <div className="rounded-lg border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700">
            {error}
          </div>
        )}

        {/* Auto mode explanation */}
        {response && !isLoading && (
          <AutoModeInfo response={response} requestedMode={requestedMode} />
        )}

        {/* Results */}
        {(isLoading || (response && response.results.length > 0)) && (
          <section className="space-y-4">
            {!isLoading && response && (
              <p className="text-sm text-neutral-500">
                {response.results.length} result
                {response.results.length !== 1 ? "s" : ""}
              </p>
            )}
            <ResultsGrid
              results={response?.results ?? []}
              isLoading={isLoading}
            />
          </section>
        )}

        {/* Empty state */}
        {response && response.results.length === 0 && !isLoading && (
          <div className="py-20 text-center text-neutral-400">
            <p className="text-lg font-medium">No results found</p>
            <p className="text-sm mt-1">Try a different query or search mode</p>
          </div>
        )}
      </main>
    </div>
  );
}
