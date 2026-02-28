"use client";

import { AgentModeInfo } from "@/components/search/AutoModeInfo";
import { ResultsGrid } from "@/components/search/ResultsGrid";
import { SearchBar } from "@/components/search/SearchBar";
import { searchImages } from "@/lib/api";
import type { SearchMode, SearchResponse } from "@/lib/types";
import { motion, AnimatePresence } from "framer-motion";
import { useState } from "react";

export default function Home() {
  const [isLoading, setIsLoading] = useState(false);
  const [response, setResponse] = useState<SearchResponse | null>(null);
  const [requestedMode, setRequestedMode] = useState<SearchMode>("agent");
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
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b border-violet-100 bg-white/80 backdrop-blur-sm px-6 py-4 sticky top-0 z-10">
        <div className="mx-auto flex max-w-5xl items-center gap-3">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-gradient-to-br from-violet-500 to-violet-700 shadow-sm shadow-violet-200">
            <span className="text-sm font-bold text-white">p</span>
          </div>
          <div>
            <h1 className="text-lg font-semibold text-neutral-900 leading-none">
              Picosearch
            </h1>
            <p className="text-xs text-violet-400 mt-0.5">
              Multimodal asset search for marketing teams
            </p>
          </div>
        </div>
      </header>

      {/* Main */}
      <main className="mx-auto max-w-5xl px-6 py-10 space-y-8">
        <section className="space-y-6">
          <AnimatePresence>
            {!response && !isLoading && (
              <motion.div
                className="text-center space-y-3 pb-2"
                initial={{ opacity: 0, y: 16 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -8 }}
                transition={{ duration: 0.4, ease: "easeOut" }}
              >
                <h2 className="text-4xl font-bold tracking-tight bg-gradient-to-r from-violet-600 via-violet-500 to-amber-500 bg-clip-text text-transparent">
                  Find assets by concept
                </h2>
                <p className="text-neutral-500 max-w-lg mx-auto">
                  Search your media library using natural language. Switch modes
                  or let Auto pick the best approach for your query.
                </p>
              </motion.div>
            )}
          </AnimatePresence>

          <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4, delay: 0.1, ease: "easeOut" }}
          >
            <SearchBar onSearch={handleSearch} isLoading={isLoading} />
          </motion.div>
        </section>

        {/* Error */}
        <AnimatePresence>
          {error && (
            <motion.div
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              className="rounded-lg border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700"
            >
              {error}
            </motion.div>
          )}
        </AnimatePresence>

        {/* Auto mode info */}
        <AnimatePresence>
          {response && !isLoading && (
            <motion.div
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.3 }}
            >
              <AgentModeInfo response={response} requestedMode={requestedMode} />
            </motion.div>
          )}
        </AnimatePresence>

        {/* Results */}
        {(isLoading || (response && response.results.length > 0)) && (
          <section className="space-y-4">
            {!isLoading && response && (
              <p className="text-sm text-violet-400 font-medium">
                {response.results.length} result
                {response.results.length !== 1 ? "s" : ""}
              </p>
            )}
            <ResultsGrid
              results={response?.results ?? []}
              isLoading={isLoading}
              mode={requestedMode}
            />
          </section>
        )}

        {/* Empty state */}
        <AnimatePresence>
          {response && response.results.length === 0 && !isLoading && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="py-20 text-center"
            >
              <p className="text-lg font-medium text-neutral-400">No results found</p>
              <p className="text-sm mt-1 text-neutral-400">Try a different query or search mode</p>
            </motion.div>
          )}
        </AnimatePresence>
      </main>
    </div>
  );
}
