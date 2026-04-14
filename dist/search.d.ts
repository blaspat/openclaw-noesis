/**
 * Noesis — Hybrid search pipeline
 *
 * Pipeline: vector ANN → BM25 keyword → hybrid merge → MMR rerank
 */
import { OllamaClient } from "./ollama.js";
import { NoesisDB } from "./lancedb.js";
import { MemoryType, NoesisConfig, SearchResult } from "./types.js";
export interface HybridSearchOptions {
    agentId?: string;
    memoryType?: MemoryType;
    topK?: number;
    crossAgent?: boolean;
    /** Weight for vector score in hybrid merge (0–1). BM25 gets 1 - vectorWeight. */
    vectorWeight?: number;
    /** MMR lambda: 0 = max diversity, 1 = max relevance */
    mmrLambda?: number;
}
/**
 * Full hybrid search pipeline.
 *
 * 1. Embed the query with Ollama
 * 2. Run vector ANN search (topK * 2 candidates)
 * 3. Run BM25 full-text search as fallback / supplement
 * 4. Hybrid merge (vector 60% + BM25 40%)
 * 5. MMR rerank for diversity
 */
export declare function hybridSearch(query: string, ollama: OllamaClient, db: NoesisDB, config: NoesisConfig, options?: HybridSearchOptions): Promise<SearchResult[]>;
//# sourceMappingURL=search.d.ts.map