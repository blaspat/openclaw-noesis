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
export async function hybridSearch(
  query: string,
  ollama: OllamaClient,
  db: NoesisDB,
  config: NoesisConfig,
  options: HybridSearchOptions = {}
): Promise<SearchResult[]> {
  const topK = options.topK ?? config.topK;
  const vectorWeight = options.vectorWeight ?? 0.6;
  const bm25Weight = 1 - vectorWeight;
  const mmrLambda = options.mmrLambda ?? 0.7;

  // 1. Embed the query
  const queryEmbedding = await ollama.embed(query);

  // 2. Vector search (expanded candidate set)
  const vectorResults = await db.vectorSearch(
    queryEmbedding,
    topK * 2,
    options.agentId,
    options.memoryType,
    options.crossAgent
  );

  // 3. BM25 full-text search
  const bm25Results = await db.fullTextSearch(
    query,
    topK,
    options.agentId,
    options.memoryType,
    options.crossAgent
  );

  // 4. Hybrid merge (active table only — expired entries live in archive, not here)
  const scoreMap = new Map<string, { result: typeof vectorResults[0]; score: number }>();

  // Index vector scores
  for (const r of vectorResults) {
    scoreMap.set(r.id, { result: r, score: r.score * vectorWeight });
  }

  // Add BM25 scores (union merge)
  for (const r of bm25Results) {
    const existing = scoreMap.get(r.id);
    if (existing) {
      existing.score += r.score * bm25Weight;
    } else {
      scoreMap.set(r.id, { result: r, score: r.score * bm25Weight });
    }
  }

  // Sort by hybrid score
  const merged = Array.from(scoreMap.values())
    .sort((a, b) => b.score - a.score)
    .slice(0, topK * 2);

  // 5. Always include high-priority entries (priority >= 75) regardless of score
  const priorityBoost = new Set(merged.map((m) => m.result.id));
  const highPriorityResults = await db.vectorSearch(queryEmbedding, topK, options.agentId, options.memoryType, options.crossAgent);
  for (const r of highPriorityResults) {
    if (r.priority >= 75 && !priorityBoost.has(r.id)) {
      merged.push({ result: r, score: 0 }); // score 0 but will be sorted to top by priority boost
    }
  }

  // Sort again with priority boost
  const withPriority = merged.sort((a, b) => {
    const ap = (a.result as any).priority ?? 0;
    const bp = (b.result as any).priority ?? 0;
    if (ap >= 75 && bp < 75) return -1;
    if (bp >= 75 && ap < 75) return 1;
    return b.score - a.score;
  }).slice(0, topK * 2);

  // 6. MMR rerank
  const reranked = mmrRerank(withPriority.map((m) => m.result), queryEmbedding, topK, mmrLambda);

  return reranked.map((r) => ({
    id: r.id,
    agentId: r.agentId,
    sessionId: r.sessionId,
    content: r.content,
    memoryType: r.memoryType as MemoryType,
    priority: (r as any).priority ?? 0,
    expiresAt: (r as any).expiresAt ?? 0,
    createdAt: r.createdAt,
    sourcePath: r.sourcePath,
    score: scoreMap.get(r.id)?.score ?? r.score,
    tags: r.tags,
  }));
}

/**
 * Maximal Marginal Relevance reranking.
 *
 * Balances relevance to query vs. diversity among selected results.
 *
 * @param candidates - Pre-ranked candidate results with scores
 * @param queryEmbedding - The original query embedding
 * @param topK - How many to return
 * @param lambda - 0 = max diversity, 1 = max relevance (default 0.7)
 */
function mmrRerank(
  candidates: Array<{ id: string; agentId: string; sessionId: string; content: string; memoryType: string; createdAt: number; sourcePath: string; tags: string[]; score: number }>,
  queryEmbedding: number[],
  topK: number,
  lambda: number
): typeof candidates {
  if (candidates.length === 0) return [];
  if (candidates.length <= topK) return candidates;

  // Use content-based pseudo-embeddings (BM25 term vectors) for MMR diversity
  // Since we don't have stored embeddings in search results, we use content TF-IDF approximation
  const contentVectors = candidates.map((c) => buildTermVector(c.content));
  const queryTermVector = buildTermVector(candidates.map((c) => c.content).join(" "));

  const selected: number[] = [];
  const remaining = new Set(candidates.map((_, i) => i));

  while (selected.length < topK && remaining.size > 0) {
    let bestIdx = -1;
    let bestScore = -Infinity;

    for (const idx of remaining) {
      const relevanceScore = candidates[idx].score;

      // Max similarity to already-selected candidates (penalize redundancy)
      let maxSimToSelected = 0;
      for (const selIdx of selected) {
        const sim = cosineSimilarity(contentVectors[idx], contentVectors[selIdx]);
        if (sim > maxSimToSelected) maxSimToSelected = sim;
      }

      const mmrScore = lambda * relevanceScore - (1 - lambda) * maxSimToSelected;
      if (mmrScore > bestScore) {
        bestScore = mmrScore;
        bestIdx = idx;
      }
    }

    if (bestIdx === -1) break;
    selected.push(bestIdx);
    remaining.delete(bestIdx);
  }

  return selected.map((i) => candidates[i]);
}

/**
 * Build a sparse TF-IDF-like term vector for MMR diversity calculation.
 */
function buildTermVector(text: string): Map<string, number> {
  const words = text.toLowerCase().replace(/[^a-z0-9\s]/g, " ").split(/\s+/).filter(Boolean);
  const counts = new Map<string, number>();
  for (const word of words) {
    if (word.length > 2) { // skip tiny words
      counts.set(word, (counts.get(word) ?? 0) + 1);
    }
  }
  // Normalize
  const total = words.length || 1;
  for (const [k, v] of counts) counts.set(k, v / total);
  return counts;
}

/**
 * Cosine similarity between two sparse term vectors.
 */
function cosineSimilarity(a: Map<string, number>, b: Map<string, number>): number {
  let dot = 0;
  let normA = 0;
  let normB = 0;

  for (const [k, v] of a) {
    normA += v * v;
    const bv = b.get(k) ?? 0;
    dot += v * bv;
  }
  for (const [, v] of b) {
    normB += v * v;
  }

  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  return denom === 0 ? 0 : dot / denom;
}
