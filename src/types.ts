/**
 * Noesis — Type definitions
 */

export interface MemoryEntry {
  id: string;
  agentId: string;
  sessionId: string;
  content: string;
  chunk: string;
  embedding: number[];
  tags: string[];
  memoryType: MemoryType;
  /** 0 = lowest, 100 = highest. High-priority memories surface in every recall regardless of score. */
  priority?: number;
  /** Unix timestamp (ms) when this entry expires. 0 = never. */
  expiresAt?: number;
  createdAt: number;
  sourcePath: string;
  checksum: string;
}

export type MemoryType = "fact" | "decision" | "preference" | "context" | "session";

export type MemoryPriority = 0 | 25 | 50 | 75 | 100;

export interface SearchResult {
  id: string;
  agentId: string;
  sessionId: string;
  content: string;
  memoryType: MemoryType;
  priority: number;
  createdAt: number;
  expiresAt: number;
  sourcePath: string;
  score: number;
  tags: string[];
}

export interface NoesisConfig {
  lanceDbPath: string;
  ollamaEndpoint: string;
  embeddingModel: string;
  chunkSize: number;
  chunkOverlap: number;
  topK: number;
  autoMigrate: boolean;
  indexQmdSessions: boolean;
  watchMemoryDirs: boolean;
  gitLfsEnabled: boolean;
  gitLfsRepo: string;
  annNprobe: number;
  annNumSubvectors: number;
  /** Default TTL in days for new entries. 0 = no expiry. Default: 90 */
  defaultTtlDays: number;
  /** Enable automatic TTL cleanup on startup. Default: true */
  autoCleanup: boolean;
  /** Run TTL cleanup on this interval (hours). 0 = disabled. Default: 6 */
  cleanupIntervalHours: number;
  /** Register Noesis as the active context engine (assemble + ingest hooks). Default: true */
  contextEngineEnabled: boolean;
  /** Minimum priority for Assemble hook injection. Entries >= this always enter context. Default: 75 */
  assembleInjectPriority: number;
  /** Max entries to inject via Assemble hook. Default: 20 */
  assembleMaxEntries: number;
  /** Max age (days) for Assemble injection. 0 = no limit. Default: 30 */
  assembleMaxAgeDays: number;
  /** Path to dedicated error log file. Default: ~/.openclaw/noesis/error.log */
  errorLogPath: string;
  /** Hybrid search: weight for vector score (0–1). BM25 gets 1-vectorWeight. Default: 0.6. Note: for short factual queries, 1.0 (pure vector) may outperform hybrid. */
  vectorWeight: number;
  /** Opt-in cross-encoder reranking using Ollama embeddings. More accurate relevance scoring but doubles embed API calls at query time. Default: false */
  rerank: boolean;
  /** Optional external reranker API endpoint (e.g. Jina cloud reranker). When set, rerank uses this endpoint instead of local Ollama approximation. */
  rerankerEndpoint?: string;
}

export const DEFAULT_CONFIG: NoesisConfig = {
  lanceDbPath: "~/.openclaw/noesis/db",
  ollamaEndpoint: "http://localhost:11434",
  embeddingModel: "nomic-embed-text",
  chunkSize: 512,
  chunkOverlap: 64,
  topK: 6,
  autoMigrate: false,
  indexQmdSessions: true,
  watchMemoryDirs: false,
  gitLfsEnabled: false,
  gitLfsRepo: "<username>/openclaw-noesis-data",
  annNprobe: 16,
  annNumSubvectors: 96,
  defaultTtlDays: 90,
  autoCleanup: true,
  cleanupIntervalHours: 6,
  contextEngineEnabled: true,
  assembleInjectPriority: 75,
  assembleMaxEntries: 20,
  assembleMaxAgeDays: 30,
  errorLogPath: "~/.openclaw/noesis/error.log",
  vectorWeight: 0.6,
  rerank: false,
};

export interface ImportResult {
  indexed: number;
  skipped: number;
  errors: number;
  agentId: string;
}

export interface NoesisStats {
  totalEntries: number;
  byAgent: Record<string, number>;
  byMemoryType: Record<string, number>;
  byPriority: Record<string, number>;
  dbPath: string;
  embeddingModel: string;
  ollamaEndpoint: string;
  expiredEntries: number;
  expiredAndDeleted: number;
  archiveEntries: number;
  byAgentArchive: Record<string, number>;
  byMemoryTypeArchive: Record<string, number>;
}
