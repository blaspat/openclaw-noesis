/**
 * openclaw-noesis — Semantic memory plugin for OpenClaw
 *
 * Features:
 * - Semantic search via embeddings + vector store
 * - Cross-session memory recall
 * - Per-agent-session memory indexing
 */

import { definePluginEntry } from "openclaw/plugin-sdk/plugin-entry";
import { Type } from "@sinclair/typebox";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface MemoryEntry {
  id: string;
  sessionId: string;
  agentId: string;
  content: string;
  embedding: number[];
  createdAt: number;
  tags: string[];
}

interface SearchResult {
  id: string;
  sessionId: string;
  agentId: string;
  content: string;
  score: number;
  createdAt: number;
}

// ---------------------------------------------------------------------------
// Config defaults
// ---------------------------------------------------------------------------

interface NoesisConfig {
  embeddingProvider: string;
  embeddingModel: string;
  topK: number;
  indexOnLoad: boolean;
}

const DEFAULT_CONFIG: NoesisConfig = {
  embeddingProvider: "openai",
  embeddingModel: "text-embedding-3-small",
  topK: 5,
  indexOnLoad: true,
};

// ---------------------------------------------------------------------------
// Vector store (in-memory)
// ---------------------------------------------------------------------------

class VectorStore {
  private entries: Map<string, MemoryEntry> = new Map();
  private sessionIndex: Map<string, Set<string>> = new Map();
  private agentIndex: Map<string, Set<string>> = new Map();

  insert(entry: MemoryEntry): void {
    this.entries.set(entry.id, entry);
    if (!this.sessionIndex.has(entry.sessionId)) {
      this.sessionIndex.set(entry.sessionId, new Set());
    }
    this.sessionIndex.get(entry.sessionId)!.add(entry.id);
    if (!this.agentIndex.has(entry.agentId)) {
      this.agentIndex.set(entry.agentId, new Set());
    }
    this.agentIndex.get(entry.agentId)!.add(entry.id);
  }

  search(queryEmbedding: number[], topK: number, agentId?: string): SearchResult[] {
    const candidates = agentId
      ? Array.from(this.agentIndex.get(agentId) ?? [])
      : Array.from(this.entries.keys());

    const scored = candidates
      .map((id) => {
        const entry = this.entries.get(id)!;
        const score = cosineSimilarity(queryEmbedding, entry.embedding);
        return { id, score, entry };
      })
      .filter((s) => s.score > 0)
      .sort((a, b) => b.score - a.score)
      .slice(0, topK);

    return scored.map((s) => ({
      id: s.entry.id,
      sessionId: s.entry.sessionId,
      agentId: s.entry.agentId,
      content: s.entry.content,
      score: s.score,
      createdAt: s.entry.createdAt,
    }));
  }

  getBySession(sessionId: string): MemoryEntry[] {
    const ids = this.sessionIndex.get(sessionId) ?? [];
    return Array.from(ids).map((id) => this.entries.get(id)!).filter(Boolean);
  }

  getByAgent(agentId: string): MemoryEntry[] {
    const ids = this.agentIndex.get(agentId) ?? [];
    return Array.from(ids).map((id) => this.entries.get(id)!).filter(Boolean);
  }

  getAll(): MemoryEntry[] {
    return Array.from(this.entries.values());
  }

  delete(id: string): void {
    const entry = this.entries.get(id);
    if (!entry) return;
    this.entries.delete(id);
    this.sessionIndex.get(entry.sessionId)?.delete(id);
    this.agentIndex.get(entry.agentId)?.delete(id);
  }

  count(): number {
    return this.entries.size;
  }
}

function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length) return 0;
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  return denom === 0 ? 0 : dot / denom;
}

function generateId(): string {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
}

// ---------------------------------------------------------------------------
// Fake embedding generator
// In production, replace with a real call to the configured embedding provider.
// ---------------------------------------------------------------------------

function generateEmbedding(content: string): number[] {
  const dim = 1536;
  const vec = new Float32Array(dim);
  let hash = 0;
  for (let i = 0; i < content.length; i++) {
    hash = (hash * 31 + content.charCodeAt(i)) >>> 0;
  }
  for (let i = 0; i < dim; i++) {
    hash = (hash * 1664525 + 1013904223) >>> 0;
    vec[i] = ((hash >>> 0) / 0xffffffff) * 2 - 1;
  }
  let norm = 0;
  for (let i = 0; i < dim; i++) norm += vec[i] * vec[i];
  norm = Math.sqrt(norm) || 1;
  for (let i = 0; i < dim; i++) vec[i] /= norm;
  return Array.from(vec);
}

// ---------------------------------------------------------------------------
// Plugin entry
// ---------------------------------------------------------------------------

export default definePluginEntry({
  id: "noesis",
  name: "Noesis Memory",
  description:
    "Semantic memory plugin for OpenClaw — cross-session recall, embedding-based search, and memory indexing per agent session",
  register(api) {
    const cfg: NoesisConfig = {
      embeddingProvider:
        (api.pluginConfig["embeddingProvider"] as string) ?? DEFAULT_CONFIG.embeddingProvider,
      embeddingModel:
        (api.pluginConfig["embeddingModel"] as string) ?? DEFAULT_CONFIG.embeddingModel,
      topK: ((api.pluginConfig["topK"] as number) ?? DEFAULT_CONFIG.topK) as number,
      indexOnLoad:
        (api.pluginConfig["indexOnLoad"] as boolean) ?? DEFAULT_CONFIG.indexOnLoad,
    };

    const store = new VectorStore();

    api.logger.info(`[noesis] Plugin loaded with config: ${JSON.stringify(cfg)}`);

    // ---------------------------------------------------------------------
    // Tool: noesis_index — store a memory entry
    // ---------------------------------------------------------------------
    api.registerTool(
      {
        name: "noesis_index",
        label: "Noesis Index",
        description:
          "Store a memory entry in Noesis semantic memory. Embeds and indexes the content for cross-session semantic search.",
        parameters: Type.Object({
          content: Type.String(),
          sessionId: Type.Optional(Type.String()),
          agentId: Type.Optional(Type.String()),
          tags: Type.Optional(Type.Array(Type.String())),
        }),
        async execute(toolCallId: string, params: any) {
          const sessionId = params.sessionId ?? "default";
          const agentId = params.agentId ?? api.id ?? "noesis";
          const content = params.content as string;
          const tags: string[] = params.tags ?? [];

          const embedding = generateEmbedding(content);

          const entry: MemoryEntry = {
            id: generateId(),
            sessionId,
            agentId,
            content,
            embedding,
            createdAt: Date.now(),
            tags,
          };

          store.insert(entry);
          api.logger.debug(`[noesis] Indexed entry ${entry.id} for agent=${agentId}`);

          return {
            content: [{ type: "text", text: `Memory indexed (id: ${entry.id}). ${store.count()} total entries.` }],
            details: { id: entry.id, count: store.count() },
          };
        },
      },
      { optional: true }
    );

    // ---------------------------------------------------------------------
    // Tool: noesis_search — semantic search across memories
    // ---------------------------------------------------------------------
    api.registerTool(
      {
        name: "noesis_search",
        label: "Noesis Search",
        description:
          "Search Noesis semantic memory. Returns the top-K most semantically similar memory entries.",
        parameters: Type.Object({
          query: Type.String(),
          agentId: Type.Optional(Type.String()),
          topK: Type.Optional(Type.Number()),
        }),
        async execute(toolCallId: string, params: any) {
          const query = params.query as string;
          const agentId = params.agentId as string | undefined;
          const topK = (params.topK as number | undefined) ?? cfg.topK;

          const queryEmbedding = generateEmbedding(query);
          const results = store.search(queryEmbedding, topK, agentId);

          if (results.length === 0) {
            return {
              content: [{ type: "text", text: "No results found." }],
              details: { count: 0 },
            };
          }

          const lines = results
            .map((r, i) => `${i + 1}. [score=${r.score.toFixed(3)}] ${r.content}`)
            .join("\n");

          return {
            content: [
              {
                type: "text",
                text: `Found ${results.length} result(s):\n${lines}`,
              },
            ],
            details: { count: results.length, results },
          };
        },
      },
      { optional: true }
    );

    // ---------------------------------------------------------------------
    // Tool: noesis_recall — cross-session memory recall
    // ---------------------------------------------------------------------
    api.registerTool(
      {
        name: "noesis_recall",
        label: "Noesis Recall",
        description:
          "Recall memories from a specific session or agent. Useful for cross-session continuity.",
        parameters: Type.Object({
          sessionId: Type.Optional(Type.String()),
          agentId: Type.Optional(Type.String()),
          limit: Type.Optional(Type.Number()),
        }),
        async execute(toolCallId: string, params: any) {
          const sessionId = params.sessionId as string | undefined;
          const agentId = params.agentId as string | undefined;
          const limit = (params.limit as number | undefined) ?? 50;

          let entries: MemoryEntry[];
          if (sessionId) {
            entries = store.getBySession(sessionId);
          } else if (agentId) {
            entries = store.getByAgent(agentId);
          } else {
            entries = store.getAll();
          }

          entries = entries.slice(0, limit);

          if (entries.length === 0) {
            return {
              content: [{ type: "text", text: "No memories found." }],
              details: { count: 0 },
            };
          }

          const lines = entries
            .map((e) => `[${new Date(e.createdAt).toISOString()}] ${e.content}`)
            .join("\n");

          return {
            content: [
              {
                type: "text",
                text: `Recalled ${entries.length} memory entries:\n${lines}`,
              },
            ],
            details: { count: entries.length },
          };
        },
      },
      { optional: true }
    );

    // ---------------------------------------------------------------------
    // Tool: noesis_status — plugin status
    // ---------------------------------------------------------------------
    api.registerTool(
      {
        name: "noesis_status",
        label: "Noesis Status",
        description: "Returns the current status and statistics of the Noesis memory plugin.",
        parameters: Type.Object({}),
        async execute(toolCallId: string, params: any) {
          return {
            content: [
              {
                type: "text",
                text: `Noesis Memory plugin is running. Total entries: ${store.count()}`,
              },
            ],
            details: { totalEntries: store.count() },
          };
        },
      },
      { optional: true }
    );

    api.logger.info(`[noesis] Plugin registered successfully`);
  },
});
