/**
 * Noesis — Local-first semantic memory plugin for OpenClaw
 *
 * Memory slot: plugins.slots.memory = "noesis"
 * Stack: LanceDB (vector store) + HuggingFace Transformers.js (embeddings) + BM25 + MMR
 * No cloud. No API keys. Fully local.
 */

import { definePluginEntry } from "openclaw/plugin-sdk/plugin-entry";
import { Type } from "@sinclair/typebox";
import { randomUUID } from "crypto";
import os from "os";
import path from "path";
import fs from "fs";

import { DEFAULT_CONFIG, MemoryEntry, MemoryType, NoesisConfig } from "./types.js";
import { autoConfigTransformers, EmbeddingClient, contentChecksum, chunkText } from "./transformers.js";
import { NoesisDB } from "./lancedb.js";
import { hybridSearch, cosineSimilarityDense } from "./search.js";
import { importMarkdownFiles, importAllAgents } from "./migrator.js";
import { SessionWatcher, startSessionScanner, SessionScanner } from "./watcher.js";
import {
  logError,
  logFatal,
  installGlobalErrorHandlers,
  configureErrorLog,
  withErrorLog,
} from "./logger.js";

// ─── plugin state (module-level singletons) ────────────────────────────────

// ─── noise patterns — messages matching these are dropped before indexing ────────

const NOISE_PATTERNS = [
  "HEARTBEAT_OK",
  "A scheduled reminder has been triggered",
  "Execute your Session Startup sequence now",
  "Queued messages from",
  "Assistant output is not needed",
  "NO_REPLY",
];

export function isNoiseMessage(text: string): boolean {
  const trimmed = text.trim();
  return NOISE_PATTERNS.some(
    (p) => trimmed === p || trimmed.startsWith(p)
  );
}

// ─── last-cleanup timestamp helpers ────────────────────────────────────────────

const LAST_CLEANUP_FILE = path.join(os.homedir(), ".openclaw", "noesis", ".last-cleanup");

/**
 * Read the last cleanup timestamp from disk. Returns 0 if file doesn't exist.
 */
function getLastCleanupTimestamp(): number {
  try {
    if (fs.existsSync(LAST_CLEANUP_FILE)) {
      return Number(fs.readFileSync(LAST_CLEANUP_FILE, "utf-8").trim());
    }
  } catch {
    // File missing or unreadable — treat as 0 (never cleaned)
  }
  return 0;
}

/**
 * Persist the last cleanup timestamp to disk.
 */
function setLastCleanupTimestamp(ts: number): void {
  try {
    const dir = path.dirname(LAST_CLEANUP_FILE);
    if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
    fs.writeFileSync(LAST_CLEANUP_FILE, String(ts), "utf-8");
  } catch {
    // Best-effort — non-fatal if this fails
  }
}

/**
 * Log a warning if the gap since last cleanup is unexpectedly large (> 24h).
 */
function warnOnLargeCleanupGap(log: (msg: string) => void): void {
  const last = getLastCleanupTimestamp();
  if (last === 0) return; // never cleaned — first run, normal
  const gapHours = (Date.now() - last) / (1000 * 60 * 60);
  if (gapHours > 24) {
    log(`[noesis] Warning: ${gapHours.toFixed(1)}h since last TTL cleanup. ` +
      "Expired entries may have accumulated. Consider restarting the gateway to trigger auto-cleanup.");
  }
}

let db: NoesisDB | null = null;
let ollama: EmbeddingClient | null = null;
let sessionScanner: SessionScanner | null = null;
let resolvedConfig: NoesisConfig = { ...DEFAULT_CONFIG };
let initialized = false;
let cleanupIntervalId: ReturnType<typeof setInterval> | null = null;
let optimizeIntervalId: ReturnType<typeof setInterval> | null = null;

// ─── plugin entry ──────────────────────────────────────────────────────────

export default definePluginEntry({
  id: "noesis",
  name: "Noesis Memory",
  description:
    "Local-first semantic memory for OpenClaw — LanceDB + Transformers.js (WASM/SIMD), no cloud required. Hybrid search (vector + BM25 + MMR), cross-agent recall, QMD session indexing, and memory slot integration.",

  register(api) {
    // Merge user config with defaults
    resolvedConfig = buildConfig(api.pluginConfig ?? {});

    api.logger.info(`[noesis] Starting with config: model=${resolvedConfig.embeddingModel}, db=${resolvedConfig.lanceDbPath}`);

    // Initialize async — plugin tools remain available, they'll block until ready
    initPlugin(resolvedConfig, api.logger.info.bind(api.logger)).catch((err) => {
      api.logger.error(`[noesis] Initialization failed: ${err}`);
    });

    // ── Tool: noesis_index ────────────────────────────────────────────────
    api.registerTool(
      {
        name: "noesis_index",
        label: "Noesis Index",
        description:
          "Store a memory entry in Noesis semantic memory. Embeds and indexes the content for future semantic search. Idempotent — duplicate content is skipped automatically.",
        parameters: Type.Object({
          content: Type.String({ description: "Text content to store" }),
          agentId: Type.Optional(Type.String({ description: "Agent ID (defaults to current agent)" })),
          sessionId: Type.Optional(Type.String({ description: "Session ID" })),
          tags: Type.Optional(Type.Array(Type.String(), { description: "Optional tags" })),
          memoryType: Type.Optional(
            Type.Union(
              [
                Type.Literal("fact"),
                Type.Literal("decision"),
                Type.Literal("preference"),
                Type.Literal("context"),
                Type.Literal("session"),
              ],
              { description: "Memory type: fact | decision | preference | context | session" }
            )
          ),
          priority: Type.Optional(
            Type.Number({
              description: "Priority 0-100. High priority (>=75) entries always surface in recall. Default: 0",
            })
          ),
          ttlDays: Type.Optional(
            Type.Number({
              description: "Time-to-live in days. Entry auto-expires after this many days. 0 = never. Default: from config (90)",
            })
          ),
          sourcePath: Type.Optional(Type.String({ description: "Source file path if applicable" })),
        }),
        async execute(_toolCallId: string, params: any) {
          try {
            await ensureInitialized();
            const content = String(params.content ?? "").trim();
            if (!content) {
              return { content: [{ type: "text", text: "Error: content is required." }], details: {} };
            }

            const agentId = String(params.agentId ?? "unknown");
            const sessionId = String(params.sessionId ?? "manual");
            const tags: string[] = Array.isArray(params.tags) ? params.tags.map(String) : [];
            const memoryType = (params.memoryType ?? "fact") as MemoryType;
            const autoPriorityByType: Record<MemoryType, number> = {
              decision: 85,
              preference: 80,
              context: 60,
              fact: 30,
              session: 20,
            };
            const autoPriority = autoPriorityByType[memoryType] ?? 30;
            const priority = params.priority !== undefined && params.priority !== null
              ? Math.min(100, Math.max(0, Number(params.priority)))
              : autoPriority;
            const ttlDays = Number(params.ttlDays ?? resolvedConfig.defaultTtlDays);
            const expiresAt = ttlDays > 0 ? Date.now() + ttlDays * 86400 * 1000 : 0;
            const sourcePath = String(params.sourcePath ?? "");

            const chunks = chunkText(content, resolvedConfig.chunkSize, resolvedConfig.chunkOverlap);
            const embeddings = await ollama!.embedBatch(chunks);

            const entries = chunks.map((chunk, i) => ({
              id: randomUUID(),
              agentId,
              sessionId,
              content,
              chunk,
              embedding: embeddings[i],
              memoryType,
              priority,
              expiresAt,
              createdAt: Date.now(),
              sourcePath,
              checksum: contentChecksum(content + chunk, agentId),
              tags,
            }));

            const inserted = await db!.upsertEntries(entries);

            return {
              content: [
                {
                  type: "text",
                  text: inserted > 0
                    ? `Indexed ${inserted} chunk(s) for agent '${agentId}'. Total entries: ${await db!.count()}`
                    : `Content already indexed (checksum match). No duplicates stored.`,
                },
              ],
              details: { inserted, chunks: chunks.length },
            };
          } catch (err) {
            logError("noesis_index failed", { error: err, tool: "noesis_index", agentId: String(params.agentId ?? "unknown") });
            throw err;
          }
        },
      },
      { optional: true }
    );

    // ── Tool: noesis_search ───────────────────────────────────────────────
    api.registerTool(
      {
        name: "noesis_search",
        label: "Noesis Search",
        description:
          "Semantic search across Noesis memory. Uses hybrid pipeline: vector ANN + BM25 keyword search, reranked with MMR for diversity.",
        parameters: Type.Object({
          query: Type.String({ description: "Search query" }),
          agentId: Type.Optional(Type.String({ description: "Filter by agent ID" })),
          memoryType: Type.Optional(
            Type.Union(
              [
                Type.Literal("fact"),
                Type.Literal("decision"),
                Type.Literal("preference"),
                Type.Literal("context"),
                Type.Literal("session"),
              ],
              { description: "Filter by memory type" }
            )
          ),
          topK: Type.Optional(Type.Number({ description: "Max results (default from config)" })),
          crossAgent: Type.Optional(Type.Boolean({ description: "Search across all agents" })),
        }),
        async execute(_toolCallId: string, params: any) {
          try {
            await ensureInitialized();
            const query = String(params.query ?? "").trim();
            if (!query) {
              return { content: [{ type: "text", text: "Error: query is required." }], details: {} };
            }

            let results = await hybridSearch(query, ollama!, db!, resolvedConfig, {
              agentId: params.agentId ? String(params.agentId) : undefined,
              memoryType: params.memoryType as MemoryType | undefined,
              topK: params.topK ? Number(params.topK) : resolvedConfig.topK,
              crossAgent: Boolean(params.crossAgent),
            });

            // Fallback to archive search if active results are sparse (< 2)
            if (results.length < 2) {
              const topK = params.topK ? Number(params.topK) : resolvedConfig.topK;
              const embedding = await ollama!.embed(query);
              const [vectorArchive, ftsArchive] = await Promise.all([
                db!.searchArchive(embedding, topK * 2, params.agentId ? String(params.agentId) : undefined, params.memoryType as MemoryType | undefined),
                db!.fullTextSearchArchive(query, topK, params.agentId ? String(params.agentId) : undefined, params.memoryType as MemoryType | undefined),
              ]);
              const scoreMap = new Map<string, typeof vectorArchive[0] & { hybridScore: number }>();
              for (const r of vectorArchive) {
                scoreMap.set(r.id, { ...r, hybridScore: r.score * 0.6 });
              }
              for (const r of ftsArchive) {
                const existing = scoreMap.get(r.id);
                if (existing) {
                  existing.hybridScore += r.score * 0.4;
                } else {
                  scoreMap.set(r.id, { ...r, hybridScore: r.score * 0.4 });
                }
              }
              const archiveResults = Array.from(scoreMap.values()).sort((a, b) => b.hybridScore - a.hybridScore).slice(0, topK);
              if (archiveResults.length > 0) {
                const activeIds = new Set(results.map(r => r.id));
                for (const ar of archiveResults) {
                  if (!activeIds.has(ar.id)) {
                    results.push({
                      id: ar.id,
                      agentId: ar.agentId,
                      sessionId: ar.sessionId,
                      content: ar.content,
                      memoryType: ar.memoryType,
                      createdAt: ar.createdAt,
                      sourcePath: ar.sourcePath,
                      tags: ar.tags,
                      score: ar.hybridScore,
                      priority: ar.priority,
                      expiresAt: ar.expiresAt,
                    } as any);
                  }
                }
              }
            }

            if (results.length === 0) {
              return {
                content: [{ type: "text", text: "No results found." }],
                details: { count: 0 },
              };
            }

            const archiveIds = new Set(
              results
                .filter(r => r.expiresAt > 0 && r.createdAt < Date.now() - 90 * 86400 * 1000)
                .map(r => r.id)
            );

            const formatted = results
              .map(
                (r, i) => {
                  const fromArchive = archiveIds.has(r.id) ? " [archived]" : "";
                  return `${i + 1}. [${r.memoryType}]${fromArchive} [score=${r.score.toFixed(3)}] ${r.content.slice(0, 300)}${r.content.length > 300 ? "…" : ""}`;
                }
              )
              .join("\n\n");

            const archivedCount = results.filter(r => archiveIds.has(r.id)).length;
            const archiveNote = archivedCount > 0 ? `(Includes ${archivedCount} archived memory\n\n)` : "";

            return {
              content: [{ type: "text", text: `Found ${results.length} result(s)${archivedCount > 0 ? ` (including ${archivedCount} archived)` : ""}:\n\n${archiveNote}${formatted}` }],
              details: { count: results.length, results },
            };
          } catch (err) {
            logError("noesis_search failed", { error: err, tool: "noesis_search", agentId: params.agentId ? String(params.agentId) : undefined });
            throw err;
          }
        },
      },
      { optional: true }
    );

    // ── Tool: noesis_search_archive ──────────────────────────────────────
    api.registerTool(
      {
        name: "noesis_search_archive",
        label: "Noesis Search Archive",
        description:
          "Search archived (expired) memories. These are entries that reached their TTL and were moved to long-term archive. Use this for 'search older memories' queries.",
        parameters: Type.Object({
          query: Type.String({ description: "Search query for archived memories" }),
          agentId: Type.Optional(Type.String({ description: "Filter by agent ID" })),
          memoryType: Type.Optional(
            Type.Union([Type.Literal("fact"), Type.Literal("decision"), Type.Literal("preference"), Type.Literal("context"), Type.Literal("session")])
          ),
          topK: Type.Optional(Type.Number({ description: "Max results (default 5)" })),
        }),
        async execute(_toolCallId: string, params: any) {
          await ensureInitialized();
          const queryText = String(params.query ?? "").trim();
          if (!queryText) return { content: [{ type: "text", text: "Error: query is required." }], details: {} };
          const topK = params.topK ? Number(params.topK) : 5;
          const embedding = await ollama!.embed(queryText);
          const [vectorResults, ftsResults] = await Promise.all([
            db!.searchArchive(embedding, topK * 2, params.agentId ? String(params.agentId) : undefined, params.memoryType as MemoryType | undefined),
            db!.fullTextSearchArchive(queryText, topK, params.agentId ? String(params.agentId) : undefined, params.memoryType as MemoryType | undefined),
          ]);
          const scoreMap = new Map<string, typeof vectorResults[0] & { hybridScore: number }>();
          for (const r of vectorResults) {
            scoreMap.set(r.id, { ...r, hybridScore: r.score * 0.6 });
          }
          for (const r of ftsResults) {
            const existing = scoreMap.get(r.id);
            if (existing) {
              existing.hybridScore += r.score * 0.4;
            } else {
              scoreMap.set(r.id, { ...r, hybridScore: r.score * 0.4 });
            }
          }
          const merged = Array.from(scoreMap.values()).sort((a, b) => b.hybridScore - a.hybridScore).slice(0, topK);
          if (merged.length === 0) {
            return { content: [{ type: "text", text: "No archived memories found." }], details: { count: 0 } };
          }
          const formatted = merged.map((r, i) => `${i + 1}. [${r.memoryType}] [archived] ${r.content.slice(0, 250)}${r.content.length > 250 ? "…" : ""}`).join("\n\n");
          return { content: [{ type: "text", text: `Found ${merged.length} archived memory(ies):\n\n${formatted}` }], details: { count: merged.length } };
        },
      },
      { optional: true }
    );

    // ── Tool: noesis_recall ───────────────────────────────────────────────
    api.registerTool(
      {
        name: "noesis_recall",
        label: "Noesis Recall",
        description:
          "Recall recent memory entries for an agent or session. Ordered by creation time, newest first.",
        parameters: Type.Object({
          agentId: Type.Optional(Type.String({ description: "Filter by agent ID" })),
          sessionId: Type.Optional(Type.String({ description: "Filter by session ID" })),
          limit: Type.Optional(Type.Number({ description: "Max entries to return (default 20)" })),
        }),
        async execute(_toolCallId: string, params: any) {
          await ensureInitialized();
          const entries = await db!.recall(
            params.agentId ? String(params.agentId) : undefined,
            params.sessionId ? String(params.sessionId) : undefined,
            params.limit ? Number(params.limit) : 20
          );

          if (entries.length === 0) {
            return {
              content: [{ type: "text", text: "No memory entries found." }],
              details: { count: 0 },
            };
          }

          const formatted = entries
            .map(
              (e) =>
                `[${new Date(e.createdAt).toISOString()}] [${e.memoryType}] ${e.content.slice(0, 200)}${e.content.length > 200 ? "…" : ""}`
            )
            .join("\n\n");

          return {
            content: [{ type: "text", text: `Recalled ${entries.length} entry(ies):\n\n${formatted}` }],
            details: { count: entries.length },
          };
        },
      },
      { optional: true }
    );

    // ── Tool: noesis_import ───────────────────────────────────────────────
    api.registerTool(
      {
        name: "noesis_import",
        label: "Noesis Import",
        description:
          "Trigger MD → LanceDB migration for one agent or all agents. Scans memory/*.md and MEMORY.md files, chunks, embeds, and upserts. Idempotent.",
        parameters: Type.Object({
          agentId: Type.Optional(
            Type.String({ description: "Agent to import (omit to import all agents)" })
          ),
        }),
        async execute(_toolCallId: string, params: any) {
          try {
            await ensureInitialized();

            const logLines: string[] = [];
            const log = (msg: string) => {
              api.logger.info(msg);
              logLines.push(msg);
            };

            if (params.agentId) {
              const result = await importMarkdownFiles(
                String(params.agentId),
                db!,
                ollama!,
                resolvedConfig,
                log
              );
              return {
                content: [
                  {
                    type: "text",
                    text: `Import complete for '${result.agentId}': ${result.indexed} indexed, ${result.skipped} skipped, ${result.errors} errors`,
                  },
                ],
                details: result,
              };
            } else {
              const results = await importAllAgents(db!, ollama!, resolvedConfig, log);
              const totals = results.reduce(
                (acc, r) => ({
                  indexed: acc.indexed + r.indexed,
                  skipped: acc.skipped + r.skipped,
                  errors: acc.errors + r.errors,
                }),
                { indexed: 0, skipped: 0, errors: 0 }
              );
              return {
                content: [
                  {
                    type: "text",
                    text: `Import complete for ${results.length} agent(s): ${totals.indexed} indexed, ${totals.skipped} skipped, ${totals.errors} errors`,
                  },
                ],
                details: { agents: results, totals },
              };
            }
          } catch (err) {
            logError("noesis_import failed", { error: err, tool: "noesis_import", agentId: params.agentId ? String(params.agentId) : undefined });
            throw err;
          }
        },
      },
      { optional: true }
    );

    // ── Tool: noesis_stats ────────────────────────────────────────────────
    api.registerTool(
      {
        name: "noesis_stats",
        label: "Noesis Stats",
        description: "Get Noesis memory statistics — entry counts, DB path, model in use.",
        parameters: Type.Object({}),
        async execute(_toolCallId: string, _params: any) {
          if (!initialized || !db) {
            return {
              content: [{ type: "text", text: "Noesis is still initializing. Try again in a moment." }],
              details: {},
            };
          }

          const stats = await db.stats();
          const lines = [
            `Noesis Memory Stats`,
            `─────────────────`,
            `Total active entries: ${stats.totalEntries}`,
            `Total archive entries: ${stats.archiveEntries}`,
            `DB path: ${stats.dbPath}`,
            `Embedding model: ${stats.embeddingModel}`,
            `Embedding model: ${stats.embeddingModel}`,
            ``,
            `Active — by agent: ${JSON.stringify(stats.byAgent, null, 2)}`,
            `Active — by type: ${JSON.stringify(stats.byMemoryType, null, 2)}`,
            `Active — by priority: ${JSON.stringify(stats.byPriority, null, 2)}`,
            ``,
            `Archive — by agent: ${JSON.stringify(stats.byAgentArchive, null, 2)}`,
            `Archive — by type: ${JSON.stringify(stats.byMemoryTypeArchive, null, 2)}`,
          ];

          return {
            content: [{ type: "text", text: lines.join("\n") }],
            details: stats,
          };
        },
      },
      { optional: true }
    );

    // ── Tool: noesis_delete ───────────────────────────────────────────────
    api.registerTool(
      {
        name: "noesis_delete",
        label: "Noesis Delete",
        description: "Delete a memory entry by ID.",
        parameters: Type.Object({
          id: Type.String({ description: "Entry ID to delete" }),
        }),
        async execute(_toolCallId: string, params: any) {
          try {
            await ensureInitialized();
            const id = String(params.id ?? "").trim();
            if (!id) {
              return { content: [{ type: "text", text: "Error: id is required." }], details: {} };
            }
            await db!.deleteById(id);
            return {
              content: [{ type: "text", text: `Deleted entry ${id}` }],
              details: {},
            };
          } catch (err) {
            logError("noesis_delete failed", { error: err, tool: "noesis_delete", agentId: undefined });
            throw err;
          }
        },
      },
      { optional: true }
    );

    // ── Tool: noesis_cleanup ───────────────────────────────────────────────
    api.registerTool(
      {
        name: "noesis_cleanup",
        label: "Noesis Cleanup",
        description: "Move all expired entries from the active table to the archive table. Expired = TTL reached. Archived entries are no longer in active search but can be retrieved with noesis_search_archive.",
        parameters: Type.Object({}),
        async execute(_toolCallId: string, _params: any) {
          try {
            await ensureInitialized();
            const archived = await db!.archiveExpired();
            return {
              content: [{ type: "text", text: `Archived ${archived} expired entries to long-term storage.` }],
              details: { archived },
            };
          } catch (err) {
            logError("noesis_cleanup failed", { error: err, tool: "noesis_cleanup" });
            throw err;
          }
        },
      },
      { optional: true }
    );

    // ── Tool: noesis_dedup ─────────────────────────────────────────────
    api.registerTool(
      {
        name: "noesis_dedup",
        label: "Noesis Dedup",
        description:
          "Find near-duplicate memory entries using semantic similarity. Computes embeddings for a sample of entries and finds pairs above the similarity threshold. Returns duplicate groups — you decide which to keep or delete. Sample is capped at 200 entries for performance.",
        parameters: Type.Object({
          threshold: Type.Optional(
            Type.Number({
              description: "Cosine similarity threshold 0.7–0.99. Higher = stricter match. Default: 0.9",
            })
          ),
          agentId: Type.Optional(Type.String({ description: "Scope dedup to a specific agent" })),
          memoryType: Type.Optional(
            Type.Union([
              Type.Literal("fact"),
              Type.Literal("decision"),
              Type.Literal("preference"),
              Type.Literal("context"),
              Type.Literal("session"),
            ])
          ),
          limit: Type.Optional(
            Type.Number({ description: "Max duplicate pairs to return. Default: 50." })
          ),
        }),
        async execute(_toolCallId: string, params: any) {
          await ensureInitialized();
          const threshold = Math.min(0.99, Math.max(0.7, Number(params.threshold ?? 0.9)));
          const agentId = params.agentId ? String(params.agentId) : undefined;
          const memoryType = params.memoryType as MemoryType | undefined;
          const maxPairs = Math.min(200, Number(params.limit ?? 50));
          const SAMPLE_SIZE = 200;

          // 1. Fetch a random sample of entries from DB
          const allEntries = await db!.exportAll(agentId, memoryType);
          const now = Date.now();
          const active = allEntries.filter((e) => e.expiresAt === 0 || e.expiresAt > now);

          // Shuffle and cap
          const shuffled = active.sort(() => Math.random() - 0.5);
          const sample = shuffled.slice(0, SAMPLE_SIZE);

          if (sample.length < 2) {
            return {
              content: [{ type: "text", text: "Not enough entries to compare. Need at least 2." }],
              details: { checked: 0, duplicates: 0 },
            };
          }

          // 2. Re-embed all sample content
          const texts = sample.map((e) => e.content);
          const embeddings = await ollama!.embedBatch(texts);

          // 3. Pairwise similarity — O(N²), capped at SAMPLE_SIZE=200 → 19,900 comparisons
          const duplicatePairs: Array<{
            idA: string;
            idB: string;
            contentA: string;
            contentB: string;
            score: number;
            agentId: string;
            memoryType: string;
          }> = [];

          for (let i = 0; i < sample.length; i++) {
            for (let j = i + 1; j < sample.length; j++) {
              const sim = cosineSimilarityDense(embeddings[i], embeddings[j]);
              if (sim >= threshold) {
                duplicatePairs.push({
                  idA: sample[i].id,
                  idB: sample[j].id,
                  contentA: sample[i].content.slice(0, 150),
                  contentB: sample[j].content.slice(0, 150),
                  score: sim,
                  agentId: sample[i].agentId,
                  memoryType: sample[i].memoryType,
                });
                if (duplicatePairs.length >= maxPairs) break;
              }
            }
            if (duplicatePairs.length >= maxPairs) break;
          }

          if (duplicatePairs.length === 0) {
            return {
              content: [
                {
                  type: "text",
                  text: `Checked ${sample.length} entries (${active.length} total active). No duplicate pairs found above threshold ${threshold}.`,
                },
              ],
              details: { checked: sample.length, total: active.length, duplicates: 0, threshold },
            };
          }

          const lines = [
            `Found ${duplicatePairs.length} duplicate pair(s) above threshold ${threshold}:`,
            `(checked ${sample.length} of ${active.length} active entries — sample is capped at 200)`,
            ``,
          ];

          for (let i = 0; i < duplicatePairs.length; i++) {
            const d = duplicatePairs[i];
            lines.push(
              `─── Pair ${i + 1} [similarity: ${d.score.toFixed(4)}] ───`,
              `[${d.memoryType}] ${d.agentId}`,
              `A (${d.idA}): ${d.contentA}${d.contentA.length >= 150 ? "…" : ""}`,
              `B (${d.idB}): ${d.contentB}${d.contentB.length >= 150 ? "…" : ""}`,
              ``
            );
          }

          lines.push(
            `To delete a duplicate, use noesis_delete with its ID.`,
            `Tip: lower the threshold (e.g. 0.85) to catch looser duplicates, or raise it to 0.95+ for near-identical only.`
          );

          return {
            content: [{ type: "text", text: lines.join("\n") }],
            details: {
              checked: sample.length,
              total: active.length,
              duplicates: duplicatePairs.length,
              threshold,
              pairs: duplicatePairs,
            },
          };
        },
      },
      { optional: true }
    );

    // ── Tool: noesis_export ───────────────────────────────────────────────
    api.registerTool(
      {
        name: "noesis_export",
        label: "Noesis Export",
        description: "Bulk export all memory entries as a JSON array.",
        parameters: Type.Object({
          agentId: Type.Optional(Type.String({ description: "Filter by agent ID" })),
          memoryType: Type.Optional(Type.String({ description: "Filter by memory type" })),
          includeExpired: Type.Optional(Type.Boolean({ description: "Include expired entries. Default: false" })),
        }),
        async execute(_toolCallId: string, params: any) {
          try {
            await ensureInitialized();
            const agentId = params.agentId ? String(params.agentId) : undefined;
            const memoryType = params.memoryType ? String(params.memoryType) : undefined;
            const includeExpired = Boolean(params.includeExpired ?? false);
            const all = await db!.exportAll(agentId, memoryType);
            const now = Date.now();
            const entries = includeExpired
              ? all
              : all.filter((e) => e.expiresAt === 0 || e.expiresAt > now);
            return {
              content: [{ type: "text", text: JSON.stringify(entries, null, 2) }],
              details: { count: entries.length, total: all.length },
            };
          } catch (err) {
            logError("noesis_export failed", { error: err, tool: "noesis_export", agentId: params.agentId ? String(params.agentId) : undefined });
            throw err;
          }
        },
      },
      { optional: true }
    );

    // ── Tool: noesis_set_priority ─────────────────────────────────────────
    api.registerTool(
      {
        name: "noesis_set_priority",
        label: "Noesis Set Priority",
        description: "Update the priority and/or TTL of an existing memory entry.",
        parameters: Type.Object({
          id: Type.String({ description: "Entry ID to update" }),
          priority: Type.Optional(Type.Number({ description: "New priority 0–100" })),
          ttlDays: Type.Optional(Type.Number({ description: "New TTL in days. 0 = never expire. Overrides expiresAt." })),
        }),
        async execute(_toolCallId: string, params: any) {
          try {
            await ensureInitialized();
            const id = String(params.id ?? "").trim();
            if (!id) return { content: [{ type: "text", text: "Error: id is required." }], details: {} };
            const priority = params.priority !== undefined ? Math.min(100, Math.max(0, Number(params.priority))) : undefined;
            const ttlDays = params.ttlDays !== undefined ? Number(params.ttlDays) : undefined;
            const expiresAt = ttlDays !== undefined ? (ttlDays > 0 ? Date.now() + ttlDays * 86400 * 1000 : 0) : undefined;
            const updated = await db!.updateEntry(id, { priority, expiresAt });
            if (!updated) return { content: [{ type: "text", text: `Entry ${id} not found.` }], details: {} };
            return {
              content: [{ type: "text", text: `Updated entry ${id}${priority !== undefined ? `, priority=${priority}` : ""}${ttlDays !== undefined ? `, ttlDays=${ttlDays}` : ""}` }],
              details: { id, priority: priority ?? updated.priority, expiresAt: expiresAt ?? updated.expiresAt },
            };
          } catch (err) {
            logError("noesis_set_priority failed", { error: err, tool: "noesis_set_priority" });
            throw err;
          }
        },
      },
      { optional: true }
    );

    // ── Memory slot tools (OpenClaw auto-recall) ──────────────────────────

    api.registerTool(
      {
        name: "memory_search",
        label: "Memory Search",
        description: "Semantic memory search (Noesis slot). Entry point for OpenClaw auto-recall.",
        parameters: Type.Object({
          query: Type.String(),
          agentId: Type.Optional(Type.String()),
          topK: Type.Optional(Type.Number()),
          crossAgent: Type.Optional(Type.Boolean()),
        }),
        async execute(_toolCallId: string, params: any) {
          await ensureInitialized();
          const query = String(params.query ?? "").trim();
          if (!query) return { content: [{ type: "text", text: "No query provided." }], details: {} };

          const results = await hybridSearch(query, ollama!, db!, resolvedConfig, {
            agentId: params.agentId ? String(params.agentId) : undefined,
            topK: params.topK ? Number(params.topK) : resolvedConfig.topK,
            crossAgent: Boolean(params.crossAgent),
            vectorWeight: resolvedConfig.vectorWeight,
            rerank: resolvedConfig.rerank,
          });

          const formatted = results
            .map((r) => `[${r.memoryType}] ${r.content}`)
            .join("\n\n---\n\n");

          return {
            content: [{ type: "text", text: formatted || "No results found." }],
            details: { count: results.length, results },
          };
        },
      },
      { optional: true }
    );

    api.registerTool(
      {
        name: "memory_get",
        label: "Memory Get",
        description: "Retrieve a specific memory entry by ID (Noesis slot).",
        parameters: Type.Object({
          id: Type.String(),
        }),
        async execute(_toolCallId: string, params: any) {
          await ensureInitialized();
          const entry = await db!.getById(String(params.id));
          if (!entry) {
            return { content: [{ type: "text", text: "Entry not found." }], details: {} };
          }
          return {
            content: [{ type: "text", text: entry.content }],
            details: entry,
          };
        },
      },
      { optional: true }
    );

    api.registerTool(
      {
        name: "memory_index",
        label: "Memory Index",
        description: "Auto-index new memory content (Noesis slot). Called by OpenClaw indexer.",
        parameters: Type.Object({
          content: Type.String(),
          agentId: Type.Optional(Type.String()),
          sessionId: Type.Optional(Type.String()),
          memoryType: Type.Optional(Type.String()),
          tags: Type.Optional(Type.Array(Type.String())),
          priority: Type.Optional(Type.Number()),
          ttlDays: Type.Optional(Type.Number()),
        }),
        async execute(_toolCallId: string, params: any) {
          await ensureInitialized();
          const content = String(params.content ?? "").trim();
          if (!content) return { content: [{ type: "text", text: "No content." }], details: {} };

          const agentId = String(params.agentId ?? "unknown");
          const sessionId = String(params.sessionId ?? "auto");
          const memoryType = (params.memoryType ?? "fact") as MemoryType;
          const tags: string[] = Array.isArray(params.tags) ? params.tags.map(String) : [];
          const autoPriorityByType: Record<MemoryType, number> = {
            decision: 85,
            preference: 80,
            context: 60,
            fact: 30,
            session: 20,
          };
          const autoPriority = autoPriorityByType[memoryType] ?? 30;
          const priority = params.priority !== undefined && params.priority !== null
            ? Math.min(100, Math.max(0, Number(params.priority)))
            : autoPriority;
          const ttlDays = Number(params.ttlDays ?? resolvedConfig.defaultTtlDays);
          const expiresAt = ttlDays > 0 ? Date.now() + ttlDays * 86400 * 1000 : 0;

          // Embed the full content once — all chunks share this embedding.
          // This avoids N separate embedding calls for an N-chunk entry and
          // ensures semantic consistency: all chunks retrieve against the same
          // content embedding, which is the text actually stored and returned.
          const [contentEmbedding] = await ollama!.embedBatch([content]);
          const contentChecksum_ = contentChecksum(content, agentId);

          const entries = chunkText(content, resolvedConfig.chunkSize, resolvedConfig.chunkOverlap).map((chunk) => ({
            id: randomUUID(),
            agentId,
            sessionId,
            content,
            chunk,
            embedding: contentEmbedding, // shared across all chunks
            memoryType,
            priority,
            expiresAt,
            createdAt: Date.now(),
            sourcePath: "",
            checksum: contentChecksum(content + chunk, agentId), // chunk in checksum keeps entries distinct per chunk
            tags,
          }));

          const inserted = await db!.upsertEntries(entries);
          return {
            content: [{ type: "text", text: `Indexed ${inserted} chunk(s).` }],
            details: { inserted },
          };
        },
      },
      { optional: true }
    );

    api.registerTool(
      {
        name: "memory_recall",
        label: "Memory Recall",
        description: "Cross-session memory recall (Noesis slot). Used by OpenClaw session recall.",
        parameters: Type.Object({
          agentId: Type.Optional(Type.String()),
          sessionId: Type.Optional(Type.String()),
          limit: Type.Optional(Type.Number()),
        }),
        async execute(_toolCallId: string, params: any) {
          await ensureInitialized();
          const entries = await db!.recall(
            params.agentId ? String(params.agentId) : undefined,
            params.sessionId ? String(params.sessionId) : undefined,
            params.limit ? Number(params.limit) : 20
          );

          const formatted = entries
            .map((e) => `[${e.memoryType}] ${e.content}`)
            .join("\n\n---\n\n");

          return {
            content: [{ type: "text", text: formatted || "No memories found." }],
            details: { count: entries.length },
          };
        },
      },
      { optional: true }
    );

    // ── Context Engine Registration ──────────────────────────────────────
    if (resolvedConfig.contextEngineEnabled) {
      api.registerContextEngine("noesis", () => {
        return {
          info: {
            id: "noesis",
            name: "Noesis Context Engine",
            ownsCompaction: false,
          },

          /** Detect memory type from message content and store in Noesis. */
          async ingest({ sessionId, message, isHeartbeat }) {
            await ensureInitialized();
            if (!db || !ollama) return { ingested: false };
            if (isHeartbeat) return { ingested: false };

            const content = typeof message === "string" ? message : (message as any)?.content ?? "";
            if (!content || content.length < 10) return { ingested: false };

            // Detect memory type from content patterns
            const lower = content.toLowerCase();
            let memoryType: MemoryType = "context";
            let priority = 60;

            if (/we decided|we're going with|let's go with|decision:|decided to|chose to|selected (?:the|a)/i.test(lower)) {
              memoryType = "decision";
              priority = 85;
            } else if (/i prefer|i always|i never|i hate|i like you|i don't like|my preference|prefer to|would rather/i.test(lower)) {
              memoryType = "preference";
              priority = 80;
            } else if (/^(?:the |a |an )?[\w\s]+ is |^fact:|^info:|^note:/im.test(lower)) {
              memoryType = "fact";
              priority = 30;
            } else if (/^(?:session|conversation) summary|^summary:/im.test(lower)) {
              memoryType = "session";
              priority = 20;
            }

            // Check for TTL from config
            const ttlDays = resolvedConfig.defaultTtlDays;
            const expiresAt = ttlDays > 0 ? Date.now() + ttlDays * 86_400_000 : 0;

            const agentId = "context-engine";
            const checksum = await contentChecksum(content, agentId);

            const entry = {
              id: randomUUID(),
              agentId,
              sessionId: sessionId ?? "unknown",
              content,
              chunk: content,
              embedding: await ollama.embed(content),
              tags: ["context-engine"],
              memoryType,
              priority,
              expiresAt,
              createdAt: Date.now(),
              sourcePath: `context-engine://${sessionId ?? "session"}`,
              checksum,
            };

            await db.upsertEntries([entry]);
            return { ingested: true };
          },

          /**
           * Assemble: inject high-priority memories (>= assembleInjectPriority)
           * into every model run as a systemPromptAddition.
           * Zero LLM cost — just priority-filtered DB reads.
           */
          async assemble({ sessionId, messages }) {
            await ensureInitialized();
            if (!db) {
              return { messages: [], estimatedTokens: 0, systemPromptAddition: "" };
            }

            // Gather conversation context from recent messages for semantic relevance
            const recentContent = messages
              .slice(-5)
              .map((m: any) => (typeof m === "string" ? m : m.content ?? ""))
              .filter(Boolean)
              .join(" ");

            // Query high-priority memories
            const memories = await db.queryByPriority(
              resolvedConfig.assembleInjectPriority,
              undefined, // cross-agent
              resolvedConfig.assembleMaxEntries,
              resolvedConfig.assembleMaxAgeDays > 0 ? resolvedConfig.assembleMaxAgeDays : undefined
            );

            if (memories.length === 0) {
              return { messages: [], estimatedTokens: 0, systemPromptAddition: "" };
            }

            // Format memories as a structured recall block
            const grouped = memories.reduce((acc: Record<string, MemoryEntry[]>, m) => {
              const key = m.memoryType ?? "unknown";
              if (!acc[key]) acc[key] = [];
              acc[key].push(m);
              return acc;
            }, {});

            const lines: string[] = [
              "[Noesis Memory — Active Context]",
              "High-priority memories (never重复 — you have these):",
              "",
            ];

            const typeLabels: Record<string, string> = {
              decision: " Decisions ",
              preference: " Preferences ",
              context: " Context ",
              fact: " Facts ",
              session: " Session ",
            };

            for (const [type, entries] of Object.entries(grouped)) {
              const label = typeLabels[type] ?? ` ${type} `;
              lines.push(`---${label}---`);
              for (const e of entries) {
                const date = new Date(e.createdAt).toLocaleDateString("en-US", { month: "short", day: "numeric" });
                lines.push(`[${date}] ${e.content}`);
              }
              lines.push("");
            }

            const addition = lines.join("\n");

            // Rough token estimate: ~4 chars per token
            const estimatedTokens = Math.ceil(addition.length / 4) + Math.ceil(recentContent.length / 4);

            return {
              messages: [], // We use systemPromptAddition instead of injecting messages
              estimatedTokens,
              systemPromptAddition: addition,
            };
          },

          /**
           * Compact: structural compression for session summarization.
           *
           * Noesis stores chunked entries with per-chunk checksums. When OpenClaw
           * compacts a session, it produces a condensed summary and writes it back.
           *
           * Limitation: OpenClaw's generic compaction doesn't know about Noesis's
           * chunk structure. If it writes back a condensed entry, Noesis will treat
           * it as a new entry (new checksum, new ID). The original fragmented
           * chunks remain in the DB until the next cleanup run picks them up via
           * expiresAt, or until a future compaction pass handles the dedup.
           *
           * A proper fix would: (1) detect that OpenClaw wrote a condensed entry
           * for this session, (2) compute its checksum, (3) check if an entry with
           * that checksum already exists via db.getByChecksum(), and (4) delete
           * the session's fragmented entries if the condensed entry is a duplicate.
           * This requires additional state tracking and is deferred for now.
           *
           * Current behavior: defer to OpenClaw's compaction output, accept that
           * fragmented chunks accumulate until TTL cleanup removes them.
           */
          async compact({ sessionId, force }) {
            return { ok: true, compacted: false };
          },

          /**
           * After turn: no proactive indexing needed.
           * Ingest hook handles message storage on new messages.
           */
          async afterTurn() {
            // No proactive indexing needed — ingest hook handles new messages
          },
        };
      });

      api.logger.info(`[noesis] Context engine registered (assembleInjectPriority=${resolvedConfig.assembleInjectPriority}, assembleMaxEntries=${resolvedConfig.assembleMaxEntries})`);
    }

    api.logger.info(`[noesis] Plugin registered — ${Object.keys(api.pluginConfig ?? {}).length} config key(s)`);

    // Graceful shutdown: clear intervals, close scanner, and disconnect LanceDB on gateway stop.
    // Also reset module-level state so repeated reloads don't leak intervals or hold stale refs.
    api.on("gateway_stop", async () => {
      api.logger.info("[noesis] Shutting down...");
      if (cleanupIntervalId !== null) { clearInterval(cleanupIntervalId); cleanupIntervalId = null; }
      if (optimizeIntervalId !== null) { clearInterval(optimizeIntervalId); optimizeIntervalId = null; }
      if (sessionScanner) { sessionScanner.close(); sessionScanner = null; }
      if (db) {
        await db.disconnect().catch((err: any) => api.logger.info(`[noesis] disconnect() error: ${err}`));
        db = null;
      }
      initialized = false;
      api.logger.info("[noesis] Shutdown complete.");
    });
  },
});

// ─── initialization ────────────────────────────────────────────────────────

// Guard: wrap a log function to suppress EPIPE if the log stream is closed.
// This prevents crashes when stdout/stderr is broken (e.g. pipe to head/jq/grep
// that closes early) or when the gateway process terminates during log output.
function makeSafeLog(log: (msg: string) => void): (msg: string) => void {
  return (msg: string) => {
    try {
      log(msg);
    } catch (err: any) {
      if (err?.code === "EPIPE" || err?.code === "EIO") return;
      throw err;
    }
  };
}


async function initPlugin(config: NoesisConfig, log: (msg: string) => void): Promise<void> {
  const safeLog = makeSafeLog(log);

  // Set up dedicated error logging first — before anything else
  configureErrorLog(config.errorLogPath);
  installGlobalErrorHandlers();
  safeLog(`[noesis] Error log: ${config.errorLogPath}`);

  // 1. Connect to LanceDB
  try {
    db = new NoesisDB(config);
  } catch (err) {
    logError("Failed to initialize NoesisDB", { error: err });
    throw err;
  }

  // 2. Auto-configure Transformers.js embedding model
  try {
    ollama = await autoConfigTransformers(config.embeddingModel, log);
  } catch (err) {
    logError("Failed to auto-configure Transformers.js embedding model", { error: err });
    throw err;
  }

  // Get embedding dimension for schema
  let embeddingDim: number;
  try {
    const testEmbed = await ollama.embed("noesis init");
    embeddingDim = testEmbed.length;
  } catch (err) {
    logError("Failed to get embedding dimension from Transformers.js", { error: err });
    throw err;
  }

  try {
    await db.connect(embeddingDim);
    safeLog(`[noesis] LanceDB connected (${embeddingDim}d embeddings, path: ${config.lanceDbPath})`);
  } catch (err) {
    logError("Failed to connect to LanceDB", { error: err });
    throw err;
  }

  initialized = true;

  // 3. Optional: auto-migrate on startup — ONLY if DB is empty (first install / fresh DB)
  if (config.autoMigrate) {
    try {
      const count = await db.count();
      if (count === 0) {
        safeLog("[noesis] autoMigrate enabled and DB is empty — importing markdown memory files...");
        await importAllAgents(db, ollama, config, safeLog);
      } else {
        safeLog(`[noesis] autoMigrate enabled but DB already has ${count} entries — skipping auto-migrate (run noesis_import to re-migrate manually)`);
      }
    } catch (err) {
      logError("Auto-migrate failed", { error: err, extra: { autoMigrate: true } });
    }
  }

  // 4. Optional: interval-based session scanner (replaces chokidar watcher)
  if (config.indexQmdSessions) {
    try {
      sessionScanner = startSessionScanner(db, ollama, config, safeLog);
    } catch (err) {
      logError("Failed to start session scanner", { error: err, extra: { indexQmdSessions: true } });
    }
  }

 
  // 5. Optional: auto-cleanup expired entries on startup
  if (config.autoCleanup) {
    warnOnLargeCleanupGap(log);
    try {
      const archived = await db.archiveExpired();
      if (archived > 0) {
        safeLog(`[noesis] Archived ${archived} expired entries.`);
        setLastCleanupTimestamp(Date.now());
      }
    } catch (err) {
      safeLog(`[noesis] Cleanup skipped: ${err}`);
      logError("Startup cleanup failed", { error: err, extra: { autoCleanup: true } });
    }
  }

  // Always run optimize on startup to reclaim LanceDB version store.
  // This removes orphaned version files from _versions/ that accumulate
  // from repeated writes. Safe to run on every startup — LanceDB handles it.
  try {
    await db.optimize();
    safeLog(`[noesis] Startup LanceDB optimize() complete.`);
  } catch (err) {
    safeLog(`[noesis] Startup optimize() skipped: ${err}`);
  }

  // 6. Optional: periodic TTL cleanup on configurable interval
  if (config.cleanupIntervalHours > 0) {
    const intervalMs = config.cleanupIntervalHours * 60 * 60 * 1000;
    cleanupIntervalId = setInterval(() => {
      db!.archiveExpired()
        .then((n) => {
          if (n > 0) {
            safeLog(`[noesis] Periodic cleanup archived ${n} entries.`);
            setLastCleanupTimestamp(Date.now());
          }
        })
        .catch((err) => {
          safeLog(`[noesis] Periodic cleanup error: ${err}`);
          logError("Periodic cleanup failed", { error: err, extra: { cleanupIntervalHours: config.cleanupIntervalHours } });
        });
    }, intervalMs);
    safeLog(`[noesis] Periodic cleanup scheduled every ${config.cleanupIntervalHours}h`);
  }

  // 7. Hardcoded 5-minute interval for LanceDB optimize() to prevent version store bloat.
  // Not configurable — this should run frequently to keep _versions/ lean.
  // Uses 1h older-than threshold (vs default 24h) to reclaim versions more aggressively
  // and prevent _versions/ accumulation from rapid writes.
  optimizeIntervalId = setInterval(() => {
    db!.optimize(60 * 60 * 1000) // 1h instead of 24h default
      .then(() => {
        safeLog(`[noesis] Periodic LanceDB optimize() complete.`);
      })
      .catch((err) => {
        safeLog(`[noesis] Periodic optimize() error: ${err}`);
        logError("Periodic optimize failed", { error: err });
      });
  }, 5 * 60 * 1000);
  safeLog(`[noesis] Periodic LanceDB optimize() scheduled every 5min`);

  safeLog("[noesis] Initialization complete.");
}

async function ensureInitialized(): Promise<void> {
  if (initialized && db && ollama) return;

  // Wait up to 30s for initialization
  const start = Date.now();
  while (!initialized) {
    if (Date.now() - start > 30_000) {
      throw new Error("[noesis] Initialization timeout — Transformers.js model may have failed to load. Check error log.");
    }
    await new Promise((r) => setTimeout(r, 200));
  }
}

// ─── config builder ────────────────────────────────────────────────────────

function buildConfig(raw: Record<string, unknown>): NoesisConfig {
  return {
    lanceDbPath: String(raw.lanceDbPath ?? DEFAULT_CONFIG.lanceDbPath),
    ollamaEndpoint: String(raw.ollamaEndpoint ?? DEFAULT_CONFIG.ollamaEndpoint),
    embeddingModel: String(raw.embeddingModel ?? DEFAULT_CONFIG.embeddingModel),
    chunkSize: Number(raw.chunkSize ?? DEFAULT_CONFIG.chunkSize),
    chunkOverlap: Number(raw.chunkOverlap ?? DEFAULT_CONFIG.chunkOverlap),
    topK: Number(raw.topK ?? DEFAULT_CONFIG.topK),
    autoMigrate: Boolean(raw.autoMigrate ?? DEFAULT_CONFIG.autoMigrate),
    indexQmdSessions: Boolean(raw.indexQmdSessions ?? DEFAULT_CONFIG.indexQmdSessions),
    sessionScanIntervalMinutes: Number(raw.sessionScanIntervalMinutes ?? DEFAULT_CONFIG.sessionScanIntervalMinutes),
    watchMemoryDirs: Boolean(raw.watchMemoryDirs ?? DEFAULT_CONFIG.watchMemoryDirs),
    gitLfsEnabled: Boolean(raw.gitLfsEnabled ?? DEFAULT_CONFIG.gitLfsEnabled),
    gitLfsRepo: String(raw.gitLfsRepo ?? DEFAULT_CONFIG.gitLfsRepo),
    annNprobe: Number(raw.annNprobe ?? DEFAULT_CONFIG.annNprobe),
    annNumSubvectors: Number(raw.annNumSubvectors ?? DEFAULT_CONFIG.annNumSubvectors),
    defaultTtlDays: Number(raw.defaultTtlDays ?? DEFAULT_CONFIG.defaultTtlDays),
    autoCleanup: Boolean(raw.autoCleanup ?? DEFAULT_CONFIG.autoCleanup),
    cleanupIntervalHours: Number(raw.cleanupIntervalHours ?? DEFAULT_CONFIG.cleanupIntervalHours),
    contextEngineEnabled: Boolean(raw.contextEngineEnabled ?? DEFAULT_CONFIG.contextEngineEnabled),
    assembleInjectPriority: Number(raw.assembleInjectPriority ?? DEFAULT_CONFIG.assembleInjectPriority),
    assembleMaxEntries: Number(raw.assembleMaxEntries ?? DEFAULT_CONFIG.assembleMaxEntries),
    assembleMaxAgeDays: Number(raw.assembleMaxAgeDays ?? DEFAULT_CONFIG.assembleMaxAgeDays),
    errorLogPath: String(raw.errorLogPath ?? DEFAULT_CONFIG.errorLogPath),
    vectorWeight: Number(raw.vectorWeight ?? DEFAULT_CONFIG.vectorWeight),
    rerank: Boolean(raw.rerank ?? DEFAULT_CONFIG.rerank),
    rerankerEndpoint: raw.rerankerEndpoint ? String(raw.rerankerEndpoint) : undefined,
  };
}
