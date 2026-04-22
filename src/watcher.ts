import fs from "fs";
import path from "path";
import os from "os";
import { randomUUID } from "crypto";
import { EmbeddingClient, chunkText, contentChecksum } from "./transformers.js";
import { NoesisDB } from "./lancedb.js";
import { MemoryEntry, MemoryType, NoesisConfig } from "./types.js";
import { isNoiseMessage } from "./index.js";

const DEBOUNCE_MS = 2000;

export interface SessionWatcher {
  close(): void;
}

const AGENTS_PATH = path.join(os.homedir(), ".openclaw", "agents");

// ─── Memory file indexer ───────────────────────────────────────────────

/**
 * Parse agentId from path like: ~/.openclaw/agents/<agentId>/workspace/memory/...
 */
function agentIdFromMemoryPath(filePath: string): string {
  const rel = path.relative(AGENTS_PATH, filePath);
  const parts = rel.split(path.sep);
  return parts[0] || "unknown";
}

/**
 * Infer memory type from markdown heading/content patterns.
 */
function inferMemoryType(content: string): MemoryType {
  const lower = content.toLowerCase();
  if (lower.includes("decided") || lower.includes("decision") || lower.includes("chose to") || lower.includes("going with")) return "decision";
  if (lower.includes("prefer") || lower.includes("like") || lower.includes("always") || lower.includes("never")) return "preference";
  if (lower.includes("fact:") || lower.includes("fact-")) return "fact";
  if (lower.includes("session") || lower.includes("yesterday") || lower.includes("last")) return "context";
  return "fact";
}

/**
 * Auto-assign priority based on memory type. Mirrors the logic in index.ts.
 */
function autoPriorityForType(memoryType: MemoryType): number {
  const map: Record<MemoryType, number> = {
    decision: 85,
    preference: 80,
    context: 60,
    fact: 30,
    session: 20,
  };
  return map[memoryType] ?? 30;
}

/**
 * Index a markdown memory file.
 * Parses by ## headings, creates entries per section.
 * Deduplicates by content checksum per agent.
 */
async function indexMemoryFile(
  filePath: string,
  db: NoesisDB,
  ollama: EmbeddingClient,
  config: NoesisConfig,
  logger?: (msg: string) => void
): Promise<number> {
  const raw = fs.readFileSync(filePath, "utf-8");
  const agentId = agentIdFromMemoryPath(filePath);
  const now = Date.now();

  // Split by ## headings
  const sections = raw.split(/(?=^##\s)/m).filter((s) => s.trim().length > 10);
  if (sections.length === 0) return 0;

  const allChunks: Array<{ content: string; chunk: string }> = [];

  for (const section of sections) {
    const lines = section.split("\n");
    const title = lines[0].replace(/^##\s*/, "").trim();
    const body = lines.slice(1).join("\n").trim();

    if (body.length < 20) continue;

    // Combine title + body as content
    const fullContent = title ? `${title}. ${body}` : body;
    const chunks = chunkText(fullContent, config.chunkSize, config.chunkOverlap);
    for (const chunk of chunks) {
      allChunks.push({ content: fullContent, chunk });
    }
  }

  if (allChunks.length === 0) return 0;

  // Deduplicate by checksum per agent
  const seen = new Set<string>();
  const unique = allChunks.filter(({ content }) => {
    const cs = contentChecksum(content, agentId);
    if (seen.has(cs)) return false;
    seen.add(cs);
    return true;
  });

  if (unique.length === 0) return 0;

  const BATCH = 8;
  let indexed = 0;

  for (let i = 0; i < unique.length; i += BATCH) {
    const batch = unique.slice(i, i + BATCH);
    try {
      const embeddings = await ollama.embedBatch(batch.map((b) => b.chunk));
      const entries: MemoryEntry[] = batch.map((b, j) => ({
        id: randomUUID(),
        agentId,
        sessionId: "memory",
        content: b.content,
        chunk: b.chunk,
        embedding: embeddings[j],
        memoryType: inferMemoryType(b.content),
        priority: autoPriorityForType(inferMemoryType(b.content)),
        expiresAt: 0,
        createdAt: now,
        sourcePath: filePath,
        checksum: contentChecksum(b.content, agentId),
        tags: [],
      }));

      const inserted = await db.upsertEntries(entries);
      indexed += inserted;
    } catch (err) {
      logger?.(`[noesis/memory-watcher] Embed batch error for ${filePath}: ${err}`);
    }
  }

  return indexed;
}

// ─── Session file indexer ────────────────────────────────────────────

/**
 * Parse agentId and sessionId from a session JSONL path.
 * Handles all patterns:
 *   ~/.openclaw/sessions/<sessionId>.jsonl
 *   ~/.openclaw/sessions/<agentId>/<sessionId>.jsonl
 *   ~/.openclaw/agents/<agentId>/sessions/<sessionId>.jsonl
 *   ~/.openclaw/agents/<agentId>/qmd/sessions/<sessionId>.jsonl
 */
export function parseSessionPath(filePath: string): { agentId: string; sessionId: string } {
  const home = os.homedir();
  const rel = filePath.startsWith(home) ? path.relative(home, filePath) : filePath;
  const parts = rel.split(path.sep).filter(Boolean);

  // Pattern: agents/<agentId>/qmd/sessions/<sessionId>.jsonl
  // e.g. ~/.openclaw/agents/kate/qmd/sessions/abc123.jsonl
  // parts = [".openclaw", "agents", "kate", "qmd", "sessions", "abc123.jsonl"]
  if (parts.length >= 6 && parts[1] === "agents" && parts[3] === "qmd" && parts[4] === "sessions") {
    return { agentId: parts[2], sessionId: path.basename(parts[5], ".jsonl") };
  }
  // Pattern: agents/<agentId>/sessions/<sessionId>.jsonl
  // e.g. ~/.openclaw/agents/kate/sessions/abc123.jsonl
  // parts = [".openclaw", "agents", "kate", "sessions", "abc123.jsonl"]
  if (parts.length >= 5 && parts[1] === "agents" && parts[3] === "sessions") {
    return { agentId: parts[2], sessionId: path.basename(parts[4], ".jsonl") };
  }
  // Pattern: sessions/<agentId>/<sessionId>.jsonl
  // e.g. ~/.openclaw/sessions/kate/abc123.jsonl
  // parts = [".openclaw", "sessions", "kate", "abc123.jsonl"]
  if (parts.length >= 4 && parts[1] === "sessions" && parts[2] !== "sessions") {
    return { agentId: parts[2], sessionId: path.basename(parts[3], ".jsonl") };
  }
  // Pattern: sessions/<sessionId>.jsonl (bare session, no agent folder)
  // e.g. ~/.openclaw/sessions/abc123.jsonl
  // parts = [".openclaw", "sessions", "abc123.jsonl"]
  if (parts.length >= 3 && parts[1] === "sessions") {
    return { agentId: "unknown", sessionId: path.basename(parts[2], ".jsonl") };
  }

  // Fallback
  return { agentId: "unknown", sessionId: path.basename(filePath, ".jsonl") };
}

/**
 * Parse a QMD JSONL session file and index its messages.
 *
 * QMD format: one JSON object per line, each representing a message event.
 * We extract assistant + user messages with content, chunk them, and embed.
 */
export async function indexSessionFile(
  filePath: string,
  db: NoesisDB,
  ollama: EmbeddingClient,
  config: NoesisConfig,
  logger?: (msg: string) => void
): Promise<void> {
  const raw = fs.readFileSync(filePath, "utf-8");
  const lines = raw.split("\n").filter((l) => l.trim());
  if (lines.length === 0) return;

  const { agentId, sessionId } = parseSessionPath(filePath);

  const texts: string[] = [];

  for (const line of lines) {
    try {
      const entry = JSON.parse(line) as {
        type?: string;
        message?: {
          role?: string;
          content?: string | Array<{ type: string; text?: string }>;
        };
        content?: string | Array<{ type: string; text?: string }>;
        text?: string;
      };

      // Extract text content from session JSONL format (entry.message.content)
      let text = "";
      const msg = entry.message;
      if (msg?.content) {
        if (typeof msg.content === "string") {
          text = msg.content;
        } else if (Array.isArray(msg.content)) {
          text = msg.content
            .filter((b) => b.type === "text" && b.text)
            .map((b) => b.text!)
            .join(" ");
        }
      } else if (typeof entry.content === "string") {
        text = entry.content;
      } else if (Array.isArray(entry.content)) {
        text = entry.content
          .filter((b) => b.type === "text" && b.text)
          .map((b) => b.text!)
          .join(" ");
      } else if (typeof entry.text === "string") {
        text = entry.text;
      }

      // Filter out noise messages before indexing.
      if (isNoiseMessage(text)) return;

      const cleaned = text.trim().replace(/[\n\t\r]+/g, ' ');
      if (cleaned.length > 20) {
        texts.push(cleaned);
      }
    } catch {
      // skip malformed lines
    }
  }

  if (texts.length === 0) return;

  // Chunk and embed
  const allChunks: Array<{ content: string; chunk: string }> = [];
  for (const text of texts) {
    const chunks = chunkText(text, config.chunkSize, config.chunkOverlap);
    for (const chunk of chunks) {
      allChunks.push({ content: text, chunk });
    }
  }

  const BATCH = 8;
  let indexed = 0;

  for (let i = 0; i < allChunks.length; i += BATCH) {
    const batch = allChunks.slice(i, i + BATCH);
    try {
      const embeddings = await ollama.embedBatch(batch.map((b) => b.chunk));
      const entries: MemoryEntry[] = batch.map((b, j) => ({
        id: randomUUID(),
        agentId,
        sessionId,
        content: b.content,
        chunk: b.chunk,
        embedding: embeddings[j],
        memoryType: "session" as MemoryType,
        priority: autoPriorityForType("session"),
        expiresAt: 0,
        createdAt: Date.now(),
        sourcePath: filePath,
        checksum: contentChecksum(b.content, agentId),
        tags: ["qmd-session"],
      }));

      const inserted = await db.upsertEntries(entries);
      indexed += inserted;
    } catch (err) {
      logger?.(`[noesis/watcher] Embed batch error for ${filePath}: ${err}`);
    }
  }

  if (indexed > 0) {
    logger?.(`[noesis/watcher] Indexed ${indexed} chunk(s) from session ${sessionId} (agent: ${agentId})`);
  }
}

// ─── Interval-based session scanner (replaces chokidar watcher) ─────────────

const SESSION_SCAN_INTERVAL_DEFAULT_MS = 5 * 60 * 1000; // 5 minutes

export interface SessionScanner {
  close: () => void;
}

/**
 * Periodically scan agent session directories for new or changed session files
 * and index them. Uses setInterval instead of filesystem watchers for reliability.
 * Deduplication via content checksum ensures idempotent re-indexing.
 */
export function startSessionScanner(
  db: NoesisDB,
  ollama: EmbeddingClient,
  config: NoesisConfig,
  logger?: (msg: string) => void
): SessionScanner {
  const home = os.homedir();
  const agentsPath = path.join(home, ".openclaw", "agents");
  const intervalMs = (config.sessionScanIntervalMinutes > 0
    ? config.sessionScanIntervalMinutes
    : 5) * 60 * 1000;

  // Track in-flight index operations to avoid concurrent processing of the same file
  const inflight = new Set<string>();

  // Concurrency limit for agent scans to avoid overwhelming the embedder
  const AGENT_CONCURRENCY = 3;

  /** Scan a single agent's session and memory files. */
  async function scanAgent(
    agentDir: string,
    agentsPath: string,
    inflight: Set<string>,
    db: NoesisDB,
    ollama: EmbeddingClient,
    config: NoesisConfig,
    logger?: (msg: string) => void
  ): Promise<void> {
    // Scan session files: agents/<agent>/sessions/*.jsonl
    const sessionsDir = path.join(agentsPath, agentDir, "sessions");
    if (fs.existsSync(sessionsDir)) {
      let files: string[] = [];
      try {
        files = fs.readdirSync(sessionsDir).filter(
          (f) => f.endsWith(".jsonl") && !f.includes(".lock") && !f.includes(".checkpoint") && !f.includes(".reset")
        );
      } catch {
        // continue
      }
      if (files.length > 0) {
        logger?.(`[noesis/scanner] ${agentDir}: scanning ${files.length} session file(s)`);
      }
      // Index session files concurrently (bounded by AGENT_CONCURRENCY via Promise.allSettled at the batch level)
      const sessionResults = await Promise.allSettled(
        files.map((file) => {
          const filePath = path.join(sessionsDir, file);
          if (inflight.has(filePath)) return Promise.resolve();
          inflight.add(filePath);
          return indexSessionFile(filePath, db, ollama, config, logger).finally(() => inflight.delete(filePath));
        })
      );
      for (const r of sessionResults) {
        if (r.status === "rejected") {
          logger?.(`[noesis/scanner] Session index error: ${r.reason}`);
        }
      }
    }

    // Scan memory files: agents/<agent>/workspace/memory/*.md
    const memoryDir = path.join(agentsPath, agentDir, "workspace", "memory");
    if (fs.existsSync(memoryDir)) {
      let files: string[] = [];
      try {
        files = fs.readdirSync(memoryDir).filter(
          (f) => f.endsWith(".md") && !f.startsWith(".")
        );
      } catch {
        // continue
      }
      const memResults = await Promise.allSettled(
        files.map((file) => {
          const filePath = path.join(memoryDir, file);
          if (inflight.has(filePath)) return Promise.resolve();
          inflight.add(filePath);
          return indexMemoryFile(filePath, db, ollama, config, logger).finally(() => inflight.delete(filePath));
        })
      );
      for (const r of memResults) {
        if (r.status === "rejected") {
          logger?.(`[noesis/scanner] Memory index error: ${r.reason}`);
        }
      }
    }
  }

  let _isScanning = false;

  const scanDirs = async () => {
    // Skip this tick if a previous scan is still running — prevents overlap
    if (_isScanning) {
      logger?.(`[noesis/scanner] Skipping tick: previous scan still in progress`);
      return;
    }
    _isScanning = true;
    try {
      if (!fs.existsSync(agentsPath)) return;
      let agentDirs: string[] = [];
      try {
        agentDirs = fs.readdirSync(agentsPath).filter((d) => !d.startsWith("."));
      } catch {
        return;
      }
      // Process agents in bounded concurrent batches
      for (let i = 0; i < agentDirs.length; i += AGENT_CONCURRENCY) {
        const batch = agentDirs.slice(i, i + AGENT_CONCURRENCY);
        await Promise.allSettled(
          batch.map((agentDir) => scanAgent(agentDir, agentsPath, inflight, db, ollama, config, logger))
        );
      }
    } finally {
      _isScanning = false;
    }
  };

  // Run once on startup after a brief delay to let the plugin settle
  const startupTimeout = setTimeout(() => {
    scanDirs().catch((err) => {
      logger?.(`[noesis/scanner] Startup scan error: ${err}`);
    });
  }, 3000);

  const intervalId = setInterval(() => {
    scanDirs().catch((err) => {
      logger?.(`[noesis/scanner] Periodic scan error: ${err}`);
    });
  }, intervalMs);

  logger?.(`[noesis/scanner] Session scanner started (interval: ${config.sessionScanIntervalMinutes ?? 5}min)`);

  return {
    close() {
      clearTimeout(startupTimeout);
      clearInterval(intervalId);
      inflight.clear();
    },
  };
}
