/**
 * Noesis — QMD session file watcher
 *
 * Watches for new/updated QMD session JSONL files and auto-indexes them.
 * QMD session files live at: ~/.openclaw/sessions/<sessionId>.jsonl
 */

import fs from "fs";
import path from "path";
import os from "os";
import { randomUUID } from "crypto";
import { watch as chokidarWatch } from "chokidar";
import { OllamaClient, chunkText, contentChecksum } from "./ollama.js";
import { NoesisDB } from "./lancedb.js";
import { MemoryEntry, MemoryType, NoesisConfig } from "./types.js";

const QMD_SESSIONS_PATH = path.join(os.homedir(), ".openclaw", "sessions");
const DEBOUNCE_MS = 2000;

export interface SessionWatcher {
  close(): void;
}

/**
 * Start watching QMD session files and indexing them as they're written.
 *
 * Returns a handle with a close() method for cleanup.
 */
export function startQmdWatcher(
  db: NoesisDB,
  ollama: OllamaClient,
  config: NoesisConfig,
  logger?: (msg: string) => void
): SessionWatcher {
  // Ensure sessions directory exists
  if (!fs.existsSync(QMD_SESSIONS_PATH)) {
    fs.mkdirSync(QMD_SESSIONS_PATH, { recursive: true });
  }

  const pending = new Map<string, NodeJS.Timeout>();

  const watcher = chokidarWatch(`${QMD_SESSIONS_PATH}/**/*.jsonl`, {
    persistent: true,
    ignoreInitial: false, // index existing files on startup
    awaitWriteFinish: { stabilityThreshold: 500, pollInterval: 100 },
  });

  const handleFile = (filePath: string) => {
    // Debounce — wait for writes to settle
    const existing = pending.get(filePath);
    if (existing) clearTimeout(existing);
    pending.set(
      filePath,
      setTimeout(async () => {
        pending.delete(filePath);
        try {
          await indexSessionFile(filePath, db, ollama, config, logger);
        } catch (err) {
          logger?.(`[noesis/watcher] Error indexing ${filePath}: ${err}`);
        }
      }, DEBOUNCE_MS)
    );
  };

  watcher.on("add", handleFile);
  watcher.on("change", handleFile);
  watcher.on("error", (err) => logger?.(`[noesis/watcher] Watcher error: ${err}`));

  logger?.(`[noesis/watcher] Watching QMD sessions at: ${QMD_SESSIONS_PATH}`);

  return {
    close() {
      watcher.close();
      for (const t of pending.values()) clearTimeout(t);
      pending.clear();
    },
  };
}

const AGENTS_PATH = path.join(os.homedir(), ".openclaw", "agents");

/**
 * Watch agent memory directories for changes and auto-index .md files.
 * Watches: ~/.openclaw/agents/<agentId>/workspace/memory/*.md
 *
 * On add/change: parse markdown, split by ## headings, infer memory type,
 * chunk, embed, and upsert to LanceDB. Checksum dedup prevents duplicates.
 */
export function startMemoryWatcher(
  db: NoesisDB,
  ollama: OllamaClient,
  config: NoesisConfig,
  logger?: (msg: string) => void
): void {
  if (!fs.existsSync(AGENTS_PATH)) {
    logger?.(`[noesis/memory-watcher] Agents dir not found: ${AGENTS_PATH}`);
    return;
  }

  // Build glob pattern for all agent memory dirs
  const globPattern = `${AGENTS_PATH}/*/workspace/memory/*.md`;

  const pending = new Map<string, NodeJS.Timeout>();

  const watcher = chokidarWatch(globPattern, {
    persistent: true,
    ignoreInitial: false, // index existing files on startup
    awaitWriteFinish: { stabilityThreshold: 1000, pollInterval: 100 },
  });

  const handleFile = (filePath: string) => {
    const existing = pending.get(filePath);
    if (existing) clearTimeout(existing);
    pending.set(
      filePath,
      setTimeout(async () => {
        pending.delete(filePath);
        try {
          await indexMemoryFile(filePath, db, ollama, config, logger);
        } catch (err) {
          logger?.(`[noesis/memory-watcher] Error indexing ${filePath}: ${err}`);
        }
      }, DEBOUNCE_MS)
    );
  };

  watcher.on("add", handleFile);
  watcher.on("change", handleFile);
  watcher.on("error", (err) => logger?.(`[noesis/memory-watcher] Watcher error: ${err}`));

  logger?.(`[noesis/memory-watcher] Watching agent memory dirs: ${globPattern}`);
}

/**
 * Parse a markdown memory file and index its sections.
 *
 * Splits by ## headings, infers memory type per section,
 * chunks, embeds, and upserts to LanceDB.
 */
async function indexMemoryFile(
  filePath: string,
  db: NoesisDB,
  ollama: OllamaClient,
  config: NoesisConfig,
  logger?: (msg: string) => void
): Promise<void> {
  // Extract agentId from path: ~/.openclaw/agents/<agentId>/workspace/memory/<file>
  const relative = path.relative(AGENTS_PATH, filePath);
  const parts = relative.split(path.sep);
  const agentId = parts.length >= 4 ? parts[0] : "unknown";

  const raw = fs.readFileSync(filePath, "utf-8");
  const sections = splitByHeadings(raw);
  if (sections.length === 0) return;

  const allChunks: Array<{ content: string; chunk: string }> = [];
  for (const section of sections) {
    if (!section.trim() || section.trim().length < 20) continue;
    const type = inferMemoryType(section);
    const chunks = chunkText(section, config.chunkSize, config.chunkOverlap);
    for (const chunk of chunks) {
      allChunks.push({ content: section, chunk });
    }
  }

  if (allChunks.length === 0) return;

  const BATCH = 8;
  let indexed = 0;

  for (let i = 0; i < allChunks.length; i += BATCH) {
    const batch = allChunks.slice(i, i + BATCH);
    try {
      const embeddings = await ollama.embedBatch(batch.map((b) => b.chunk));
      const autoPriorityByType: Record<MemoryType, number> = {
        decision: 85, preference: 80, context: 60, fact: 30, session: 20,
      };
      const entries: MemoryEntry[] = batch.map((b, j) => ({
        id: randomUUID(),
        agentId,
        sessionId: "memory-dir",
        content: b.content,
        chunk: b.chunk,
        embedding: embeddings[j],
        memoryType: inferMemoryType(b.content),
        createdAt: Date.now(),
        sourcePath: filePath,
        checksum: contentChecksum(b.content, agentId),
        tags: ["memory-dir"],
        priority: autoPriorityByType[inferMemoryType(b.content)] ?? 30,
      }));
      const n = await db.upsertEntries(entries);
      indexed += n;
    } catch (err) {
      logger?.(`[noesis/memory-watcher] Embed batch error for ${filePath}: ${err}`);
    }
  }

  if (indexed > 0) {
    logger?.(`[noesis/memory-watcher] Indexed ${indexed} chunk(s) from ${path.basename(filePath)} (agent: ${agentId})`);
  }
}

// ─── helpers (duplicated from migrator.ts to avoid circular deps) ──────────────

function splitByHeadings(text: string): string[] {
  const lines = text.split("\n");
  const sections: string[] = [];
  let current: string[] = [];
  for (const line of lines) {
    if (line.startsWith("## ") && current.length > 0) {
      sections.push(current.join("\n").trim());
      current = [line];
    } else {
      current.push(line);
    }
  }
  if (current.length > 0) sections.push(current.join("\n").trim());
  return sections.filter(Boolean);
}

function inferMemoryType(content: string): MemoryType {
  const lower = content.toLowerCase();
  if (/\bdecid|decision|chose|choice\b/.test(lower)) return "decision";
  if (/\bprefer|always|never|style|like to|don't like\b/.test(lower)) return "preference";
  if (/\bsession|today|yesterday|this morning|last night\b/.test(lower)) return "context";
  return "fact";
}

/**
 * Parse a QMD JSONL session file and index its messages.
 *
 * QMD format: one JSON object per line, each representing a message event.
 * We extract assistant + user messages with content, chunk them, and embed.
 */
async function indexSessionFile(
  filePath: string,
  db: NoesisDB,
  ollama: OllamaClient,
  config: NoesisConfig,
  logger?: (msg: string) => void
): Promise<void> {
  const raw = fs.readFileSync(filePath, "utf-8");
  const lines = raw.split("\n").filter((l) => l.trim());
  if (lines.length === 0) return;

  // Extract session ID and agent ID from file path
  // Expected pattern: ~/.openclaw/sessions/<agentId>/<sessionId>.jsonl
  // or: ~/.openclaw/sessions/<sessionId>.jsonl
  const parts = path.relative(QMD_SESSIONS_PATH, filePath).split(path.sep);
  const agentId = parts.length >= 2 ? parts[0] : "unknown";
  const sessionId = path.basename(filePath, ".jsonl");

  const texts: string[] = [];

  for (const line of lines) {
    try {
      const entry = JSON.parse(line) as {
        role?: string;
        content?: string | Array<{ type: string; text?: string }>;
        type?: string;
        text?: string;
      };

      // Extract text content from various QMD event shapes
      let text = "";
      if (typeof entry.content === "string") {
        text = entry.content;
      } else if (Array.isArray(entry.content)) {
        text = entry.content
          .filter((b) => b.type === "text" && b.text)
          .map((b) => b.text!)
          .join(" ");
      } else if (typeof entry.text === "string") {
        text = entry.text;
      }

      if (text.trim().length > 20) {
        texts.push(text.trim());
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
        memoryType: "session",
        createdAt: Date.now(),
        sourcePath: filePath,
        checksum: contentChecksum(b.content, agentId),
        tags: ["qmd-session"],
        priority: 20,
      }));
      const n = await db.upsertEntries(entries);
      indexed += n;
    } catch (err) {
      logger?.(`[noesis/watcher] Embed batch error for ${filePath}: ${err}`);
    }
  }

  if (indexed > 0) {
    logger?.(`[noesis/watcher] Indexed ${indexed} chunk(s) from session ${sessionId} (agent: ${agentId})`);
  }
}
