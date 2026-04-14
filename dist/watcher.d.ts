/**
 * Noesis — QMD session file watcher
 *
 * Watches for new/updated QMD session JSONL files and auto-indexes them.
 * QMD session files live at: ~/.openclaw/sessions/<sessionId>.jsonl
 */
import { OllamaClient } from "./ollama.js";
import { NoesisDB } from "./lancedb.js";
import { NoesisConfig } from "./types.js";
export interface SessionWatcher {
    close(): void;
}
/**
 * Start watching QMD session files and indexing them as they're written.
 *
 * Returns a handle with a close() method for cleanup.
 */
export declare function startQmdWatcher(db: NoesisDB, ollama: OllamaClient, config: NoesisConfig, logger?: (msg: string) => void): SessionWatcher;
/**
 * Watch agent memory directories for changes and auto-index .md files.
 * Watches: ~/.openclaw/agents/<agentId>/workspace/memory/*.md
 *
 * On add/change: parse markdown, split by ## headings, infer memory type,
 * chunk, embed, and upsert to LanceDB. Checksum dedup prevents duplicates.
 */
export declare function startMemoryWatcher(db: NoesisDB, ollama: OllamaClient, config: NoesisConfig, logger?: (msg: string) => void): void;
//# sourceMappingURL=watcher.d.ts.map