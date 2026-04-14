/**
 * Noesis — MD → LanceDB migrator
 *
 * Scans agent memory files, chunks them, embeds via Ollama,
 * and upserts to LanceDB. Fully idempotent via checksum dedup.
 */
import { OllamaClient } from "./ollama.js";
import { NoesisDB } from "./lancedb.js";
import { ImportResult, NoesisConfig } from "./types.js";
/**
 * Import markdown memory files for a single agent.
 *
 * Scans: ~/.openclaw/agents/<agentId>/workspace/memory/*.md
 * Also scans: ~/.openclaw/agents/<agentId>/workspace/MEMORY.md
 */
export declare function importMarkdownFiles(agentId: string, db: NoesisDB, ollama: OllamaClient, config: NoesisConfig, logger?: (msg: string) => void): Promise<ImportResult>;
/**
 * Import from all agents found in ~/.openclaw/agents/
 */
export declare function importAllAgents(db: NoesisDB, ollama: OllamaClient, config: NoesisConfig, logger?: (msg: string) => void): Promise<ImportResult[]>;
//# sourceMappingURL=migrator.d.ts.map