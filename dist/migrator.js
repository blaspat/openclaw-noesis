/**
 * Noesis — MD → LanceDB migrator
 *
 * Scans agent memory files, chunks them, embeds via Ollama,
 * and upserts to LanceDB. Fully idempotent via checksum dedup.
 */
import fs from "fs";
import path from "path";
import os from "os";
import { randomUUID } from "crypto";
import { chunkText, contentChecksum } from "./ollama.js";
const BATCH_SIZE = 10; // embed in batches to control Ollama load
/**
 * Import markdown memory files for a single agent.
 *
 * Scans: ~/.openclaw/agents/<agentId>/workspace/memory/*.md
 * Also scans: ~/.openclaw/agents/<agentId>/workspace/MEMORY.md
 */
export async function importMarkdownFiles(agentId, db, ollama, config, logger) {
    const result = { indexed: 0, skipped: 0, errors: 0, agentId };
    const agentBase = path.join(os.homedir(), ".openclaw", "agents", agentId, "workspace");
    const filesToScan = [];
    // Add MEMORY.md if exists
    const memoryMd = path.join(agentBase, "MEMORY.md");
    if (fs.existsSync(memoryMd))
        filesToScan.push(memoryMd);
    // Add daily memory files
    const memoryDir = path.join(agentBase, "memory");
    if (fs.existsSync(memoryDir)) {
        const files = fs.readdirSync(memoryDir)
            .filter((f) => f.endsWith(".md"))
            .map((f) => path.join(memoryDir, f));
        filesToScan.push(...files);
    }
    if (filesToScan.length === 0) {
        logger?.(`[noesis/migrator] No markdown files found for agent '${agentId}'`);
        return result;
    }
    logger?.(`[noesis/migrator] Found ${filesToScan.length} file(s) for agent '${agentId}'`);
    // Collect all chunks across all files
    const allChunks = [];
    for (const filePath of filesToScan) {
        try {
            const raw = fs.readFileSync(filePath, "utf-8");
            const sections = splitByHeadings(raw);
            for (const section of sections) {
                if (!section.trim())
                    continue;
                const chunks = chunkText(section, config.chunkSize, config.chunkOverlap);
                for (const chunk of chunks) {
                    allChunks.push({
                        chunk,
                        content: section,
                        sourcePath: filePath,
                        memoryType: inferMemoryType(section),
                    });
                }
            }
        }
        catch (err) {
            logger?.(`[noesis/migrator] Error reading ${filePath}: ${err}`);
            result.errors++;
        }
    }
    if (allChunks.length === 0) {
        logger?.(`[noesis/migrator] No chunks to index for agent '${agentId}'`);
        return result;
    }
    // Process in batches
    for (let i = 0; i < allChunks.length; i += BATCH_SIZE) {
        const batch = allChunks.slice(i, i + BATCH_SIZE);
        try {
            const texts = batch.map((b) => b.chunk);
            const embeddings = await ollama.embedBatch(texts);
            const entries = batch.map((b, j) => {
                const checksum = contentChecksum(b.content, agentId);
                return {
                    id: randomUUID(),
                    agentId,
                    sessionId: "migration",
                    content: b.content,
                    chunk: b.chunk,
                    embedding: embeddings[j],
                    memoryType: b.memoryType,
                    priority: 0,
                    expiresAt: 0,
                    createdAt: Date.now(),
                    sourcePath: b.sourcePath,
                    checksum,
                    tags: [],
                };
            });
            const inserted = await db.upsertEntries(entries);
            result.indexed += inserted;
            result.skipped += entries.length - inserted;
            logger?.(`[noesis/migrator] Batch ${Math.ceil(i / BATCH_SIZE) + 1}: +${inserted} indexed, ${entries.length - inserted} skipped`);
        }
        catch (err) {
            logger?.(`[noesis/migrator] Batch error: ${err}`);
            result.errors += batch.length;
        }
    }
    // Try to build ANN index after a large import
    await db.ensureAnnIndex().catch(() => { });
    logger?.(`[noesis/migrator] Done for '${agentId}': ${result.indexed} indexed, ${result.skipped} skipped, ${result.errors} errors`);
    return result;
}
/**
 * Import from all agents found in ~/.openclaw/agents/
 */
export async function importAllAgents(db, ollama, config, logger) {
    const agentsDir = path.join(os.homedir(), ".openclaw", "agents");
    if (!fs.existsSync(agentsDir))
        return [];
    const agents = fs.readdirSync(agentsDir)
        .filter((name) => {
        const p = path.join(agentsDir, name);
        return fs.statSync(p).isDirectory();
    });
    const results = [];
    for (const agentId of agents) {
        const result = await importMarkdownFiles(agentId, db, ollama, config, logger);
        results.push(result);
    }
    return results;
}
// ─── helpers ──────────────────────────────────────────────────────────────────
/**
 * Split a markdown document into sections by ## headings.
 */
function splitByHeadings(text) {
    const lines = text.split("\n");
    const sections = [];
    let current = [];
    for (const line of lines) {
        if (line.startsWith("## ") && current.length > 0) {
            sections.push(current.join("\n").trim());
            current = [line];
        }
        else {
            current.push(line);
        }
    }
    if (current.length > 0)
        sections.push(current.join("\n").trim());
    return sections.filter(Boolean);
}
/**
 * Heuristically infer memory type from section content.
 */
function inferMemoryType(content) {
    const lower = content.toLowerCase();
    if (/\bdecid|decision|chose|choice\b/.test(lower))
        return "decision";
    if (/\bprefer|always|never|style|like to|don't like\b/.test(lower))
        return "preference";
    if (/\bsession|today|yesterday|this morning|last night\b/.test(lower))
        return "context";
    return "fact";
}
//# sourceMappingURL=migrator.js.map