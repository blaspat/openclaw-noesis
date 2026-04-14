/**
 * Noesis — LanceDB storage layer
 *
 * Uses @lancedb/lancedb (Node.js native SDK v0.27+).
 * Single persistent connection for connection pooling.
 */
import * as lancedb from "@lancedb/lancedb";
import { Schema, Field, Utf8, Int64, FixedSizeList, Float32, List } from "apache-arrow";
import os from "os";
import fs from "fs";
const TABLE_NAME = "memories";
const ARCHIVE_TABLE_NAME = "memories_archive";
export class NoesisDB {
    config;
    conn = null;
    table = null;
    archiveTable = null;
    embeddingDim = 768;
    dbPath;
    indexCreated = false;
    constructor(config) {
        this.config = config;
        this.dbPath = resolvePath(config.lanceDbPath);
    }
    /**
     * Connect to LanceDB (idempotent — call once at plugin startup).
     */
    async connect(embeddingDim = 768) {
        if (this.conn)
            return; // already connected
        this.embeddingDim = embeddingDim;
        // Ensure db directory exists
        fs.mkdirSync(this.dbPath, { recursive: true });
        this.conn = await lancedb.connect(this.dbPath);
        this.table = await this.openOrCreateTable();
        this.archiveTable = await this.openOrCreateArchiveTable();
    }
    /**
     * Disconnect — used for graceful shutdown.
     */
    async disconnect() {
        this.table = null;
        this.archiveTable = null;
        this.conn = null;
    }
    async openOrCreateTable() {
        if (!this.conn)
            throw new Error("Not connected");
        const existing = await this.conn.tableNames();
        if (existing.includes(TABLE_NAME)) {
            return this.conn.openTable(TABLE_NAME);
        }
        // Create table with explicit schema
        const schema = this.buildSchema(this.embeddingDim);
        const tbl = await this.conn.createEmptyTable(TABLE_NAME, schema);
        return tbl;
    }
    async openOrCreateArchiveTable() {
        if (!this.conn)
            throw new Error("Not connected");
        const existing = await this.conn.tableNames();
        if (existing.includes(ARCHIVE_TABLE_NAME)) {
            return this.conn.openTable(ARCHIVE_TABLE_NAME);
        }
        const schema = this.buildSchema(this.embeddingDim);
        return this.conn.createEmptyTable(ARCHIVE_TABLE_NAME, schema);
    }
    buildSchema(dim) {
        return new Schema([
            new Field("id", new Utf8(), false),
            new Field("agentId", new Utf8(), false),
            new Field("sessionId", new Utf8(), false),
            new Field("content", new Utf8(), false),
            new Field("chunk", new Utf8(), false),
            new Field("embedding", new FixedSizeList(dim, new Field("item", new Float32(), true)), false),
            new Field("memoryType", new Utf8(), false),
            new Field("priority", new Int64(), false),
            new Field("expiresAt", new Int64(), false),
            new Field("createdAt", new Int64(), false),
            new Field("sourcePath", new Utf8(), false),
            new Field("checksum", new Utf8(), false),
            new Field("tags", new List(new Field("item", new Utf8(), true)), false),
        ]);
    }
    getTable() {
        if (!this.table)
            throw new Error("[noesis] Not connected to LanceDB. Call connect() first.");
        return this.table;
    }
    getArchiveTable() {
        if (!this.archiveTable)
            throw new Error("[noesis] Not connected to archive table. Call connect() first.");
        return this.archiveTable;
    }
    /**
     * Insert a batch of memory entries (upsert by checksum — idempotent).
     * Returns count of actually inserted entries.
     */
    async upsertEntries(entries) {
        if (entries.length === 0)
            return 0;
        const tbl = this.getTable();
        // Dedup: check existing checksums in batch
        const checksums = entries.map((e) => e.checksum);
        const quotedChecksums = checksums.map((c) => `'${c}'`).join(", ");
        let existingChecksums = new Set();
        try {
            const existing = await tbl
                .query()
                .where(`checksum IN (${quotedChecksums})`)
                .select(["checksum"])
                .limit(checksums.length + 1)
                .toArray();
            existingChecksums = new Set(existing.map((r) => String(r.checksum)));
        }
        catch {
            // table might be empty — that's fine
        }
        const newEntries = entries.filter((e) => !existingChecksums.has(e.checksum));
        if (newEntries.length === 0)
            return 0;
        const rows = newEntries.map((e) => ({
            id: e.id,
            agentId: e.agentId,
            sessionId: e.sessionId,
            content: e.content,
            chunk: e.chunk,
            embedding: e.embedding,
            memoryType: e.memoryType,
            createdAt: BigInt(e.createdAt),
            sourcePath: e.sourcePath,
            checksum: e.checksum,
            priority: e.priority ?? 0,
            expiresAt: e.expiresAt ?? 0,
            tags: Array.isArray(e.tags) ? e.tags.map(String) : [],
        }));
        await tbl.add(rows);
        return newEntries.length;
    }
    /**
     * Vector similarity search using ANN index.
     * Returns up to topK*2 candidates for hybrid merge.
     */
    async vectorSearch(embedding, topK, agentId, memoryType, crossAgent) {
        const tbl = this.getTable();
        let query = tbl
            .vectorSearch(embedding)
            .select(["id", "agentId", "sessionId", "content", "memoryType", "createdAt", "sourcePath", "tags"])
            .limit(topK * 2)
            .nprobes(this.config.annNprobe);
        const filters = [];
        if (agentId && !crossAgent) {
            filters.push(`agentId = '${agentId}'`);
        }
        if (memoryType) {
            filters.push(`memoryType = '${memoryType}'`);
        }
        if (filters.length > 0) {
            query = query.where(filters.join(" AND "));
        }
        const results = await query.toArray();
        return results.map((r) => ({
            id: String(r.id),
            agentId: String(r.agentId),
            sessionId: String(r.sessionId),
            content: String(r.content),
            memoryType: String(r.memoryType),
            createdAt: Number(r.createdAt),
            sourcePath: String(r.sourcePath),
            tags: Array.isArray(r.tags) ? r.tags.map(String) : [],
            score: typeof r._distance === "number" ? 1 / (1 + r._distance) : 0,
            priority: Number(r.priority ?? 0),
            expiresAt: Number(r.expiresAt ?? 0),
        }));
    }
    /**
     * Full-text keyword search (BM25 via LanceDB FTS).
     */
    async fullTextSearch(queryText, topK, agentId, memoryType, crossAgent) {
        const tbl = this.getTable();
        try {
            // Ensure FTS index exists on content column
            await this.ensureFtsIndex();
            let query = tbl.search(queryText, "fts")
                .select(["id", "agentId", "sessionId", "content", "memoryType", "createdAt", "sourcePath", "tags"])
                .limit(topK);
            const filters = [];
            if (agentId && !crossAgent) {
                filters.push(`agentId = '${agentId}'`);
            }
            if (memoryType) {
                filters.push(`memoryType = '${memoryType}'`);
            }
            if (filters.length > 0) {
                query = query.where(filters.join(" AND "));
            }
            const results = await query.toArray();
            return results.map((r, i) => ({
                id: String(r.id),
                agentId: String(r.agentId),
                sessionId: String(r.sessionId),
                content: String(r.content),
                memoryType: String(r.memoryType),
                createdAt: Number(r.createdAt),
                sourcePath: String(r.sourcePath),
                tags: Array.isArray(r.tags) ? r.tags.map(String) : [],
                score: 1 / (1 + i), // rank-based score
                priority: Number(r.priority ?? 0),
                expiresAt: Number(r.expiresAt ?? 0),
            }));
        }
        catch {
            // FTS not available or no index yet — return empty
            return [];
        }
    }
    /**
     * Retrieve by ID.
     */
    async getById(id) {
        const tbl = this.getTable();
        const results = await tbl
            .query()
            .where(`id = '${id}'`)
            .select(["id", "agentId", "sessionId", "content", "chunk", "embedding", "memoryType", "priority", "expiresAt", "createdAt", "sourcePath", "checksum", "tags"])
            .limit(1)
            .toArray();
        if (results.length === 0)
            return null;
        const r = results[0];
        return rowToEntry(r);
    }
    /**
     * Recall entries by agent/session, ordered by createdAt desc.
     */
    async recall(agentId, sessionId, limit = 50) {
        const tbl = this.getTable();
        const filters = [];
        if (agentId)
            filters.push(`agentId = '${agentId}'`);
        if (sessionId)
            filters.push(`sessionId = '${sessionId}'`);
        let query = tbl
            .query()
            .select(["id", "agentId", "sessionId", "content", "chunk", "embedding", "memoryType", "priority", "expiresAt", "createdAt", "sourcePath", "checksum", "tags"])
            .limit(limit);
        if (filters.length > 0) {
            query = query.where(filters.join(" AND "));
        }
        const results = await query.toArray();
        return results
            .map(rowToEntry)
            .sort((a, b) => b.createdAt - a.createdAt);
    }
    /**
     * Delete a single entry by ID.
     */
    async deleteById(id) {
        const tbl = this.getTable();
        await tbl.delete(`id = '${id}'`);
    }
    /**
     * Update priority and/or expiresAt on an existing entry.
     * Returns the updated entry, or null if not found.
     */
    async updateEntry(id, patch) {
        const tbl = this.getTable();
        const now = Date.now();
        try {
            const rows = await tbl
                .query()
                .where(`id = '${id}'`)
                .limit(1)
                .toArray();
            if (rows.length === 0)
                return null;
            const r = rows[0];
            const updates = {};
            if (patch.priority !== undefined)
                updates["priority"] = String(patch.priority);
            if (patch.expiresAt !== undefined)
                updates["expiresAt"] = String(patch.expiresAt);
            if (Object.keys(updates).length === 0)
                return rowToEntry(r);
            await tbl.update({ where: `id = '${id}'`, values: updates });
            // Refetch after update
            const refreshed = await tbl.query().where(`id = '${id}'`).limit(1).toArray();
            return refreshed.length > 0 ? rowToEntry(refreshed[0]) : null;
        }
        catch {
            return null;
        }
    }
    /**
     * Export all entries, optionally filtered by agentId / memoryType.
     */
    async exportAll(agentId, memoryType) {
        const tbl = this.getTable();
        const filters = [];
        if (agentId)
            filters.push(`agentId = '${agentId}'`);
        if (memoryType)
            filters.push(`memoryType = '${memoryType}'`);
        let query = tbl.query().select(["id", "agentId", "sessionId", "content", "chunk", "memoryType", "priority", "expiresAt", "createdAt", "sourcePath", "checksum", "tags"]).limit(100_000);
        if (filters.length > 0)
            query = query.where(filters.join(" AND "));
        const rows = await query.toArray();
        return rows.map(rowToEntry);
    }
    /**
     * Count entries (total or by agentId filter).
     */
    async count(agentId) {
        const tbl = this.getTable();
        if (agentId) {
            return tbl.countRows(`agentId = '${agentId}'`);
        }
        return tbl.countRows();
    }
    /**
     * Count entries grouped by a string column.
     */
    async countByColumn(column) {
        const tbl = this.getTable();
        try {
            const rows = await tbl.query().select([column]).limit(100_000).toArray();
            const counts = {};
            for (const row of rows) {
                const key = String(row[column] ?? "unknown");
                counts[key] = (counts[key] ?? 0) + 1;
            }
            return counts;
        }
        catch {
            return {};
        }
    }
    /**
     * Move expired entries from active table to archive table.
     * Expired = expiresAt > 0 AND expiresAt < now.
     * Entries are MOVED (not copy + delete), so they leave no trace in active table.
     * Archive is searchable via noesis_search_archive.
     * Returns count of entries archived.
     */
    async archiveExpired() {
        const tbl = this.getTable();
        const now = Date.now();
        let archived = 0;
        try {
            // Select full rows for expired entries
            const toArchive = await tbl
                .query()
                .where(`expiresAt > 0 AND expiresAt < ${now}`)
                .select(["id", "agentId", "sessionId", "content", "chunk", "embedding", "memoryType", "priority", "expiresAt", "createdAt", "sourcePath", "checksum", "tags"])
                .limit(10_000)
                .toArray();
            if (toArchive.length === 0)
                return 0;
            // Insert into archive
            const archiveTbl = this.getArchiveTable();
            const rows = toArchive.map((r) => ({
                id: String(r.id),
                agentId: String(r.agentId),
                sessionId: String(r.sessionId),
                content: String(r.content),
                chunk: String(r.chunk ?? ""),
                embedding: r.embedding,
                memoryType: String(r.memoryType ?? "fact"),
                priority: Number(r.priority ?? 0),
                expiresAt: Number(r.expiresAt ?? 0),
                createdAt: Number(r.createdAt ?? 0),
                sourcePath: String(r.sourcePath ?? ""),
                checksum: String(r.checksum ?? ""),
                tags: Array.isArray(r.tags) ? r.tags.map(String) : [],
            }));
            await archiveTbl.add(rows);
            // Delete from active table
            const ids = toArchive.map((r) => `'${r.id}'`).join(", ");
            await tbl.delete(`id IN (${ids})`);
            archived = toArchive.length;
        }
        catch (err) {
            // archive table not ready — skip silently
        }
        return archived;
    }
    /**
     * Count entries in the archive table.
     */
    async countArchive() {
        try {
            return this.getArchiveTable().countRows();
        }
        catch {
            return 0;
        }
    }
    /**
     * Count archive entries grouped by column (e.g. by agentId, by memoryType).
     */
    async countByColumnArchive(column) {
        try {
            const tbl = this.getArchiveTable();
            const rows = await tbl.query().select([column]).limit(100_000).toArray();
            const counts = {};
            for (const row of rows) {
                const key = String(row[column] ?? "unknown");
                counts[key] = (counts[key] ?? 0) + 1;
            }
            return counts;
        }
        catch {
            return {};
        }
    }
    /**
     * Search archive table (for "older memories" queries).
     * Returns top-K results from archive, sorted by score.
     */
    async searchArchive(embedding, topK, agentId, memoryType, crossAgent) {
        const tbl = this.getArchiveTable();
        try {
            await this.ensureAnnIndexArchive();
            const filters = [];
            if (agentId && !crossAgent)
                filters.push(`agentId = '${agentId}'`);
            if (memoryType)
                filters.push(`memoryType = '${memoryType}'`);
            let query = tbl
                .vectorSearch(embedding)
                .select(["id", "agentId", "sessionId", "content", "memoryType", "createdAt", "sourcePath", "tags", "priority", "expiresAt"])
                .limit(topK);
            if (filters.length > 0) {
                query = query.where(filters.join(" AND "));
            }
            const results = await query.toArray();
            return results.map((r) => ({
                id: String(r.id),
                agentId: String(r.agentId),
                sessionId: String(r.sessionId),
                content: String(r.content),
                memoryType: String(r.memoryType),
                createdAt: Number(r.createdAt),
                sourcePath: String(r.sourcePath),
                tags: Array.isArray(r.tags) ? r.tags.map(String) : [],
                score: typeof r._distance === "number" ? 1 / (1 + r._distance) : 0,
                priority: Number(r.priority ?? 0),
                expiresAt: Number(r.expiresAt ?? 0),
            }));
        }
        catch {
            return [];
        }
    }
    /**
     * Full-text search in archive table.
     */
    async fullTextSearchArchive(queryText, topK, agentId, memoryType, crossAgent) {
        const tbl = this.getArchiveTable();
        try {
            await this.ensureFtsIndexArchive();
            const filters = [];
            if (agentId && !crossAgent)
                filters.push(`agentId = '${agentId}'`);
            if (memoryType)
                filters.push(`memoryType = '${memoryType}'`);
            let query = tbl.search(queryText, "fts")
                .select(["id", "agentId", "sessionId", "content", "memoryType", "createdAt", "sourcePath", "tags", "priority", "expiresAt"])
                .limit(topK);
            if (filters.length > 0) {
                query = query.where(filters.join(" AND "));
            }
            const results = await query.toArray();
            return results.map((r, i) => ({
                id: String(r.id),
                agentId: String(r.agentId),
                sessionId: String(r.sessionId),
                content: String(r.content),
                memoryType: String(r.memoryType),
                createdAt: Number(r.createdAt),
                sourcePath: String(r.sourcePath),
                tags: Array.isArray(r.tags) ? r.tags.map(String) : [],
                score: 1 / (1 + i),
                priority: Number(r.priority ?? 0),
                expiresAt: Number(r.expiresAt ?? 0),
            }));
        }
        catch {
            return [];
        }
    }
    /**
     * Get aggregate stats.
     */
    async stats() {
        const total = await this.count();
        const archiveTotal = await this.countArchive();
        const [byAgent, byMemoryType, byPriority, byAgentArchive, byMemoryTypeArchive] = await Promise.all([
            this.countByColumn("agentId"),
            this.countByColumn("memoryType"),
            this.countByColumn("priority"),
            this.countByColumnArchive("agentId"),
            this.countByColumnArchive("memoryType"),
        ]);
        return {
            totalEntries: total,
            byAgent,
            byMemoryType,
            byPriority,
            dbPath: this.dbPath,
            embeddingModel: this.config.embeddingModel,
            ollamaEndpoint: this.config.ollamaEndpoint,
            expiredEntries: 0,
            expiredAndDeleted: 0,
            archiveEntries: archiveTotal,
            byAgentArchive,
            byMemoryTypeArchive,
        };
    }
    /**
     * Ensure IVF-PQ ANN index exists on the embedding column.
     * Called after initial data ingestion (requires at least 256 rows).
     */
    async ensureAnnIndex() {
        if (this.indexCreated)
            return;
        const tbl = this.getTable();
        const count = await tbl.countRows();
        if (count < 256)
            return; // IVF-PQ needs enough data to build partitions
        try {
            const indices = await tbl.listIndices();
            const hasVectorIndex = indices.some((idx) => idx.columns?.includes("embedding") || idx.name?.includes("embedding"));
            if (!hasVectorIndex) {
                await tbl.createIndex("embedding", {
                    config: lancedb.Index.ivfPq({
                        numSubVectors: this.config.annNumSubvectors,
                    }),
                });
            }
            this.indexCreated = true;
        }
        catch {
            // Index creation is best-effort — ANN just won't be used
        }
    }
    /**
     * Ensure IVF-PQ ANN index exists on the archive table's embedding column.
     */
    async ensureAnnIndexArchive() {
        if (this.archiveIndexCreated)
            return;
        const tbl = this.getArchiveTable();
        const count = await tbl.countRows();
        if (count < 256)
            return;
        try {
            const indices = await tbl.listIndices();
            const hasVectorIndex = indices.some((idx) => idx.columns?.includes("embedding") || idx.name?.includes("embedding"));
            if (!hasVectorIndex) {
                await tbl.createIndex("embedding", {
                    config: lancedb.Index.ivfPq({
                        numSubVectors: this.config.annNumSubvectors,
                    }),
                });
            }
            this.archiveIndexCreated = true;
        }
        catch {
            // Index creation is best-effort
        }
    }
    /**
     * Ensure FTS index exists on the content column.
     */
    async ensureFtsIndex() {
        const tbl = this.getTable();
        try {
            const indices = await tbl.listIndices();
            const hasFtsIndex = indices.some((idx) => idx.columns?.includes("content") && idx.indexType === "FTS");
            if (!hasFtsIndex) {
                await tbl.createIndex("content", {
                    config: lancedb.Index.fts(),
                });
            }
        }
        catch {
            // FTS index creation is best-effort
        }
    }
    async ensureFtsIndexArchive() {
        const tbl = this.getArchiveTable();
        try {
            const indices = await tbl.listIndices();
            const hasFtsIndex = indices.some((idx) => idx.columns?.includes("content") && idx.indexType === "FTS");
            if (!hasFtsIndex) {
                await tbl.createIndex("content", {
                    config: lancedb.Index.fts(),
                });
            }
        }
        catch {
            // FTS index creation is best-effort
        }
    }
    /**
     * Query entries by minimum priority threshold, optionally filtered by agent
     * and recency. Used by the context engine Assemble hook to inject high-priority
     * memories into every model run.
     */
    async queryByPriority(minPriority, agentId, limit = 50, maxAgeDays) {
        const tbl = this.getTable();
        const now = Date.now();
        let query = tbl
            .query()
            .select(["id", "agentId", "sessionId", "content", "chunk", "embedding", "memoryType", "priority", "expiresAt", "createdAt", "sourcePath", "checksum", "tags"])
            .where(`priority >= ${minPriority}`)
            .limit(limit);
        const results = await query.toArray();
        const entries = results.map(rowToEntry);
        // Filter out expired
        const active = entries.filter(e => e.expiresAt === 0 || e.expiresAt > now);
        // Filter by recency if maxAgeDays set
        const cutoff = maxAgeDays ? now - maxAgeDays * 24 * 60 * 60 * 1000 : 0;
        const recent = cutoff ? active.filter(e => e.createdAt >= cutoff) : active;
        // Sort by priority desc, then createdAt desc
        return recent.sort((a, b) => {
            if (b.priority !== a.priority)
                return (b.priority ?? 0) - (a.priority ?? 0);
            return b.createdAt - a.createdAt;
        });
    }
}
// ─── helpers ──────────────────────────────────────────────────────────────────
function resolvePath(p) {
    return p.replace(/^~/, os.homedir());
}
function rowToEntry(r) {
    let embedding = [];
    if (r.embedding) {
        if (r.embedding instanceof Float32Array || r.embedding instanceof Array) {
            embedding = Array.from(r.embedding);
        }
    }
    return {
        id: String(r.id ?? ""),
        agentId: String(r.agentId ?? ""),
        sessionId: String(r.sessionId ?? ""),
        content: String(r.content ?? ""),
        chunk: String(r.chunk ?? ""),
        embedding,
        memoryType: (r.memoryType ?? "fact"),
        priority: Number(r.priority ?? 0),
        expiresAt: Number(r.expiresAt ?? 0),
        createdAt: Number(r.createdAt ?? 0),
        sourcePath: String(r.sourcePath ?? ""),
        checksum: String(r.checksum ?? ""),
        tags: Array.isArray(r.tags) ? r.tags.map(String) : [],
    };
}
//# sourceMappingURL=lancedb.js.map