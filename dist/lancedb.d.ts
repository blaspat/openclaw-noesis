/**
 * Noesis — LanceDB storage layer
 *
 * Uses @lancedb/lancedb (Node.js native SDK v0.27+).
 * Single persistent connection for connection pooling.
 */
import { MemoryEntry, MemoryType, NoesisConfig, NoesisStats } from "./types.js";
export declare class NoesisDB {
    private config;
    private conn;
    private table;
    private archiveTable;
    private embeddingDim;
    private dbPath;
    private indexCreated;
    constructor(config: NoesisConfig);
    /**
     * Connect to LanceDB (idempotent — call once at plugin startup).
     */
    connect(embeddingDim?: number): Promise<void>;
    /**
     * Disconnect — used for graceful shutdown.
     */
    disconnect(): Promise<void>;
    private openOrCreateTable;
    private openOrCreateArchiveTable;
    private buildSchema;
    private getTable;
    private getArchiveTable;
    /**
     * Insert a batch of memory entries (upsert by checksum — idempotent).
     * Returns count of actually inserted entries.
     */
    upsertEntries(entries: MemoryEntry[]): Promise<number>;
    /**
     * Vector similarity search using ANN index.
     * Returns up to topK*2 candidates for hybrid merge.
     */
    vectorSearch(embedding: number[], topK: number, agentId?: string, memoryType?: MemoryType, crossAgent?: boolean): Promise<Array<{
        id: string;
        agentId: string;
        sessionId: string;
        content: string;
        memoryType: string;
        createdAt: number;
        sourcePath: string;
        tags: string[];
        score: number;
        priority: number;
        expiresAt: number;
    }>>;
    /**
     * Full-text keyword search (BM25 via LanceDB FTS).
     */
    fullTextSearch(queryText: string, topK: number, agentId?: string, memoryType?: MemoryType, crossAgent?: boolean): Promise<Array<{
        id: string;
        agentId: string;
        sessionId: string;
        content: string;
        memoryType: string;
        createdAt: number;
        sourcePath: string;
        tags: string[];
        score: number;
        priority: number;
        expiresAt: number;
    }>>;
    /**
     * Retrieve by ID.
     */
    getById(id: string): Promise<MemoryEntry | null>;
    /**
     * Recall entries by agent/session, ordered by createdAt desc.
     */
    recall(agentId?: string, sessionId?: string, limit?: number): Promise<MemoryEntry[]>;
    /**
     * Delete a single entry by ID.
     */
    deleteById(id: string): Promise<void>;
    /**
     * Update priority and/or expiresAt on an existing entry.
     * Returns the updated entry, or null if not found.
     */
    updateEntry(id: string, patch: {
        priority?: number;
        expiresAt?: number;
    }): Promise<MemoryEntry | null>;
    /**
     * Export all entries, optionally filtered by agentId / memoryType.
     */
    exportAll(agentId?: string, memoryType?: string): Promise<MemoryEntry[]>;
    /**
     * Count entries (total or by agentId filter).
     */
    count(agentId?: string): Promise<number>;
    /**
     * Count entries grouped by a string column.
     */
    countByColumn(column: string): Promise<Record<string, number>>;
    /**
     * Move expired entries from active table to archive table.
     * Expired = expiresAt > 0 AND expiresAt < now.
     * Entries are MOVED (not copy + delete), so they leave no trace in active table.
     * Archive is searchable via noesis_search_archive.
     * Returns count of entries archived.
     */
    archiveExpired(): Promise<number>;
    /**
     * Count entries in the archive table.
     */
    countArchive(): Promise<number>;
    /**
     * Count archive entries grouped by column (e.g. by agentId, by memoryType).
     */
    countByColumnArchive(column: string): Promise<Record<string, number>>;
    /**
     * Search archive table (for "older memories" queries).
     * Returns top-K results from archive, sorted by score.
     */
    searchArchive(embedding: number[], topK: number, agentId?: string, memoryType?: MemoryType, crossAgent?: boolean): Promise<Array<{
        id: string;
        agentId: string;
        sessionId: string;
        content: string;
        memoryType: string;
        createdAt: number;
        sourcePath: string;
        tags: string[];
        score: number;
        priority: number;
        expiresAt: number;
    }>>;
    /**
     * Full-text search in archive table.
     */
    fullTextSearchArchive(queryText: string, topK: number, agentId?: string, memoryType?: MemoryType, crossAgent?: boolean): Promise<Array<{
        id: string;
        agentId: string;
        sessionId: string;
        content: string;
        memoryType: string;
        createdAt: number;
        sourcePath: string;
        tags: string[];
        score: number;
        priority: number;
        expiresAt: number;
    }>>;
    /**
     * Get aggregate stats.
     */
    stats(): Promise<NoesisStats>;
    /**
     * Ensure IVF-PQ ANN index exists on the embedding column.
     * Called after initial data ingestion (requires at least 256 rows).
     */
    ensureAnnIndex(): Promise<void>;
    /**
     * Ensure IVF-PQ ANN index exists on the archive table's embedding column.
     */
    ensureAnnIndexArchive(): Promise<void>;
    /**
     * Ensure FTS index exists on the content column.
     */
    private ensureFtsIndex;
    private ensureFtsIndexArchive;
    /**
     * Query entries by minimum priority threshold, optionally filtered by agent
     * and recency. Used by the context engine Assemble hook to inject high-priority
     * memories into every model run.
     */
    queryByPriority(minPriority: number, agentId?: string, limit?: number, maxAgeDays?: number): Promise<MemoryEntry[]>;
}
//# sourceMappingURL=lancedb.d.ts.map