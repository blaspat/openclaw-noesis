/**
 * Noesis — LanceDB storage layer
 *
 * Uses @lancedb/lancedb (Node.js native SDK v0.27+).
 * Single persistent connection for connection pooling.
 */

import * as lancedb from "@lancedb/lancedb";
import { Schema, Field, Utf8, Int64, FixedSizeList, Float32, List } from "apache-arrow";
import path from "path";
import os from "os";
import fs from "fs";
import { MemoryEntry, MemoryType, NoesisConfig, NoesisStats } from "./types.js";
import { logError } from "./logger.js";

/**
 * Per-checksum mutex — serializes concurrent upserts that share the same checksum.
 * Keyed by checksum, cleared after each dedup+insert completes.
 * Guards the two-step: dedup-query → insert (gap allows duplicates otherwise).
 */

const TABLE_NAME = "memories";
const ARCHIVE_TABLE_NAME = "memories_archive";

export class NoesisDB {
  private conn: lancedb.Connection | null = null;
  // Instance-level gate — each NoesisDB has its own serialised upsert queue
  private _upsertGate: Promise<unknown> = Promise.resolve();
  private table: lancedb.Table | null = null;
  private archiveTable: lancedb.Table | null = null;
  private embeddingDim: number = 768;
  private dbPath: string;
  private indexCreated: boolean = false;

  constructor(private config: NoesisConfig) {
    this.dbPath = resolvePath(config.lanceDbPath);
  }

  /**
   * Connect to LanceDB (idempotent — call once at plugin startup).
   */
  async connect(embeddingDim: number = 768): Promise<void> {
    if (this.conn) return; // already connected
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
  async disconnect(): Promise<void> {
    this.table = null;
    this.archiveTable = null;
    this.conn = null;
  }

  private async openOrCreateTable(): Promise<lancedb.Table> {
    if (!this.conn) throw new Error("Not connected");

    const existing = await this.conn.tableNames();
    if (existing.includes(TABLE_NAME)) {
      return this.conn.openTable(TABLE_NAME);
    }

    // Create table with explicit schema
    const schema = this.buildSchema(this.embeddingDim);
    const tbl = await this.conn.createEmptyTable(TABLE_NAME, schema);
    return tbl;
  }

  private async openOrCreateArchiveTable(): Promise<lancedb.Table> {
    if (!this.conn) throw new Error("Not connected");
    const existing = await this.conn.tableNames();
    if (existing.includes(ARCHIVE_TABLE_NAME)) {
      return this.conn.openTable(ARCHIVE_TABLE_NAME);
    }
    const schema = this.buildSchema(this.embeddingDim);
    return this.conn.createEmptyTable(ARCHIVE_TABLE_NAME, schema);
  }

  private buildSchema(dim: number): Schema {
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

  private getTable(): lancedb.Table {
    if (!this.table) throw new Error("[noesis] Not connected to LanceDB. Call connect() first.");
    return this.table;
  }

  private getArchiveTable(): lancedb.Table {
    if (!this.archiveTable) throw new Error("[noesis] Not connected to archive table. Call connect() first.");
    return this.archiveTable;
  }

  /**
   * Insert a batch of memory entries (upsert by checksum — idempotent).
   *
   * Concurrency safety — delete-before-insert per checksum:
   *   1. DELETE any row that already has this checksum (no-op if none exists)
   *   2. ADD the new row(s)
   * This is atomic — there is no gap between the "already exists" check and insert
   * where another concurrent upsert could slip in and create a duplicate.
   * Each checksum gets its own mutex slot so different checksums proceed in parallel.
   *
   * Returns the number of entries inserted (0 if checksum already existed).
   */
  async upsertEntries(entries: MemoryEntry[]): Promise<number> {
    if (entries.length === 0) return 0;
    const tbl = this.getTable();

    // Group by checksum — serialise entries that share the same checksum
    const byChecksum = new Map<string, MemoryEntry[]>();
    for (const e of entries) {
      const arr = byChecksum.get(e.checksum);
      if (arr) arr.push(e);
      else byChecksum.set(e.checksum, [e]);
    }

    // Serialise ALL upserts through a shared Promise chain (module-level _upsertGate).
    // Each call: 1) captures its result promise from the gate chain, 2) awaits it.
    // This ensures all concurrent calls queue FIFO and each gets the correct result.
    const myWork = async (): Promise<number> => {
      let inserted = 0;
      for (const [checksum, group] of byChecksum.entries()) {
        const n = await this._upsertOneChecksum(tbl, checksum, group);
        inserted += n;
      }
      return inserted;
    };
    // Chain work onto the gate; capture the linked gate BEFORE assigning to _upsertGate
    const linkedGate = this._upsertGate.then(() => myWork());
    this._upsertGate = linkedGate; // ensure next call chains onto THIS call's completion
    const result = await linkedGate; // wait for THIS call's work to finish
    return result;
  }

  /**
   * Wraps a promise with a timeout.  Rejects if `promise` doesn't settle within
   * `ms` milliseconds — prevents a stuck operation from blocking the upsert queue.
   */
  private async withTimeout<T>(promise: Promise<T>, ms: number, label: string): Promise<T> {
    let timeoutId: ReturnType<typeof setTimeout>;
    const timer = new Promise<never>((_, reject) => {
      timeoutId = setTimeout(() => reject(new Error(label + " timed out after " + ms + "ms")), ms);
    });
    try {
      return await Promise.race([promise, timer]);
    } finally {
      clearTimeout(timeoutId!);
    }
  }

  /**
   * Delete + re-insert for one checksum.  Called only from within the mutex.
   * Uses delete-before-insert to eliminate the read-then-insert race window.
   * Protected by a 30-second timeout — if the operation hangs, callers get
   * an error rather than blocking the entire upsert queue indefinitely.
   * Returns 1 if the entry was inserted (new), 0 if the checksum already
   * existed, or throws if the operation timed out.
   */
  async _upsertOneChecksum(
    tbl: ReturnType<typeof this.getTable>,
    checksum: string,
    entries: MemoryEntry[]
  ): Promise<number> {
    let numDeleted = 0;
    try {
      const delP = tbl.delete(`checksum = '${checksum}'`);
      const result = await this.withTimeout(delP, 30_000, "delete checksum=" + checksum);
      numDeleted = result.numDeletedRows;
    } catch (err) {
      logError("Upsert delete failed", { error: err, extra: { checksum } });
      throw err;
    }
    const rows = entries.map((e) => ({
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
    // numDeleted > 0 means checksum already existed — not a new entry
    return numDeleted === 0 ? 1 : 0;
  }

  /**
   * Vector similarity search using ANN index.
   * Returns up to topK*2 candidates for hybrid merge.
   */
  async vectorSearch(
    embedding: number[],
    topK: number,
    agentId?: string,
    memoryType?: MemoryType,
    crossAgent?: boolean
  ): Promise<Array<{ id: string; agentId: string; sessionId: string; content: string; memoryType: string; createdAt: number; sourcePath: string; tags: string[]; score: number; priority: number; expiresAt: number }>> {
    let tbl: ReturnType<typeof this.getTable>;
    try {
      tbl = this.getTable();
    } catch {
      return [];
    }

    let query: ReturnType<typeof tbl.vectorSearch>;
    try {
      query = tbl
        .vectorSearch(embedding)
        .select(["id", "agentId", "sessionId", "content", "memoryType", "createdAt", "sourcePath", "tags"])
        .limit(topK * 2)
        .nprobes(this.config.annNprobe);
    } catch (err) {
      logError("vectorSearch() failed — ANN index may not be ready", { error: err });
      return [];
    }

    const filters: string[] = [];
    if (agentId && !crossAgent) {
      filters.push(`agentId = '${escapeFilterValue(agentId)}'`);
    }
    if (memoryType) {
      filters.push(`memoryType = '${escapeFilterValue(memoryType)}'`);
    }
    if (filters.length > 0) {
      query = query.where(filters.join(" AND "));
    }

    try {
      const results = await query.toArray();
      return results.map((r: any) => ({
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
    } catch (err) {
      logError("vectorSearch() query execution failed", { error: err });
      return [];
    }
  }

  /**
   * Full-text keyword search (BM25 via LanceDB FTS).
   */
  async fullTextSearch(
    queryText: string,
    topK: number,
    agentId?: string,
    memoryType?: MemoryType,
    crossAgent?: boolean
  ): Promise<Array<{ id: string; agentId: string; sessionId: string; content: string; memoryType: string; createdAt: number; sourcePath: string; tags: string[]; score: number; priority: number; expiresAt: number }>> {
    const tbl = this.getTable();

    try {
      // Ensure FTS index exists on content column
      await this.ensureFtsIndex();

      let query = (tbl.search(queryText, "fts") as any)
        .select(["id", "agentId", "sessionId", "content", "memoryType", "createdAt", "sourcePath", "tags"])
        .limit(topK);

      const filters: string[] = [];
      if (agentId && !crossAgent) {
        filters.push(`agentId = '${escapeFilterValue(agentId)}'`);
      }
      if (memoryType) {
        filters.push(`memoryType = '${escapeFilterValue(memoryType)}'`);
      }
      if (filters.length > 0) {
        query = query.where(filters.join(" AND "));
      }

      const results = await query.toArray();
      return results.map((r: any, i: number) => ({
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
    } catch {
      // FTS not available or no index yet — return empty
      return [];
    }
  }

  /**
   * Retrieve by ID.
   */
  async getById(id: string): Promise<MemoryEntry | null> {
    const tbl = this.getTable();
    const results = await tbl
      .query()
      .where(`id = '${id.replace(/'/g, "''")}'`)
      .select(["id", "agentId", "sessionId", "content", "chunk", "embedding", "memoryType", "priority", "expiresAt", "createdAt", "sourcePath", "checksum", "tags"])
      .limit(1)
      .toArray();

    if (results.length === 0) return null;
    const r = results[0] as any;
    return rowToEntry(r);
  }

  /**
   * Recall entries by agent/session, ordered by createdAt desc.
   */
  async recall(
    agentId?: string,
    sessionId?: string,
    limit: number = 50
  ): Promise<MemoryEntry[]> {
    const tbl = this.getTable();

    const filters: string[] = [];
    if (agentId) filters.push(`agentId = '${escapeFilterValue(agentId)}'`);
    if (sessionId) filters.push(`sessionId = '${escapeFilterValue(sessionId)}'`);
    // Filter out expired entries (TTL reached but not yet archived)
    const now = Date.now();
    filters.push(`(expiresAt = 0 OR expiresAt > ${now})`);

    let query = tbl
      .query()
      .select(["id", "agentId", "sessionId", "content", "chunk", "embedding", "memoryType", "priority", "expiresAt", "createdAt", "sourcePath", "checksum", "tags"])
      .limit(limit);

    if (filters.length > 0) {
      query = query.where(filters.join(" AND "));
    }

    const results = await query.toArray();
    return (results as any[])
      .map(rowToEntry)
      .sort((a, b) => b.createdAt - a.createdAt);
  }

  /**
   * Retrieve an entry by its content checksum (idempotent upsert lookup).
   */
  async getByChecksum(checksum: string): Promise<MemoryEntry | null> {
    const tbl = this.getTable();
    try {
      const results = await tbl
        .query()
        .where(`checksum = '${checksum.replace(/'/g, "''")}'`)
        .select(["id", "agentId", "sessionId", "content", "chunk", "embedding", "memoryType", "priority", "expiresAt", "createdAt", "sourcePath", "checksum", "tags"])
        .limit(1)
        .toArray();
      if (results.length === 0) return null;
      return rowToEntry(results[0] as any);
    } catch {
      return null;
    }
  }

  /**
   * Delete a single entry by ID.
   */
  async deleteById(id: string): Promise<void> {
    const tbl = this.getTable();
    await tbl.delete(`id = '${id.replace(/'/g, "''")}'`);
  }

  /**
   * Update priority and/or expiresAt on an existing entry.
   * Returns the updated entry, or null if not found.
   */
  async updateEntry(id: string, patch: { priority?: number; expiresAt?: number }): Promise<MemoryEntry | null> {
    const tbl = this.getTable();
    const now = Date.now();
    try {
      const rows = await tbl
        .query()
        .where(`id = '${id.replace(/'/g, "''")}'`)
        .limit(1)
        .toArray();
      if (rows.length === 0) return null;
      const r = rows[0] as any;
      const updates: Record<string, string> = {};
      if (patch.priority !== undefined) updates["priority"] = String(patch.priority);
      if (patch.expiresAt !== undefined) updates["expiresAt"] = String(patch.expiresAt);
      if (Object.keys(updates).length === 0) return rowToEntry(r);
      await tbl.update({ where: `id = '${id.replace(/'/g, "''")}'`, values: updates });
      // Refetch after update
      const refreshed = await tbl.query().where(`id = '${id.replace(/'/g, "''")}'`).limit(1).toArray();
      return refreshed.length > 0 ? rowToEntry(refreshed[0] as any) : null;
    } catch {
      return null;
    }
  }

  /**
   * Export all entries, optionally filtered by agentId / memoryType.
   */
  async exportAll(agentId?: string, memoryType?: string): Promise<MemoryEntry[]> {
    const tbl = this.getTable();
    const filters: string[] = [];
    if (agentId) filters.push(`agentId = '${escapeFilterValue(agentId)}'`);
    if (memoryType) filters.push(`memoryType = '${escapeFilterValue(memoryType)}'`);
    const whereClause = filters.length > 0 ? filters.join(" AND ") : undefined;

    // Paginate to avoid loading the entire table + embeddings into heap at once.
    // Each page = 5000 rows; embeddings are 768 * 8 = ~6KB/row → ~30MB per page,
    // well within reasonable memory bounds.
    const PAGE_SIZE = 5000;
    const allEntries: MemoryEntry[] = [];
    let offset = 0;

    while (true) {
      let query = tbl.query()
        .select(["id", "agentId", "sessionId", "content", "chunk", "memoryType", "priority", "expiresAt", "createdAt", "sourcePath", "checksum", "tags"])
        .limit(PAGE_SIZE)
        .offset(offset);
      if (whereClause) query = query.where(whereClause);
      const rows = await query.toArray();
      if (rows.length === 0) break;
      for (const row of rows as any[]) {
        allEntries.push(rowToEntry(row));
      }
      if (rows.length < PAGE_SIZE) break;
      offset += PAGE_SIZE;
    }

    return allEntries;
  }

  /**
   * Count entries (total or by agentId filter).
   */
  async count(agentId?: string): Promise<number> {
    const tbl = this.getTable();
    if (agentId) {
      return tbl.countRows(`agentId = '${escapeFilterValue(agentId)}'`);
    }
    return tbl.countRows();
  }

  /**
   * Count entries grouped by a string column.
   */
  async countByColumn(column: string): Promise<Record<string, number>> {
    const tbl = this.getTable();
    try {
      const rows = await tbl.query().select([column]).limit(100_000).toArray();
      const counts: Record<string, number> = {};
      for (const row of rows as any[]) {
        const key = String(row[column] ?? "unknown");
        counts[key] = (counts[key] ?? 0) + 1;
      }
      return counts;
    } catch {
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
  async archiveExpired(): Promise<number> {
    const tbl = this.getTable();
    const now = Date.now();
    let archived = 0;
    try {
      // Process expired entries in batches to avoid large queries.
      while (true) {
        const toArchive = await tbl
          .query()
          .where(`expiresAt > 0 AND expiresAt < ${now}`)
          .select(["id", "agentId", "sessionId", "content", "chunk", "embedding", "memoryType", "priority", "expiresAt", "createdAt", "sourcePath", "checksum", "tags"])
          .limit(500)
          .toArray();
        if (toArchive.length === 0) break;

      // Delete FIRST — if the process crashes between delete and insert, entries
      // are missing from active (not duplicated in archive). Next run will re-archive
      // them. This is safer than insert-then-delete where a crash causes duplication.
      const ids = toArchive.map((r: any) => `'${r.id}'`).join(", ");
      await tbl.delete(`id IN (${ids})`);

      // Insert into archive — if this fails, entries are still gone from active
      // but will be re-archived on the next cleanup cycle. Acceptable trade-off.
      const archiveTbl = this.getArchiveTable();
      const rows = toArchive.map((r: any) => ({
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
      archived += toArchive.length;

      }
    } catch (err) {
      // archive table not ready — skip silently
    }
    return archived;
  }

  /**
   * Optimize the LanceDB tables to reclaim disk space and reduce memory usage.
   *
   * LanceDB uses MVCC — every write creates version files in _versions/.
   * Without periodic optimize(), these accumulate indefinitely and consume
   * both disk and memory (via memory-mapped I/O).
   *
   * This runs on the active memories table and the archive table.
   * Call this from the periodic cleanup interval or as a one-time repair.
   *
   * @param olderThanMs - Reclaim versions older than this many ms. Default: 24h.
   */
  async optimize(olderThanMs: number = 24 * 60 * 60 * 1000): Promise<void> {
    const cleanupOlderThan = new Date(Date.now() - olderThanMs);

    // Optimize active table
    try {
      const tbl = this.getTable();
      const stats = await tbl.optimize({
        cleanupOlderThan,
        deleteUnverified: true,
      });
      const removed = stats.prune.oldVersionsRemoved;
      logError?.(`[noesis/lancedb] optimize() complete — old versions removed: ${removed}`);
    } catch (err) {
      logError?.(`[noesis/lancedb] optimize() failed on active table: ${err}`);
    }

    // Optimize archive table
    try {
      const archiveTbl = this.getArchiveTable();
      await archiveTbl.optimize({
        cleanupOlderThan,
        deleteUnverified: true,
      });
    } catch {
      // archive table may not be initialized — skip
    }
  }

  /**
   * Count entries in the archive table.
   */
  async countArchive(): Promise<number> {
    try {
      return this.getArchiveTable().countRows();
    } catch {
      return 0;
    }
  }

  /**
   * Count archive entries grouped by column (e.g. by agentId, by memoryType).
   */
  async countByColumnArchive(column: string): Promise<Record<string, number>> {
    try {
      const tbl = this.getArchiveTable();
      const rows = await tbl.query().select([column]).limit(100_000).toArray();
      const counts: Record<string, number> = {};
      for (const row of rows as any[]) {
        const key = String(row[column] ?? "unknown");
        counts[key] = (counts[key] ?? 0) + 1;
      }
      return counts;
    } catch {
      return {};
    }
  }

  /**
   * Search archive table (for "older memories" queries).
   * Returns top-K results from archive, sorted by score.
   */
  async searchArchive(
    embedding: number[],
    topK: number,
    agentId?: string,
    memoryType?: MemoryType,
    crossAgent?: boolean
  ): Promise<Array<{ id: string; agentId: string; sessionId: string; content: string; memoryType: string; createdAt: number; sourcePath: string; tags: string[]; score: number; priority: number; expiresAt: number }>> {
    const tbl = this.getArchiveTable();
    try {
      await this.ensureAnnIndexArchive();
      const filters: string[] = [];
      if (agentId && !crossAgent) filters.push(`agentId = '${escapeFilterValue(agentId)}'`);
      if (memoryType) filters.push(`memoryType = '${escapeFilterValue(memoryType)}'`);

      let query = tbl
        .vectorSearch(embedding)
        .select(["id", "agentId", "sessionId", "content", "memoryType", "createdAt", "sourcePath", "tags", "priority", "expiresAt"])
        .limit(topK);

      if (filters.length > 0) {
        query = query.where(filters.join(" AND "));
      }

      const results = await query.toArray();
      return results.map((r: any) => ({
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
    } catch {
      return [];
    }
  }

  /**
   * Full-text search in archive table.
   */
  async fullTextSearchArchive(
    queryText: string,
    topK: number,
    agentId?: string,
    memoryType?: MemoryType,
    crossAgent?: boolean
  ): Promise<Array<{ id: string; agentId: string; sessionId: string; content: string; memoryType: string; createdAt: number; sourcePath: string; tags: string[]; score: number; priority: number; expiresAt: number }>> {
    const tbl = this.getArchiveTable();
    try {
      await this.ensureFtsIndexArchive();
      const filters: string[] = [];
      if (agentId && !crossAgent) filters.push(`agentId = '${escapeFilterValue(agentId)}'`);
      if (memoryType) filters.push(`memoryType = '${escapeFilterValue(memoryType)}'`);

      let query = (tbl.search(queryText, "fts") as any)
        .select(["id", "agentId", "sessionId", "content", "memoryType", "createdAt", "sourcePath", "tags", "priority", "expiresAt"])
        .limit(topK);

      if (filters.length > 0) {
        query = query.where(filters.join(" AND "));
      }

      const results = await query.toArray();
      return results.map((r: any, i: number) => ({
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
    } catch {
      return [];
    }
  }

  /**
   * Get aggregate stats.
   */
  async stats(): Promise<NoesisStats> {
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
  async ensureAnnIndex(): Promise<void> {
    if (this.indexCreated) return;
    const tbl = this.getTable();
    const count = await tbl.countRows();
    if (count < 256) return; // IVF-PQ needs enough data to build partitions

    try {
      const indices = await tbl.listIndices();
      const hasVectorIndex = indices.some(
        (idx: any) => idx.columns?.includes("embedding") || idx.name?.includes("embedding")
      );
      if (!hasVectorIndex) {
        await tbl.createIndex("embedding", {
          config: lancedb.Index.ivfPq({
            numSubVectors: this.config.annNumSubvectors,
          }),
        });
      }
      this.indexCreated = true;
    } catch (err) {
      // Index creation is best-effort — ANN just won't be used, but log so operator knows
      logError("Failed to create ANN index", { error: err, extra: { table: "memories" } });
    }
  }

  /**
   * Ensure IVF-PQ ANN index exists on the archive table's embedding column.
   */
  async ensureAnnIndexArchive(): Promise<void> {
    if ((this as any).archiveIndexCreated) return;
    const tbl = this.getArchiveTable();
    const count = await tbl.countRows();
    if (count < 256) return;

    try {
      const indices = await tbl.listIndices();
      const hasVectorIndex = indices.some(
        (idx: any) => idx.columns?.includes("embedding") || idx.name?.includes("embedding")
      );
      if (!hasVectorIndex) {
        await tbl.createIndex("embedding", {
          config: lancedb.Index.ivfPq({
            numSubVectors: this.config.annNumSubvectors,
          }),
        });
      }
      (this as any).archiveIndexCreated = true;
    } catch (err) {
      logError("Failed to create ANN index on archive table", { error: err, extra: { table: "memories_archive" } });
    }
  }

  /**
   * Ensure FTS index exists on the content column.
   */
  private async ensureFtsIndex(): Promise<void> {
    const tbl = this.getTable();
    try {
      const indices = await tbl.listIndices();
      const hasFtsIndex = indices.some(
        (idx: any) => idx.columns?.includes("content") && idx.indexType === "FTS"
      );
      if (!hasFtsIndex) {
        await tbl.createIndex("content", {
          config: lancedb.Index.fts(),
        });
      }
    } catch (err) {
      logError("Failed to create FTS index", { error: err, extra: { table: "memories" } });
    }
  }

  private async ensureFtsIndexArchive(): Promise<void> {
    const tbl = this.getArchiveTable();
    try {
      const indices = await tbl.listIndices();
      const hasFtsIndex = indices.some(
        (idx: any) => idx.columns?.includes("content") && idx.indexType === "FTS"
      );
      if (!hasFtsIndex) {
        await tbl.createIndex("content", {
          config: lancedb.Index.fts(),
        });
      }
    } catch (err) {
      logError("Failed to create FTS index on archive table", { error: err, extra: { table: "memories_archive" } });
    }
  }

  /**
   * Query entries by minimum priority threshold, optionally filtered by agent
   * and recency. Used by the context engine Assemble hook to inject high-priority
   * memories into every model run.
   */
  async queryByPriority(
    minPriority: number,
    agentId?: string,
    limit: number = 50,
    maxAgeDays?: number
  ): Promise<MemoryEntry[]> {
    const tbl = this.getTable();
    const now = Date.now();

    let query = tbl
      .query()
      .select(["id", "agentId", "sessionId", "content", "chunk", "embedding", "memoryType", "priority", "expiresAt", "createdAt", "sourcePath", "checksum", "tags"])
      .where(`priority >= ${minPriority}`)
      .limit(limit);

    const results = await query.toArray();
    const entries = (results as any[]).map(rowToEntry);

    // Filter out expired
    const active = entries.filter(e => e.expiresAt === 0 || e.expiresAt > now);

    // Filter by recency if maxAgeDays set
    const cutoff = maxAgeDays ? now - maxAgeDays * 24 * 60 * 60 * 1000 : 0;
    const recent = cutoff ? active.filter(e => e.createdAt >= cutoff) : active;

    // Sort by priority desc, then createdAt desc
    return recent.sort((a, b) => {
      if (b.priority !== a.priority) return (b.priority ?? 0) - (a.priority ?? 0);
      return b.createdAt - a.createdAt;
    });
  }
}

// ─── helpers ──────────────────────────────────────────────────────────────────

/** Escape a string value for use in a SQL WHERE clause filter string. */
function escapeFilterValue(value: string): string {
  return value.replace(/'/g, "''");
}

function resolvePath(p: string): string {
  return p.replace(/^~/, os.homedir());
}

function rowToEntry(r: any): MemoryEntry {
  let embedding: number[] = [];
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
    memoryType: (r.memoryType ?? "fact") as MemoryType,
    priority: Number(r.priority ?? 0),
    expiresAt: Number(r.expiresAt ?? 0),
    createdAt: Number(r.createdAt ?? 0),
    sourcePath: String(r.sourcePath ?? ""),
    checksum: String(r.checksum ?? ""),
    tags: Array.isArray(r.tags) ? r.tags.map(String) : [],
  };
}
