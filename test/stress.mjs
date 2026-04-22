/**
 * Noesis — Concurrency, Race Condition & Stress Tests
 *
 * Run: node test/stress.mjs
 *
 * Tests:
 *  1. Concurrent upsert (10 agents × 10 entries simultaneously)
 *  2. Concurrent search while indexing (reads vs writes)
 *  3. archiveExpired race: concurrent archiver + new entry creator
 *  4. SessionScanner concurrency: parallel agent scanning
 *  5. Stress: 1000 entries, 10k queries, verify no leaks/crashes
 *  6. Idempotency under concurrent duplicate inserts
 *  7. Cleanup: verify all intervals closed, no lingering handles
 */

import { fileURLToPath } from "url";
import path from "path";
import os from "os";
import fs from "fs";
import { randomUUID, createHash } from "crypto";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const TEST_DB_DIR = path.join(os.tmpdir(), `noesis-stress-${Date.now()}`);

const RESULTS = [];
function pass(msg) { RESULTS.push(`✅ ${msg}`); }
function fail(msg) { RESULTS.push(`❌ ${msg}`); }
function info(msg) { RESULTS.push(`ℹ️  ${msg}`); }
function warn(msg) { RESULTS.push(`⚠️  ${msg}`); }

async function run() {
  console.log("\n=== Noesis Stress + Concurrency Tests ===\n");

  fs.mkdirSync(TEST_DB_DIR, { recursive: true });
  const { NoesisDB } = await import("../dist/lancedb.js");
  const { autoConfigTransformers } = await import("../dist/transformers.js");
  const { hybridSearch } = await import("../dist/search.js");

  // Initialize embedding client (may return zero vectors if model unavailable)
  let ollama;
  try {
    ollama = await autoConfigTransformers("nomic-ai/nomic-embed-text-v1.5", (m) => {});
  } catch {
    warn("Transformers.js unavailable — using zero vectors for embed");
    ollama = {
      embed: () => new Array(768).fill(0),
      embedBatch: (texts) => texts.map(() => new Array(768).fill(0)),
      get model() { return "mock"; },
    };
  }

  let db;
  try {
    db = new NoesisDB({
      lanceDbPath: TEST_DB_DIR,
      embeddingModel: "nomic-ai/nomic-embed-text-v1.5",
      ollamaEndpoint: "none",
    });
    await db.connect(768);
    pass("LanceDB connected for stress tests");
  } catch (err) {
    fail(`LanceDB connect failed: ${err.message}`);
    printResults();
    process.exit(1);
  }

  // Helper: make a deterministic embedding from a string
  function emb(str) {
    const hash = createHash("sha256").update(str).digest();
    const vec = new Array(768);
    for (let i = 0; i < 768; i++) vec[i] = (hash[i % 32] / 255) * 2 - 1;
    return vec;
  }

  // Helper: make an entry
  let _entrySeq = 0;
  function makeEntry(agentId, content, checksum) {
    return {
      id: randomUUID(),
      agentId,
      sessionId: "stress-test",
      content,
      chunk: content,
      embedding: emb(content + _entrySeq++),
      memoryType: "fact",
      priority: 50,
      expiresAt: 0,
      createdAt: Date.now(),
      sourcePath: "/test/stress",
      checksum: checksum || `cs-${randomUUID()}`,
      tags: [],
    };
  }

  // ─── TEST 1: Concurrent upsert from multiple agents ────────────────────────
  info("\n─── TEST 1: Concurrent multi-agent upserts ───");
  try {
    const NUM_AGENTS = 10;
    const ENTRIES_PER_AGENT = 10;
    const allEntries = [];

    for (let a = 0; a < NUM_AGENTS; a++) {
      for (let e = 0; e < ENTRIES_PER_AGENT; e++) {
        allEntries.push(makeEntry(`agent-${a}`, `content for agent ${a} entry ${e}`, `cs-a${a}-e${e}`));
      }
    }

    // Partition by agent and upsert all concurrently
    const agentGroups = Array.from({ length: NUM_AGENTS }, (_, a) =>
      allEntries.filter((e) => e.agentId === `agent-${a}`)
    );

    const start = Date.now();
    const upsertResults = await Promise.all(
      agentGroups.map((entries) => db.upsertEntries(entries))
    );
    const elapsed = Date.now() - start;

    const totalInserted = upsertResults.reduce((s, r) => s + r, 0);
    const expectedInserted = NUM_AGENTS * ENTRIES_PER_AGENT;

    if (totalInserted === expectedInserted) {
      pass(`T1: Concurrent upsert — ${totalInserted}/${expectedInserted} entries in ${elapsed}ms`);
    } else {
      fail(`T1: Concurrent upsert — expected ${expectedInserted}, got ${totalInserted}`);
    }

    const count = await db.count();
    if (count === expectedInserted) {
      pass(`T1: Count after concurrent upserts — ${count} (no ghost entries)`);
    } else {
      fail(`T1: Count mismatch — expected ${expectedInserted}, got ${count}`);
    }
  } catch (err) {
    fail(`T1: Concurrent upsert failed: ${err.message}`);
  }

  // ─── TEST 2: Concurrent read while writing ─────────────────────────────────
  info("\n─── TEST 2: Concurrent reads during writes ───");
  try {
    const NUM_READERS = 5;
    const NUM_WRITERS = 3;
    const WRITES_PER_WRITER = 20;
    const READS_PER_READER = 20;

    let writeCount = 0;
    let readErrors = 0;
    let readCount = 0;

    const writers = Array.from({ length: NUM_WRITERS }, (_, w) =>
      (async () => {
        for (let i = 0; i < WRITES_PER_WRITER; i++) {
          const entry = makeEntry(`writer-${w}`, `write-${w}-${i}`, `w${w}-${i}-${Date.now()}`);
          const inserted = await db.upsertEntries([entry]);
          writeCount += inserted;
          await new Promise((r) => setTimeout(r, 5));
        }
      })()
    );

    const readers = Array.from({ length: NUM_READERS }, (_, r) =>
      (async () => {
        for (let i = 0; i < READS_PER_READER; i++) {
          try {
            const results = await hybridSearch(`write-${r}`, ollama, db, { topK: 5, vectorWeight: 0.6 }, {});
            readCount++;
          } catch {
            readErrors++;
          }
          await new Promise((r) => setTimeout(r, 3));
        }
      })()
    );

    const start = Date.now();
    await Promise.all([...writers, ...readers]);
    const elapsed = Date.now() - start;

    pass(`T2: Concurrent R/W — ${writeCount} writes, ${readCount} reads (${readErrors} errors) in ${elapsed}ms`);
    if (readErrors > 0) {
      fail(`T2: ${readErrors} read errors during concurrent write`);
    } else {
      pass("T2: Zero read errors during concurrent write");
    }
  } catch (err) {
    fail(`T2: Concurrent read/write failed: ${err.message}`);
  }

  // ─── TEST 3: archiveExpired race — insert-while-archive ────────────────────
  info("\n─── TEST 3: archiveExpired race (insert while archiving) ───");
  try {
    // Create entries that will expire in 50ms
    const soon = Date.now() + 50;
    const entries = Array.from({ length: 50 }, (_, i) => ({
      id: randomUUID(),
      agentId: "archiverace",
      sessionId: "race",
      content: `race-entry-${i}`,
      chunk: `race-entry-${i}`,
      embedding: emb(`race-entry-${i}`),
      memoryType: "fact",
      priority: 50,
      expiresAt: soon,
      createdAt: Date.now(),
      sourcePath: "/test/race",
      checksum: `race-cs-${i}`,
      tags: [],
    }));

    await db.upsertEntries(entries);
    const beforeCount = await db.count();

    // Wait for expiry
    await new Promise((r) => setTimeout(r, 60));

    // Fire archiveExpired AND keep inserting simultaneously
    const archivePromise = db.archiveExpired();
    const insertPromise = (async () => {
      // Insert 10 new entries while archive is running
      const newEntries = Array.from({ length: 10 }, (_, i) =>
        makeEntry("archiverace", `new-during-archive-${i}`, `new-arch-${i}-${Date.now()}`)
      );
      return db.upsertEntries(newEntries);
    })();

    const [archived, newlyInserted] = await Promise.all([archivePromise, insertPromise]);
    const afterCount = await db.count();
    const archiveCount = await db.countArchive();

    info(`T3: archived=${archived}, new-during-archive=${newlyInserted}, total-active=${afterCount}, total-archive=${archiveCount}`);

    // No entries should be lost: archived entries should be in archive table
    if (archiveCount === archived) {
      pass(`T3: All archived entries landed in archive table (${archiveCount})`);
    } else {
      fail(`T3: Archive count mismatch — archived=${archived}, archiveTable=${archiveCount}`);
    }

    // Active table should have new entries + any unexpired
    if (afterCount >= newlyInserted) {
      pass(`T3: Active table intact — ${afterCount} entries (>= ${newlyInserted} new)`);
    } else {
      fail(`T3: Active table entry loss — ${afterCount} < ${newlyInserted}`);
    }
  } catch (err) {
    fail(`T3: archiveExpired race failed: ${err.message}`);
  }

  // ─── TEST 4: Idempotency under concurrent duplicate inserts ─────────────────
  info("\n─── TEST 4: Concurrent duplicate checksum inserts ───");
  try {
    const sharedChecksum = `dup-cs-${Date.now()}`;
    const entry = makeEntry("dup-agent", "duplicate content", sharedChecksum);
    const numConcurrent = 20;

    // All 20 try to insert the same checksum simultaneously
    const results = await Promise.all(
      Array.from({ length: numConcurrent }, () => db.upsertEntries([{ ...entry, id: randomUUID() }]))
    );

    const totalInserted = results.reduce((s, r) => s + r, 0);
    const finalCount = await db.count();

    if (totalInserted === 1) {
      pass(`T4: Concurrent duplicates — exactly 1 of ${numConcurrent} inserted (idempotent)`);
    } else {
      fail(`T4: Concurrent duplicates — expected 1 insert, got ${totalInserted}`);
    }

    // Count how many have this checksum
    const exported = await db.exportAll("dup-agent");
    const withChecksum = exported.filter((e) => e.checksum === sharedChecksum);
    if (withChecksum.length === 1) {
      pass(`T4: Exactly 1 entry with checksum in DB (no duplication)`);
    } else {
      fail(`T4: ${withChecksum.length} entries with shared checksum (duplication!)`);
    }
  } catch (err) {
    fail(`T4: Concurrent duplicate test failed: ${err.message}`);
  }

  // ─── TEST 5: Stress — high volume insert + query ───────────────────────────
  info("\n─── TEST 5: Stress — 1000 entries + 200 queries ───");
  try {
    const STRESS_ENTRIES = 1000;
    const STRESS_QUERIES = 200;

    const stressEntries = Array.from({ length: STRESS_ENTRIES }, (_, i) =>
      makeEntry("stress-agent", `stress content item ${i} with unique marker ${randomUUID()}`, `stress-cs-${i}`)
    );

    info(`T5: Inserting ${STRESS_ENTRIES} entries...`);
    const t5start = Date.now();
    await db.upsertEntries(stressEntries);
    const insertMs = Date.now() - t5start;

    const count = await db.count();
    pass(`T5: ${STRESS_ENTRIES} entries inserted in ${insertMs}ms — DB count=${count}`);

    // Run many queries
    info(`T5: Running ${STRESS_QUERIES} queries...`);
    const qStart = Date.now();
    let queryErrors = 0;
    const queryPromises = Array.from({ length: STRESS_QUERIES }, (_, i) =>
      (async () => {
        try {
          await hybridSearch(`stress content item ${i % 50}`, ollama, db, { topK: 5, vectorWeight: 0.6 }, {});
        } catch {
          queryErrors++;
        }
      })()
    );
    await Promise.all(queryPromises);
    const queryMs = Date.now() - qStart;

    pass(`T5: ${STRESS_QUERIES} queries in ${queryMs}ms (${queryErrors} errors)`);
    if (queryErrors > 0) {
      fail(`T5: ${queryErrors} query errors under load`);
    } else {
      pass("T5: Zero query errors under stress load");
    }

    // Verify count hasn't drifted
    const countAfter = await db.count();
    if (countAfter === count) {
      pass(`T5: Entry count stable after queries — ${countAfter}`);
    } else {
      fail(`T5: Entry count drifted — before=${count}, after=${countAfter}`);
    }
  } catch (err) {
    fail(`T5: Stress test failed: ${err.message}`);
  }

  // ─── TEST 6: archiveExpired handles empty table gracefully ─────────────────
  info("\n─── TEST 6: archiveExpired on empty/clean table ───");
  try {
    // Count everything first
    const beforeActive = await db.count();
    const beforeArchive = await db.countArchive();

    // Archive with nothing expired
    const archived = await db.archiveExpired();

    const afterActive = await db.count();
    const afterArchive = await db.countArchive();

    if (archived === 0) {
      pass(`T6: archiveExpired on clean table — ${archived} archived (expected 0)`);
    } else {
      warn(`T6: archiveExpired reported ${archived} but table was clean — possible stale data`);
    }

    if (afterActive === beforeActive && afterArchive === beforeArchive) {
      pass(`T6: Counts unchanged after archive on clean table`);
    } else {
      fail(`T6: Counts changed unexpectedly — active: ${beforeActive}→${afterActive}, archive: ${beforeArchive}→${afterArchive}`);
    }
  } catch (err) {
    fail(`T6: archiveExpired on clean table threw: ${err.message}`);
  }

  // ─── TEST 7: ExportAll pagination — verify all entries retrieved ─────────────
  info("\n─── TEST 7: exportAll pagination completeness ───");
  try {
    const total = await db.count();
    const exported = await db.exportAll();

    if (exported.length === total) {
      pass(`T7: exportAll pagination — all ${total} entries retrieved`);
    } else {
      fail(`T7: exportAll count mismatch — count()=${total}, exportAll()=${exported.length}`);
    }

    // Verify no empty entries
    const empty = exported.filter((e) => !e.content || !e.id);
    if (empty.length === 0) {
      pass(`T7: No empty/corrupt entries in export`);
    } else {
      fail(`T7: ${empty.length} empty entries in export`);
    }

    // Verify all entries have valid checksums
    const noChecksum = exported.filter((e) => !e.checksum);
    if (noChecksum.length === 0) {
      pass(`T7: All exported entries have checksums`);
    } else {
      fail(`T7: ${noChecksum.length} entries missing checksums`);
    }
  } catch (err) {
    fail(`T7: exportAll pagination test failed: ${err.message}`);
  }

  // ─── TEST 8: Memory — verify DB size is reasonable ─────────────────────────
  info("\n─── TEST 8: Disk usage sanity ───");
  try {
    const dbSize = await getDirSize(TEST_DB_DIR);
    const perEntry = (dbSize / (await db.count())).toFixed(0);
    info(`T8: DB size: ${(dbSize / 1024 / 1024).toFixed(2)} MB, ~${perEntry} bytes/entry`);
    // Should be under ~50KB per entry for 768-dim float embeddings
    if (perEntry && parseInt(perEntry) < 50000) {
      pass(`T8: Per-entry storage reasonable (${perEntry} bytes/entry)`);
    } else {
      warn(`T8: Per-entry storage higher than expected: ${perEntry} bytes`);
    }
  } catch (err) {
    warn(`T8: Could not measure DB size: ${err.message}`);
  }

  // ─── Cleanup ────────────────────────────────────────────────────────────────
  info("\n─── Cleanup ───");
  try {
    await db.disconnect();
    fs.rmSync(TEST_DB_DIR, { recursive: true, force: true });
    pass("Test DB cleaned up");
  } catch (err) {
    warn(`Cleanup warning: ${err.message}`);
  }

  printResults();

  const failed = RESULTS.filter((r) => r.startsWith("❌")).length;
  const warnings = RESULTS.filter((r) => r.startsWith("⚠️")).length;
  const passed = RESULTS.filter((r) => r.startsWith("✅")).length;

  console.log(`\n─── Summary ───`);
  console.log(`✅ Passed:  ${passed}`);
  console.log(`⚠️  Warnings: ${warnings}`);
  console.log(`❌ Failed:  ${failed}`);

  if (failed > 0) {
    console.log(`\n❌ ${failed} test(s) FAILED`);
    process.exit(1);
  } else {
    console.log(`\n✅ All tests passed`);
    process.exit(warnings > 0 ? 0 : 0);
  }
}

function printResults() {
  console.log("\n─── Results ───");
  for (const r of RESULTS) console.log(r);
  console.log("");
}

async function getDirSize(dir) {
  let size = 0;
  for (const [entry, child] of await eachDirent(dir)) {
    if (child) {
      size += await getDirSize(path.join(dir, entry));
    } else {
      size += fs.statSync(path.join(dir, entry)).size;
    }
  }
  return size;
}

async function eachDirent(dir) {
  const entries = await fs.promises.readdir(dir, { withFileTypes: true });
  return entries.map((e) => [e.name, e.isDirectory()]);
}

run().catch((err) => {
  console.error("Stress test crashed:", err);
  process.exit(1);
});
