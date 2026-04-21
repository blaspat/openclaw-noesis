/**
 * Noesis Concurrency + Race Condition Tests
 * ==========================================
 * Tests that concurrent upsertEntries calls are safe and idempotent.
 *
 * Run:  node test/concurrency.mjs
 */

import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { createRequire } from "module";
import { randomUUID } from "crypto";

const require = createRequire(import.meta.url);
const __dirname = path.dirname(fileURLToPath(import.meta.url));

// Suppress LanceDB Rust warnings — we only care about our own test output
const originalWrite = process.stderr.write.bind(process.stderr);
process.stderr.write = (chunk) => {
  if (typeof chunk === "string" && chunk.includes("Deprecation warning")) return true;
  return originalWrite(chunk);
};

const TEST_DB_DIR = `/tmp/noesis-concurrency-test-${Date.now()}`;

async function runTests() {
  const { NoesisDB } = await import("../dist/lancedb.js");

  let passed = 0;
  let failed = 0;

  function pass(msg) {
    console.log(`  ✅ ${msg}`);
    passed++;
  }
  function fail(msg) {
    console.log(`  ❌ ${msg}`);
    failed++;
  }

  // ── Setup ──────────────────────────────────────────────────────────────
  console.log("\n=== Concurrency Tests ===");
  fs.mkdirSync(TEST_DB_DIR, { recursive: true });
  const db = new NoesisDB({
    lanceDbPath: TEST_DB_DIR,
    embeddingModel: "nomic-ai/nomic-embed-text-v1.5",
    ollamaEndpoint: "none",
  });
  await db.connect(768);
  console.log(`  DB: ${TEST_DB_DIR}\n`);

  // ── T1: Same checksum, 20 concurrent calls — expect exactly 1 row in DB ──
  console.log("T1: 20 concurrent upserts with identical checksum");
  const tsT1 = Date.now();
  const checksumT1 = `race-test-${tsT1}-t1`;
  const entryT1 = {
    id: randomUUID(),
    agentId: "kate",
    sessionId: "concurrency",
    content: "T1 content",
    chunk: "T1 chunk",
    embedding: new Array(768).fill(0.01),
    memoryType: "fact",
    priority: 50,
    expiresAt: 0,
    createdAt: tsT1,
    sourcePath: "/test/concurrency.mjs",
    checksum: checksumT1,
    tags: [],
  };

  const resultsT1 = await Promise.all(
    Array.from({ length: 20 }, () => db.upsertEntries([entryT1]))
  );
  const totalInsertedT1 = resultsT1.reduce((s, n) => s + n, 0);
  const countT1 = await db.count();
  const exportedT1 = await db.exportAll();
  const t1Entries = exportedT1.filter((e) => e.checksum === checksumT1);
  const callersGot1 = resultsT1.filter((n) => n === 1).length;
  const callersGot0 = resultsT1.filter((n) => n === 0).length;

  if (totalInsertedT1 === 1) {
    pass(`T1: returned total inserted=1 (${callersGot1} callers got 1, ${callersGot0} got 0)`);
  } else {
    fail(`T1: expected total inserted=1, got ${totalInsertedT1}`);
  }

  if (countT1 === 1) {
    pass(`T1: DB has exactly 1 entry (not 20 duplicates)`);
  } else {
    fail(`T1: DB has ${countT1} entries — duplicates created!`);
  }

  if (t1Entries.length === 1) {
    pass(`T1: exportAll returns exactly 1 T1 entry`);
  } else {
    fail(`T1: exportAll returns ${t1Entries.length} T1 entries`);
  }

  // ── T2: Same checksum, 50 concurrent calls — same test but louder ──
  console.log("\nT2: 50 concurrent upserts with identical checksum");
  const tsT2 = Date.now();
  const checksumT2 = `race-test-${tsT2}-t2`;
  const entryT2 = {
    id: randomUUID(),
    agentId: "kate",
    sessionId: "concurrency",
    content: "T2 content",
    chunk: "T2 chunk",
    embedding: new Array(768).fill(0.02),
    memoryType: "fact",
    priority: 50,
    expiresAt: 0,
    createdAt: tsT2,
    sourcePath: "/test/concurrency.mjs",
    checksum: checksumT2,
    tags: [],
  };

  const resultsT2 = await Promise.all(
    Array.from({ length: 50 }, () => db.upsertEntries([{ ...entryT2, id: randomUUID() }]))
  );
  const totalInsertedT2 = resultsT2.reduce((s, n) => s + n, 0);
  const exportedT2 = await db.exportAll();
  const t2Entries = exportedT2.filter((e) => e.checksum === checksumT2);

  if (totalInsertedT2 === 1) {
    pass(`T2: returned total inserted=1`);
  } else {
    fail(`T2: expected total inserted=1, got ${totalInsertedT2}`);
  }

  if (t2Entries.length === 1) {
    pass(`T2: DB has exactly 1 T2 entry (no duplicates)`);
  } else {
    fail(`T2: DB has ${t2Entries.length} T2 entries — RACE CONDITION!`);
  }

  // ── T3: All different checksums — all 50 should insert ──
  console.log("\nT3: 50 concurrent upserts with different checksums");
  const tsT3 = Date.now();
  const checksumsT3 = Array.from({ length: 50 }, (_, i) => `race-test-${tsT3}-t3-${i}`);
  const entriesT3 = checksumsT3.map((checksum, i) => ({
    id: randomUUID(),
    agentId: "kate",
    sessionId: "concurrency",
    content: `T3 content ${i}`,
    chunk: `T3 chunk ${i}`,
    embedding: new Array(768).fill(0.03 + i * 0.001),
    memoryType: "fact",
    priority: 50,
    expiresAt: 0,
    createdAt: tsT3,
    sourcePath: "/test/concurrency.mjs",
    checksum,
    tags: [],
  }));

  const resultsT3 = await Promise.all(entriesT3.map((e) => db.upsertEntries([e])));
  const totalInsertedT3 = resultsT3.reduce((s, n) => s + n, 0);
  const exportedT3 = await db.exportAll();
  const t3Entries = exportedT3.filter((e) => e.checksum.startsWith(`race-test-${tsT3}-t3`));

  if (totalInsertedT3 === 50) {
    pass(`T3: all 50 different checksums inserted (returned total=50)`);
  } else {
    fail(`T3: expected total inserted=50, got ${totalInsertedT3}`);
  }

  if (t3Entries.length === 50) {
    pass(`T3: DB has exactly 50 T3 entries`);
  } else {
    fail(`T3: DB has ${t3Entries.length} T3 entries`);
  }

  // ── T4: Mixed — 10 groups of 5 identical checksums each ──
  console.log("\nT4: 10 checksum groups × 5 concurrent each (50 total calls, 10 unique entries)");
  const tsT4 = Date.now();
  const groupsT4 = Array.from({ length: 10 }, (_, g) => ({
    checksum: `race-test-${tsT4}-t4-${g}`,
    content: `T4 group ${g}`,
  }));

  const callsT4 = groupsT4.flatMap((group) =>
    Array.from({ length: 5 }, () => ({
      id: randomUUID(),
      agentId: "kate",
      sessionId: "concurrency",
      content: group.content,
      chunk: group.content,
      embedding: new Array(768).fill(0.04),
      memoryType: "fact",
      priority: 50,
      expiresAt: 0,
      createdAt: tsT4,
      sourcePath: "/test/concurrency.mjs",
      checksum: group.checksum,
      tags: [],
    }))
  );

  const resultsT4 = await Promise.all(callsT4.map((e) => db.upsertEntries([e])));
  const totalInsertedT4 = resultsT4.reduce((s, n) => s + n, 0);
  const exportedT4 = await db.exportAll();
  const t4Entries = exportedT4.filter((e) => e.checksum.startsWith(`race-test-${tsT4}-t4`));

  if (totalInsertedT4 === 10) {
    pass(`T4: returned total inserted=10 (1 per group)`);
  } else {
    fail(`T4: expected total inserted=10, got ${totalInsertedT4}`);
  }

  if (t4Entries.length === 10) {
    pass(`T4: DB has exactly 10 T4 entries (1 per group)`);
  } else {
    fail(`T4: DB has ${t4Entries.length} T4 entries`);
  }

  // ── T5: Upsert then immediate concurrent update of same checksum ──
  console.log("\nT5: Insert entry, then 20 concurrent updates to the same checksum");
  const tsT5 = Date.now();
  const checksumT5 = `race-test-${tsT5}-t5`;
  const baseEntryT5 = {
    id: randomUUID(),
    agentId: "kate",
    sessionId: "concurrency",
    content: "T5 original",
    chunk: "T5 chunk",
    embedding: new Array(768).fill(0.05),
    memoryType: "fact",
    priority: 50,
    expiresAt: 0,
    createdAt: tsT5,
    sourcePath: "/test/concurrency.mjs",
    checksum: checksumT5,
    tags: [],
  };
  await db.upsertEntries([baseEntryT5]);

  const updatedEntriesT5 = Array.from({ length: 20 }, (_, i) => ({
    ...baseEntryT5,
    id: randomUUID(), // new ID each time
    content: `T5 updated ${i}`,
    priority: i,
  }));
  const resultsT5 = await Promise.all(updatedEntriesT5.map((e) => db.upsertEntries([e])));
  const totalInsertedT5 = resultsT5.reduce((s, n) => s + n, 0);
  const exportedT5 = await db.exportAll();
  const t5Entries = exportedT5.filter((e) => e.checksum === checksumT5);

  if (totalInsertedT5 === 0) {
    pass(`T5: concurrent update after initial insert — all returned 0 (already existed)`);
  } else {
    fail(`T5: expected total inserted=0 for updates, got ${totalInsertedT5}`);
  }

  if (t5Entries.length === 1) {
    pass(`T5: DB has exactly 1 T5 entry after concurrent updates`);
  } else {
    fail(`T5: DB has ${t5Entries.length} T5 entries — update race condition!`);
  }

  // ── Summary ────────────────────────────────────────────────────────────
  console.log(`\n─── Results ───`);
  console.log(`  Passed: ${passed}`);
  console.log(`  Failed: ${failed}`);

  // Cleanup
  try { await db.close?.(); } catch {}
  fs.rmSync(TEST_DB_DIR, { recursive: true, force: true });

  process.exit(failed > 0 ? 1 : 0);
}

runTests().catch((err) => {
  console.error("Test crashed:", err);
  process.exit(1);
});
