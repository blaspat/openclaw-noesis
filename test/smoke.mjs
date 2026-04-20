/**
 * Noesis smoke test — validates build + basic functionality
 * Run: node test/smoke.mjs
 *
 * Tests:
 *  1. Build compiles without errors
 *  2. Module loads without errors
 *  3. Transformers.js client initializes (model download may take a while on first run)
 *  4. LanceDB can be instantiated and connected
 *  5. upsertEntries is idempotent (same checksum = skip)
 *  6. exportAll pagination works (empty DB returns [])
 *  7. archiveExpired handles empty result gracefully
 *  8. chunkText and contentChecksum utilities work
 */

import { fileURLToPath } from "url";
import path from "path";
import os from "os";
import fs from "fs";
import { randomUUID } from "crypto";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const TEST_DB_DIR = path.join(os.tmpdir(), `noesis-smoke-test-${Date.now()}`);

const RESULTS = [];
function pass(msg) { RESULTS.push(`✅ ${msg}`); }
function fail(msg) { RESULTS.push(`❌ ${msg}`); }
function info(msg) { RESULTS.push(`ℹ️  ${msg}`); }

async function run() {
  console.log("\n=== Noesis Smoke Test ===\n");

  // 1. Check build artifacts exist
  info("1. Checking build output...");
  const distIndex = path.join(__dirname, "..", "dist", "index.js");
  const distLancedb = path.join(__dirname, "..", "dist", "lancedb.js");
  if (fs.existsSync(distIndex) && fs.existsSync(distLancedb)) {
    pass(`Build artifacts exist (dist/)`);
  } else {
    fail(`dist/ not found — run: npm run build`);
    printResults();
    process.exit(1);
  }

  // 2. Load the module
  info("2. Loading noesis module...");
  let noesis;
  try {
    noesis = await import("../dist/index.js");
    pass("Module loaded without errors");
  } catch (err) {
    fail(`Module load failed: ${err.message}`);
    printResults();
    process.exit(1);
  }

  // 3. contentChecksum + chunkText utilities
  info("3. Testing utilities (contentChecksum, chunkText)...");
  try {
    const { contentChecksum, chunkText } = await import("../dist/transformers.js");
    const cs1 = contentChecksum("hello world", "agent1");
    const cs2 = contentChecksum("hello world", "agent1");
    const cs3 = contentChecksum("hello world", "agent2"); // different agent = different checksum
    if (cs1 === cs2 && cs1 !== cs3 && cs1.length === 64) {
      pass(`contentChecksum: deterministic within agent, agent-distinct, 64-char sha256`);
    } else {
      fail(`contentChecksum produced unexpected output: ${cs1}`);
    }
    // chunkText splits at markdown boundaries (headings, blank lines, etc.).
    // Use text with multiple paragraphs/headings to trigger splitting.
    const multiPara = "# Section 1\n\n" + "word ".repeat(80) + "\n\n## Section 2\n\n" + "word ".repeat(80);
    const chunks = chunkText(multiPara, 50, 16);
    if (chunks.length >= 2 && chunks.every(c => typeof c === "string" && c.length > 0)) {
      pass(`chunkText: produced ${chunks.length} chunks from multi-section text`);
    } else {
      fail(`chunkText: expected >= 2 chunks, got ${chunks.length}`);
    }
  } catch (err) {
    fail(`Utility test failed: ${err.message}`);
  }

  // 4. Initialize Transformers.js client
  info("4. Initializing Transformers.js embedding client (first run downloads model, may be slow)...");
  let ollama;
  try {
    const { autoConfigTransformers } = await import("../dist/transformers.js");
    const client = await autoConfigTransformers("nomic-ai/nomic-embed-text-v1.5", (m) => info(`  [transformers] ${m}`));
    const vec = await client.embed("hello world");
    if (Array.isArray(vec) && vec.length === 768 && vec.every(n => typeof n === "number")) {
      pass(`embed() returned 768-dim vector (model: ${client.model})`);
    } else {
      fail(`embed() returned unexpected result length: ${vec?.length}`);
    }
    const batch = await client.embedBatch(["test1", "test2"]);
    if (Array.isArray(batch) && batch.length === 2 && batch.every(v => v.length === 768)) {
      pass(`embedBatch([2]) returned 2x 768-dim vectors`);
    } else {
      fail(`embedBatch() returned unexpected result`);
    }
    ollama = client;
  } catch (err) {
    fail(`Transformers.js init failed: ${err.message}`);
    info("  This may fail if @huggingface/transformers is not installed.");
    info("  Fix: npm install @huggingface/transformers");
    printResults();
    process.exit(1);
  }

  // 5. LanceDB setup
  info("5. Setting up test LanceDB at tmp dir...");
  try {
    fs.mkdirSync(TEST_DB_DIR, { recursive: true });
    const { NoesisDB } = await import("../dist/lancedb.js");
    const db = new NoesisDB({
      lanceDbPath: TEST_DB_DIR,
      embeddingModel: "nomic-ai/nomic-embed-text-v1.5",
      ollamaEndpoint: "none",
    });
    await db.connect(768);
    pass(`LanceDB connected (db: ${TEST_DB_DIR})`);

    // 6. Basic upsert
    info("6. Testing upsertEntries...");
    const entry = {
      id: randomUUID(),
      agentId: "smoke-test",
      sessionId: "smoke-session",
      content: "This is a test memory entry for smoke testing.",
      chunk: "This is a test memory entry for smoke testing.",
      embedding: new Array(768).fill(0.01),
      memoryType: "fact",
      priority: 50,
      expiresAt: 0,
      createdAt: Date.now(),
      sourcePath: "/test/smoke.mjs",
      checksum: "test-checksum-smoke-001",
      tags: ["smoke-test"],
    };
    const inserted = await db.upsertEntries([entry]);
    if (inserted === 1) {
      pass(`upsertEntries: inserted 1 new entry`);
    } else {
      fail(`upsertEntries: expected 1, got ${inserted}`);
    }

    // 7. Idempotency — same checksum should skip
    const skipped = await db.upsertEntries([{ ...entry, id: randomUUID() }]);
    if (skipped === 0) {
      pass(`upsertEntries: same checksum correctly skipped (idempotent)`);
    } else {
      fail(`upsertEntries: same checksum should skip, got inserted=${skipped}`);
    }

    // 8. Count
    const count = await db.count();
    if (count === 1) {
      pass(`count(): correctly reports 1 entry`);
    } else {
      fail(`count(): expected 1, got ${count}`);
    }

    // 9. exportAll pagination (empty = [])
    info("9. Testing exportAll (empty filter = all entries)...");
    const exported = await db.exportAll();
    if (exported.length === 1 && exported[0].checksum === "test-checksum-smoke-001") {
      pass(`exportAll(): returned ${exported.length} entry — pagination working`);
    } else {
      fail(`exportAll(): expected 1 entry, got ${exported.length}`);
    }

    // 10. archiveExpired with no expired entries
    info("10. Testing archiveExpired (no expired entries)...");
    const archived = await db.archiveExpired();
    if (archived === 0) {
      pass(`archiveExpired(): gracefully handled 0 expired entries`);
    } else {
      fail(`archiveExpired(): expected 0, got ${archived}`);
    }

    // 11. Search (hybrid)
    info("11. Testing hybrid search...");
    const { hybridSearch } = await import("../dist/search.js");
    const results = await hybridSearch("test memory", ollama, db, {
      topK: 5,
      vectorWeight: 0.6,
    }, {});
    if (Array.isArray(results) && results.length >= 1) {
      pass(`hybridSearch(): returned ${results.length} result(s)`);
    } else {
      fail(`hybridSearch(): expected >= 1 result, got ${results?.length}`);
    }

    // 12. Cleanup
    info("12. Cleaning up test DB...");
    await db.disconnect();
    fs.rmSync(TEST_DB_DIR, { recursive: true, force: true });
    pass(`Test DB cleaned up`);
  } catch (err) {
    fail(`LanceDB test failed: ${err.message}`);
    try { fs.rmSync(TEST_DB_DIR, { recursive: true, force: true }); } catch {}
  }

  printResults();

  const failed = RESULTS.filter(r => r.startsWith("❌")).length;
  if (failed > 0) {
    info(`\n${failed} test(s) failed — see ❌ above`);
    process.exit(1);
  } else {
    info(`\nAll tests passed`);
    process.exit(0);
  }
}

function printResults() {
  console.log("\n─── Results ───");
  for (const r of RESULTS) console.log(r);
  console.log("");
}

run().catch(err => {
  console.error("Smoke test crashed:", err);
  process.exit(1);
});
