# Noesis — Local-First Semantic Memory for OpenClaw

Local-first semantic memory + context engine for OpenClaw. No API keys. No cloud. Memory layer + proactive context assembly (zero LLM cost).

Noesis gives your OpenClaw agents a persistent, searchable memory layer backed by [LanceDB](https://lancedb.com) and [Ollama](https://ollama.ai). It slots directly into OpenClaw's memory system so the auto-recall loop just works — your agents remember things across sessions, across restarts, and across each other.

---

## Table of Contents

- [Why Noesis?](#why-noesis)
- [Stack](#stack)
- [Installation](#installation)
  - [1. Install the plugin](#1-install-the-plugin)
  - [2. Set the memory slot](#2-set-the-memory-slot)
  - [3. Start Ollama](#3-start-ollama)
  - [4. Restart the gateway](#4-restart-the-gateway)
- [How It Works](#how-it-works)
  - [Active + Archive Architecture](#active--archive-architecture)
  - [Hybrid Search Pipeline](#hybrid-search-pipeline)
  - [Memory Types](#memory-types)
  - [When Agents Write to Noesis](#when-agents-write-to-noesis)
  - [Multi-Agent Isolation](#multi-agent-isolation)
  - [Context Engine (Assemble + Ingest)](#context-engine-assemble--ingest-hooks)
  - [ANN Index](#ann-index)
- [Tools](#tools)
  - [Agent tools](#agent-tools-use-these-in-your-prompts)
  - [Memory slot tools](#memory-slot-tools-used-by-openclaw-auto-recall)
- [Data Model](#data-model)
  - [Active Table](#active-table-memories)
  - [Archive Table](#archive-table-memories_archive)
- [Migrating Existing Memory Files](#migrating-existing-memory-files)
- [Configuration](#configuration)
  - [Embedding Model Choice](#embedding-model-choice)
- [Performance](#performance)
- [Git LFS Persistence](#git-lfs-persistence)
- [Requirements](#requirements)
- [Python CLI Dependencies](#python-cli-dependencies)
- [Architecture Notes](#architecture-notes)
- [License](#license)

---

## Why Noesis?

QMD covers the basics well. Noesis builds on it with features for more structured, long-lived memory:

- **Active + archive architecture** — expired entries move to a separate archive table instead of diluting search. Clean active search, opt-in older memory.
- **Auto-priority by memory type** — decisions get 85, preferences get 80, context gets 60, facts get 30, sessions get 20. Important memories always surface.
- **Soft-delete on TTL** — when entries expire, they're archived (not deleted). Still retrievable via archive search, gone from active results.
- **Structured memory types** — `fact`, `decision`, `preference`, `context`, `session`. Search is filtered by type, recall is more targeted.
- **Agent-aware search bias** — results from the calling agent are boosted, so each agent gets memory that's actually theirs.
- **Memory management tools** — `noesis_stats`, `noesis_cleanup`, `noesis_set_priority`, `noesis_export` give you programmatic control.
- **Periodic automatic cleanup** — TTL cleanup runs every 6 hours by default, no manual intervention needed.
- **Archive search fallback** — when active results are sparse (< 2), archive is queried automatically and results merged in. Older memories surface without explicit "search old memories" prompt.
- **LanceDB disk-based vector store** — purpose-built for vectors, IVF-PQ ANN indexing, ARM64 native, sub-30ms at scale.

---

## Stack

| Layer | Technology |
|-------|-----------|
| Vector store | LanceDB (disk-based, ARM64 native) |
| Embeddings | Ollama (local, no API key) |
| Default model | `nomic-embed-text` (768d, fast, ~274MB) |
| Alt model | `mxbai-embed-large` (1024d, higher accuracy) |
| Search | Vector ANN + BM25 FTS + MMR rerank |
| Session indexing | QMD session watcher |
| Plugin type | Memory slot (`plugins.slots.memory: "noesis"`) |

---

## Installation

### 1. Install the plugin

```bash
# Via ClawHub
openclaw plugins install clawhub:@blaspat/openclaw-noesis
```

### 2. Set the memory slot and context engine slot

```bash
openclaw config set plugins.slots.memory noesis
openclaw config set plugins.slots.contextEngine noesis
```

Or edit your OpenClaw config:

```json
{
  "plugins": {
    "entries": {
      "noesis": {
        "enabled": true,
        "config": {
          "ollamaEndpoint": "http://localhost:11434",
          "embeddingModel": "nomic-embed-text",
          "topK": 6,
          "indexQmdSessions": true,
          "watchMemoryDirs": true,
          "defaultTtlDays": 90,
          "autoCleanup": true,
          "cleanupIntervalHours": 6
        }
      }
    },
    "slots": {
      "memory": "noesis",
      "contextEngine": "noesis"
    }
  }
}
```

### 3. Start Ollama

```bash
ollama serve
```

Noesis auto-detects Ollama on startup and pulls the embedding model if it's not already downloaded. No manual steps needed.

### 4. Restart the gateway

```bash
openclaw gateway restart
```

That's it. Your agents now have persistent semantic memory.

---

## How It Works

### Active + Archive Architecture

Noesis maintains two LanceDB tables:

**`memories` (active table)**
- Only entries where `expiresAt = 0 OR expiresAt > now`
- Clean, fast search — no dilution from old entries
- All write operations go here

**`memories_archive` (archive table)**
- Entries where `expiresAt > 0 AND expiresAt < now` (TTL reached)
- Long-term storage, searchable via `noesis_search_archive`
- Active search falls back to archive when results are sparse (< 2)
- Both tables have ANN indexes for fast vector search

When TTL expires:
1. Entry is **deleted from `memories`** (no longer in active search)
2. Same entry is **inserted into `memories_archive`** (long-term storage)
3. User can retrieve it via archive search when needed

### Hybrid Search Pipeline

Every `noesis_search` call runs through this pipeline:

```
Query text
   ↓
Ollama embeddings (~20ms, local)
   ↓
LanceDB IVF-PQ vector search on active table (topK × 2 candidates)
   ↓
BM25 full-text search (keyword fallback/supplement)
   ↓
Hybrid merge (vector 60% + BM25 40%)
   ↓
MMR rerank (λ=0.7, balances relevance + diversity)
   ↓
If results < 2 → silently query archive table + merge results
   ↓
Top-K results with full content (lossless)
```

### Memory Types

| Type | Default Priority | Used for |
|------|-----------------|---------|
| `decision` | 85 | Things that were decided, logged for recall |
| `preference` | 80 | How things should be done |
| `context` | 60 | Project or session context |
| `fact` | 30 | Objective info (names, settings, world knowledge) |
| `session` | 20 | Auto-indexed from QMD session transcripts |

Higher priority entries surface first. Manual priority override always wins over auto-priority.

### When Agents Write to Noesis

Noesis auto-indexes without needing explicit agent calls. Memory flows in through these paths:

**1. Session transcripts (automatic)**

When `indexQmdSessions: true`, a background watcher picks up new QMD session files across all session locations: `~/.openclaw/sessions/`, `~/.openclaw/agents/<agentId>/sessions/`, and `~/.openclaw/agents/<agentId>/qmd/sessions/`. Session content is chunked and stored as `memoryType: session` entries under the agent's session ID.

**2. Agent memory dirs (automatic)**

When `watchMemoryDirs: true`, Noesis watches `~/.openclaw/agents/*/workspace/memory/*.md`. Any `.md` file in an agent's memory directory is checksummed, chunked, and indexed — deduplicated by content hash. This includes `MEMORY.md`, daily notes, and anything else an agent writes to disk.

**3. Manual via `noesis_index`**

Agents can call `noesis_index` directly with text, tags, `memoryType`, `priority`, and `ttlDays`. Used for explicitly important entries — a decision the agent wants to guarantee will be remembered.

### Multi-Agent Isolation

Each entry stores an `agentId`. By default, search only returns entries from the querying agent. Pass `crossAgent: true` to search across all agents (read-only — agents can't write each other's memories).

### Context Engine (Assemble + Ingest Hooks)

Noesis also registers as a context engine — set `plugins.slots.contextEngine = "noesis"` to activate. This replaces OpenClaw's built-in LLM summarization with proactive, zero-LLM context management.

**Ingest hook:** Every new message is analyzed for memory content. Pattern-matched to detect decisions ("we decided X", priority 85), preferences ("I prefer Y", priority 80), facts (topic sentences, priority 30), and sessions (summaries, priority 20). Everything else stored as context (priority 60). Stored with TTL, indexed in LanceDB.

**Assemble hook:** Before each model run, queries Noesis for all entries with `priority >= assembleInjectPriority` (default: 75). Formats them as a `systemPromptAddition` block the model sees on every turn. Decisions and preferences are always in context — the model never needs to be told twice. Zero LLM cost.

**Compact hook:** Delegates to OpenClaw's built-in auto-compaction (`ownsCompaction: false`). The `/compact` command continues to work. Noesis doesn't add LLM summarization cost.

**Why this vs Lossless-Claw:** Lossless-Claw fires 5-10+ LLM calls per compaction (multi-level DAG condensation). Noesis context engine fires 0. Decisions and preferences are always preserved; lower-priority content stays retrievable via `noesis_search`.

### ANN Index

Both `memories` and `memories_archive` tables get IVF-PQ ANN indexes when they reach ≥256 rows:

- **nprobe = 16** — 16 partition probes per query. >90% recall at ~20ms on HDD.
- **num_sub_vectors = 96** — PQ compression for 768-dim vectors.
- Indexes are persistent on disk and survive restarts.
- Archive table gets its own ANN index for fast archive search.

---

## Tools

### Agent tools (use these in your prompts)

| Tool | Description |
|------|-------------|
| `noesis_index` | Store a memory entry (text + metadata + priority + TTL). Auto-priority by type if not specified. |
| `noesis_search` | Semantic search with filters. Falls back to archive automatically when active results < 2. |
| `noesis_recall` | List recent entries by agent or session, newest first. |
| `noesis_import` | Trigger MD → LanceDB migration for one agent or all agents. |
| `noesis_stats` | Entry counts, breakdown by agent/type/priority, archive stats. |
| `noesis_delete` | Delete entry by ID. |
| `noesis_cleanup` | Move all expired entries from active → archive table. Manual trigger. |
| `noesis_export` | Bulk export all active entries as JSON (with optional agent/type filters). |
| `noesis_set_priority` | Update priority and/or TTL on an existing entry. |
| `noesis_search_archive` | Search archived memories explicitly. For "search older memories" queries. |

### Memory slot tools (used by OpenClaw auto-recall)

| Tool | Description |
|------|-------------|
| `memory_search` | Auto-recall entry point |
| `memory_get` | Retrieve by ID |
| `memory_index` | Auto-index new content |
| `memory_recall` | Cross-session recall |

## Migrating Existing Memory Files

If you have existing markdown memory files (`MEMORY.md`, `memory/*.md`), import them:

```bash
# Import one agent
noesis_import agentId=<agentId>

# Import all agents
noesis_import
```

Or use the standalone Python CLI:

```bash
pip install lancedb pyarrow numpy requests

python3 scripts/import_memory.py --agent <agentId>
python3 scripts/import_memory.py --all
python3 scripts/import_memory.py --agent <agentId> --chunk-size 256 --model mxbai-embed-large
```

---

## Configuration

| Key | Default | Description |
|-----|---------|-------------|
| `lanceDbPath` | `~/.openclaw/noesis/db` | LanceDB storage directory |
| `ollamaEndpoint` | `http://localhost:11434` | Ollama API endpoint |
| `embeddingModel` | `nomic-embed-text` | Embedding model (see below) |
| `chunkSize` | `512` | Chunk size in words |
| `chunkOverlap` | `64` | Overlap between chunks |
| `topK` | `6` | Default search result count |
| `autoMigrate` | `false` | Auto-import markdown files on startup |
| `indexQmdSessions` | `true` | Watch + auto-index QMD sessions |
| `watchMemoryDirs` | `true` | Watch agent memory dirs for changes and auto-index .md files |
| `defaultTtlDays` | `90` | Default TTL in days for new entries. `0` = never expire. |
| `autoCleanup` | `true` | On startup, move expired entries to archive. |
| `cleanupIntervalHours` | `6` | Run TTL cleanup on this interval (hours). `0` = disabled. |
| `contextEngineEnabled` | `true` | Register Noesis as context engine (assemble + ingest hooks). |
| `assembleInjectPriority` | `75` | Min priority for Assemble hook injection. Entries >= this always enter context. |
| `assembleMaxEntries` | `20` | Max entries to inject via Assemble hook. |
| `assembleMaxAgeDays` | `30` | Max age (days) for Assemble injection. `0` = no limit. |
| `gitLfsEnabled` | `false` | Enable Git LFS persistence for LanceDB backups |
| `gitLfsRepo` | `<username>/openclaw-noesis-data` | GitHub repo for Git LFS snapshots |
| `annNprobe` | `16` | IVF-PQ search probes (higher = more accurate, slower) |
| `annNumSubvectors` | `96` | IVF-PQ compression granularity |

### Embedding Model Choice

**`nomic-embed-text` (default, 768d, ~274MB)**
- Faster inference (~20ms on CPU)
- Excellent for short queries and agent memory recall
- Recommended for most setups

**`mxbai-embed-large` (alternative, 1024d, ~834MB)**
- Higher accuracy on longer contexts
- Better MTEB scores overall
- Use when retrieval accuracy matters more than speed

Switch model:
```json
{ "plugins": { "entries": { "noesis": { "config": { "embeddingModel": "mxbai-embed-large" } } } } }
```

---

## Performance

| Setup | Query latency | Notes |
|-------|--------------|-------|
| No index, small dataset | ~50–200ms | Fine for <1K entries |
| IVF-PQ index, HDD | ~20–50ms | Recommended baseline |
| IVF-PQ index, NVMe SSD | <10ms | Automatic when hardware allows |

Embedding latency with `nomic-embed-text` on a typical VPS/laptop CPU: **~20ms per query**.

---

## Git LFS Persistence

To back up your LanceDB data to GitHub:

```json
{
  "gitLfsEnabled": true,
  "gitLfsRepo": "yourusername/noesis-data"
}
```

Set up the data repo once:

```bash
git init noesis-data && cd noesis-data
git lfs install
git lfs track "*.lance" "*.vectordata"
git add .gitattributes && git commit -m "init"
git remote add origin https://github.com/yourusername/noesis-data.git
git push -u origin main
```

Then push snapshots manually or via cron:

```bash
cp ~/.openclaw/noesis/db/*.lance ./noesis-data/
cd noesis-data && git add . && git commit -m "snapshot $(date -Iseconds)" && git push
```

---

## Requirements

- **OpenClaw** ≥ 2026.4.0
- **Node.js** ≥ 18 (22 recommended)
- **Ollama** running locally (`ollama serve`)
- **Disk space**: ~5KB per memory entry + model size (~274MB for nomic-embed-text)

---

## Python CLI Dependencies

The standalone import CLI requires:

```bash
pip install lancedb pyarrow numpy requests
```

The main plugin uses the `@lancedb/lancedb` Node.js SDK directly — no Python runtime needed for normal operation.

---

## Architecture Notes

```
OpenClaw Gateway
  └── Noesis Plugin (Node.js)
        ├── @lancedb/lancedb  → disk-based vector store (memories + memories_archive)
        ├── Ollama HTTP API   → local embeddings
        ├── chokidar          → QMD session file watcher + memory dir watcher
        └── Hybrid search     → vector + BM25 + MMR with archive fallback
```

Memory slot integration means OpenClaw's auto-recall loop calls `memory_search` automatically — no manual tool calls needed. Just talk to your agent normally.

---

---

## Bug Fixes & Improvements

### Critical fixes
- `queryByPriority` — query was built but never executed (toArray called on wrong object)
- Deduplication in `indexMemoryFile` — broken logic (`seen.add` after `return false`, dedup never worked)
- QMD session entries — always had `priority=0`, priority surfacing non-functional
- `parseSessionPath` — wrong path index handling for `qmd/sessions/` pattern
- Archive detection heuristic — incorrectly flagged active table entries as archived by age alone
- Periodic cleanup interval — reference was local, leaked on hot reload
- `memory_index` tool — `priority` and `ttlDays` params accepted but silently ignored

### Hardening
- SQL injection in all filter string interpolations — added `escapeFilterValue()` across all queries
- `embedBatch` — single embedding failure crashed entire batch; now uses `Promise.allSettled` with zero-vector fallback
- `watchMemoryDirs` default — corrected to `false` (was mismatched across schema/README/DEFAULT_CONFIG)
- Session path parsing — corrected all path patterns and removed duplicate unreachable branches
- Archive detection — rebuilt with explicit `archiveIds` set for correct tagging

---

## License

See [LICENSE.md](./LICENSE.md) for the full MIT license text.

---

*Built for the [OpenClaw](https://openclaw.ai) community. Feedback and PRs welcome.*
test version-bump fix Thu Apr 16 11:10:04 WIB 2026
# test
