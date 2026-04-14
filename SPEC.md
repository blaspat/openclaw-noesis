# Noesis — Production Memory Plugin for OpenClaw

**Version:** 1.0.0
**Status:** Feature-complete — memory layer + context engine (assemble/ingest hooks).
**Repo:** https://github.com/blaspat/openclaw-noesis

---

## What is Noesis?

Noesis is a **production-grade memory plugin** for OpenClaw that provides:

- Cross-agent, cross-session semantic memory search
- Full local operation (no LLM API calls) — Ollama embeddings only
- LanceDB for persistent vector storage (active + archive tables)
- Active + archive architecture — expired entries move to archive, not deleted
- Auto-priority by memory type (decision=85, preference=80, context=60, fact=30, session=20)
- Periodic TTL cleanup on configurable interval (default: every 6 hours)
- Archive search fallback — automatic when active results are sparse
- QMD session watcher for automatic session indexing
- Memory dir watcher for automatic .md file indexing
- Memory slot integration — replaces auto-recall loop, not just a tool
- Git LFS persistence to GitHub

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        OpenClaw Gateway                          │
│  ┌────────────┐  ┌──────────────┐  ┌────────────────────────┐   │
│  │   QMD      │  │   Noesis     │  │  memory-core (builtin)│   │
│  │ (session)  │  │  (memory     │  │                      │   │
│  │            │  │   slot)      │  │                      │   │
│  └─────┬──────┘  └──────┬───────┘  └────────────────────────┘   │
│        │                │                                      │
│        ↓                ↓                                      │
│  ┌──────────────────────────────────────┐                      │
│  │      Noesis Plugin (Node.js/TS)      │                      │
│  │  ┌────────────┐  ┌────────────────┐  │                      │
│  │  │  Ollama    │  │   LanceDB      │  │                      │
│  │  │ (embedder) │  │  (vector store)│  │                      │
│  │  └────────────┘  └────────────────┘  │                      │
│  │         ↓                ↓           │                      │
│  │  ┌──────────────┐ ┌───────────────┐  │                      │
│  │  │ memories     │ │memories_     │  │                      │
│  │  │ (active)     │ │archive        │  │                      │
│  │  └──────────────┘ └───────────────┘  │                      │
│  └──────────────────────────────────────┘                      │
│        │                ↓                                      │
│        │         ┌──────────────┐                              │
│        │         │  Git LFS     │                              │
│        │         │  + GitHub    │                              │
│        │         └──────────────┘                              │
└────────┴───────────────────────────────────────────────────────┘
```

---

## Key Design Decisions

### Active + Archive Architecture

Entries have a TTL (`expiresAt`). When TTL is reached:

1. Entry is **deleted from `memories`** (active table)
2. Same entry is **inserted into `memories_archive`** (long-term archive)
3. Active search never sees expired entries — no dilution
4. Archive is searchable via `noesis_search_archive` or automatic fallback in `noesis_search`

This replaces the "20% penalty" approach. Users explicitly retrieve from archive when they want older memories.

### Auto-Priority by Memory Type

When `noesis_index` is called without an explicit `priority`, the system auto-assigns based on `memoryType`:

| Memory Type | Auto-Priority |
|-------------|---------------|
| decision    | 85            |
| preference  | 80            |
| context     | 60            |
| fact        | 30            |
| session     | 20            |

Manual priority always overrides auto-priority.

### Archive Search Fallback

`noesis_search` first queries the active table. If results < 2, it automatically queries the archive table and merges results. Users get relevant memories regardless of age — no need to explicitly ask for "old memories."

---

## Context Engine (Assemble + Ingest Hooks)

Noesis registers as a context engine (`plugins.slots.contextEngine = "noesis"`). This replaces OpenClaw's built-in LLM summarization with proactive, zero-LLM context management.

### Ingest Hook

Every new message is analyzed for memory content:
- Pattern `we decided|we're going with|let's go with` → `decision`, priority=85
- Pattern `i prefer|i always|i never` → `preference`, priority=80
- Pattern `^(the |a )?[\w\s]+ is |^fact:|^info:` → `fact`, priority=30
- Pattern `^summary:|^session summary` → `session`, priority=20
- Everything else → `context`, priority=60

Stored with TTL (`defaultTtlDays`), indexed in LanceDB.

### Assemble Hook

Before every model run:
1. Query Noesis for entries with `priority >= assembleInjectPriority` (default: 75)
2. Format as a structured `systemPromptAddition` block
3. Model enters every conversation already knowing decisions and preferences
4. Zero LLM cost — just priority-filtered DB reads

### Compact Hook

`ownsCompaction: false` — delegates to OpenClaw's built-in auto-compaction. This avoids breaking the `/compact` command while letting OpenClaw handle overflow recovery.

### Why This Approach?

- **Legacy engine**: LLM summarization every overflow (expensive)
- **Lossless-Claw**: Multi-level DAG condensation (5-10 LLM calls per overflow)
- **Noesis context engine**: Priority-filtered injection (zero LLM calls)

Decisions (85) and preferences (80) are **always in context** — the model never needs to be told twice. Low-priority content stays in Noesis for on-demand retrieval via `noesis_search`.

---

## Data Model

### LanceDB Table: `memories` (active)

| Column | Type | Description |
|--------|------|-------------|
| `id` | string | UUID v4 |
| `agentId` | string | "claire", "luna", "kate", "patrick" |
| `sessionId` | string | OpenClaw session ID |
| `content` | string | Full original text (lossless) |
| `chunk` | string | Chunked for embedding (for reference) |
| `embedding` | float32[768/1024] | Vector from Ollama |
| `memoryType` | string | "fact" / "decision" / "preference" / "context" / "session" |
| `priority` | int64 | Priority 0–100. Auto from type if not set. ≥75 always surfaces. |
| `expiresAt` | int64 | Unix ms. 0 = never. On expiry, moved to archive. |
| `createdAt` | int64 | Unix timestamp (ms) |
| `sourcePath` | string | Original .md or QMD session path |
| `checksum` | string | SHA-256(content + agentId) for dedup |
| `tags` | string[] | Manual + auto tags |

### LanceDB Table: `memories_archive`

Identical schema to `memories`. Contains entries after TTL expires. Both tables have independent ANN indexes for fast search.

---

## Feature Checklist

### Core Memory
- [x] Active + archive table architecture
- [x] TTL with soft-delete (move to archive, not penalty)
- [x] Auto-priority by memory type
- [x] Manual priority override
- [x] Memory types: fact, decision, preference, context, session
- [x] Per-entry metadata: agentId, sessionId, sourcePath, tags, checksum
- [x] Checksum-based dedup on upsert

### Search
- [x] Hybrid vector + BM25 + MMR search
- [x] Archive search fallback (automatic when active results < 2)
- [x] Explicit `noesis_search_archive` tool
- [x] Priority boost for ≥75 entries
- [x] Agent-aware bias in search
- [x] Cross-agent search with `crossAgent: true`
- [x] Memory type filtering
- [x] ANN index on both active and archive tables

### Ingestion
- [x] Manual `noesis_index` tool
- [x] QMD session file watcher (`startQmdWatcher`)
- [x] Agent memory dir watcher (`startMemoryWatcher`)
- [x] `noesis_import` for bulk MD file import
- [x] `noesis_export` for JSON backup
- [x] `noesis_import` via standalone Python CLI

### Management
- [x] `noesis_stats` with per-agent, per-type breakdown + archive stats
- [x] `noesis_cleanup` manual trigger (moves expired → archive)
- [x] `noesis_delete` entry by ID
- [x] `noesis_set_priority` update priority/TTL
- [x] Periodic TTL cleanup via `cleanupIntervalHours` config
- [x] Startup TTL cleanup via `autoCleanup` config

### Integration
- [x] Memory slot (`plugins.slots.memory: "noesis"`)
- [x] OpenClaw auto-recall integration
- [x] Ollama auto-config + auto-pull
- [x] Git LFS persistence path
- [x] Multi-agent with agentId isolation

### Not Implemented (future phase)
- [ ] Context engine (`kind: "context-engine"`) — controls context assembly/compaction
- [ ] Archive TTL — eventually delete archive entries after extended period
- [ ] Priority decay — entries slowly lose priority over time
- [ ] Memory summarization on compact

---

## Tools Reference

### Agent Tools

| Tool | Parameters | Description |
|------|------------|-------------|
| `noesis_index` | `content`, `agentId?`, `sessionId?`, `tags?`, `memoryType?`, `priority?`, `ttlDays?` | Store memory. Auto-priority by type if priority not set. |
| `noesis_search` | `query`, `agentId?`, `memoryType?`, `topK?`, `crossAgent?` | Search. Falls back to archive if active < 2 results. |
| `noesis_recall` | `agentId?`, `sessionId?`, `limit?` | Get recent entries by agent/session. |
| `noesis_stats` | — | Counts + breakdowns (by agent, type, priority, archive stats). |
| `noesis_delete` | `id` | Delete entry by ID. |
| `noesis_cleanup` | — | Move expired entries to archive (manual trigger). |
| `noesis_export` | `agentId?`, `memoryType?` | Bulk export as JSON. |
| `noesis_set_priority` | `id`, `priority?`, `ttlDays?` | Update priority and/or TTL. |
| `noesis_search_archive` | `query`, `agentId?`, `memoryType?`, `topK?` | Explicit archive search. |
| `noesis_import` | `agentId?` | Trigger MD → LanceDB migration. |

### Memory Slot Tools (OpenClaw auto-recall)

| Tool | Used by | Description |
|------|---------|-------------|
| `memory_search` | OpenClaw auto-recall | Entry point for memory prompts |
| `memory_get` | OpenClaw memory retrieval | Get specific entry |
| `memory_index` | OpenClaw indexer | Auto-index new content |
| `memory_recall` | OpenClaw session recall | Cross-session recall |

---

## Configuration

| Key | Default | Description |
|-----|---------|-------------|
| `lanceDbPath` | `~/.openclaw/noesis/db` | LanceDB storage directory |
| `ollamaEndpoint` | `http://localhost:11434` | Ollama API endpoint |
| `embeddingModel` | `nomic-embed-text` | Embedding model |
| `chunkSize` | `512` | Chunk size in words |
| `chunkOverlap` | `64` | Overlap between chunks |
| `topK` | `6` | Default search result count |
| `autoMigrate` | `false` | Auto-import markdown files on startup |
| `indexQmdSessions` | `true` | Watch + auto-index QMD sessions |
| `watchMemoryDirs` | `false` | Watch agent memory dirs for changes |
| `defaultTtlDays` | `90` | Default TTL. `0` = never. |
| `autoCleanup` | `true` | On startup, move expired → archive |
| `cleanupIntervalHours` | `6` | Periodic cleanup interval (hours). `0` = disabled |
| `gitLfsEnabled` | `false` | Enable Git LFS backups |
| `gitLfsRepo` | `blaspat/openclaw-noesis-data` | GitHub repo for LFS |
| `annNprobe` | `16` | IVF-PQ search probes |
| `annNumSubvectors` | `96` | IVF-PQ compression |

---

## Embedding Models

| Model | Dimensions | Size | Speed | Best for |
|-------|------------|------|-------|----------|
| `nomic-embed-text` | 768 | ~274MB | ~20ms/query | General-purpose, fast iteration |
| `mxbai-embed-large` | 1024 | ~834MB | Slower | High-accuracy retrieval |

---

## Installation

```bash
# 1. Install Noesis plugin
npm install @blaspat/openclaw-noesis

# Or via ClawHub
openclaw plugins install clawhub:@blaspat/openclaw-noesis

# 2. Set memory slot
openclaw config set plugins.slots.memory noesis

# 3. Ensure Ollama is running
ollama serve

# 4. Restart gateway
openclaw gateway restart
```

---

## Repo Layout

```
openclaw-noesis/
├── src/
│   ├── index.ts       # Plugin entry point + tools
│   ├── lancedb.ts    # LanceDB operations (active + archive tables)
│   ├── ollama.ts     # Ollama embeddings + auto-config
│   ├── migrator.ts   # MD → LanceDB migration
│   ├── search.ts     # Hybrid search pipeline
│   ├── watcher.ts    # QMD session + memory dir watchers
│   └── types.ts      # TypeScript interfaces + config
├── dist/             # Compiled JS output
├── scripts/
│   ├── import_memory.py    # Standalone migration CLI
│   └── lancedb_server.py   # LanceDB server runner
├── LICENSE.md
├── README.md          # Public-facing README
├── SPEC.md            # This file
└── openclaw.plugin.json
```

---

## What's Next (Future Phases)

**Phase 2: Context Engine** ✅ **IMPLEMENTED (v1.3.0)**
- Ingest: auto-detect memory type from message content ✅
- Assemble: inject priority ≥75 memories as systemPromptAddition ✅
- Compact: delegates to runtime (no LLM cost) ✅

**Phase 3: Archive Lifecycle**
- Archive entries auto-purge after extended period (e.g., 1 year)
- Or allow users to set archive-specific TTL

**Phase 4: Priority Decay**
- Entries slowly lose priority over time unless recalled/boosted