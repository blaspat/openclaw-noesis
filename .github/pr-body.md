## What this PR adds

**High priority:**
- `memory_index`: embed content once and share across all chunks (N→1 embed calls)
- `embed()`: graceful zero-vector fallback on failure (robustness)
- ANN/FTS index creation: log via `logError` instead of silent `console.warn`
- Hybrid search: configurable `vectorWeight` (default 0.6, can set 1.0 for pure vector)

**Medium priority:**
- Cross-encoder reranking (opt-in `rerank: true`): re-embed query + candidates with Ollama, blend 60/40 with original score
- `upsertEntries` dedup: batch checksums in groups of 100 to avoid query size limits
- Last-cleanup timestamp: persisted to `~/.openclaw/noesis/.last-cleanup`, gap warning on startup

**Low priority:**
- `compact` hook: documented limitation around chunk fragmentation
- QMD watcher: detect mid-index file growth, re-debounce
- `recall()`: filter expired entries via SQL
- `getByChecksum()`: new DB method for future compact hook fix

**Docs:**
- README.md: added TODO section for future improvements

## Testing
- TypeScript compiles clean
- `cosineSimilarityDense` math verified: identical=1.0, zero=0.0, orthogonal sensible
- `scoreMap` populated before MMR/rerank, fallback for unmapped IDs works
- No behavioral changes unless features are explicitly enabled via config

## Notes
- Cross-encoder reranking is an Ollama approximation (no native /rerank endpoint), opt-in via `rerank: true`
- Zero-vector fallback in `embed()` can contaminate ANN search if Ollama is down, but plugin won't crash