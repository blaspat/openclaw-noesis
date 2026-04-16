## Summary

### Bug fixes (7 items)
- `queryByPriority` never executed — Assemble hook always returned empty
- Deduplication in `indexMemoryFile` was broken (seen.add after return)
- QMD session entries always had priority=0
- Session path parsing wrong indices in `parseSessionPath`
- Archive detection heuristic incorrectly flagged active entries
- Periodic cleanup interval leaked on hot reload
- `memory_index` tool silently ignored `priority` and `ttlDays` params

### Hardening (4 items)
- SQL injection in all filter string interpolations — added `escapeFilterValue()`
- `embedBatch` crash on partial failure — now uses `Promise.allSettled` with zero-vector fallback
- `watchMemoryDirs` default corrected to false (mismatched across schema/README/DEFAULT_CONFIG)

### Auto-versioning system
- New `scripts/bump-version.js` — bumps patch/minor/major, syncs package.json + openclaw.plugin.json
- New `scripts/github-commit.js` — creates verified commits via GitHub REST API (no GPG/SSH needed)
- New `.github/workflows/release.yml` — push to main bumps version, publishes to npm (trusted publisher OIDC), creates GitHub Release
- Removed stale `scripts/sync-version.js`
- Dropped `prepublishOnly` from package.json scripts

### Workflow fixes
- Removed unicode comment chars that caused YAML parser errors
- Stripped all comments from release.yml
- Fixed 422 error in github-commit.js — switched to base64-encoded input
- Removed NPM_TOKEN secret — uses npm trusted publisher OIDC instead

## Testing
- Build passes (`npm run build`)
- All 11 items addressed
- No new issues introduced