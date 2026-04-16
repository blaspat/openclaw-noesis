DONE — ready for production

## CI/CD Setup (as of 2026-04-15)

Two-workflow setup for release automation:
- `version-bump.yml` — bumps version on PR lifecycle events (opened, synchronize, labeled, etc.)
- `release.yml` — builds, publishes to npm, and creates GitHub Release on push to main

No PATs, no GPG, no signing keys. Works with branch protection rules.