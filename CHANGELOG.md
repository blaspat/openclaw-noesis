# Changelog

## [1.5.1](https://github.com/blaspat/openclaw-noesis/compare/v1.5.0...v1.5.1) (2026-04-20)


### Bug Fixes

* Fix LanceDB version store memory leak by adding periodic `optimize()` calls
  * Add `db.optimize()` method to reclaim `_versions/` directory (was 4.4GB)
  * Run `optimize()` on every startup to immediately reclaim bloated version store
  * Run `optimize()` on hardcoded 5-minute interval to keep version store lean


## [1.5.0](https://github.com/blaspat/openclaw-noesis/compare/v1.4.5...v1.5.0) (2026-04-20)


### Features

* **BREAKING** Replace Ollama with HuggingFace Transformers.js for CPU embeddings ([`ad39308`](https://github.com/blaspat/openclaw-noesis/commit/ad39308))
  * New `transformers.ts` backend — WASM/SIMD in-process, no Ollama server needed
  * Auto-downloads `nomic-ai/nomic-embed-text-v1.5` ONNX (q4f16) from HuggingFace on first use
  * Graceful degradation with zero vectors on embed failure
  * Timeout protection (60s model load, 30s per embed)


## [1.2.0](https://github.com/blaspat/openclaw-noesis/compare/v1.1.10...v1.2.0) (2026-04-16)


### Features

* dedicated error log file with structured JSON entries ([3e83931](https://github.com/blaspat/openclaw-noesis/commit/3e839318109ea0db4d9ccb7bebc7d21bc40d94af))


### Bug Fixes

* race condition in session/memory watcher + add noesis_dedup tool ([7502180](https://github.com/blaspat/openclaw-noesis/commit/750218097543cdbb28433c22028e9142518a759f))
* race condition in watchers + add noesis_dedup semantic dedup tool ([506dfe4](https://github.com/blaspat/openclaw-noesis/commit/506dfe4ff43b5caf7726f71cab416f3442c67519))
