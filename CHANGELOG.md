# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0]

### Added

- `tessera query` accepts `-k/--top-k` to control how many results are returned.

## [0.1.0]

### Added

- Shared-space `Encoder` interface with a deterministic, offline `HashingEncoder` and
  an optional `ClipEncoder` (the `clip` extra).
- `InMemoryVectorStore` with save/load, plus an optional `FaissVectorStore` (the
  `faiss` extra).
- `DenseRetriever` and a `MultimodalRetriever` that fuses text and image indexes.
- `weighted_fusion` and `reciprocal_rank_fusion` fusion strategies.
- `TemplateGenerator` and a `build_prompt` helper for grounded generation.
- `RagPipeline` orchestrating index → retrieve → answer.
- JSONL corpus loader and a `tessera` command-line interface.

[Unreleased]: https://github.com/nathanm82/tessera/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/nathanm82/tessera/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/nathanm82/tessera/releases/tag/v0.1.0
