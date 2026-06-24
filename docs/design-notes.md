# Design notes

Some decisions worth recording, and the reasoning behind them.

## Dependency-light core, optional heavy extras

The core depends only on numpy. Real encoders (CLIP) and ANN backends (FAISS) are
optional extras, imported lazily so that `import tessera` never drags in torch. This
keeps installs fast, makes the test suite run offline in seconds, and lets the library
be embedded somewhere that can't afford a multi-gigabyte dependency tree.

The price is the default `HashingEncoder`: its embeddings are deterministic but not
semantically meaningful. That is an intentional trade — it exists so the *plumbing* is
testable end to end. Real quality comes from swapping the encoder, not rewriting the
pipeline.

## Why a shared embedding space

Rather than maintaining a projection/adapter between separate text and image spaces,
every encoder commits to one space across modalities. CLIP already gives us that for
free, and it removes a whole class of "which space am I in?" bugs. The cost is that an
encoder must support both modalities to do cross-modal retrieval; encoders that
support only text simply leave the image index empty.

## Serializable payloads

Stores keep payloads as plain dicts (a serialized `Chunk`) rather than live objects.
That is what lets `InMemoryVectorStore.save` / `load` round-trip an index to disk as a
`.npy` matrix plus a JSON sidecar, with no pickle and no custom codecs.

## Fusion over concatenation

A text index and an image index report scores on different scales, so naively merging
their hits favors whichever scale is larger. `weighted_fusion` makes the blend
explicit, and `reciprocal_rank_fusion` ignores magnitudes entirely by working from
ranks — useful when you genuinely can't compare two retrievers' scores.

## Open questions

- A reranking stage between retrieval and generation would help precision.
- Persisting a whole `RagPipeline` (not just the store) is not implemented yet.
- Chunking is whitespace-token based; a sentence-aware splitter would be better for
  prose. `# TODO: revisit chunk boundaries for long documents`.
