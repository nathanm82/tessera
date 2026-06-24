# Architecture

`tessera` is a small pipeline of four stages, each behind an interface so it can be
replaced independently.

```
Documents ──► Corpus.to_chunks ──► Encoder ──► VectorStore
                                                   │
                                Query ──► Encoder ──┤
                                                   ▼
                                          MultimodalRetriever ──► Generator ──► Answer
```

## Stages

**Encoder** (`tessera/encoders`)
Maps text and images into one shared, L2-normalized embedding space. `HashingEncoder`
is deterministic and dependency-free; `ClipEncoder` (optional) provides real
cross-modal semantics. Encoders register themselves by name in `ENCODERS`.

**Vector store** (`tessera/stores`)
Holds embedding vectors plus a JSON-serializable payload and ranks them against a
query. `InMemoryVectorStore` is a numpy brute-force index that can be saved and
reloaded; `FaissVectorStore` (optional) swaps in FAISS for larger corpora.

**Retrieval** (`tessera/retrieval`)
`DenseRetriever` indexes one modality. `MultimodalRetriever` keeps a text index and an
image index over the same encoder and **fuses** their results, since both live in the
same space. Fusion strategies (`weighted_fusion`, `reciprocal_rank_fusion`) combine
ranked lists whose score scales may differ.

**Generation** (`tessera/generation`)
`Generator.generate` turns retrieved context into a `GeneratedAnswer`. `build_prompt`
produces the exact text an LLM would receive; `TemplateGenerator` is an offline,
deterministic fallback that cites its sources.

## Data types

`Document` → split into `Chunk`s → embedded into vectors → retrieved as
`RetrievalResult` (chunk + score) → summarized into a `GeneratedAnswer`.

## The shared-space contract

Every encoder returns L2-normalized `float32` rows of a single `dim`, regardless of
modality. That single rule is what makes cross-modal retrieval work: a dot product is
cosine similarity, and a text query and an image chunk are directly comparable.
