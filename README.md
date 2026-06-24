# tessera

[![CI](https://github.com/nathanm82/tessera/actions/workflows/ci.yml/badge.svg)](https://github.com/nathanm82/tessera/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Ruff](https://img.shields.io/badge/lint-ruff-261230)](https://github.com/astral-sh/ruff)

Multimodal retrieval-augmented generation, built from small, swappable pieces.

`tessera` indexes corpora that mix **images and text**, retrieves the most relevant
pieces for a query, and assembles grounded context for a language model. Text and
images are embedded into a single shared space, so a text query can surface image
chunks and vice versa.

The core depends only on **numpy** and ships a deterministic hashing encoder, so the
whole pipeline — indexing, retrieval, generation — runs offline with no model
downloads. When you want real cross-modal semantics, drop in the CLIP encoder via an
optional extra; nothing else in your code changes.

## Why another RAG library?

Most multimodal retrieval code is a notebook glued to one model and one vector store.
`tessera` keeps the moving parts behind small interfaces:

- **Encoders** map text/images to a shared embedding space.
- **Stores** hold vectors and rank them.
- **Retrievers** turn a query into ranked, source-attributed chunks.
- **Generators** turn retrieved context into an answer.

Each is registered by name and swappable, so you can start fully offline and upgrade
one stage at a time.

## Install

```bash
pip install tessera-rag                 # core (numpy only)
pip install "tessera-rag[clip]"         # + open_clip / torch for real semantics
pip install "tessera-rag[faiss]"        # + faiss-cpu for large indexes
```

Or from a checkout:

```bash
pip install -e ".[dev]"
```

## Usage

```python
from tessera import Document, RagPipeline

pipeline = RagPipeline()
pipeline.add_documents([
    Document(id="reef", text="The Great Barrier Reef is the largest coral reef system."),
    Document(id="coral", text="Coral reefs support about a quarter of all marine species."),
])

answer = pipeline.answer("tell me about coral reefs", top_k=2)
print(answer.text)
for source in answer.sources:
    print(source.score, source.chunk.doc_id)
```

There is also a small CLI:

```bash
tessera index corpus.jsonl
tessera query corpus.jsonl "a calm forest"
```

## Documentation

- [Usage guide](docs/usage.md)
- [Architecture](docs/architecture.md)
- [Design notes](docs/design-notes.md)
- [API reference](docs/api-reference.md)
- [Examples](examples/)

## License

MIT — see [LICENSE](LICENSE).
