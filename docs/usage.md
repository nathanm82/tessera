# Usage guide

## Building a pipeline

`RagPipeline` ties together an encoder, a multimodal retriever, and a generator. The
defaults are fully offline:

```python
from tessera import RagPipeline, PipelineConfig

pipeline = RagPipeline(PipelineConfig(chunk_size=128, top_k=5))
```

`PipelineConfig` controls chunking (`chunk_size`, `chunk_overlap`), how many results
to return (`top_k`), and how strongly text vs. image hits are weighted when both are
present (`text_weight`, `image_weight`).

## Adding documents

A `Document` carries text, an image reference (a path), or both:

```python
from tessera import Document

pipeline.add_documents([
    Document(id="d1", text="a short caption", image="/path/to/photo.jpg"),
    Document(id="d2", text="text-only document"),
])
```

Documents are chunked lazily the first time you retrieve, or eagerly with
`pipeline.index()`. Adding more documents marks the index stale so the next retrieve
rebuilds it.

## Retrieving and answering

```python
results = pipeline.retrieve("query text", top_k=3)
for result in results:
    print(result.score, result.chunk.doc_id, result.chunk.content)

answer = pipeline.answer("query text")
print(answer.text)        # grounded answer
print(answer.sources)     # the chunks it used
```

## Choosing an encoder

```python
from tessera.encoders import get_encoder

pipeline = RagPipeline(encoder=get_encoder("hashing", dim=512))  # offline default
pipeline = RagPipeline(encoder=get_encoder("clip"))              # needs the clip extra
```

## Loading a corpus from disk

JSON Lines is the built-in format — one object per line with an `id` and at least one
of `text` / `image`; extra keys become metadata:

```json
{"id": "1", "text": "a caption", "image": "/imgs/1.jpg", "source": "wiki"}
```

```python
from tessera.io import load_jsonl

pipeline.add_documents(load_jsonl("corpus.jsonl"))
```

## Command line

```bash
tessera index corpus.jsonl              # report document and chunk counts
tessera query corpus.jsonl "a query"    # retrieve, printing score / doc id / snippet
tessera query corpus.jsonl "a query" --top-k 10
```

A missing corpus file exits with a non-zero status and a one-line error.
