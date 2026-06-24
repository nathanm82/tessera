# API reference

A summary of the public surface re-exported from `tessera`.

## Pipeline

### `RagPipeline(config=None, encoder=None, generator=None)`

- `add_documents(documents)` — queue documents for indexing.
- `index()` — (re)build the retriever from queued documents.
- `retrieve(query, query_modality=Modality.TEXT, top_k=None) -> list[RetrievalResult]`
- `answer(query, query_modality=Modality.TEXT, top_k=None) -> GeneratedAnswer`

### `PipelineConfig`

Fields: `encoder` (str), `chunk_size`, `chunk_overlap`, `top_k`, `text_weight`,
`image_weight`. Validated on construction.

## Data types

- `Document(id, text=None, image=None, metadata={})` — `modalities()` returns the
  modalities present.
- `Chunk(id, doc_id, modality, content, metadata={})` — `to_dict()` / `from_dict()`.
- `RetrievalResult(chunk, score)`
- `GeneratedAnswer(text, sources=[])`
- `Modality.TEXT` / `Modality.IMAGE`

## Encoders

- `Encoder` — base class: `dim`, `supported_modalities`, `supports(modality)`,
  `encode_text(texts)`, `encode_image(images)`, `encode(items, modality)`.
- `HashingEncoder(dim=256, image_window=8)` — offline default.
- `get_encoder(name, **kwargs)` / `ENCODERS` — registry of encoder factories.
- `ClipEncoder(...)` — optional, requires the `clip` extra.

## Stores

- `InMemoryVectorStore` — `add(ids, vectors, payloads)`, `search(query, top_k=5)`,
  `save(path)`, `load(path)`, `len()`.
- `FaissVectorStore` — optional, requires the `faiss` extra.
- `SearchHit(id, score, payload)`

## Retrieval

- `DenseRetriever(encoder, modality=Modality.TEXT, store=None)`
- `MultimodalRetriever(encoder, text_weight=0.5, image_weight=0.5)`
- `weighted_fusion(result_lists, weights=None)`
- `reciprocal_rank_fusion(result_lists, k=60)`

## Generation

- `Generator` — base class with `generate(query, context)`.
- `TemplateGenerator(max_sources=3)` — `prompt(query, context)` and `generate(...)`.
- `build_prompt(query, context, *, max_chars=500)`
