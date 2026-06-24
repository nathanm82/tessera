# Quickstart

The smallest end-to-end example: build a pipeline with the default offline encoder,
add three text documents, retrieve the best matches, and print a grounded answer.

```bash
python examples/quickstart/quickstart.py
```

No model downloads or extra dependencies are needed — the default `HashingEncoder`
and `TemplateGenerator` run entirely offline. Swap in `RagPipeline(encoder=...)` with
a CLIP encoder for real semantics.
