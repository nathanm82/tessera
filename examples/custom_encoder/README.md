# Custom encoder

`tessera` doesn't care where embeddings come from — implement the small `Encoder`
interface and register your class so it's selectable by name.

```bash
python examples/custom_encoder/custom_encoder.py
```

The interface you implement:

- `dim` — the embedding dimensionality
- `supported_modalities` — which of `Modality.TEXT` / `Modality.IMAGE` you handle
- `encode_text` / `encode_image` — return L2-normalized `float32` rows

Register with `ENCODERS.register("my-encoder", MyEncoder)` and build it anywhere via
`get_encoder("my-encoder", ...)`.
