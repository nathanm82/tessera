# Image + text search

Indexes documents that carry both a caption and an image, then runs a single text
query that ranks chunks from both modalities — the payoff of a shared embedding
space.

```bash
python examples/image_search/search.py
```

To search real images, point each `Document(image=...)` at a file on disk and build
the pipeline with the CLIP encoder:

```python
from tessera import RagPipeline
from tessera.encoders import get_encoder

pipeline = RagPipeline(encoder=get_encoder("clip"))  # needs: pip install "tessera-rag[clip]"
```
