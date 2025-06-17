"""Retrieve across a mixed image + text corpus with a single query.

The corpus pairs a caption with an image for each item. With the offline hashing
encoder the "images" are stand-in byte blobs so the example needs no assets; point
``image`` at real files and switch to the CLIP encoder for meaningful image search.

Run with::

    python examples/image_search/search.py
"""

import tempfile
from pathlib import Path

from tessera import Document, RagPipeline


def _write_demo_images(directory: Path) -> dict[str, str]:
    paths = {}
    for name, seed in {"sunset": 7, "city": 42, "forest": 99}.items():
        blob = bytes((seed * i) % 256 for i in range(512))
        path = directory / f"{name}.bin"
        path.write_bytes(blob)
        paths[name] = str(path)
    return paths


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        images = _write_demo_images(Path(tmp))
        documents = [
            Document(id=name, text=f"a photo of a {name}", image=path)
            for name, path in images.items()
        ]
        pipeline = RagPipeline()
        pipeline.add_documents(documents)

        for result in pipeline.retrieve("a calm forest", top_k=4):
            chunk = result.chunk
            print(f"{result.score:.3f}  {chunk.doc_id:8s} ({chunk.modality.value})")


if __name__ == "__main__":
    main()
