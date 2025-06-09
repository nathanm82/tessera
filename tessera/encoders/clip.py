"""Optional CLIP encoder bridging open_clip into tessera's Encoder interface.

Requires the ``clip`` extra::

    pip install "tessera-rag[clip]"

This is where real cross-modal semantics come from: text and images land in CLIP's
shared space, so a text query genuinely retrieves relevant images. ``torch`` and
``open_clip`` are imported lazily, so importing tessera never pulls in the heavy
stack -- only constructing a :class:`ClipEncoder` does.
"""

from __future__ import annotations

from collections.abc import Sequence
from io import BytesIO
from typing import Any

import numpy as np
from numpy.typing import NDArray

from tessera.encoders.base import Encoder
from tessera.exceptions import EncoderError
from tessera.similarity import l2_normalize
from tessera.types import Modality


class ClipEncoder(Encoder):
    """Wraps an open_clip checkpoint as a tessera encoder."""

    supported_modalities = frozenset({Modality.TEXT, Modality.IMAGE})

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: str = "cpu",
    ) -> None:
        try:
            import open_clip
            import torch
        except ImportError as exc:  # pragma: no cover - exercised only with the extra
            raise EncoderError(
                "ClipEncoder requires the 'clip' extra: pip install 'tessera-rag[clip]'"
            ) from exc
        self._torch = torch
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=device
        )
        model.eval()
        self._model: Any = model
        self._preprocess: Any = preprocess
        self._tokenizer: Any = open_clip.get_tokenizer(model_name)
        self._device = device
        self._dim = int(getattr(model.visual, "output_dim", 512))

    @property
    def dim(self) -> int:
        return self._dim

    def encode_text(self, texts: Sequence[str]) -> NDArray[np.float32]:
        tokens = self._tokenizer(list(texts))
        with self._torch.no_grad():
            features = self._model.encode_text(tokens.to(self._device))
        return l2_normalize(features.cpu().numpy().astype(np.float32))

    def encode_image(self, images: Sequence[object]) -> NDArray[np.float32]:
        from PIL import Image

        tensors = []
        for image in images:
            if isinstance(image, bytes):
                pil = Image.open(BytesIO(image)).convert("RGB")
            elif isinstance(image, str):
                pil = Image.open(image).convert("RGB")
            else:
                raise EncoderError(f"cannot read image of type {type(image).__name__}")
            tensors.append(self._preprocess(pil))
        batch = self._torch.stack(tensors).to(self._device)
        with self._torch.no_grad():
            features = self._model.encode_image(batch)
        return l2_normalize(features.cpu().numpy().astype(np.float32))
