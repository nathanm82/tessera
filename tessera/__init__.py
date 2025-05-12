"""tessera: multimodal retrieval-augmented generation framework.

The top-level namespace re-exports the handful of objects most users need: build a
:class:`RagPipeline`, feed it :class:`Document` objects, and call ``answer``.
"""

from tessera.__about__ import __version__
from tessera.config import PipelineConfig
from tessera.corpus import Corpus
from tessera.encoders import Encoder, HashingEncoder, get_encoder
from tessera.generation import GeneratedAnswer, Generator, TemplateGenerator
from tessera.pipeline import RagPipeline
from tessera.retrieval import DenseRetriever, MultimodalRetriever
from tessera.types import Chunk, Document, Modality, RetrievalResult

__all__ = [
    "Chunk",
    "Corpus",
    "DenseRetriever",
    "Document",
    "Encoder",
    "GeneratedAnswer",
    "Generator",
    "HashingEncoder",
    "Modality",
    "MultimodalRetriever",
    "PipelineConfig",
    "RagPipeline",
    "RetrievalResult",
    "TemplateGenerator",
    "__version__",
    "get_encoder",
]
