"""The end-to-end RAG pipeline: index a corpus, retrieve, and answer."""

from __future__ import annotations

from collections.abc import Iterable

from tessera.config import PipelineConfig
from tessera.corpus import Corpus
from tessera.encoders import get_encoder
from tessera.encoders.base import Encoder
from tessera.generation.base import Generator
from tessera.generation.template import TemplateGenerator
from tessera.retrieval.multimodal import MultimodalRetriever
from tessera.types import Document


class RagPipeline:
    """Wires an encoder, a multimodal retriever, and a generator into one object.

    The defaults are fully offline (hashing encoder, template generator), so a
    pipeline can be constructed and exercised without any model downloads. Pass a
    real encoder or generator to upgrade either stage independently.
    """

    def __init__(
        self,
        config: PipelineConfig | None = None,
        encoder: Encoder | None = None,
        generator: Generator | None = None,
    ) -> None:
        self.config = config or PipelineConfig()
        self.encoder = encoder or get_encoder(self.config.encoder)
        self.generator = generator or TemplateGenerator()
        self.retriever = self._new_retriever()
        self._corpus = Corpus()
        self._indexed = False

    def _new_retriever(self) -> MultimodalRetriever:
        return MultimodalRetriever(
            self.encoder,
            text_weight=self.config.text_weight,
            image_weight=self.config.image_weight,
        )

    def add_documents(self, documents: Iterable[Document]) -> None:
        """Queue documents for indexing; call :meth:`index` (or retrieve) to commit."""
        self._corpus.extend(documents)
        self._indexed = False

    def index(self) -> None:
        """(Re)build the retriever from every queued document."""
        self.retriever = self._new_retriever()
        chunks = self._corpus.to_chunks(self.config.chunk_size, self.config.chunk_overlap)
        self.retriever.index(chunks)
        self._indexed = True
