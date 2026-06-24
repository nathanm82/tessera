"""Generation turns retrieved context into a grounded answer."""

from __future__ import annotations

from tessera.generation.base import GeneratedAnswer, Generator, build_prompt
from tessera.generation.template import TemplateGenerator

__all__ = ["GeneratedAnswer", "Generator", "TemplateGenerator", "build_prompt"]
