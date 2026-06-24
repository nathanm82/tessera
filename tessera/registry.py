"""A tiny string-keyed registry for pluggable components.

Encoders, stores, and generators register themselves by name so the framework can
assemble a pipeline from a config without importing every backend up front.
"""

from __future__ import annotations

from typing import Callable, Dict, Generic, List, TypeVar

from tessera.exceptions import RegistrationError

T = TypeVar("T")


class Registry(Generic[T]):
    """Maps lowercased names to factory callables that build a component."""

    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._factories: Dict[str, Callable[..., T]] = {}

    def register(self, name: str, factory: Callable[..., T]) -> None:
        """Register ``factory`` under ``name`` (case-insensitive)."""
        key = name.lower()
        if key in self._factories:
            raise RegistrationError(f"{self._kind} '{name}' is already registered")
        self._factories[key] = factory

    def create(self, name: str, **kwargs: object) -> T:
        """Instantiate the component registered under ``name``."""
        key = name.lower()
        if key not in self._factories:
            known = ", ".join(self.names()) or "<none>"
            raise RegistrationError(f"unknown {self._kind} '{name}'; known: {known}")
        return self._factories[key](**kwargs)

    def names(self) -> List[str]:
        """Return the registered names, sorted."""
        return sorted(self._factories)

    def __contains__(self, name: str) -> bool:
        return name.lower() in self._factories
