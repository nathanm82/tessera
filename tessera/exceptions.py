"""Exception hierarchy for tessera.

Everything raised by the framework derives from :class:`TesseraError`, so callers
can catch the whole family with a single ``except``.
"""

from __future__ import annotations


class TesseraError(Exception):
    """Base class for all tessera errors."""


class RegistrationError(TesseraError):
    """Raised when registering or resolving a named component fails."""


class EncoderError(TesseraError):
    """Raised when an encoder cannot embed its input."""


class StoreError(TesseraError):
    """Raised for vector-store level failures."""


class DimensionMismatchError(StoreError):
    """Raised when a vector does not match the store's dimensionality."""
