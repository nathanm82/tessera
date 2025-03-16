import pytest

from tessera.encoders import ENCODERS, HashingEncoder, get_encoder
from tessera.exceptions import RegistrationError


def test_hashing_is_registered() -> None:
    assert "hashing" in ENCODERS


def test_get_encoder_builds_hashing() -> None:
    enc = get_encoder("hashing", dim=32)
    assert isinstance(enc, HashingEncoder)
    assert enc.dim == 32


def test_unknown_encoder_raises() -> None:
    with pytest.raises(RegistrationError):
        get_encoder("does-not-exist")
