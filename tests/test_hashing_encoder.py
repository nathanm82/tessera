import numpy as np

from tessera.encoders.hashing import HashingEncoder
from tessera.types import Modality


def test_dim_and_supported_modalities() -> None:
    enc = HashingEncoder(dim=64)
    assert enc.dim == 64
    assert enc.supports(Modality.TEXT)
    assert enc.supports(Modality.IMAGE)


def test_text_embeddings_are_deterministic() -> None:
    enc = HashingEncoder(dim=128)
    a = enc.encode_text(["a red bicycle"])
    b = enc.encode_text(["a red bicycle"])
    assert np.array_equal(a, b)


def test_text_embeddings_are_unit_norm() -> None:
    enc = HashingEncoder(dim=128)
    vecs = enc.encode_text(["hello world", "another caption here"])
    norms = np.linalg.norm(vecs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_image_embeddings_have_encoder_dim() -> None:
    enc = HashingEncoder(dim=32)
    vecs = enc.encode_image([b"\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09"])
    assert vecs.shape == (1, 32)
    assert np.isclose(np.linalg.norm(vecs[0]), 1.0, atol=1e-5)


def test_different_text_differs() -> None:
    enc = HashingEncoder(dim=256)
    vecs = enc.encode_text(["sunset over the ocean", "a cat on a keyboard"])
    assert not np.allclose(vecs[0], vecs[1])


def test_encode_dispatch_matches_direct_calls() -> None:
    enc = HashingEncoder(dim=64)
    via_dispatch = enc.encode(["a caption"], Modality.TEXT)
    direct = enc.encode_text(["a caption"])
    assert np.array_equal(via_dispatch, direct)
