import tessera


def test_version_is_exported() -> None:
    assert isinstance(tessera.__version__, str)


def test_public_surface_is_importable() -> None:
    expected = {
        "RagPipeline",
        "PipelineConfig",
        "Document",
        "Chunk",
        "Modality",
        "Encoder",
        "HashingEncoder",
        "MultimodalRetriever",
        "TemplateGenerator",
        "get_encoder",
    }
    assert expected.issubset(set(tessera.__all__))
    for name in expected:
        assert hasattr(tessera, name)
