from pathlib import Path

import pytest

from tessera.io import load_jsonl


def test_load_jsonl_parses_records(tmp_path: Path) -> None:
    path = tmp_path / "corpus.jsonl"
    path.write_text(
        '{"id": "1", "text": "hello", "lang": "en"}\n'
        "\n"
        '{"id": 2, "image": "/tmp/a.png"}\n',
        encoding="utf-8",
    )
    docs = list(load_jsonl(path))
    assert len(docs) == 2
    assert docs[0].id == "1"
    assert docs[0].text == "hello"
    assert docs[0].metadata == {"lang": "en"}
    assert docs[1].id == "2"
    assert docs[1].image == "/tmp/a.png"


def test_missing_id_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"text": "no id here"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="id"):
        list(load_jsonl(path))


def test_invalid_json_reports_line(tmp_path: Path) -> None:
    path = tmp_path / "broken.jsonl"
    path.write_text('{"id": "1"}\nnot json\n', encoding="utf-8")
    with pytest.raises(ValueError, match="line 2"):
        list(load_jsonl(path))
