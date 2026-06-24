import subprocess
import sys
from pathlib import Path


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "tessera.cli", *args],
        capture_output=True,
        text=True,
        check=False,
    )


def test_version_flag() -> None:
    result = _run("--version")
    assert result.returncode == 0
    assert "tessera" in result.stdout


def test_index_reports_counts(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text('{"id": "1", "text": "a b c d e f"}\n', encoding="utf-8")
    result = _run("index", str(corpus))
    assert result.returncode == 0
    assert "documents" in result.stdout


def test_query_returns_best_doc(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        '{"id": "1", "text": "the cat sat on the mat"}\n'
        '{"id": "2", "text": "quarterly stock prices fell"}\n',
        encoding="utf-8",
    )
    result = _run("query", str(corpus), "cat on the mat")
    assert result.returncode == 0
    assert "1" in result.stdout


def test_missing_corpus_exits_nonzero(tmp_path: Path) -> None:
    result = _run("index", str(tmp_path / "nope.jsonl"))
    assert result.returncode == 2
    assert "tessera:" in result.stderr
