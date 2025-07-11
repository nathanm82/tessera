"""Command-line interface for tessera."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from tessera import __version__
from tessera.corpus import Corpus
from tessera.io import load_jsonl
from tessera.pipeline import RagPipeline


def _cmd_index(args: argparse.Namespace) -> int:
    documents = list(load_jsonl(args.corpus))
    chunks = Corpus(documents).to_chunks()
    print(f"{len(documents)} documents -> {len(chunks)} chunks")
    return 0


def _cmd_query(args: argparse.Namespace) -> int:
    documents = list(load_jsonl(args.corpus))
    pipeline = RagPipeline()
    pipeline.add_documents(documents)
    results = pipeline.retrieve(args.query)
    if not results:
        print("(no results)")
        return 0
    for result in results:
        snippet = " ".join(result.chunk.content.split())[:80]
        print(f"{result.score:.4f}\t{result.chunk.doc_id}\t{snippet}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tessera",
        description="Multimodal retrieval-augmented generation toolkit.",
    )
    parser.add_argument("--version", action="version", version=f"tessera {__version__}")
    sub = parser.add_subparsers(dest="command")

    index_parser = sub.add_parser("index", help="report chunk statistics for a JSONL corpus")
    index_parser.add_argument("corpus", help="path to a JSONL corpus file")
    index_parser.set_defaults(func=_cmd_index)

    query_parser = sub.add_parser("query", help="retrieve chunks for a query over a corpus")
    query_parser.add_argument("corpus", help="path to a JSONL corpus file")
    query_parser.add_argument("query", help="the natural-language query")
    query_parser.set_defaults(func=_cmd_query)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "command", None) is None:
        parser.print_help()
        return 0
    try:
        return int(args.func(args))
    except (FileNotFoundError, ValueError) as exc:
        print(f"tessera: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
