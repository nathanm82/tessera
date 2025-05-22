"""Command-line interface for tessera."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from tessera import __version__
from tessera.corpus import Corpus
from tessera.io import load_jsonl


def _cmd_index(args: argparse.Namespace) -> int:
    documents = list(load_jsonl(args.corpus))
    chunks = Corpus(documents).to_chunks()
    print(f"{len(documents)} documents -> {len(chunks)} chunks")
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

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "command", None) is None:
        parser.print_help()
        return 0
    handler = args.func
    return int(handler(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
