"""Command-line interface for tessera."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from tessera import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tessera",
        description="Multimodal retrieval-augmented generation toolkit.",
    )
    parser.add_argument("--version", action="version", version=f"tessera {__version__}")
    parser.add_subparsers(dest="command")
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
