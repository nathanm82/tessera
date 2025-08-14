"""Micro-benchmark for index build and retrieval latency.

Uses the offline hashing encoder so the numbers reflect tessera's own overhead rather
than a model's forward pass.

Run with::

    python benchmarks/bench_retrieval.py --docs 2000 --queries 100
"""

from __future__ import annotations

import argparse
import time

from tessera import Document, RagPipeline

_WORDS = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"]


def _synthetic_docs(n: int) -> list[Document]:
    docs = []
    for i in range(n):
        text = " ".join(_WORDS[(i + j) % len(_WORDS)] for j in range(20))
        docs.append(Document(id=str(i), text=text))
    return docs


def main() -> None:
    parser = argparse.ArgumentParser(description="benchmark tessera retrieval")
    parser.add_argument("--docs", type=int, default=2000)
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    pipeline = RagPipeline()
    pipeline.add_documents(_synthetic_docs(args.docs))

    start = time.perf_counter()
    pipeline.index()
    index_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    for _ in range(args.queries):
        pipeline.retrieve("alpha beta gamma", top_k=args.top_k)
    query_ms = (time.perf_counter() - start) * 1000

    print(f"indexed {args.docs} docs in {index_ms:.1f} ms")
    print(
        f"{args.queries} queries in {query_ms:.1f} ms "
        f"({query_ms / max(args.queries, 1):.2f} ms/query)"
    )


if __name__ == "__main__":
    main()
