# Benchmarks

Quick latency checks for the offline path. These measure tessera's own indexing and
retrieval overhead with the hashing encoder — not model inference.

```bash
python benchmarks/bench_retrieval.py --docs 2000 --queries 100 --top-k 5
```

The in-memory store is brute-force (a single matmul per query), so retrieval time
scales linearly with corpus size. Switch to the `faiss` extra when that becomes the
bottleneck.
