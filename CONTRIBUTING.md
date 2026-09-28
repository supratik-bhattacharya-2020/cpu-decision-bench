# Contributing

Keep changes small and readable.

## Add a model

1. Copy one JSON file in `models/`.
2. Pin the upstream model, GGUF repository, revision, filename, byte size, and
   SHA-256.
3. Record the license and access requirements.
4. Run the smoke suite locally and report whether every alias is one token.
5. Do not commit model weights or caches.

Distinguish total from effective parameters and quantization formats. If a
quantizer did not publish its source checkpoint revision, disclose that gap
rather than pinning today's upstream revision as if it were the original.

## Add a dataset

1. Add a converter in `src/decisionbench/datasets.py`.
2. Preserve stable source IDs and the exact source revision.
3. Use deterministic selection when only a subset is included.
4. Add attribution and redistribution limits to `THIRD_PARTY.md`.
5. Add tests for row count, labels, and deterministic output.

Run:

```powershell
.\.venv\Scripts\python -m pytest -q
.\.venv\Scripts\decisionbench verify
```

Keep frozen benchmark and result bytes unchanged. New results/reports need new
paths and explicit LF writers; extend `results/publication.json` when publishing
evidence. Verify a Git checkout, not just local working-tree hashes. Do not
commit in-progress runs, caches, or weights.
