import json

import pytest

from decisionbench.report import create_comparison_report


def test_comparison_rejects_different_benchmarks(tmp_path):
    runs = tmp_path / "runs"
    for index, digest in enumerate(("a" * 64, "b" * 64)):
        run = runs / str(index)
        run.mkdir(parents=True)
        (run / "manifest.json").write_text(json.dumps({
            "status": "complete",
            "benchmark": {"sha256": digest},
            "model": {
                "id": str(index),
                "gguf_sha256": "c" * 64,
                "load_seconds": 1.0,
            },
        }))
        (run / "summary.json").write_text(json.dumps({
            "overall": {
                "rows": 1,
                "coverage": 1.0,
                "accuracy": 1.0,
                "macro_f1": 1.0,
                "allowed_token_mass": {"mean": 1.0},
                "latency_seconds": {"p50": 1.0},
                "peak_process_rss_bytes": {"max": 1},
            }
        }))
        (run / "predictions.jsonl").write_text("{}\n")
        (run / "SHA256SUMS").write_text("")
    with pytest.raises(ValueError, match="hashes differ"):
        create_comparison_report(runs, tmp_path / "report", title="Smoke")
