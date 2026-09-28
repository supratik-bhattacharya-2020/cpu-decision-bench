import json

import pytest

from decisionbench.runner import derive_run, run_benchmark
from decisionbench.schema import write_jsonl_create


def row(identifier):
    return {
        "id": identifier,
        "dataset": "test",
        "state": "A fact.",
        "question": "Which option?",
        "options": [
            {"id": "left", "description": "Left"},
            {"id": "right", "description": "Right"},
        ],
        "label": "left",
    }


class FakeEngine:
    def __init__(self, fail=None):
        self.fail = fail
        self.metadata = {"id": "fake", "gguf_sha256": "a" * 64}

    def score(self, item):
        if item["id"] == self.fail:
            raise RuntimeError("declared failure")
        return {
            "id": item["id"],
            "dataset": item["dataset"],
            "option_ids": ["left", "right"],
            "probabilities": [0.8, 0.2],
            "prediction_id": "left",
            "total_seconds": 0.1,
        }


def test_runner_writes_evidence_files(tmp_path):
    benchmark = tmp_path / "benchmark.jsonl"
    write_jsonl_create(benchmark, [row("one"), row("two")])
    output = tmp_path / "run"
    run_benchmark(benchmark, output, FakeEngine())
    assert {path.name for path in output.iterdir()} == {
        "manifest.json",
        "predictions.jsonl",
        "summary.json",
        "SHA256SUMS",
    }
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    with pytest.raises(ValueError, match="complete"):
        run_benchmark(benchmark, output, FakeEngine())


def test_runner_records_explicit_errors(tmp_path):
    benchmark = tmp_path / "benchmark.jsonl"
    write_jsonl_create(benchmark, [row("one"), row("two")])
    output = tmp_path / "run"
    summary = run_benchmark(benchmark, output, FakeEngine(fail="two"))
    assert summary["overall"]["coverage"] == 0.5
    assert summary["overall"]["errors"][0]["error"] == "RuntimeError: declared failure"
    predictions = [
        json.loads(line) for line in (output / "predictions.jsonl").read_text().splitlines()
    ]
    assert predictions[1]["error"]["message"] == "declared failure"


def test_runner_resumes_missing_ids(tmp_path):
    benchmark = tmp_path / "benchmark.jsonl"
    write_jsonl_create(benchmark, [row("one"), row("two")])
    output = tmp_path / "run"
    output.mkdir()
    manifest = {
        "schema_version": 1,
        "status": "incomplete",
        "benchmark": {"sha256": __import__("decisionbench.schema", fromlist=["sha256_file"]).sha256_file(benchmark)},
        "model": {"id": "fake", "gguf_sha256": "a" * 64},
    }
    (output / "manifest.json").write_text(json.dumps(manifest))
    (output / "predictions.jsonl").write_text(
        json.dumps(FakeEngine().score(row("one"))) + "\n"
    )
    run_benchmark(benchmark, output, FakeEngine())
    assert len((output / "predictions.jsonl").read_text().splitlines()) == 2


def test_derive_run_reuses_matching_rows(tmp_path):
    source_benchmark = tmp_path / "source.jsonl"
    write_jsonl_create(source_benchmark, [row("one"), row("two")])
    source_run = tmp_path / "source-run"
    run_benchmark(source_benchmark, source_run, FakeEngine())
    subset = tmp_path / "subset.jsonl"
    write_jsonl_create(subset, [row("two")])
    output = tmp_path / "derived"
    summary = derive_run(source_run, subset, output)
    assert summary["overall"]["rows"] == 1
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["derived_from"]["run"] == str(source_run)
