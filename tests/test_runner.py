import json
import math

import pytest

from decisionbench.runner import derive_run, run_benchmark, verify_run
from decisionbench.schema import sha256_file, sha256_text, write_json_create, write_jsonl_create


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
        self.environment = {"python": "test", "packages": {"test": "1"}}

    def score(self, item):
        if item["id"] == self.fail:
            raise RuntimeError("declared failure")
        return {
            "id": item["id"],
            "dataset": item["dataset"],
            "option_ids": ["left", "right"],
            "probabilities": [0.8, 0.2],
            "alias_token_ids": [1, 2],
            "option_logits": [math.log(0.8), math.log(0.2)],
            "prediction_id": "left",
            "total_seconds": 0.1,
            "forward_seconds": 0.05,
            "input_tokens": 3,
            "peak_process_rss_bytes": 1000,
            "allowed_token_mass": 0.9,
            "prompt": {"text": "test", "sha256": sha256_text("test")},
            "model": self.metadata,
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
    verify_run(output, benchmark)
    for path in output.iterdir():
        assert b"\r\n" not in path.read_bytes()
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
        "schema_version": 2,
        "summary_version": 2,
        "status": "incomplete",
        "benchmark": {"sha256": sha256_file(benchmark), "rows": 2},
        "model": FakeEngine().metadata,
        "environment": FakeEngine().environment,
    }
    (output / "manifest.json").write_text(json.dumps(manifest))
    (output / "predictions.jsonl").write_text(
        json.dumps(FakeEngine().score(row("one"))) + "\n"
    )
    run_benchmark(benchmark, output, FakeEngine())
    assert len((output / "predictions.jsonl").read_text().splitlines()) == 2
    verify_run(output, benchmark)


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
    verify_run(output, subset)


@pytest.mark.parametrize("field", [
    "threads", "context_tokens", "benchmark_date", "prompt_version",
    "instruction_sha256", "implementation_sha256", "n_gpu_layers",
])
def test_resume_rejects_changed_configuration(tmp_path, field):
    benchmark = tmp_path / "benchmark.jsonl"
    write_jsonl_create(benchmark, [row("one")])
    output = tmp_path / "run"
    output.mkdir()
    engine = FakeEngine()
    write_json_create(output / "manifest.json", {
        "schema_version": 2,
        "status": "incomplete",
        "benchmark": {"sha256": sha256_file(benchmark)},
        "model": {**engine.metadata, field: "different"},
        "environment": engine.environment,
    })
    with pytest.raises(ValueError, match="configuration differs"):
        run_benchmark(benchmark, output, engine)
    assert not (output / "predictions.jsonl").exists()


@pytest.mark.parametrize("phase", ["summary", "manifest"])
def test_resume_finishes_interrupted_finalization(tmp_path, phase):
    benchmark = tmp_path / "benchmark.jsonl"
    write_jsonl_create(benchmark, [row("one")])
    output = tmp_path / "run"
    run_benchmark(benchmark, output, FakeEngine())
    before = (output / "predictions.jsonl").read_bytes()
    (output / "SHA256SUMS").unlink()
    if phase == "summary":
        manifest = json.loads((output / "manifest.json").read_text())
        manifest["status"] = "incomplete"
        (output / "manifest.json").write_text(json.dumps(manifest))
    run_benchmark(benchmark, output, FakeEngine(fail="one"))
    assert (output / "predictions.jsonl").read_bytes() == before
    verify_run(output, benchmark)


def test_derive_rejects_changed_gold(tmp_path):
    source = tmp_path / "source.jsonl"
    write_jsonl_create(source, [row("one")])
    run = tmp_path / "run"
    run_benchmark(source, run, FakeEngine())
    changed = tmp_path / "changed.jsonl"
    write_jsonl_create(changed, [{**row("one"), "label": "right"}])
    with pytest.raises(ValueError, match="Subset rows differ"):
        derive_run(run, changed, tmp_path / "derived")


def test_resume_rejects_runtime_change(tmp_path):
    benchmark = tmp_path / "benchmark.jsonl"
    write_jsonl_create(benchmark, [row("one")])
    output = tmp_path / "run"
    run_benchmark(benchmark, output, FakeEngine())
    (output / "SHA256SUMS").unlink()
    engine = FakeEngine()
    engine.environment["python"] = "changed"
    with pytest.raises(ValueError, match="runtime environment differs"):
        run_benchmark(benchmark, output, engine)


def test_resume_rejects_legacy_and_truncated_evidence(tmp_path):
    benchmark = tmp_path / "benchmark.jsonl"
    write_jsonl_create(benchmark, [row("one")])
    output = tmp_path / "run"
    run_benchmark(benchmark, output, FakeEngine())
    (output / "SHA256SUMS").unlink()
    with (output / "predictions.jsonl").open("ab") as stream:
        stream.write(b'{"id":')
    before = (output / "predictions.jsonl").read_bytes()
    with pytest.raises(ValueError, match="incomplete final line"):
        run_benchmark(benchmark, output, FakeEngine())
    assert (output / "predictions.jsonl").read_bytes() == before
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["schema_version"] = 1
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Legacy runs"):
        run_benchmark(benchmark, output, FakeEngine())
