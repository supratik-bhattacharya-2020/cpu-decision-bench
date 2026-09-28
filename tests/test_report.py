import json

import pytest

from decisionbench.report import create_comparison_report
from decisionbench.runner import run_benchmark, verify_run
from decisionbench.schema import write_checksums, write_jsonl_create
from test_runner import FakeEngine, row


def bundle(tmp_path, model_id="fake", engine=None):
    gold = tmp_path / f"{model_id}.jsonl"
    write_jsonl_create(gold, [row("one")])
    engine = engine or FakeEngine()
    engine.metadata["id"] = model_id
    run = tmp_path / "runs" / model_id
    run_benchmark(gold, run, engine)
    return gold, run


def rehash(run):
    (run / "SHA256SUMS").unlink()
    write_checksums(run, ("manifest.json", "predictions.jsonl", "summary.json"))


def test_comparison_rejects_different_benchmarks(tmp_path):
    gold, run = bundle(tmp_path)
    manifest_path = run / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["benchmark"]["sha256"] = "b" * 64
    manifest_path.write_text(json.dumps(manifest))
    rehash(run)
    with pytest.raises(ValueError, match="hashes differ"):
        create_comparison_report(run.parent, tmp_path / "report", gold_path=gold, title="Smoke")


def test_comparison_rejects_missing_bundle_file(tmp_path):
    gold, run = bundle(tmp_path)
    (run / "summary.json").unlink()
    with pytest.raises(ValueError, match="Incomplete run bundle"):
        create_comparison_report(run.parent, tmp_path / "report", gold_path=gold, title="Smoke")


@pytest.mark.parametrize("rehash_after_edit", [False, True])
def test_report_rejects_altered_summary(tmp_path, rehash_after_edit):
    gold, run = bundle(tmp_path)
    path = run / "summary.json"
    summary = json.loads(path.read_text())
    summary["overall"]["accuracy"] = 0.0
    path.write_text(json.dumps(summary))
    if rehash_after_edit:
        rehash(run)
    with pytest.raises(ValueError, match="Checksum mismatch|does not match predictions"):
        verify_run(run, gold)


def test_report_preserves_all_invalid_rows_and_pending_model(tmp_path):
    gold, run = bundle(tmp_path, engine=FakeEngine(fail="one"))
    result = create_comparison_report(
        run.parent, tmp_path / "report", gold_path=gold, title="Smoke",
        model_ids=["fake", "pending"],
    )
    metrics = result["models"][0]["datasets"]["test"]
    assert metrics["rows"] == 1 and metrics["accuracy"] == 0 and metrics["coverage"] == 0
    assert result["models"][1] == {"model_id": "pending", "status": "not_run"}
    assert "0 / 1" in (tmp_path / "report" / "REPORT.md").read_text()


def test_legacy_cli_report_infers_recorded_gold(tmp_path):
    _, run = bundle(tmp_path)
    result = create_comparison_report(run.parent, tmp_path / "report", title="Smoke")
    assert result["models"][0]["status"] == "complete"


def test_report_rejects_mixed_settings(tmp_path):
    gold, run = bundle(tmp_path)
    engine = FakeEngine()
    engine.metadata["threads"] = 99
    bundle(tmp_path, "second", engine)
    with pytest.raises(ValueError, match="settings differ"):
        create_comparison_report(run.parent, tmp_path / "report", gold_path=gold, title="Smoke")
