import json
from pathlib import Path
import shutil

import pytest

from decisionbench.schema import write_checksums
from decisionbench.verify import verify_publication


ROOT = Path(__file__).parents[1]


def test_published_evidence_is_recomputable():
    result = verify_publication(ROOT)
    assert result["verified_runs"] == 20
    assert result["verified_reports"] == 6
    assert result["inference_performed"] is False


def test_readme_core_table_matches_complete_active_report():
    report = json.loads(
        (ROOT / "results" / "core-v1-active-six-summary" / "summary.json").read_text()
    )
    models = report["models"]
    active_ids = {
        json.loads(path.read_text())["id"] for path in (ROOT / "models").glob("*.json")
    }
    assert len(models) == 6
    assert {model["model_id"] for model in models} == active_ids
    tasks = {
        "JevBench easy": "jevbench-easy",
        "JevBench original": "jevbench-original",
        "JevBench hard": "jevbench-hard",
        "BANKING77 subset": "banking77-12-intent",
        "BoolQ subset": "boolq-balanced",
        "WANLI subset": "wanli-balanced",
        "MASSIVE subset": "massive-5-language-12-intent",
        "MMLU-Pro subset": "mmlu-pro-domain-balanced",
        "Authored robustness (exploratory)": "owned-robustness",
    }
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    table = readme.split("## Full core v1 results", 1)[1].split("## Smoke status", 1)[0]
    for model in models:
        assert model["status"] == "complete"
        assert set(model["datasets"]) == set(tasks.values())
        assert sum(metrics["rows"] for metrics in model["datasets"].values()) == 1071
        assert all(metrics["coverage"] == 1 for metrics in model["datasets"].values())
    for label, dataset in tasks.items():
        counts = [
            f"{model['datasets'][dataset]['correct_predictions']}/{model['datasets'][dataset]['rows']}"
            for model in models
        ]
        assert "| " + " | ".join([label, *counts]) + " |" in table


@pytest.mark.parametrize("tamper", ["missing_run", "invented_csv"])
def test_publication_rejects_missing_or_misreported_evidence(tmp_path, tamper):
    for directory in ("models", "benchmarks"):
        shutil.copytree(ROOT / directory, tmp_path / directory)
    catalogue = json.loads((ROOT / "results" / "publication.json").read_text())
    (tmp_path / "results").mkdir()
    shutil.copy2(ROOT / "results" / "publication.json", tmp_path / "results" / "publication.json")
    paths = (
        [suite["runs"] for suite in catalogue["suites"]]
        + catalogue["historical_reports"] + catalogue["active_reports"]
    )
    for relative in paths:
        shutil.copytree(ROOT / relative, tmp_path / relative)
    if tamper == "missing_run":
        (tmp_path / "results" / "smoke-v1" / "unexpected").mkdir()
        match = "run list differs"
    else:
        directory = tmp_path / catalogue["active_reports"][0]
        (directory / "summary.csv").write_text("fabricated\n")
        (directory / "SHA256SUMS").unlink()
        write_checksums(directory, ("summary.json", "summary.csv", "REPORT.md"))
        match = "Rendered report differs"
    with pytest.raises(ValueError, match=match):
        verify_publication(tmp_path)
