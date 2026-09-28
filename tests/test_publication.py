import json
from pathlib import Path
import shutil

import pytest

from decisionbench.schema import write_checksums
from decisionbench.verify import verify_publication


ROOT = Path(__file__).parents[1]


def test_published_evidence_is_recomputable():
    result = verify_publication(ROOT)
    assert result["verified_runs"] == 12
    assert result["verified_reports"] == 4
    assert result["inference_performed"] is False


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
