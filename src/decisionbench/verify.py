"""Check published evidence without loading a model or running inference."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from .datasets import banking77_correction
from .models import read_model_manifest
from .runner import verify_run
from .report import _write_report
from .schema import canonical_json, sha256_file, verify_checksums


def verify_publication(root: Path, banking77_source: Path | None = None) -> dict:
    catalogue = json.loads((root / "results" / "publication.json").read_text(encoding="utf-8"))
    models = {}
    for path in (root / "models").rglob("*.json"):
        model = read_model_manifest(path)
        if model["id"] in models:
            raise ValueError(f"Duplicate model id: {model['id']}")
        models[model["id"]] = model
    for relative, expected in catalogue["frozen_files"].items():
        if sha256_file(root / relative) != expected:
            raise ValueError(f"Frozen file hash differs: {relative}")
    verified = {}
    for suite in catalogue["suites"]:
        directory = root / suite["runs"]
        actual = {path.name for path in directory.iterdir() if path.is_dir()}
        if actual != set(suite["models"]):
            raise ValueError(f"Published run list differs: {directory}")
        for model_id in suite["models"]:
            run = directory / model_id
            verified[run.relative_to(root).as_posix()] = verify_run(
                run, root / suite["gold"], models[model_id],
            )
    for relative in catalogue["historical_reports"]:
        verify_checksums(root / relative, {"summary.json", "summary.csv", "REPORT.md"})
    active_ids = {
        read_model_manifest(path)["id"] for path in (root / "models").glob("*.json")
    }
    for relative in catalogue["active_reports"]:
        directory = root / relative
        verify_checksums(directory, {"summary.json", "summary.csv", "REPORT.md"})
        report = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        ids = [entry["model_id"] for entry in report["models"]]
        if len(ids) != len(set(ids)) or set(ids) != active_ids:
            raise ValueError(f"Report does not contain the active model lineup: {relative}")
        for entry in report["models"]:
            if entry["status"] == "not_run":
                matching = [
                    key for key, (manifest, _) in verified.items()
                    if manifest["model"]["id"] == entry["model_id"]
                    and manifest["benchmark"]["sha256"] == report["benchmark"]["sha256"]
                ]
                if matching:
                    raise ValueError(f"Report hides a published run: {entry['model_id']}")
                continue
            key = entry["evidence"]["run"].replace("\\", "/")
            if key not in verified:
                raise ValueError(f"Report refers to unpublished evidence: {key}")
            manifest, computed = verified[key]
            if (
                manifest["benchmark"]["sha256"] != report["benchmark"]["sha256"]
                or manifest["model"]["id"] != entry["model_id"]
                or manifest["model"]["gguf_sha256"] != entry["gguf_sha256"]
            ):
                raise ValueError(f"Report identity differs from evidence: {key}")
            for name in ("manifest", "predictions"):
                suffix = ".jsonl" if name == "predictions" else ".json"
                if sha256_file(root / key / (name + suffix)) != entry["evidence"][name + "_sha256"]:
                    raise ValueError(f"Report evidence hash differs: {key}")
            for name in ("datasets", "robustness"):
                if canonical_json(entry[name]) != canonical_json(computed[name]):
                    raise ValueError(f"Report metrics differ from predictions: {key}")
            if entry["scoring_settings"] != {
                key: manifest["model"].get(key)
                for key in ("threads", "context_tokens", "n_gpu_layers", "benchmark_date")
            }:
                raise ValueError(f"Report settings differ from evidence: {key}")
        with TemporaryDirectory() as temporary:
            regenerated = Path(temporary) / "report"
            _write_report(report, regenerated)
            for name in ("summary.csv", "REPORT.md"):
                if (directory / name).read_bytes() != (regenerated / name).read_bytes():
                    raise ValueError(f"Rendered report differs from verified metrics: {relative}/{name}")
    if banking77_source is not None:
        correction = json.loads(
            (root / "benchmarks" / "corrections" / "core-v1-banking77.json").read_text(encoding="utf-8")
        )
        if correction != banking77_correction(root / "benchmarks" / "core-v1.jsonl", banking77_source):
            raise ValueError("BANKING77 correction does not match the pinned source")
    return {
        "verified_runs": len(verified),
        "verified_reports": len(catalogue["active_reports"]) + len(catalogue["historical_reports"]),
        "inference_performed": False,
        "banking77_source_verified": banking77_source is not None,
    }
