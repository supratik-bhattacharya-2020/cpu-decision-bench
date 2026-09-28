from __future__ import annotations

import csv
import json
from pathlib import Path

from .runner import verify_run
from .schema import sha256_file, write_checksums, write_json_create


def _number(value) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _write_report(report: dict, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json_create(output_dir / "summary.json", report)
    fields = ["model_id", "dataset", "rows", "valid_predictions", "correct_predictions",
              "coverage", "accuracy", "macro_f1", "nll", "brier", "ece_10_bin"]
    lines = [
        f"# {report['title']}",
        "",
        report["warning"],
        "",
        "Accuracy = correct answers / all declared rows; coverage = valid predictions / all rows.",
        "Both are fractions from 0 to 1; higher is better. Failures remain in the denominator.",
        "F1 is shown only for tasks sharing one label set, not across unrelated questions.",
        "Timing and RAM remain diagnostic data in raw runs, not performance rankings.",
        "",
        "| Model | Dataset | Correct / rows | Coverage | Accuracy |",
        "|---|---|---:|---:|---:|",
    ]
    with (output_dir / "summary.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for model in report["models"]:
            if model["status"] == "not_run":
                lines.append(f"| {model['model_id']} | Not run | n/a | n/a | n/a |")
                continue
            for dataset, metrics in model["datasets"].items():
                writer.writerow({"model_id": model["model_id"], "dataset": dataset, **metrics})
                lines.append(
                    f"| {model['model_id']} | {dataset} | "
                    f"{metrics['correct_predictions']} / {metrics['rows']} | "
                    f"{_number(metrics['coverage'])} | {_number(metrics['accuracy'])} |"
                )
    with (output_dir / "REPORT.md").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write("\n".join(lines) + "\n")
    write_checksums(output_dir, ("summary.json", "summary.csv", "REPORT.md"))
    return report


def _model_result(run_dir: Path, gold_path: Path, expected_model: dict | None = None) -> dict:
    manifest, summary = verify_run(run_dir, gold_path, expected_model)
    return {
        "model_id": manifest["model"]["id"],
        "status": "complete",
        "gguf_sha256": manifest["model"]["gguf_sha256"],
        "scoring_settings": {
            key: manifest["model"].get(key)
            for key in ("threads", "context_tokens", "n_gpu_layers", "benchmark_date")
        },
        "prompt_contract": {
            key: manifest["model"].get(key)
            for key in ("prompt_version", "instruction_sha256", "implementation_sha256")
        },
        "evidence": {
            "run": str(run_dir),
            "manifest_sha256": sha256_file(run_dir / "manifest.json"),
            "predictions_sha256": sha256_file(run_dir / "predictions.jsonl"),
        },
        "performance_status": manifest.get("performance_status", "legacy, uncontrolled; diagnostic only"),
        "environment_recorded": manifest.get("environment") is not None,
        "datasets": summary["datasets"],
        "robustness": summary["robustness"],
    }


def create_report(gold_path: Path, predictions_path: Path, output_dir: Path) -> dict:
    if output_dir.exists():
        raise ValueError("Report output directory must be new")
    if predictions_path.name != "predictions.jsonl":
        raise ValueError("Report requires predictions.jsonl from a complete run bundle")
    model = _model_result(predictions_path.parent, gold_path)
    return _write_report({
        "schema_version": 2,
        "title": "DecisionBench report",
        "warning": "Task-specific results. No combined model ranking or controlled speed claim.",
        "benchmark": {"path": str(gold_path), "sha256": sha256_file(gold_path)},
        "models": [model],
    }, output_dir)


def create_comparison_report(
    runs_dir: Path,
    output_dir: Path,
    *,
    gold_path: Path | None = None,
    title: str,
    warning: str = "Preliminary results; do not treat a small suite as the full benchmark.",
    model_ids: list[str] | None = None,
    model_manifests: dict[str, dict] | None = None,
) -> dict:
    if output_dir.exists():
        raise ValueError("Report output directory must be new")
    directories = {path.name: path for path in runs_dir.iterdir() if path.is_dir()}
    selected = sorted(directories if model_ids is None else model_ids)
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("Comparison needs distinct model ids")
    models = []
    settings = None
    for model_id in selected:
        run_dir = directories.get(model_id)
        if run_dir is None:
            models.append({"model_id": model_id, "status": "not_run"})
            continue
        if not all((run_dir / name).is_file() for name in (
            "manifest.json", "summary.json", "predictions.jsonl", "SHA256SUMS"
        )):
            raise ValueError(f"Incomplete run bundle: {run_dir}")
        if gold_path is None:
            manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
            candidate = Path(manifest["benchmark"]["path"].replace("\\", "/"))
            if not candidate.is_file():
                raise ValueError("Recorded benchmark path is unavailable; supply --gold")
            gold_path = candidate
        result = _model_result(
            run_dir, gold_path, model_manifests[model_id] if model_manifests else None,
        )
        if result["model_id"] != model_id:
            raise ValueError(f"Run directory does not match its model id: {run_dir}")
        if settings is not None and result["scoring_settings"] != settings:
            raise ValueError("Comparison scoring settings differ")
        settings = result["scoring_settings"]
        models.append(result)
    if not any(model["status"] == "complete" for model in models):
        raise ValueError("No complete selected runs found")
    contracts = {
        json.dumps(model["prompt_contract"], sort_keys=True)
        for model in models if model["status"] == "complete"
    }
    if len(contracts) > 1:
        warning += " Mixed legacy/new implementation metadata: treat these runs as separate cohorts."
    return _write_report({
        "schema_version": 2,
        "title": title,
        "warning": warning,
        "benchmark": {"path": str(gold_path), "sha256": sha256_file(gold_path)},
        "models": models,
        "excluded_run_directories": sorted(set(directories) - set(selected)),
    }, output_dir)
