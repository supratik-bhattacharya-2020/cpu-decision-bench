from __future__ import annotations

import csv
import json
from pathlib import Path

from .metrics import by_dataset, robustness, summarize
from .schema import read_jsonl, sha256_file


def _read_predictions(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def create_report(gold_path: Path, predictions_path: Path, output_dir: Path) -> dict:
    if output_dir.exists():
        raise ValueError("Report output directory must be new")
    rows = read_jsonl(gold_path)
    predictions = _read_predictions(predictions_path)
    report = {
        "gold": {"path": str(gold_path), "sha256": sha256_file(gold_path)},
        "predictions": {"path": str(predictions_path), "sha256": sha256_file(predictions_path)},
        "overall": summarize(rows, predictions),
        "datasets": by_dataset(rows, predictions),
        "robustness": robustness(rows, predictions),
    }
    output_dir.mkdir(parents=True)
    summary = output_dir / "summary.json"
    summary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with (output_dir / "summary.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "dataset",
            "rows",
            "coverage",
            "accuracy",
            "macro_f1",
            "brier",
            "ece_10_bin",
            "p50_seconds",
            "peak_process_rss_bytes",
        ])
        for dataset, metrics in report["datasets"].items():
            writer.writerow([
                dataset,
                metrics["rows"],
                metrics["coverage"],
                metrics["accuracy"],
                metrics["macro_f1"],
                metrics["brier"],
                metrics["ece_10_bin"],
                metrics["latency_seconds"]["p50"],
                metrics["peak_process_rss_bytes"]["max"],
            ])
    lines = [
        "# DecisionBench report",
        "",
        "| Dataset | Rows | Coverage | Accuracy | Macro-F1 | Brier | ECE | p50 seconds | Peak RAM GiB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset, metrics in report["datasets"].items():
        values = [
            dataset,
            str(metrics["rows"]),
            f"{metrics['coverage']:.3f}",
            f"{metrics['accuracy']:.3f}",
            f"{metrics['macro_f1']:.3f}",
            "n/a" if metrics["brier"] is None else f"{metrics['brier']:.3f}",
            f"{metrics['ece_10_bin']:.3f}",
            "n/a" if metrics["latency_seconds"]["p50"] is None else f"{metrics['latency_seconds']['p50']:.3f}",
            (
                "n/a"
                if metrics["peak_process_rss_bytes"]["max"] is None
                else f"{metrics['peak_process_rss_bytes']['max'] / 1024 ** 3:.2f}"
            ),
        ]
        lines.append("| " + " | ".join(values) + " |")
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    hashes = [
        f"{sha256_file(output_dir / name)}  {name}"
        for name in ("summary.json", "summary.csv", "REPORT.md")
    ]
    (output_dir / "SHA256SUMS").write_text("\n".join(hashes) + "\n", encoding="ascii")
    return report


def create_comparison_report(runs_dir: Path, output_dir: Path, *, title: str) -> dict:
    if output_dir.exists():
        raise ValueError("Report output directory must be new")
    models = []
    benchmark_sha256 = None
    for run_dir in sorted(path for path in runs_dir.iterdir() if path.is_dir()):
        manifest_path = run_dir / "manifest.json"
        summary_path = run_dir / "summary.json"
        predictions_path = run_dir / "predictions.jsonl"
        checksums_path = run_dir / "SHA256SUMS"
        if not all(path.is_file() for path in (manifest_path, summary_path, predictions_path, checksums_path)):
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "complete":
            raise ValueError(f"Run is incomplete: {run_dir}")
        current_hash = manifest["benchmark"]["sha256"]
        benchmark_sha256 = benchmark_sha256 or current_hash
        if current_hash != benchmark_sha256:
            raise ValueError("Run benchmark hashes differ")
        metrics = summary["overall"]
        models.append({
            "model_id": manifest["model"]["id"],
            "gguf_sha256": manifest["model"]["gguf_sha256"],
            "rows": metrics["rows"],
            "coverage": metrics["coverage"],
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "allowed_token_mass": metrics["allowed_token_mass"]["mean"],
            "p50_seconds": metrics["latency_seconds"]["p50"],
            "peak_process_rss_bytes": metrics["peak_process_rss_bytes"]["max"],
            "load_seconds": manifest["model"]["load_seconds"],
        })
    if not models:
        raise ValueError("No complete run directories found")
    report = {
        "title": title,
        "warning": "Smoke results only; do not use as headline benchmark claims.",
        "benchmark_sha256": benchmark_sha256,
        "models": models,
    }
    output_dir.mkdir(parents=True)
    (output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    columns = [
        "model_id",
        "rows",
        "coverage",
        "accuracy",
        "macro_f1",
        "allowed_token_mass",
        "p50_seconds",
        "peak_process_rss_bytes",
        "load_seconds",
    ]
    with (output_dir / "summary.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(models)
    lines = [
        f"# {title}",
        "",
        "**Smoke results only. Do not use these seven rows as headline benchmark claims.**",
        "",
        "| Model | Coverage | Accuracy | Allowed mass | p50 seconds | Peak RAM GiB |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model in models:
        lines.append(
            "| "
            + " | ".join([
                model["model_id"],
                f"{model['coverage']:.3f}",
                f"{model['accuracy']:.3f}",
                f"{model['allowed_token_mass']:.3f}",
                f"{model['p50_seconds']:.3f}",
                f"{model['peak_process_rss_bytes'] / 1024 ** 3:.2f}",
            ])
            + " |"
        )
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    hashes = [
        f"{sha256_file(output_dir / name)}  {name}"
        for name in ("summary.json", "summary.csv", "REPORT.md")
    ]
    (output_dir / "SHA256SUMS").write_text("\n".join(hashes) + "\n", encoding="ascii")
    return report
