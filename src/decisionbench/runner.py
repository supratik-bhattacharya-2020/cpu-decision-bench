from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from .metrics import by_dataset, robustness, summarize
from .schema import (
    canonical_json,
    read_jsonl,
    read_predictions,
    sha256_file,
    validate_prediction,
    verify_checksums,
    write_checksums,
    write_json_create,
    write_jsonl_create,
)

RUN_FILES = {"manifest.json", "predictions.jsonl", "summary.json"}
CONFIG_FIELDS = (
    "id", "gguf_sha256", "threads", "context_tokens", "n_gpu_layers",
    "chat_format", "prompt_rendering", "benchmark_date", "prompt_version", "instruction_sha256",
    "implementation_sha256",
)


def scoring_configuration(model: dict) -> dict:
    return {key: model.get(key) for key in CONFIG_FIELDS}


def _write_json_atomic(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    write_json_create(temporary, value)
    temporary.replace(path)


def _read_predictions(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return read_predictions(path)


def _validate_resume(manifest: dict, benchmark_path: Path, engine) -> None:
    if manifest.get("schema_version") != 2:
        raise ValueError("Legacy runs cannot be safely resumed; start a new run directory")
    if manifest.get("benchmark", {}).get("sha256") != sha256_file(benchmark_path):
        raise ValueError("Cannot resume: benchmark hash differs")
    if scoring_configuration(manifest["model"]) != scoring_configuration(engine.metadata):
        raise ValueError("Cannot resume: model or scoring configuration differs")
    if manifest.get("environment") != engine.environment:
        raise ValueError("Cannot resume: runtime environment differs")


def _check_rows(rows: list[dict], predictions: list[dict], model: dict) -> None:
    row_map = {row["id"]: row for row in rows}
    for prediction in predictions:
        row = row_map.get(prediction["id"])
        if row is None:
            raise ValueError(f"Unknown prediction id: {prediction['id']}")
        if scoring_configuration(prediction.get("model", {})) != scoring_configuration(model):
            raise ValueError(f"Prediction model/configuration differs: {row['id']}")
        if prediction.get("error"):
            error = prediction["error"]
            if (
                not isinstance(error, dict)
                or not isinstance(error.get("type"), str)
                or not isinstance(error.get("message"), str)
                or prediction.get("option_ids") != [option["id"] for option in row["options"]]
                or prediction.get("dataset") != row["dataset"]
                or "probabilities" in prediction
                or "prediction_id" in prediction
            ):
                raise ValueError(f"Malformed error prediction: {row['id']}")
        else:
            required = {
                "dataset", "alias_token_ids", "option_logits", "prompt",
                "input_tokens", "total_seconds", "forward_seconds",
                "peak_process_rss_bytes", "allowed_token_mass",
            }
            if not required <= prediction.keys():
                raise ValueError(f"Prediction evidence fields missing: {sorted(required - prediction.keys())}")
            validate_prediction(prediction, row)


def verify_run(
    run_dir: Path, benchmark_path: Path, expected_model: dict | None = None,
) -> tuple[dict, dict]:
    verify_checksums(run_dir, RUN_FILES)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError(f"Run is incomplete: {run_dir}")
    if expected_model is not None:
        model = manifest["model"]
        if (
            model["id"] != expected_model["id"]
            or model["gguf_sha256"] != expected_model["gguf"]["sha256"]
            or model["gguf_bytes"] != expected_model["gguf"]["bytes"]
        ):
            raise ValueError(f"Run model differs from its pinned manifest: {run_dir}")
    if manifest["benchmark"]["sha256"] != sha256_file(benchmark_path):
        raise ValueError(f"Run benchmark hashes differ: {run_dir}")
    rows = read_jsonl(benchmark_path)
    predictions = read_predictions(run_dir / "predictions.jsonl")
    if (
        manifest["benchmark"]["rows"] != len(rows)
        or manifest.get("prediction_rows") != len(rows)
        or len(predictions) != len(rows)
        or {p["id"] for p in predictions} != {r["id"] for r in rows}
    ):
        raise ValueError(f"Completed run has missing or unexpected rows: {run_dir}")
    _check_rows(rows, predictions, manifest["model"])
    computed = {
        "overall": summarize(rows, predictions),
        "datasets": by_dataset(rows, predictions),
        "robustness": robustness(rows, predictions),
    }
    saved = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    if manifest.get("summary_version") == 2:
        if canonical_json(saved) != canonical_json(computed):
            raise ValueError(f"Saved summary does not match predictions: {run_dir}")
    else:
        # Old F1 combined unrelated labels. Recompute it; verify the original counts/losses.
        keys = ("rows", "valid_predictions", "coverage", "accuracy", "nll", "brier", "ece_10_bin")
        for section in ("overall", *computed["datasets"]):
            old = saved["overall"] if section == "overall" else saved["datasets"][section]
            new = computed["overall"] if section == "overall" else computed["datasets"][section]
            if any(old[key] != new[key] for key in keys):
                raise ValueError(f"Legacy summary does not match predictions: {run_dir}")
    return manifest, computed


def run_benchmark(benchmark_path: Path, output_dir: Path, engine) -> dict:
    rows = read_jsonl(benchmark_path)
    if not rows:
        raise ValueError("Benchmark must contain at least one row")
    row_ids = {row["id"] for row in rows}
    manifest_path = output_dir / "manifest.json"
    predictions_path = output_dir / "predictions.jsonl"
    existing = output_dir.exists()
    if not existing:
        output_dir.mkdir(parents=True)
    lock_path = output_dir / ".writer.lock"
    with lock_path.open("x"):
        pass
    try:
        if existing:
            if not manifest_path.is_file():
                raise ValueError("Existing output directory has no run manifest")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("status") == "complete" and (output_dir / "SHA256SUMS").exists():
                verify_run(output_dir, benchmark_path)
                raise ValueError("Run is already complete")
            _validate_resume(manifest, benchmark_path, engine)
        else:
            manifest = {
                "schema_version": 2,
                "summary_version": 2,
                "status": "incomplete",
                "started_at": datetime.now(timezone.utc).isoformat(),
                "benchmark": {
                    "path": str(benchmark_path),
                    "sha256": sha256_file(benchmark_path),
                    "rows": len(rows),
                },
                "model": engine.metadata,
                "environment": engine.environment,
                "performance_status": "uncontrolled; diagnostic timing and memory only",
            }
            _write_json_atomic(manifest_path, manifest)
        predictions = _read_predictions(predictions_path)
        completed = {prediction["id"] for prediction in predictions}
        unknown = completed - row_ids
        if unknown:
            raise ValueError(f"Existing predictions contain unknown ids: {sorted(unknown)[:5]}")
        _check_rows(rows, predictions, manifest["model"])
        return _append_and_finish(rows, predictions, completed, output_dir, engine, manifest)
    finally:
        lock_path.unlink()


def _append_and_finish(rows, predictions, completed, output_dir, engine, manifest):
    predictions_path = output_dir / "predictions.jsonl"
    manifest_path = output_dir / "manifest.json"
    with predictions_path.open("a", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            if row["id"] in completed:
                continue
            try:
                prediction = engine.score(row)
                _check_rows([row], [prediction], manifest["model"])
            except (ValueError, RuntimeError, MemoryError) as error:
                prediction = {
                    "id": row["id"],
                    "dataset": row["dataset"],
                    "option_ids": [option["id"] for option in row["options"]],
                    "error": {"type": type(error).__name__, "message": str(error)},
                    "model": engine.metadata,
                }
            stream.write(json.dumps(prediction, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            predictions.append(prediction)
    summary = {
        "overall": summarize(rows, predictions),
        "datasets": by_dataset(rows, predictions),
        "robustness": robustness(rows, predictions),
    }
    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        if json.loads(summary_path.read_text(encoding="utf-8")) != summary:
            raise ValueError("Existing partial summary differs from the predictions")
    else:
        write_json_create(summary_path, summary)
    if manifest["status"] != "complete":
        manifest["status"] = "complete"
        manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
        manifest["prediction_rows"] = len(predictions)
        _write_json_atomic(manifest_path, manifest)
    write_checksums(output_dir, sorted(RUN_FILES))
    return summary


def derive_run(source_run: Path, benchmark_path: Path, output_dir: Path) -> dict:
    if output_dir.exists():
        raise ValueError("Derived run output directory must be new")
    rows = read_jsonl(benchmark_path)
    source_manifest = json.loads((source_run / "manifest.json").read_text(encoding="utf-8"))
    source_predictions_path = source_run / "predictions.jsonl"
    source_benchmark = Path(source_manifest["benchmark"]["path"])
    verify_run(source_run, source_benchmark)
    source_rows = {row["id"]: row for row in read_jsonl(source_benchmark)}
    if any(row != source_rows.get(row["id"]) for row in rows):
        raise ValueError("Subset rows differ from the source benchmark")
    source_predictions = {
        prediction["id"]: prediction for prediction in _read_predictions(source_predictions_path)
    }
    missing = [row["id"] for row in rows if row["id"] not in source_predictions]
    if missing:
        raise ValueError(f"Source run is missing {len(missing)} required rows")
    predictions = [source_predictions[row["id"]] for row in rows]
    output_dir.mkdir(parents=True)
    predictions_path = output_dir / "predictions.jsonl"
    write_jsonl_create(predictions_path, predictions)
    manifest = {
        "schema_version": 2,
        "summary_version": 2,
        "status": "complete",
        "started_at": source_manifest["started_at"],
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "benchmark": {
            "path": str(benchmark_path),
            "sha256": sha256_file(benchmark_path),
            "rows": len(rows),
        },
        "model": source_manifest["model"],
        "environment": source_manifest.get("environment"),
        "performance_status": "derived; diagnostic timing and memory only",
        "prediction_rows": len(predictions),
        "derived_from": {
            "run": str(source_run),
            "benchmark_sha256": source_manifest["benchmark"]["sha256"],
            "predictions_sha256": sha256_file(source_predictions_path),
        },
    }
    _write_json_atomic(output_dir / "manifest.json", manifest)
    summary = {
        "overall": summarize(rows, predictions),
        "datasets": by_dataset(rows, predictions),
        "robustness": robustness(rows, predictions),
    }
    write_json_create(output_dir / "summary.json", summary)
    write_checksums(output_dir, sorted(RUN_FILES))
    return summary
