from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from .metrics import by_dataset, robustness, summarize
from .schema import read_jsonl, sha256_file


def _write_json_atomic(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read_predictions(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _validate_resume(manifest: dict, benchmark_path: Path, engine) -> None:
    if manifest.get("benchmark", {}).get("sha256") != sha256_file(benchmark_path):
        raise ValueError("Cannot resume: benchmark hash differs")
    if manifest.get("model", {}).get("gguf_sha256") != engine.metadata["gguf_sha256"]:
        raise ValueError("Cannot resume: GGUF hash differs")
    if manifest.get("model", {}).get("id") != engine.metadata["id"]:
        raise ValueError("Cannot resume: model id differs")


def run_benchmark(benchmark_path: Path, output_dir: Path, engine) -> dict:
    rows = read_jsonl(benchmark_path)
    row_ids = {row["id"] for row in rows}
    manifest_path = output_dir / "manifest.json"
    predictions_path = output_dir / "predictions.jsonl"
    if output_dir.exists():
        if not manifest_path.is_file():
            raise ValueError("Existing output directory has no run manifest")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        _validate_resume(manifest, benchmark_path, engine)
        if manifest.get("status") == "complete":
            raise ValueError("Run is already complete")
    else:
        output_dir.mkdir(parents=True)
        manifest = {
            "schema_version": 1,
            "status": "incomplete",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "benchmark": {
                "path": str(benchmark_path),
                "sha256": sha256_file(benchmark_path),
                "rows": len(rows),
            },
            "model": engine.metadata,
        }
        _write_json_atomic(manifest_path, manifest)
    predictions = _read_predictions(predictions_path)
    completed = {prediction.get("id") for prediction in predictions}
    if None in completed or len(completed) != len(predictions):
        raise ValueError("Existing predictions contain missing or duplicate ids")
    unknown = completed - row_ids
    if unknown:
        raise ValueError(f"Existing predictions contain unknown ids: {sorted(unknown)[:5]}")
    with predictions_path.open("a", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            if row["id"] in completed:
                continue
            try:
                prediction = engine.score(row)
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
        raise ValueError("Run summary already exists for an incomplete run")
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    manifest["status"] = "complete"
    manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
    manifest["prediction_rows"] = len(predictions)
    _write_json_atomic(manifest_path, manifest)
    hashes = [
        f"{sha256_file(output_dir / name)}  {name}"
        for name in ("manifest.json", "predictions.jsonl", "summary.json")
    ]
    (output_dir / "SHA256SUMS").write_text("\n".join(hashes) + "\n", encoding="ascii")
    return summary
