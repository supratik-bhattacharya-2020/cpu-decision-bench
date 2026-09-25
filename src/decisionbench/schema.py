from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Iterable


def canonical_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_row(row: dict) -> None:
    required = {"id", "dataset", "state", "question", "options", "label"}
    if not required <= row.keys():
        raise ValueError(f"Missing row fields: {sorted(required - row.keys())}")
    if not all(isinstance(row[key], str) and row[key] for key in ("id", "dataset", "question", "label")):
        raise ValueError("id, dataset, question, and label must be nonempty strings")
    if not isinstance(row["state"], (str, dict, list)) or not row["state"]:
        raise ValueError("state must be a nonempty string, object, or array")
    json.dumps(row["state"], ensure_ascii=False, allow_nan=False)
    options = row["options"]
    if not isinstance(options, list) or not 2 <= len(options) <= 26:
        raise ValueError("options must contain 2-26 entries")
    ids = []
    for option in options:
        if not isinstance(option, dict):
            raise ValueError("Each option must be an object")
        if not isinstance(option.get("id"), str) or not option["id"]:
            raise ValueError("Each option needs a nonempty string id")
        if not isinstance(option.get("description"), str) or not option["description"]:
            raise ValueError("Each option needs a nonempty description")
        ids.append(option["id"])
    if len(ids) != len(set(ids)):
        raise ValueError("Option ids must be unique")
    if row["label"] not in ids:
        raise ValueError("label must match an option id")


def validate_prediction(prediction: dict, row: dict) -> None:
    expected = [option["id"] for option in row["options"]]
    if prediction.get("id") != row["id"] or prediction.get("option_ids") != expected:
        raise ValueError("Prediction does not align with the benchmark row")
    probabilities = prediction.get("probabilities")
    if not isinstance(probabilities, list) or len(probabilities) != len(expected):
        raise ValueError("Prediction probabilities have the wrong shape")
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(value) or value < 0 for value in probabilities):
        raise ValueError("Prediction probabilities must be finite and nonnegative")
    if abs(sum(probabilities) - 1.0) > 1e-5:
        raise ValueError("Prediction probabilities must sum to one")


def validate_model_manifest(manifest: dict) -> None:
    required = {
        "id",
        "family",
        "parameters_billion",
        "upstream",
        "gguf",
        "license",
        "benchmark_context_tokens",
        "alias_compatibility",
    }
    if not required <= manifest.keys():
        raise ValueError(f"Missing model fields: {sorted(required - manifest.keys())}")
    if not isinstance(manifest["parameters_billion"], (int, float)) or not 0 < manifest["parameters_billion"] <= 4.5:
        raise ValueError("Model must be between 0 and 4.5 billion parameters")
    for key in ("id", "family", "upstream", "license"):
        if not isinstance(manifest[key], str) or not manifest[key]:
            raise ValueError(f"{key} must be a nonempty string")
    if not isinstance(manifest["benchmark_context_tokens"], int) or manifest["benchmark_context_tokens"] < 1:
        raise ValueError("benchmark_context_tokens must be positive")
    compatibility = manifest["alias_compatibility"]
    if (
        not isinstance(compatibility, dict)
        or compatibility.get("status") not in {"verified", "incompatible", "unverified"}
        or not isinstance(compatibility.get("detail"), str)
        or not compatibility["detail"]
    ):
        raise ValueError("alias_compatibility must contain a supported status and detail")
    gguf = manifest["gguf"]
    gguf_required = {"repository", "revision", "filename", "bytes", "sha256", "quantization"}
    if not isinstance(gguf, dict) or not gguf_required <= gguf.keys():
        raise ValueError("gguf metadata is incomplete")
    if not isinstance(gguf["bytes"], int) or gguf["bytes"] < 1:
        raise ValueError("gguf bytes must be positive")
    if not isinstance(gguf["sha256"], str) or len(gguf["sha256"]) != 64:
        raise ValueError("gguf sha256 must contain 64 hexadecimal characters")


def read_jsonl(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = set()
    for row in rows:
        validate_row(row)
        if row["id"] in ids:
            raise ValueError(f"Duplicate row id: {row['id']}")
        ids.add(row["id"])
    return rows


def write_jsonl_create(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
