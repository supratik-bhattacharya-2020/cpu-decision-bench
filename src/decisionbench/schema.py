from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
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
    if not isinstance(prediction, dict):
        raise ValueError("Prediction must be an object")
    expected = [option["id"] for option in row["options"]]
    if prediction.get("id") != row["id"] or prediction.get("option_ids") != expected:
        raise ValueError("Prediction does not align with the benchmark row")
    if "dataset" in prediction and prediction["dataset"] != row["dataset"]:
        raise ValueError("Prediction dataset differs from the benchmark row")
    if prediction.get("error"):
        raise ValueError("Prediction records an inference error")
    probabilities = prediction.get("probabilities")
    if not isinstance(probabilities, list) or len(probabilities) != len(expected):
        raise ValueError("Prediction probabilities have the wrong shape")
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(value) or value < 0 for value in probabilities):
        raise ValueError("Prediction probabilities must be finite and nonnegative")
    if abs(sum(probabilities) - 1.0) > 1e-5:
        raise ValueError("Prediction probabilities must sum to one")
    best = expected[max(range(len(probabilities)), key=probabilities.__getitem__)]
    if prediction.get("prediction_id") != best:
        raise ValueError("prediction_id must match the first highest-probability option")
    if "alias_token_ids" in prediction:
        aliases = prediction["alias_token_ids"]
        if (
            not isinstance(aliases, list) or len(aliases) != len(expected)
            or any(type(value) is not int or value < 0 for value in aliases)
            or len(aliases) != len(set(aliases))
        ):
            raise ValueError("Alias token ids must be distinct nonnegative integers")
    if "option_logits" in prediction:
        logits = prediction["option_logits"]
        if not isinstance(logits, list) or len(logits) != len(expected) or any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) for value in logits
        ):
            raise ValueError("Option logits must be finite and align with the options")
        weights = [math.exp(value - max(logits)) for value in logits]
        total = sum(weights)
        if any(abs(p - w / total) > 1e-7 for p, w in zip(probabilities, weights)):
            raise ValueError("Probabilities do not match the option logits")
    if "prompt" in prediction:
        prompt = prediction["prompt"]
        if not isinstance(prompt, dict) or not isinstance(prompt.get("text"), str):
            raise ValueError("Prediction prompt must include text")
        if prompt.get("sha256") != sha256_text(prompt["text"]):
            raise ValueError("Prediction prompt hash differs from its text")
    for key in ("total_seconds", "forward_seconds", "peak_process_rss_bytes", "input_tokens"):
        if key in prediction:
            value = prediction[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{key} must be finite and nonnegative")
    for key in ("input_tokens", "peak_process_rss_bytes"):
        if key in prediction and type(prediction[key]) is not int:
            raise ValueError(f"{key} must be an integer")
    if (
        "total_seconds" in prediction and "forward_seconds" in prediction
        and prediction["forward_seconds"] > prediction["total_seconds"]
    ):
        raise ValueError("Forward time cannot exceed total decision time")
    if "allowed_token_mass" in prediction:
        mass = prediction["allowed_token_mass"]
        if isinstance(mass, bool) or not isinstance(mass, (int, float)) or not math.isfinite(mass) or not 0 <= mass <= 1 + 1e-8:
            raise ValueError("Allowed-token mass must be between zero and one")


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
    parameters = manifest["parameters_billion"]
    if isinstance(parameters, bool) or not isinstance(parameters, (int, float)) or not math.isfinite(parameters) or parameters <= 0:
        raise ValueError("Model parameter count must be finite and positive")
    if parameters > 4 and manifest.get("parameter_basis") != "effective":
        raise ValueError("Models above 4B must explicitly declare effective sizing")
    if manifest.get("parameter_basis") == "effective":
        effective = manifest.get("effective_parameters_billion")
        if isinstance(effective, bool) or not isinstance(effective, (int, float)) or not 0 < effective <= 4:
            raise ValueError("Effective-size models must declare a count up to 4B")
    for key in ("id", "family", "upstream", "license"):
        if not isinstance(manifest[key], str) or not manifest[key]:
            raise ValueError(f"{key} must be a nonempty string")
    if manifest.get("upstream_revision") is not None and not re.fullmatch(
        r"[0-9a-f]{40}", manifest["upstream_revision"]
    ):
        raise ValueError("Upstream revision must be a pinned 40-character commit")
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
    if not isinstance(gguf["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", gguf["sha256"]):
        raise ValueError("gguf sha256 must contain 64 hexadecimal characters")
    if not isinstance(gguf["revision"], str) or not re.fullmatch(r"[0-9a-f]{40}", gguf["revision"]):
        raise ValueError("GGUF revision must be a pinned 40-character commit")


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


def write_json_create(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def read_predictions(path: Path) -> list[dict]:
    data = path.read_bytes()
    if data and not data.endswith(b"\n"):
        raise ValueError(f"Prediction file has an incomplete final line: {path}")
    rows = [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]
    ids = [row.get("id") if isinstance(row, dict) else None for row in rows]
    if any(not isinstance(identifier, str) or not identifier for identifier in ids):
        raise ValueError("Every prediction must have an id")
    if len(ids) != len(set(ids)):
        raise ValueError("Prediction ids must be unique")
    return rows


def write_checksums(directory: Path, names: Iterable[str]) -> None:
    with (directory / "SHA256SUMS").open("x", encoding="ascii", newline="\n") as stream:
        for name in names:
            stream.write(f"{sha256_file(directory / name)}  {name}\n")


def verify_checksums(directory: Path, required: set[str] | None = None) -> None:
    recorded = {}
    for line in (directory / "SHA256SUMS").read_text(encoding="ascii").splitlines():
        parts = line.split("  ", 1)
        if len(parts) != 2 or not re.fullmatch(r"[0-9a-f]{64}", parts[0]):
            raise ValueError(f"Invalid checksum entry in {directory}")
        digest, name = parts
        if not name or "/" in name or "\\" in name or name in {".", ".."} or name in recorded:
            raise ValueError(f"Invalid or duplicate checksum filename: {name}")
        recorded[name] = digest
        if sha256_file(directory / name) != digest:
            raise ValueError(f"Checksum mismatch: {directory / name}")
    if not recorded or required is not None and set(recorded) != required:
        raise ValueError(f"Checksum file list differs in {directory}")
