import hashlib
import json
from pathlib import Path

import pytest

from decisionbench.models import fetch_model, verify_model_file
from decisionbench.schema import validate_model_manifest


def test_all_model_manifests_are_valid():
    root = Path(__file__).parents[1]
    manifests = sorted((root / "models").glob("*.json"))
    assert len(manifests) == 6
    for path in manifests:
        validate_model_manifest(json.loads(path.read_text()))


def _manifest(data):
    return {
        "id": "tiny",
        "family": "test",
        "parameters_billion": 0.1,
        "upstream": "test/upstream",
        "license": "MIT",
        "benchmark_context_tokens": 32,
        "alias_compatibility": {"status": "unverified", "detail": "test"},
        "gguf": {
            "repository": "test/repository",
            "revision": "a" * 40,
            "filename": "tiny.gguf",
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "quantization": "test",
        },
    }


def test_existing_model_file_is_verified(tmp_path):
    data = b"model"
    manifest_path = tmp_path / "model.json"
    manifest_path.write_text(json.dumps(_manifest(data)))
    model_path = tmp_path / "cache" / "tiny" / "tiny.gguf"
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(data)
    assert fetch_model(manifest_path, tmp_path / "cache") == model_path


def test_model_hash_mismatch_is_rejected(tmp_path):
    data = b"model"
    model_path = tmp_path / "tiny.gguf"
    model_path.write_bytes(b"wrong")
    with pytest.raises(ValueError, match="SHA-256 differs"):
        verify_model_file(model_path, _manifest(data))
