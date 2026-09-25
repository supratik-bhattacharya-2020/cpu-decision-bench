from __future__ import annotations

import hashlib
import json
from pathlib import Path
import urllib.request

from .schema import validate_model_manifest


def read_model_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    validate_model_manifest(manifest)
    return manifest


def verify_model_file(path: Path, manifest: dict) -> None:
    gguf = manifest["gguf"]
    if not path.is_file():
        raise ValueError(f"Model file not found: {path}")
    if path.stat().st_size != gguf["bytes"]:
        raise ValueError(f"Model byte size differs for {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != gguf["sha256"]:
        raise ValueError(f"Model SHA-256 differs for {path}")


def fetch_model(manifest_path: Path, cache: Path) -> Path:
    manifest = read_model_manifest(manifest_path)
    gguf = manifest["gguf"]
    destination = cache / manifest["id"] / gguf["filename"]
    if destination.exists():
        verify_model_file(destination, manifest)
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    if temporary.exists():
        raise ValueError(f"Partial download already exists: {temporary}")
    url = (
        f"https://huggingface.co/{gguf['repository']}/resolve/"
        f"{gguf['revision']}/{gguf['filename']}"
    )
    request = urllib.request.Request(url, headers={"User-Agent": "decisionbench/0.1"})
    digest = hashlib.sha256()
    received = 0
    completed = False
    try:
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("xb") as stream:
            while block := response.read(8 * 1024 * 1024):
                received += len(block)
                if received > gguf["bytes"]:
                    raise ValueError(f"Model download exceeded declared size: {url}")
                digest.update(block)
                stream.write(block)
        if received != gguf["bytes"]:
            raise ValueError(f"Expected {gguf['bytes']} bytes, received {received}")
        if digest.hexdigest() != gguf["sha256"]:
            raise ValueError("Downloaded model SHA-256 does not match the manifest")
        temporary.replace(destination)
        completed = True
    finally:
        if not completed and temporary.exists():
            temporary.unlink()
    return destination
