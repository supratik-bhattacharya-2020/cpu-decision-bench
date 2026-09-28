import json
import csv
from pathlib import Path

from decisionbench.datasets import (
    build_jevbench_pilot,
    build_owned,
    build_smoke,
    convert_jevbench,
    convert_banking77,
    BANKING77_INTENTS,
)
from decisionbench.schema import read_jsonl, sha256_file


def test_owned_suite_has_expected_variants():
    root = Path(__file__).parents[1]
    rows = build_owned(root / "benchmarks" / "owned" / "base36.jsonl")
    assert len(rows) == 180
    assert {row["provenance"]["variant"] for row in rows} == {
        "original",
        "option_reversal",
        "question_reworded",
        "irrelevant_context",
        "missing_evidence",
    }
    assert sum(row["label"] == "insufficient" for row in rows) == 36


def test_jevbench_conversion(tmp_path):
    source = {
        "expected": "yes",
        "family": "policy",
        "group": "g",
        "id": "case",
        "labels": ["no", "yes"],
        "provenance": {},
        "question": {
            "criteria": {"false": "A condition is missing.", "true": "All conditions hold."},
            "instructions": "Is this allowed?",
            "type": "noul",
        },
        "split": "public",
        "state": "All conditions hold.",
    }
    files = {}
    for split, count in (("easy", 48), ("hard", 111), ("original", 72)):
        path = tmp_path / f"{split}.jsonl"
        rows = [{**source, "id": f"{split}-{index}"} for index in range(count)]
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        files[split] = path
    converted = convert_jevbench(files)
    assert len(converted) == 231
    assert converted[0]["options"][1]["description"] == "All conditions hold."


def test_jevbench_score_criteria_list(tmp_path):
    source = {
        "expected": "1",
        "family": "ordinal",
        "group": None,
        "labels": ["0", "1"],
        "provenance": {},
        "question": {
            "criteria": ["No violation", "One violation"],
            "instructions": "How many violations?",
            "type": "score",
        },
        "split": "public",
        "state": "One rule was broken.",
    }
    files = {}
    for split, count in (("easy", 48), ("hard", 111), ("original", 72)):
        path = tmp_path / f"{split}.jsonl"
        rows = [{**source, "id": f"{split}-{index}"} for index in range(count)]
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        files[split] = path
    converted = convert_jevbench(files)
    assert converted[0]["options"][1] == {"id": "1", "description": "One violation"}


def test_frozen_core_v1_matches_manifest():
    root = Path(__file__).parents[1]
    manifest = json.loads((root / "benchmarks" / "manifests" / "core-v1.json").read_text())
    benchmark = root / manifest["benchmark_file"]["path"]
    rows = read_jsonl(benchmark)
    assert len(rows) == manifest["rows"] == 1071
    assert sha256_file(benchmark) == manifest["benchmark_file"]["sha256"]
    assert max(len(row["options"]) for row in rows) <= 26


def test_smoke_suite_covers_seven_sources(tmp_path):
    root = Path(__file__).parents[1]
    output = tmp_path / "smoke.jsonl"
    result = build_smoke(root / "benchmarks" / "core-v1.jsonl", output)
    rows = read_jsonl(output)
    assert result["rows"] == len(rows) == 7
    assert len({row["dataset"] for row in rows}) == 7


def test_jevbench_pilot_balances_tiers(tmp_path):
    root = Path(__file__).parents[1]
    output = tmp_path / "pilot.jsonl"
    result = build_jevbench_pilot(root / "benchmarks" / "core-v1.jsonl", output)
    rows = read_jsonl(output)
    assert result["rows"] == len(rows) == 30
    assert result["tiers"] == {"easy": 10, "original": 10, "hard": 10}


def test_banking77_records_original_csv_index(tmp_path):
    path = tmp_path / "source.csv"
    records = [{"text": "excluded", "category": "other"}]
    for index in range(10):
        records.extend({"text": f"{label}-{index}", "category": label} for label in BANKING77_INTENTS)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["text", "category"])
        writer.writeheader()
        writer.writerows(records)
    rows = convert_banking77(path)
    assert len(rows) == 120
    for row in rows:
        source = records[row["provenance"]["source_row"]]
        assert row["state"] == source["text"]
        assert row["label"] == source["category"]
