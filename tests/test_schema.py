import json

import pytest

from decisionbench.schema import read_jsonl, validate_row, write_jsonl_create


def row():
    return {
        "id": "test/1",
        "dataset": "test",
        "state": "The payment was duplicated.",
        "question": "Which team should handle this?",
        "options": [
            {"id": "billing", "description": "Billing support"},
            {"id": "access", "description": "Account access support"},
        ],
        "label": "billing",
    }


def test_valid_row():
    validate_row(row())


def test_label_must_be_an_option():
    invalid = row()
    invalid["label"] = "missing"
    with pytest.raises(ValueError, match="option id"):
        validate_row(invalid)


def test_jsonl_is_create_only(tmp_path):
    path = tmp_path / "rows.jsonl"
    write_jsonl_create(path, [row()])
    assert read_jsonl(path) == [row()]
    with pytest.raises(FileExistsError):
        write_jsonl_create(path, [row()])


def test_duplicate_ids_are_rejected(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text(json.dumps(row()) + "\n" + json.dumps(row()) + "\n")
    with pytest.raises(ValueError, match="Duplicate"):
        read_jsonl(path)

