import json

import pytest

from decisionbench.schema import read_jsonl, read_predictions, validate_prediction, validate_row, write_jsonl_create


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


@pytest.mark.parametrize("field,value", [
    ("total_seconds", -1), ("forward_seconds", float("nan")),
    ("input_tokens", float("inf")), ("peak_process_rss_bytes", True),
    ("allowed_token_mass", 1.1),
])
def test_prediction_metadata_is_validated(field, value):
    prediction = {
        "id": "test/1", "option_ids": ["billing", "access"],
        "probabilities": [0.9, 0.1], "prediction_id": "billing", field: value,
    }
    with pytest.raises(ValueError):
        validate_prediction(prediction, row())


def test_unfinished_prediction_line_is_not_silently_accepted(tmp_path):
    path = tmp_path / "predictions.jsonl"
    path.write_bytes(b'{"id":"one"}\n{"id":')
    with pytest.raises(ValueError, match="incomplete final line"):
        read_predictions(path)


def test_prediction_logits_must_match_probabilities():
    prediction = {
        "id": "test/1", "option_ids": ["billing", "access"],
        "probabilities": [0.9, 0.1], "prediction_id": "billing",
        "option_logits": [0.0, 0.0],
    }
    with pytest.raises(ValueError, match="do not match"):
        validate_prediction(prediction, row())
