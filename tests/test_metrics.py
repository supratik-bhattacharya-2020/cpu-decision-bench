import pytest

from decisionbench.metrics import robustness, summarize


ROWS = [
    {
        "id": "a",
        "dataset": "boolq-balanced",
        "state": "x",
        "question": "q",
        "options": [{"id": "yes", "description": "Yes"}, {"id": "no", "description": "No"}],
        "label": "yes",
    },
    {
        "id": "b",
        "dataset": "boolq-balanced",
        "state": "y",
        "question": "q",
        "options": [{"id": "yes", "description": "Yes"}, {"id": "no", "description": "No"}],
        "label": "no",
    },
]


def prediction(identifier, probabilities, choice):
    return {
        "id": identifier,
        "option_ids": ["yes", "no"],
        "probabilities": probabilities,
        "prediction_id": choice,
        "total_seconds": 1.0,
        "allowed_token_mass": 0.9,
    }


def test_metrics_keep_missing_rows_in_denominator():
    report = summarize(ROWS, [prediction("a", [0.8, 0.2], "yes")])
    assert report["coverage"] == 0.5
    assert report["accuracy"] == 0.5
    assert report["nll"] is None
    assert report["errors"] == [{"id": "b", "status": "missing"}]


def test_complete_metrics():
    report = summarize(ROWS, [
        prediction("a", [0.8, 0.2], "yes"),
        prediction("b", [0.1, 0.9], "no"),
    ])
    assert report["accuracy"] == 1.0
    assert report["macro_f1"] == 1.0
    assert report["brier"] == pytest.approx(0.05)


def test_robustness_aligns_semantic_option_ids():
    def owned(identifier, variant, options, label):
        return {
            "id": identifier,
            "dataset": "owned-robustness",
            "group_id": "case",
            "state": "Evidence",
            "question": "Choose.",
            "options": [{"id": value, "description": value} for value in options],
            "label": label,
            "provenance": {"variant": variant},
        }

    original = owned("owned/case/original", "original", ["a", "b", "insufficient"], "a")
    reordered = owned(
        "owned/case/option_reversal",
        "option_reversal",
        ["insufficient", "b", "a"],
        "a",
    )
    missing = owned(
        "owned/case/missing_evidence",
        "missing_evidence",
        ["a", "b", "insufficient"],
        "insufficient",
    )
    predictions = [
        {
            "id": original["id"],
            "option_ids": ["a", "b", "insufficient"],
            "probabilities": [0.8, 0.1, 0.1],
            "prediction_id": "a",
        },
        {
            "id": reordered["id"],
            "option_ids": ["insufficient", "b", "a"],
            "probabilities": [0.1, 0.1, 0.8],
            "prediction_id": "a",
        },
        {
            "id": missing["id"],
            "option_ids": ["a", "b", "insufficient"],
            "probabilities": [0.1, 0.1, 0.8],
            "prediction_id": "insufficient",
        },
    ]
    report = robustness([original, reordered, missing], predictions)
    assert report["argmax_flip_rate"] == 0.0
    assert report["mean_absolute_probability_movement"] == pytest.approx(0.0)
    assert report["missing_evidence_error_rate"] == 0.0
    assert report["valid_pair_coverage"] == 1.0
    predictions[1]["prediction_id"] = "b"
    predictions[2]["error"] = {"type": "test", "message": "invalid"}
    report = robustness([original, reordered, missing], predictions)
    assert report["valid_pair_coverage"] == 0.0
    assert report["argmax_flip_rate"] is None
    assert report["missing_evidence_error_rate"] == 1.0


def test_no_macro_f1_for_heterogeneous_or_dynamic_classes():
    rows = [{**row, "dataset": "mmlu-pro-domain-balanced"} for row in ROWS]
    predictions = [prediction("a", [0.8, 0.2], "yes"), prediction("b", [0.1, 0.9], "no")]
    assert summarize(rows, predictions)["macro_f1"] is None
    rows[0]["dataset"] = "boolq-balanced"
    assert summarize(rows, predictions)["macro_f1"] is None


def test_argmax_mismatch_is_an_invalid_prediction():
    report = summarize(ROWS[:1], [prediction("a", [0.01, 0.99], "yes")])
    assert report["accuracy"] == 0 and report["coverage"] == 0
