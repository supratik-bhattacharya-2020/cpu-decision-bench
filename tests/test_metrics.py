import pytest

from decisionbench.metrics import robustness, summarize


ROWS = [
    {
        "id": "a",
        "dataset": "demo",
        "state": "x",
        "question": "q",
        "options": [{"id": "yes", "description": "Yes"}, {"id": "no", "description": "No"}],
        "label": "yes",
    },
    {
        "id": "b",
        "dataset": "demo",
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
