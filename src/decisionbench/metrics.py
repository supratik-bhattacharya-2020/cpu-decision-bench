from __future__ import annotations

from collections import defaultdict
import math
import statistics

from .schema import validate_prediction, validate_row


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _macro_f1(gold: list[str], predicted: list[str | None]) -> float:
    scores = []
    for label in sorted(set(gold)):
        true_positive = sum(left == label and right == label for left, right in zip(gold, predicted))
        false_positive = sum(left != label and right == label for left, right in zip(gold, predicted))
        false_negative = sum(left == label and right != label for left, right in zip(gold, predicted))
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(2 * true_positive / denominator if denominator else 0.0)
    return statistics.mean(scores)


def summarize(rows: list[dict], predictions: list[dict]) -> dict:
    for row in rows:
        validate_row(row)
    by_prediction = {prediction["id"]: prediction for prediction in predictions}
    if len(by_prediction) != len(predictions):
        raise ValueError("Prediction ids must be unique")
    unknown = set(by_prediction) - {row["id"] for row in rows}
    if unknown:
        raise ValueError(f"Unknown prediction ids: {sorted(unknown)[:5]}")
    gold_ids = []
    predicted_ids: list[str | None] = []
    correct = []
    nll = []
    brier = []
    confidences = []
    confidence_correct = []
    latencies = []
    masses = []
    memories = []
    input_tokens = []
    errors = []
    for row in rows:
        gold_ids.append(row["label"])
        prediction = by_prediction.get(row["id"])
        if prediction is None:
            predicted_ids.append(None)
            correct.append(False)
            errors.append({"id": row["id"], "status": "missing"})
            continue
        try:
            validate_prediction(prediction, row)
            option_ids = prediction["option_ids"]
            probabilities = prediction["probabilities"]
            predicted = prediction.get("prediction_id") or option_ids[max(range(len(probabilities)), key=probabilities.__getitem__)]
            if predicted not in option_ids:
                raise ValueError("prediction_id is outside the declared options")
            predicted_ids.append(predicted)
            is_correct = predicted == row["label"]
            correct.append(is_correct)
            gold_index = option_ids.index(row["label"])
            nll.append(-math.log(max(probabilities[gold_index], 1e-12)))
            brier.append(sum((value - float(index == gold_index)) ** 2 for index, value in enumerate(probabilities)))
            confidence = max(probabilities)
            confidences.append(confidence)
            confidence_correct.append((confidence, is_correct))
            if isinstance(prediction.get("total_seconds"), (int, float)):
                latencies.append(float(prediction["total_seconds"]))
            if isinstance(prediction.get("allowed_token_mass"), (int, float)):
                masses.append(float(prediction["allowed_token_mass"]))
            if isinstance(prediction.get("peak_process_rss_bytes"), (int, float)):
                memories.append(float(prediction["peak_process_rss_bytes"]))
            if isinstance(prediction.get("input_tokens"), int):
                input_tokens.append(prediction["input_tokens"])
        except (ValueError, TypeError) as error:
            predicted_ids.append(None)
            correct.append(False)
            errors.append({"id": row["id"], "status": "invalid", "error": str(error)})
    bins = []
    for index in range(10):
        lower = index / 10
        upper = (index + 1) / 10
        selected = [(confidence, value) for confidence, value in confidence_correct
                    if lower <= confidence < upper or index == 9 and confidence == 1]
        if selected:
            bins.append({
                "lower": lower,
                "upper": upper,
                "n": len(selected),
                "mean_confidence": statistics.mean(value[0] for value in selected),
                "accuracy": statistics.mean(value[1] for value in selected),
            })
    ece = (
        sum(
            item["n"] / len(confidence_correct)
            * abs(item["mean_confidence"] - item["accuracy"])
            for item in bins
        )
        if confidence_correct
        else None
    )
    total_seconds = sum(latencies)
    return {
        "rows": len(rows),
        "valid_predictions": len(rows) - len(errors),
        "coverage": (len(rows) - len(errors)) / len(rows) if rows else 0.0,
        "accuracy": sum(correct) / len(rows) if rows else None,
        "macro_f1": _macro_f1(gold_ids, predicted_ids) if rows else None,
        "nll": statistics.mean(nll) if len(nll) == len(rows) else None,
        "brier": statistics.mean(brier) if len(brier) == len(rows) else None,
        "ece_10_bin": ece,
        "reliability_bins": bins,
        "latency_seconds": {
            "p50": _percentile(latencies, 0.50),
            "p95": _percentile(latencies, 0.95),
            "total": total_seconds,
            "decisions_per_second": len(latencies) / total_seconds if total_seconds else None,
        },
        "allowed_token_mass": {
            "mean": statistics.mean(masses) if masses else None,
            "p50": _percentile(masses, 0.50),
        },
        "peak_process_rss_bytes": {
            "max": max(memories) if memories else None,
            "p50": _percentile(memories, 0.50),
        },
        "input_tokens": {
            "mean": statistics.mean(input_tokens) if input_tokens else None,
            "p50": _percentile(input_tokens, 0.50),
        },
        "errors": errors,
    }


def by_dataset(rows: list[dict], predictions: list[dict]) -> dict:
    prediction_map = {prediction["id"]: prediction for prediction in predictions}
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["dataset"]].append(row)
    return {
        dataset: summarize(dataset_rows, [prediction_map[row["id"]] for row in dataset_rows if row["id"] in prediction_map])
        for dataset, dataset_rows in sorted(grouped.items())
    }


def robustness(rows: list[dict], predictions: list[dict]) -> dict | None:
    owned = [row for row in rows if row.get("dataset") == "owned-robustness"]
    if not owned:
        return None
    prediction_map = {prediction["id"]: prediction for prediction in predictions}
    groups = defaultdict(dict)
    for row in owned:
        groups[row["group_id"]][row["provenance"]["variant"]] = row
    comparisons = 0
    flips = 0
    probability_changes = []
    missing_rows = 0
    missing_errors = 0
    for variants in groups.values():
        original = variants.get("original")
        if original is None:
            continue
        original_prediction = prediction_map.get(original["id"])
        if original_prediction and isinstance(original_prediction.get("probabilities"), list):
            original_probabilities = dict(zip(original_prediction["option_ids"], original_prediction["probabilities"]))
            for variant, row in variants.items():
                if variant in {"original", "missing_evidence"}:
                    continue
                prediction = prediction_map.get(row["id"])
                if not prediction or not isinstance(prediction.get("probabilities"), list):
                    continue
                probabilities = dict(zip(prediction["option_ids"], prediction["probabilities"]))
                if set(probabilities) != set(original_probabilities):
                    continue
                comparisons += 1
                flips += prediction.get("prediction_id") != original_prediction.get("prediction_id")
                probability_changes.extend(
                    abs(probabilities[option_id] - original_probabilities[option_id])
                    for option_id in original_probabilities
                )
        missing = variants.get("missing_evidence")
        if missing:
            missing_rows += 1
            prediction = prediction_map.get(missing["id"])
            if not prediction or prediction.get("prediction_id") != "insufficient":
                missing_errors += 1
    return {
        "paired_variant_comparisons": comparisons,
        "argmax_flip_rate": flips / comparisons if comparisons else None,
        "mean_absolute_probability_movement": (
            statistics.mean(probability_changes) if probability_changes else None
        ),
        "missing_evidence_rows": missing_rows,
        "missing_evidence_error_rate": missing_errors / missing_rows if missing_rows else None,
    }
