from __future__ import annotations

import json

from .schema import sha256_text, validate_row

LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
PROMPT_VERSION = "decisionbench-choice-v1"
SYSTEM_PROMPT = (
    "Apply the question to the evidence. Choose exactly one listed option. "
    "Respond with only its option label."
)


def payload(row: dict) -> dict:
    validate_row(row)
    return {
        "evidence": row["state"],
        "question": row["question"],
        "options": [
            {"label": LABELS[index], "description": option["description"]}
            for index, option in enumerate(row["options"])
        ],
    }


def messages(row: dict) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(payload(row), ensure_ascii=False, allow_nan=False)},
    ]


def render_plain(row: dict) -> str:
    body = payload(row)
    lines = [
        SYSTEM_PROMPT,
        "",
        "Evidence:",
        json.dumps(body["evidence"], ensure_ascii=False, allow_nan=False),
        "",
        "Question:",
        body["question"],
        "",
        "Options:",
    ]
    lines.extend(f"{option['label']}. {option['description']}" for option in body["options"])
    lines.extend(["", "Answer:"])
    return "\n".join(lines)


def prompt_record(text: str) -> dict:
    return {"version": PROMPT_VERSION, "sha256": sha256_text(text), "text": text}

