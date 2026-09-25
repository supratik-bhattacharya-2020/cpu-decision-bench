from decisionbench.prompt import LABELS, messages, render_plain


ROW = {
    "id": "route/1",
    "dataset": "owned",
    "state": "Customer was charged twice.",
    "question": "Which team should handle this?",
    "options": [
        {"id": "billing", "description": "Billing support"},
        {"id": "access", "description": "Account access support"},
        {"id": "shipping", "description": "Shipping support"},
    ],
    "label": "billing",
}


def test_prompt_maps_options_to_letters():
    text = render_plain(ROW)
    assert "A. Billing support" in text
    assert "B. Account access support" in text
    assert "C. Shipping support" in text
    assert text.endswith("Answer:")
    assert [option["label"] for option in __import__("json").loads(messages(ROW)[1]["content"])["options"]] == list(LABELS[:3])

