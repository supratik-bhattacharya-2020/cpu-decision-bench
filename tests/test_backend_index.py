from types import SimpleNamespace

import numpy

from decisionbench.engine import CpuEngine
from test_runner import row


def test_engine_reads_last_evaluated_position(monkeypatch):
    scores = numpy.zeros((8, 4))
    scores[2] = [1.0, 5.0, 3.0, 4.0]
    engine = CpuEngine.__new__(CpuEngine)
    engine.context_tokens = 8
    engine.metadata = {}
    engine._llm = SimpleNamespace(
        reset=lambda: None, eval=lambda _: None, scores=scores, n_tokens=3,
    )
    monkeypatch.setattr(engine, "_render", lambda _: ("test", False))
    monkeypatch.setattr(engine, "_tokenize", lambda *_, **__: [0, 1, 2])
    monkeypatch.setattr(engine, "_alias_ids", lambda *_, **__: [0, 1])
    monkeypatch.setattr(engine, "_special_token_text", str)
    prediction = engine.score(row("one"))
    assert prediction["option_logits"] == [1.0, 5.0]
    assert prediction["prediction_id"] == "right"
    assert prediction["full_vocab_argmax_id"] == 1
