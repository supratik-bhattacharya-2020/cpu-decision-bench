import math

import pytest

from decisionbench.engine import softmax


def test_softmax_is_normalized():
    result = softmax([4.0, 2.0, 1.0])
    assert sum(result) == pytest.approx(1.0)
    assert result[0] > result[1] > result[2]


def test_softmax_rejects_nonfinite_values():
    with pytest.raises(ValueError):
        softmax([1.0, math.inf])

