import numpy


def test_llama_score_buffer_uses_last_evaluated_position():
    scores = numpy.zeros((8, 4))
    scores[2] = [1.0, 2.0, 3.0, 4.0]
    n_tokens = 3
    vocabulary = scores[n_tokens - 1]
    assert vocabulary.tolist() == [1.0, 2.0, 3.0, 4.0]
    assert scores[-1].tolist() == [0.0, 0.0, 0.0, 0.0]
