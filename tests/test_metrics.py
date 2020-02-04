from jitsdp import metrics
import numpy as np

from pytest import approx


def test_prequential_recalls():
    fading_factor = 1
    targets = [0, 1, 0, 0, 1, 1]
    predictions = [1, 0, 0, 0, 1, 1]
    expected = np.array(
        [[0, 0], [0, 0], [.5, 0], [.6666666, 0], [.6666666, .5], [.6666666, .6666666]])
    actual = metrics.prequential_recalls(targets, predictions, fading_factor)
    assert expected == approx(actual)
