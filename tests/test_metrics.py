from jitsdp import metrics
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from pytest import approx


def test_prequential_recalls():
    fading_factor = 1
    targets = [0, 1, 0, 0, 1, 1]
    predictions = [1, 0, 0, 0, 1, 1]
    expected = {
        'timestep': [0, 1, 2, 3, 4, 5],
        'r0': [0, 0, .5, .6666666, .6666666, .6666666],
        'r1': [0, 0,  0,        0,       .5, .6666666]        
    }
    expected = pd.DataFrame(expected)
    actual = metrics.prequential_recalls(targets, predictions, fading_factor)
    assert_frame_equal(expected, actual)
