from jitsdp import metrics
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from pytest import approx


def test_prequential_recalls():
    fading_factor = 1
    results = {
        'timestep': [0, 1, 2, 3, 4, 5],
        'target': [0, 1, 0, 0, 1, 1],
        'prediction': [None, 0, 0, 0, 1, 1],
    }
    expected = results.copy()
    expected.update({
        'r0': [0, 0, .5, .6666666, .6666666, .6666666],
        'r1': [0, 0,  0,        0,       .5, .6666666],    
    })
    results = pd.DataFrame(results)
    expected = pd.DataFrame(expected)
    actual = metrics.prequential_recalls(results, fading_factor)
    assert_frame_equal(expected, actual)
