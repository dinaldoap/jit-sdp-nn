# coding=utf-8
from jitsdp import metrics
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from pytest import approx


def test_prequential_recalls():
    fading_factor = .9
    results = {
        'timestep': [0, 1, 2, 3, 4, 5],
        'target': [0, 1, 0, 0, 1, 1],
        'prediction': [None, 0, 0, 0, 1, 1],
    }
    expected = results.copy()
    expected.update({
        'r0': [0, 0, .526315789, .701107011, .701107011, .701107011],
        'r1': [0, 0,          0,          0, .526315789, .701107011],
    })
    results = pd.DataFrame(results)
    expected = pd.DataFrame(expected)
    actual = metrics.prequential_recalls(results, fading_factor)
    assert_frame_equal(expected, actual)


def test_prequential_recalls_difference():
    recalls = {
        'r0': [0, 0, .526315789, .701107011, .701107011, .701107011],
        'r1': [0, 0,          0,          0, .526315789, .701107011],
    }
    expected = recalls.copy()
    expected.update({
        'r0-r1': [0, 0, .526315789, .701107011, .174791222, .0],
    })
    recalls = pd.DataFrame(recalls)
    expected = pd.DataFrame(expected)
    recalls_difference = metrics.prequential_recalls_difference(recalls)
    assert_frame_equal(expected, recalls_difference)


def test_prequential_gmean():
    recalls = {
        'r0': [0, 0, .526315789, .701107011, .701107011, .701107011],
        'r1': [0, 0,          0,          0, .526315789, .701107011],
    }
    expected = recalls.copy()
    expected.update({
        'g-mean': [0, 0,       0,          0, .607456739, .701107011],
    })
    recalls = pd.DataFrame(recalls)
    expected = pd.DataFrame(expected)
    actual = metrics.prequential_gmean(recalls)
    assert_frame_equal(expected, actual)


def test_prequential_proportions():
    fading_factor = .9
    results = {
        'timestep': [0, 1, 2, 3, 4, 5],
        'target': [1, 0, 0, 0, 1, 0],
        'probability': [1, 0, 0, 0, 1, .5],
        'prediction': [1, 0, 0, 0, 1, 1],
        'ma': [1, .4736842105, .2988929889, .2119802268, .4044101487, .5315211105],
    }
    expected = results.copy()
    expected.update({
        't1': [1, .4736842105, .2988929889, .2119802268, .4044101487, .3181008155],
        's1': [1, .4736842105, .2988929889, .2119802268, .4044101487, .4248109630],
        'p1': [1, .4736842105, .2988929889, .2119802268, .4044101487, .5315211105],        
    })
    threshold = .5
    expected.update({
        'th-ma': [abs(threshold-ma) for ma in results['ma']]
    })
    results = pd.DataFrame(results)
    expected = pd.DataFrame(expected)
    actual = metrics.prequential_proportions(results, fading_factor, threshold)
    assert_frame_equal(expected, actual)
