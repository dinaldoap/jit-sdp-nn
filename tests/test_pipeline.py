from jitsdp import evaluation
from jitsdp import metrics
from jitsdp.data import FEATURES
from jitsdp.pipeline import _combine, _tune_threshold


import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim

from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal, assert_series_equal
from pytest import approx


def create_pipeline():
    pipeline = evaluation.create_pipeline(
        {'n_epochs': 100, 'normal_proportion': .5, 'ensemble_size': 1, 'model': 'mlp', 'threshold': 0, 'orb': 0, 'f_val': .0})
    return pipeline


def create_data():
    n_samples = 100
    half_samples = n_samples // 2
    features = np.random.rand(n_samples, len(FEATURES))
    features[half_samples:, :] = features[half_samples:, :] + 1
    data = pd.DataFrame(features, columns=FEATURES)
    targets = [0] * half_samples + [1] * half_samples
    data['target'] = np.array(targets, dtype=np.int64)
    data['soft_target'] = data['target']
    return data


def test_train_predict():
    pipeline = create_pipeline()
    data = create_data()
    list(pipeline.train(data))
    target_prediction = pipeline.predict(data)

    # metrics
    expected_gmean = 1.
    expected_recalls = np.array([1., 1.])
    gmean, recalls = metrics.gmean_recalls(target_prediction)
    assert expected_gmean == gmean
    assert_array_equal(expected_recalls, recalls)

    # probability
    assert 0.5 == target_prediction['probability'].round().mean()


def test_combine():
    prediction = {
        'probability0': [0, .5],
        'probability1': [.5, 1],
    }
    expected = dict(prediction)
    expected.update({
        'probability': [.25, .75],
    })
    prediction = pd.DataFrame(prediction)
    expected = pd.DataFrame(expected)
    combined_prediction = _combine(prediction=prediction)

    assert_frame_equal(expected, combined_prediction)


def test_tune_threshold():
    val_probabilities = pd.Series([0, .1], name='probability')
    test_probabilities = pd.Series([.2, .3], name='probability')
    expected = pd.Series([.05, .15], name='threshold',
                         index=test_probabilities.index)
    thresholds = _tune_threshold(val_probabilities=val_probabilities,
                                 test_probabilities=test_probabilities, normal_proportion=.5)

    assert_series_equal(expected, thresholds)
