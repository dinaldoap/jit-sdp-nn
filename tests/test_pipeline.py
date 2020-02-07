from jitsdp.evaluation import create_pipeline
from jitsdp.data import FEATURES
from jitsdp import metrics


import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim

from pytest import approx
from numpy.testing import assert_array_equal


def test_train_predict():
    pipeline = create_pipeline({'epochs': 100})
    n_samples = 100
    half_samples = n_samples // 2
    features = np.random.rand(n_samples, len(FEATURES))
    features[half_samples:, :] = features[half_samples:, :] + 1
    data = pd.DataFrame(features, columns=FEATURES)
    targets = [0] * half_samples + [1] * half_samples
    data['target'] = np.array(targets, dtype=np.int64)
    pipeline.train(data)
    target_prediction = pipeline.predict(data)

    # metrics
    expected_gmean = 1.
    expected_recalls = np.array([1., 1.])
    gmean, recalls = metrics.gmean_recalls(target_prediction)
    assert expected_gmean == gmean
    assert_array_equal(expected_recalls, recalls)

    # probability
    assert 0.5 == approx(target_prediction['probability'].median(), abs=.05)