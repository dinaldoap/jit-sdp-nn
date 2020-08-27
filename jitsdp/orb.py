from jitsdp.pipeline import MultiflowBaseEstimator

import numpy as np
import pandas as pd
import time
from skmultiflow.meta import OzaBaggingClassifier
from skmultiflow.trees import HoeffdingTreeClassifier
from skmultiflow.utils import get_dimensions


class ORB(OzaBaggingClassifier):

    def __init__(self, features):
        super().__init__(base_estimator=HoeffdingTreeClassifier(), n_estimators=20)
        self.features = features
        self.estimators = [MultiflowBaseEstimator(
            estimator) for estimator in self.ensemble]
        # parameters
        self.decay_factor = .99
        self.ma_window_size = 100
        self.th = .4
        self.l0 = 10
        self.l1 = 12
        self.m = 1.5
        # state
        self.observed_classes = set()
        self.ma_window = None
        self.sum_target = 0
        self.count_target = 0
        self.old_random_state = self._random_state
        self._random_state = RandomStateWrapper(self)

    @property
    def trained(self):
        return len(self.observed_classes) == 2

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        for features, target in zip(X, y):
            self.observed_classes.update(y)
            self.update_lambda_obf(target)
            super().partial_fit([features], [target], classes, sample_weight)

    def update_lambda_obf(self, target):
        self.update_lambda(target)
        self.update_obf(target)

    def update_lambda(self, target):
        self.sum_target = target + self.decay_factor * self.sum_target
        self.count_target = 1 + self.decay_factor * self.count_target
        p1 = self.sum_target / self.count_target
        p0 = 1 - p1
        self.lambda_ = 1
        if not self.trained:
            return
        if target == 1 and p1 < p0:
            self.lambda_ = p0 / p1
        if target == 0 and p0 < p1:
            self.lambda_ = p1 / p0

    def update_obf(self, target):
        self.obf = 1
        if not self.trained:
            return
        ma = self.th if self.ma_window is None else self.__predict(
            self.ma_window).mean()
        if target == 0 and ma > self.th:
            self.obf = ((self.m ** ma - self.m ** self.th) *
                        self.l0) / (self.m - self.m ** self.th) + 1
        if target == 1 and ma < self.th:
            self.obf = (((self.m ** (self.th - ma) - 1) * self.l1) /
                        (self.m ** self.th - 1)) + 1

    def predict(self, df_test):
        if self.trained:
            predictions = self.__predict(df_test)
        else:
            predictions = np.zeros(len(df_test))
        self.ma_window = pd.concat([self.ma_window, df_test])
        self.ma_window = self.ma_window[-self.ma_window_size:]
        prediction = df_test.copy()
        prediction['prediction'] = predictions
        prediction['probability'] = prediction['prediction']
        prediction = _track_rf(prediction, self)
        prediction = _track_time(prediction)
        return prediction

    def __predict(self, df_test):
        return super().predict(df_test[self.features].values)


class RandomStateWrapper():
    def __init__(self, orb):
        self.orb = orb

    def poisson(self):
        k = self.orb.old_random_state.poisson(self.orb.lambda_)
        return int(k * self.orb.obf)


def _track_rf(prediction, rf):
    properties = {
        'depth': lambda tree: tree.get_depth() if rf.trained else 0,
        'n_leaves': lambda tree: tree.get_n_leaves() if rf.trained else 0,
    }
    for name, func in properties.items():
        values = _extract_property(rf, func)
        prediction = _concat_property(prediction, name, values)
    return prediction


def _extract_property(rf, func):
    if rf.trained:
        return [func(estimator) for estimator in rf.estimators]
    else:
        return [0.]


def _concat_property(prediction, name, values):
    prop = pd.Series(values, dtype=np.float64)
    prop = prop.describe()
    prop = prop.to_frame()
    prop = prop.transpose()
    prop.columns = ['{}_{}'.format(name, column) for column in prop.columns]
    template = [prop.head(0)]
    prop = pd.concat(template + [prop] * len(prediction))
    prop.index = prediction.index
    return pd.concat([prediction, prop], axis='columns')


def _track_time(prediction):
    prediction['timestamp_test'] = time.time()
    return prediction
