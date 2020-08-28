from jitsdp.pipeline import MultiflowBaseEstimator

import mlflow
import numpy as np
import pandas as pd
import time
from skmultiflow.meta import OzaBaggingClassifier
from skmultiflow.trees import HoeffdingTreeClassifier
from skmultiflow.utils import get_dimensions


class ORB():

    def __init__(self, features, balanced_window_size):
        self.features = features
        self.balanced_window_size = balanced_window_size
        # parameters
        self.decay_factor = .99
        self.ma_window_size = 100
        self.th = .4
        self.l0 = 10
        self.l1 = 12
        self.m = 1.5
        # state
        self.observed_classes = set()
        self.observed_instances = 0
        self.ma_window = None
        self.p1 = .5
        self.oza_bag = OzaBaggingClassifier(
            base_estimator=HoeffdingTreeClassifier(), n_estimators=20)
        self.estimators = [MultiflowBaseEstimator(
            estimator) for estimator in self.oza_bag.ensemble]
        self.random_state = self.oza_bag._random_state
        RandomStateWrapper(self)

    @property
    def trained(self):
        return len(self.observed_classes) == 2

    @property
    def active(self):
        return self.observed_instances >= self.balanced_window_size

    def train(self, X, y, **kwargs):
        for features, target in zip(X, y):
            self.update_state(target, **kwargs)
            self.oza_bag.partial_fit([features], [target], classes=[
                                     0, 1], sample_weight=[self.k])
            self.observed_classes.update(y)
            self.observed_instances += 1

    def update_state(self, target, **kwargs):
        self.update_lambda(target, **kwargs)
        self.update_obf(target, **kwargs)
        self.update_k(**kwargs)

    def update_lambda(self, target, **kwargs):
        self.p1 = self.decay_factor * self.p1 + \
            (1 - self.decay_factor) * target
        p0 = 1 - self.p1
        self.lambda_ = 1
        if not self.trained or not self.active or kwargs['ma_boosting']:
            return
        if target == 1 and self.p1 < p0:
            self.lambda_ = p0 / self.p1
        if target == 0 and p0 < self.p1:
            self.lambda_ = self.p1 / p0
        if kwargs.pop('track_orb', False):
            mlflow.log_metrics({'p1': self.p1})

    def update_obf(self, target, **kwargs):
        self.obf = 1
        if not self.trained or not self.active:
            return
        ma = self.th if self.ma_window is None \
            else self.__predict(self.ma_window).mean() if kwargs['ma_update'] \
            else self.ma_window.mean()
        if target == 0 and ma > self.th:
            self.obf = ((self.m ** ma - self.m ** self.th) *
                        self.l0) / (self.m - self.m ** self.th) + 1
        if target == 1 and ma < self.th:
            self.obf = (((self.m ** (self.th - ma) - 1) * self.l1) /
                        (self.m ** self.th - 1)) + 1
        if kwargs.pop('track_orb', False):
            mlflow.log_metrics({'ma': ma,
                                'target': target,
                                'obf': self.obf})

    def update_k(self, **kwargs):
        self.k = self.random_state.poisson(self.lambda_)
        self.k = int(self.k * self.obf)
        if kwargs.pop('track_orb', False):
            mlflow.log_metrics({'k': self.k})

    def predict(self, df_test, **kwargs):
        if self.trained:
            predictions = self.__predict(df_test)
            if kwargs['ma_update']:
                self.ma_window = pd.concat([self.ma_window, df_test])
            else:
                self.ma_window = predictions if self.ma_window is None else np.concatenate(
                    [self.ma_window, predictions])
            self.ma_window = self.ma_window[-self.ma_window_size:]
        else:
            predictions = np.zeros(len(df_test))
        prediction = df_test.copy()
        prediction['prediction'] = predictions
        prediction['probability'] = prediction['prediction']
        prediction = _track_rf(prediction, self)
        prediction = _track_time(prediction)
        return prediction

    def __predict(self, df_test):
        return self.oza_bag.predict(df_test[self.features].values)


class RandomStateWrapper():
    def __init__(self, orb):
        self.orb = orb

    def poisson(self):
        return int(self.orb.k > 0)


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
