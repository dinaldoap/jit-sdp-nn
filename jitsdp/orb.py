from jitsdp.pipeline import MultiflowBaseEstimator
from jitsdp.utils import track_forest, track_metric, track_time

import mlflow
import numpy as np
import pandas as pd
from skmultiflow.meta import OzaBaggingClassifier
from skmultiflow.utils import get_dimensions


class ORB():

    def __init__(self, features, decay_factor, ma_window_size, th, l0, l1, m, base_estimator, n_estimators, rate_driven, rate_driven_grace_period):
        self.features = features
        # parameters
        self.decay_factor = decay_factor
        self.ma_window_size = ma_window_size
        self.th = th
        self.l0 = l0
        self.l1 = l1
        self.m = m
        self.rate_driven = rate_driven
        self.rate_driven_grace_period = rate_driven_grace_period
        # state
        self.observed_classes = set()
        self.observed_weight_window = None
        self.ma = th
        self.ma_window = None
        self.ma_instance_window = None
        self.p1 = .5
        self.oza_bag = OzaBaggingClassifier(
            base_estimator=base_estimator, n_estimators=n_estimators)
        self.estimators = [MultiflowBaseEstimator(
            estimator) for estimator in self.oza_bag.ensemble]

    @property
    def trained(self):
        return len(self.observed_classes) == 2

    def train(self, X, y, **kwargs):
        for features, target in zip(X, y):
            self.update_state(target, **kwargs)
            self.oza_bag.partial_fit([features], [target], classes=[
                                     0, 1], sample_weight=[self.k])
            self.observed_classes.update(y)
            self.observed_weight_window = None if self.observed_weight_window is None else self.observed_weight_window + self.k

    def update_state(self, target, **kwargs):
        self.update_lambda(target, **kwargs)
        self.update_obf(target, **kwargs)
        self.update_k(**kwargs)
        if kwargs.pop('track_orb', False) and self.trained:
            mlflow.log_metrics({'ma': self.ma,
                                'target': target,
                                'lambda': self.lambda_,
                                'obf': self.obf,
                                'p1': self.p1,
                                })

    def update_lambda(self, target, **kwargs):
        self.p1 = self.decay_factor * self.p1 + \
            (1 - self.decay_factor) * target
        p0 = 1 - self.p1
        self.lambda_ = 1
        if not self.trained or self.rate_driven:
            return
        if target == 1 and self.p1 < p0:
            self.lambda_ = p0 / self.p1
        if target == 0 and p0 < self.p1:
            self.lambda_ = self.p1 / p0

    def update_obf(self, target, **kwargs):
        self.obf = 1
        if not self.trained:
            return
        self.update_ma(**kwargs)
        if target == 0 and self.ma > self.th:
            self.obf = ((self.m ** self.ma - self.m ** self.th) *
                        self.l0) / (self.m - self.m ** self.th) + 1
        if target == 1 and self.ma < self.th:
            self.obf = (((self.m ** (self.th - self.ma) - 1) * self.l1) /
                        (self.m ** self.th - 1)) + 1

    def update_ma(self, **kwargs):
        if self.ma_window is None:
            self.ma = self.th
        else:
            if self.rate_driven:
                outdated_predictions = self.observed_weight_window >= self.rate_driven_grace_period
                if np.any(outdated_predictions):
                    self.ma_window[outdated_predictions], _ = self.__predict(
                        self.ma_instance_window[outdated_predictions])
                    self.observed_weight_window[outdated_predictions] = 0
            self.ma = self.ma_window.mean()

    def update_k(self, **kwargs):
        self.k = np.random.poisson(self.lambda_)
        self.k = self.k * self.obf

    def predict(self, df_test, **kwargs):
        if self.trained:
            predictions, probabilities = self.__predict(df_test)
            if self.rate_driven:
                self.ma_instance_window = self.__update_window(
                    self.ma_instance_window, df_test, pd.concat)
                self.observed_weight_window = self.__update_window(
                    self.observed_weight_window, np.zeros(len(predictions)), np.concatenate)
            self.ma_window = self.__update_window(
                self.ma_window, predictions, np.concatenate)
        else:
            probabilities = np.zeros(len(df_test))
            predictions = probabilities
        prediction = df_test.copy()
        prediction['prediction'] = predictions
        prediction['probability'] = probabilities
        if kwargs['track_forest']:
            prediction = track_forest(prediction, self)
        prediction = track_metric(prediction, 'tr1', self.p1)
        prediction = track_metric(prediction, 'ma', self.ma)
        if kwargs['track_time']:
            prediction = track_time(prediction)
        return prediction

    def __update_window(self, window, input_, fconcat):
        window_size = 0 if window is None else len(window)
        input_limited_size = min(len(input_), self.ma_window_size)

        if window_size == 0 or input_limited_size == self.ma_window_size:
            return input_[-input_limited_size:]
        else:
            concat_window = fconcat([window, input_])
            return concat_window[-self.ma_window_size:]

    def __predict(self, df_test):
        probabilities = self.oza_bag.predict_proba(
            df_test[self.features].values)
        probabilities = probabilities[:, 1]
        predictions = (probabilities >= .5).round().astype('int')
        return predictions, probabilities
