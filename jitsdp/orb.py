import numpy as np
from skmultiflow.meta import OzaBaggingClassifier
from skmultiflow.trees import HoeffdingTreeClassifier
from skmultiflow.utils import get_dimensions


class ORB(OzaBaggingClassifier):

    def __init__(self):
        super().__init__(base_estimator=HoeffdingTreeClassifier(), n_estimators=20)
        # parameters
        self.decay_factor = .99
        self.ma_window_size = 100
        self.th = .4
        self.l0 = 10
        self.l1 = 12
        self.m = 1.5
        # state
        self.ma_window = np.array([self.th] * self.ma_window_size)
        self.sum_target = 0
        self.count_target = 0
        self.old_random_state = self._random_state
        self._random_state = RandomStateWrapper(self)

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        for features, target in zip(X, y):
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
        if target == 1 and p1 < p0:
            self.lambda_ = p0 / p1
        if target == 0 and p0 < p1:
            self.lambda_ = p1 / p0

    def update_obf(self, target):
        self.obf = 1
        ma = self.ma_window.mean()
        if target == 0 and ma > self.th:
            self.obf = ((self.m ** ma - self.m ** self.th) *
                        self.l0) / (self.m - self.m ** self.th) + 1
        if target == 1 and ma < self.th:
            self.obf = (((self.m ** (self.th - ma) - 1) * self.l1) /
                        (self.m ** self.th - 1)) + 1

    def predict(self, X):
        predictions = super().predict(X)
        size = min(self.ma_window_size, len(predictions))
        self.ma_window[-size:] = predictions[-size:]
        return predictions


class RandomStateWrapper():
    def __init__(self, orb):
        self.orb = orb

    def poisson(self):
        k = self.orb.old_random_state.poisson(self.orb.lambda_)
        return int(k * self.orb.obf)
