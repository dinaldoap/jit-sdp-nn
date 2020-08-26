import numpy as np
from skmultiflow.trees import HoeffdingTreeClassifier
from skmultiflow.meta import OzaBaggingClassifier


class ORB(OzaBaggingClassifier):

    def __init__(self):
        super().__init__(base_estimator=HoeffdingTreeClassifier(), n_estimators=20)
        self.ma_window_size = 100
        self.ma_window = np.array([.5] * self.ma_window_size)

    def predict(self, X):
        predictions = super().predict(X)
        size = min(self.ma_window_size, len(predictions))
        self.ma_window[-size:] = predictions[-size:]
        return predictions
