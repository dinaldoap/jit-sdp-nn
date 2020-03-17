from jitsdp import metrics
from jitsdp.mlp import MLP
from jitsdp.data import FEATURES
from jitsdp.utils import mkdir

from abc import ABCMeta, abstractmethod
import joblib
import logging
import numpy as np
import pandas as pd
import pathlib
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import NotFittedError

logger = logging.getLogger(__name__)


def set_seed(config):
    seed = config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)


def create_pipeline(config):
    estimators = [create_estimator(config)
                  for i in range(config['estimators'])]
    model = Ensemble(estimators=estimators)
    if config['threshold'] == 1:
        classifier = RateFixed(
            model=model, normal_proportion=config['normal_proportion'])
    elif config['threshold'] == 2:
        classifier = RateFixedTrain(
            model=model)
    else:
        classifier = ScoreFixed(model=model)
    if config['orb']:
        classifier = ORB(classifier=classifier,
                         normal_proportion=config['normal_proportion'])
    return classifier


def create_estimator(config):
    scaler = StandardScaler()
    criterion = nn.BCELoss()
    classifier = MLP(input_size=len(FEATURES),
                     hidden_size=len(FEATURES) // 2, drop_prob=0.2)
    optimizer = optim.Adam(params=classifier.parameters(), lr=0.003)
    return PyTorch(steps=[scaler], classifier=classifier, optimizer=optimizer, criterion=criterion,
                   features=FEATURES, target='target', soft_target='soft_target',
                   max_epochs=config['epochs'], batch_size=512, fading_factor=1)


class Model(metaclass=ABCMeta):
    @abstractmethod
    def train(self, df_train, **kwargs):
        pass

    @abstractmethod
    def predict_proba(self, df_features):
        pass

    @abstractmethod
    def save(self):
        pass

    @abstractmethod
    def load(self):
        pass


class Classifier(Model):
    @abstractmethod
    def predict(self, df_features, **kwargs):
        pass


class Threshold(Classifier):
    def __init__(self, model):
        self.model = model

    def train(self, df_train, **kwargs):
        self.model.train(df_train, **kwargs)

    def predict_proba(self, df_features):
        return self.model.predict_proba(df_features)

    def save(self):
        self.model.save()

    def load(self):
        self.model.load()


class ScoreFixed(Threshold):
    def __init__(self, model, score=.5):
        super().__init__(model=model)
        self.score = score

    def predict(self, df_features, **kwargs):
        prediction = self.predict_proba(df_features=df_features)
        prediction['prediction'] = (
            prediction['probability'] >= self.score).round().astype('int')
        return prediction


class RateFixed(Threshold):
    def __init__(self, model, normal_proportion):
        super().__init__(model=model)
        self.normal_proportion = normal_proportion

    def predict(self, df_features, **kwargs):
        df_threshold = kwargs.pop('df_threshold', None)
        val_probabilities = self.predict_proba(
            df_threshold)['probability'] if df_threshold is not None else None
        prediction = self.predict_proba(df_features=df_features)
        threshold = _tune_threshold(val_probabilities=val_probabilities,
                                    test_probabilities=prediction['probability'], normal_proportion=self.normal_proportion)
        threshold = threshold.values
        prediction['prediction'] = (
            prediction['probability'] >= threshold).round().astype('int')
        return prediction


def _tune_threshold(val_probabilities, test_probabilities, normal_proportion):
    if val_probabilities is None:
        # fixed threshold
        return pd.Series([.5] * len(test_probabilities), name='threshold', index=test_probabilities.index)

    # rolling threshold
    probabilities = pd.concat([val_probabilities, test_probabilities[:-1]])
    threshold = probabilities.rolling(len(val_probabilities)).quantile(
        quantile=normal_proportion)
    threshold = threshold.rename('threshold')
    threshold = threshold.dropna()
    threshold.index = test_probabilities.index
    return threshold


class RateFixedTrain(Threshold):
    def __init__(self, model):
        super().__init__(model=model)

    def predict(self, df_features, **kwargs):
        df_proportion = kwargs.pop('df_proportion', None)
        normal_proportion = 1 - df_proportion['soft_target'].mean()
        normal_proportion = (normal_proportion + .5) / 2
        df_threshold = kwargs.pop('df_threshold', None)
        threshold_probabilities = self.predict_proba(df_threshold)[
            'probability']
        prediction = self.predict_proba(df_features=df_features)
        threshold = _tune_threshold(val_probabilities=threshold_probabilities,
                                    test_probabilities=prediction['probability'], normal_proportion=normal_proportion)
        threshold = threshold.values
        prediction['prediction'] = (
            prediction['probability'] >= threshold).round().astype('int')
        return prediction


class ORB(Classifier):
    def __init__(self, classifier, normal_proportion):
        self.classifier = classifier
        self.th = 1 - normal_proportion
        self.m = 1.5
        self.l0 = 10
        self.l1 = 12

    def train(self, df_train, **kwargs):
        df_ma = kwargs.pop('df_ma', None)
        ma = .4
        max_epochs = 50
        for epoch in range(max_epochs):
            obf0 = 1
            obf1 = 1
            if ma > self.th:
                obf0 = ((self.m ** ma - self.m ** self.th) *
                        self.l0) / (self.m - self.m ** self.th) + 1
            elif ma < self.th:
                obf1 = (((self.m ** (self.th - ma) - 1) * self.l1) /
                        (self.m ** self.th - 1)) + 1
            new_kwargs = dict(kwargs)
            new_kwargs['weights'] = [obf0, obf1]
            new_kwargs['max_epochs'] = 1
            self.classifier.train(df_train, **new_kwargs)
            df_output = self.classifier.predict(df_ma)
            ma = df_output['prediction'].mean()

    def predict(self, df_features, **kwargs):
        return self.classifier.predict(df_features, **kwargs)

    def predict_proba(self, df_features):
        return self.classifier.predict_proba(df_features)

    def save(self):
        self.classifier.save()

    def load(self):
        self.classifier.load()


class PyTorch(Model):
    DIR = pathlib.Path('models')
    FILENAME = DIR / 'steps.cpt'

    def __init__(self, steps, classifier, optimizer, criterion, features, target, soft_target, max_epochs, batch_size, fading_factor, val_size=0.0):
        super().__init__()
        self.steps = steps
        self.classifier = classifier
        self.optimizer = optimizer
        self.criterion = criterion
        self.features = features
        self.target = target
        self.soft_target = soft_target
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.fading_factor = fading_factor
        self.val_size = val_size

    def train(self, df_train, **kwargs):
        if len(df_train) == 0:
            logger.warning('No labeled sample to train.')
            return

        X = df_train[self.features].values
        y = df_train[self.target].values
        soft_y = df_train[self.soft_target].values
        if self.has_validation():
            X_train, X_val, y_train, y_val, soft_y_train, soft_y_val = train_test_split(
                X, y, soft_y, test_size=self.val_size, shuffle=False)
            val_dataloader = self.__dataloader(X_val, y_val)
        else:
            X_train, y_train, soft_y_train = X, y, soft_y

        X_train = self.__steps_fit_transform(X_train, y_train)

        weights = kwargs.pop('weights', [1, 1])
        sampled_train_dataloader = self.__dataloader(
            X_train, soft_y_train, batch_size=self.batch_size, sampler=self.__sampler(y_train, weights))
        train_dataloader = self.__dataloader(X_train, y_train)

        if torch.cuda.is_available():
            self.classifier = self.classifier.cuda()

        self.max_epochs = kwargs.pop('max_epochs', self.max_epochs)
        train_loss = 0
        for epoch in range(self.max_epochs):
            self.classifier.train()
            for inputs, targets in sampled_train_dataloader:
                if torch.cuda.is_available():
                    inputs, targets = inputs.cuda(), targets.cuda()

                outputs = self.classifier(inputs.float())
                loss = self.criterion(outputs.view(
                    outputs.shape[0]), targets.float())
                train_loss += loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            train_loss = train_loss / len(sampled_train_dataloader)
            val_loss = None
            if self.has_validation():
                val_loss = metrics.loss(
                    self.classifier, val_dataloader, criterion=self.criterion)
                # Best classifier
                if self.classifier.val_loss is None or val_loss > self.classifier.val_loss:
                    self.classifier.epoch = epoch
                    self.classifier.val_loss = val_loss
                    self.classifier.save()

            logger.debug(
                'Epoch: {}, Train loss: {}, Val loss: {}'.format(epoch, train_loss, val_loss))

    def predict_proba(self, df_features):
        X = df_features[self.features].values
        X = self.__steps_transform(X)
        y = np.zeros(len(X))
        dataloader = self.__dataloader(X, y)

        if torch.cuda.is_available():
            self.classifier = self.classifier.cuda()

        probabilities = []
        with torch.no_grad():
            self.classifier.eval()
            for inputs, targets in dataloader:
                if torch.cuda.is_available():
                    inputs, targets = inputs.cuda(), targets.cuda()

                outputs = self.classifier(inputs.float())
                probabilities.append(outputs.detach().cpu().numpy())

        probability = df_features.copy()
        probability['probability'] = np.concatenate(probabilities)
        return probability

    def __tensor(self, X, y):
        return torch.from_numpy(X), torch.from_numpy(y)

    def __dataloader(self, X, y, batch_size=32, sampler=None):
        X, y = self.__tensor(X, y)
        dataset = data.TensorDataset(X, y)
        return data.DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    def __sampler(self, y, weights):
        normal_indices = np.flatnonzero(y == 0)
        bug_indices = np.flatnonzero(y == 1)
        age_weights = np.zeros(len(y))
        # normal commit ages
        age_weights[normal_indices] = self.__fading_weights(
            size=len(normal_indices), fading_factor=self.fading_factor, total=weights[0])
        # bug commit doesn't age
        age_weights[bug_indices] = self.__fading_weights(
            size=len(bug_indices), fading_factor=self.fading_factor, total=weights[1])
        return data.WeightedRandomSampler(weights=age_weights, num_samples=len(y), replacement=True)

    def __fading_weights(self, size, fading_factor, total):
        fading_weights = reversed(range(size))
        fading_weights = [fading_factor**x for x in fading_weights]
        fading_weights = np.array(fading_weights)
        return (total * fading_weights) / np.sum(fading_weights)

    def __steps_fit_transform(self, X, y):
        for step in self.steps:
            X = step.fit_transform(X, y)
        return X

    def __steps_transform(self, X):
        for step in self.steps:
            try:
                X = step.transform(X)
            except NotFittedError:
                logger.warning('Step {} not fitted.'.format(step))
        return X

    def has_validation(self):
        return self.val_size > 0

    @property
    def epoch(self):
        return self.classifier.epoch

    def load(self):
        self.steps = joblib.load(PyTorch.FILENAME)
        self.classifier.load()

    def save(self):
        mkdir(PyTorch.DIR)
        joblib.dump(self.steps, PyTorch.FILENAME)
        self.classifier.save()


class Ensemble(Model):
    def __init__(self, estimators):
        super().__init__()
        self.estimators = estimators

    def train(self, df_train, **kwargs):
        for estimator in self.estimators:
            estimator.train(df_train, **kwargs)

    def predict_proba(self, df_features):
        probability = df_features
        for index, estimator in enumerate(self.estimators):
            probability = estimator.predict_proba(probability)
            probability = probability.rename({
                'probability': 'probability{}'.format(index),
            },
                axis='columns')
        return _combine(probability)

    def save(self):
        for estimator in self.estimators:
            estimator.save()

    def load(self):
        for estimator in self.estimators:
            estimator.load()


def _combine(prediction):
    prediction = prediction.copy()
    probability_cols = [
        col for col in prediction.columns if 'probability' in col]
    prediction['probability'] = prediction[probability_cols].mean(
        axis='columns')
    return prediction
