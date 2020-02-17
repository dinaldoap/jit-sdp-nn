from jitsdp import metrics
from jitsdp.classifier import Classifier
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

logger = logging.getLogger(__name__)


def set_seed(config):
    seed = config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)


def create_pipeline(config):
    estimators = [create_estimator(config)
                  for i in range(config['estimators'])]
    return Ensemble(estimators=estimators, normal_proportion=config['normal_proportion'])


def create_estimator(config):
    scaler = StandardScaler()
    criterion = nn.BCELoss()
    classifier = Classifier(input_size=len(FEATURES),
                            hidden_size=len(FEATURES) // 2, drop_prob=0.2)
    optimizer = optim.Adam(params=classifier.parameters(), lr=0.003)
    return Estimator(steps=[scaler], classifier=classifier, optimizer=optimizer, criterion=criterion,
                     features=FEATURES, target='target',
                     max_epochs=config['epochs'], batch_size=512, fading_factor=1, normal_proportion=config['normal_proportion'])


class Pipeline(metaclass=ABCMeta):
    def __init__(self, normal_proportion):
        self.normal_proportion = normal_proportion

    @abstractmethod
    def train(self, labeled):
        pass

    def predict(self, features, unlabeled=None):
        val_probabilities = self.predict_proba(
            unlabeled)['probability'] if unlabeled is not None else None
        prediction = self.predict_proba(features=features)
        threshold = _tune_threshold(val_probabilities=val_probabilities,
                                    test_probabilities=prediction['probability'], normal_proportion=self.normal_proportion)
        threshold = threshold.values
        prediction['prediction'] = (
            prediction['probability'] >= threshold).round().astype('int')
        return prediction

    @abstractmethod
    def predict_proba(self, features):
        pass


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


class Estimator(Pipeline):
    DIR = pathlib.Path('models')
    FILENAME = DIR / 'steps.cpt'

    def __init__(self, steps, classifier, optimizer, criterion, features, target, max_epochs, batch_size, fading_factor, normal_proportion, val_size=0.0):
        super().__init__(normal_proportion=normal_proportion)
        self.steps = steps
        self.classifier = classifier
        self.optimizer = optimizer
        self.criterion = criterion
        self.features = features
        self.target = target
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.fading_factor = fading_factor
        self.val_size = val_size

    def train(self, labeled):
        X = labeled[self.features].values
        y = labeled[self.target].values
        if self.has_validation():
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=self.val_size, shuffle=False)
            val_dataloader = self.__dataloader(X_val, y_val)
        else:
            X_train, y_train = X, y

        X_train = self.__steps_fit_transform(X_train, y_train)

        sampled_train_dataloader = self.__dataloader(
            X_train, y_train, batch_size=self.batch_size, sampler=self.__sampler(y_train))
        train_dataloader = self.__dataloader(X_train, y_train)

        if torch.cuda.is_available():
            self.classifier = self.classifier.cuda()

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
        # Last classifier
        self.classifier.epoch = epoch
        self.classifier.val_loss = val_loss
        if not self.has_validation():
            self.classifier.save()

    def predict_proba(self, features):
        X = features[self.features].values
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

        probability = features.copy()
        probability['probability'] = np.concatenate(probabilities)
        return probability

    def __tensor(self, X, y):
        return torch.from_numpy(X), torch.from_numpy(y)

    def __dataloader(self, X, y, batch_size=32, sampler=None):
        X, y = self.__tensor(X, y)
        dataset = data.TensorDataset(X, y)
        return data.DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    def __sampler(self, y):
        normal_indices = np.flatnonzero(y == 0)
        bug_indices = np.flatnonzero(y == 1)
        age_weights = np.zeros(len(y))
        # normal commit ages
        age_weights[normal_indices] = self.__fading_weights(
            size=len(normal_indices), fading_factor=self.fading_factor)
        # bug commit doesn't age
        age_weights[bug_indices] = self.__fading_weights(
            size=len(bug_indices), fading_factor=self.fading_factor)
        return data.WeightedRandomSampler(weights=age_weights, num_samples=len(y), replacement=True)

    def __fading_weights(self, size, fading_factor):
        fading_weights = reversed(range(size))
        fading_weights = [fading_factor**x for x in fading_weights]
        fading_weights = np.array(fading_weights)
        return fading_weights / np.sum(fading_weights)

    def __steps_fit_transform(self, X, y):
        for step in self.steps:
            X = step.fit_transform(X, y)
        return X

    def __steps_transform(self, X):
        for step in self.steps:
            X = step.transform(X)
        return X

    def has_validation(self):
        return self.val_size > 0

    @property
    def epoch(self):
        return self.classifier.epoch

    def load(self):
        self.steps = joblib.load(Estimator.FILENAME)
        self.classifier.load()

    def save(self):
        mkdir(Estimator.DIR)
        joblib.dump(self.steps, Estimator.FILENAME)
        self.classifier.save()


class Ensemble(Pipeline):
    def __init__(self, estimators, normal_proportion):
        super().__init__(normal_proportion=normal_proportion)
        self.estimators = estimators

    def train(self, labeled):
        for estimator in self.estimators:
            estimator.train(labeled)

    def predict_proba(self, features):
        probability = features
        for index, estimator in enumerate(self.estimators):
            probability = estimator.predict_proba(probability)
            probability = probability.rename({
                'probability': 'probability{}'.format(index),
            },
                axis='columns')
        return _combine(probability)


def _combine(prediction):
    prediction = prediction.copy()
    probability_cols = [
        col for col in prediction.columns if 'probability' in col]
    prediction['probability'] = prediction[probability_cols].mean(
        axis='columns')
    return prediction
