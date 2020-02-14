from jitsdp import metrics
from jitsdp.utils import mkdir

from abc import ABCMeta, abstractmethod
import numpy as np
import torch
import torch.utils.data as data
from sklearn.model_selection import train_test_split
import joblib
import pathlib

import logging
logger = logging.getLogger(__name__)


class Pipeline(metaclass=ABCMeta):
    def __init__(self, normal_proportion):
        self.threshold = .5
        self.normal_proportion = normal_proportion

    def train(self, labeled, unlabeled=None):
        self._train(labeled, unlabeled)
        self.__tune_threshold(unlabeled)

    @abstractmethod
    def _train(self, labeled, unlabeled=None):
        pass

    def __tune_threshold(self, unlabeled):
        if unlabeled is None:
            return

        df_val = unlabeled[-100:]
        probabilities = self.predict(df_val)
        probabilities = probabilities['probability']
        self.threshold = probabilities.quantile(q=self.normal_proportion)

    @abstractmethod
    def predict(self, features):
        pass


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

    def _train(self, labeled, unlabeled=None):
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
            val_gmean = None
            if self.has_validation():
                val_gmean = metrics.classifier_gmean(
                    self.classifier, val_dataloader)
                # Best classifier
                if self.classifier.val_gmean is None or val_gmean > self.classifier.val_gmean:
                    self.classifier.epoch = epoch
                    self.classifier.val_gmean = val_gmean
                    self.classifier.save()

            logger.debug(
                'Epoch: {}, Train loss: {}, Val g-mean: {}'.format(epoch, train_loss, val_gmean))
        # Last classifier
        self.classifier.epoch = epoch
        self.classifier.val_gmean = val_gmean
        if not self.has_validation():
            self.classifier.save()

    def predict(self, features):
        X = features[self.features].values
        X = self.__steps_transform(X)
        y = np.zeros(len(X))
        dataloader = self.__dataloader(X, y)

        if torch.cuda.is_available():
            self.classifier = self.classifier.cuda()

        predictions = []
        probabilities = []
        with torch.no_grad():
            self.classifier.eval()
            for inputs, targets in dataloader:
                if torch.cuda.is_available():
                    inputs, targets = inputs.cuda(), targets.cuda()

                outputs = self.classifier(inputs.float())
                probabilities.append(outputs.detach().cpu().numpy())
                batch_predictions = (outputs >= self.threshold).int()
                batch_predictions = batch_predictions.view(
                    batch_predictions.shape[0])
                predictions.append(batch_predictions.detach().cpu().numpy())

        features_prediction = features.copy()
        features_prediction['prediction'] = np.concatenate(predictions)
        features_prediction['probability'] = np.concatenate(probabilities)
        return features_prediction

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

    def _train(self, labeled, unlabeled=None):
        for estimator in self.estimators:
            estimator.train(labeled, unlabeled)

    def predict(self, features):
        prediction = features
        for index, estimator in enumerate(self.estimators):
            prediction = estimator.predict(prediction)
            prediction = prediction.rename({
                'probability': 'probability{}'.format(index),
                'prediction': 'prediction{}'.format(index)
            },
                axis='columns')
        return self.__combine(prediction)

    def __combine(self, prediction):
        prediction = prediction.copy()
        probability_cols = [
            col for col in prediction.columns if 'probability' in col]
        prediction['probability'] = prediction[probability_cols].mean(
            axis='columns')
        prediction['prediction'] = (
            prediction['probability'] >= self.threshold).round().astype('int')
        return prediction
