# coding=utf-8
from jitsdp import metrics as met
from jitsdp.mlp import MLP
from jitsdp.data import FEATURES
from jitsdp.utils import mkdir

from abc import ABCMeta, abstractmethod
import joblib
import logging
import time
import numpy as np
import pandas as pd
import pathlib
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import NotFittedError

logger = logging.getLogger(__name__)


def set_seed(config):
    seed = config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)


def create_pipeline(config):
    map_fn = {
        'mlp': create_mlp_model,
        'nb': create_nb_model,
        'rf': create_rf_model,
        'lr': create_lr_model,
        'svm': create_svm_model,
    }
    fn_create_model = map_fn[config['model']]
    if config['ensemble_size'] > 1:
        models = [fn_create_model(config)
                  for i in range(config['ensemble_size'])]
        model = Ensemble(models=models)
    else:
        model = fn_create_model(config)
    if config['threshold'] == 1:
        classifier = RateFixed(
            model=model, normal_proportion=(1 - config['borb_th']))
    elif config['threshold'] == 2:
        classifier = RateFixedTrain(
            model=model)
    else:
        classifier = ScoreFixed(model=model)
    if config['borb']:
        classifier = ORB(classifier=classifier,
                         max_sample_size=config['borb_max_sample_size'],
                         th=config['borb_th'],
                         l0=config['borb_l0'],
                         l1=config['borb_l1'],
                         m=config['borb_m'])
    return classifier


def create_mlp_model(config):
    scaler = StandardScaler()
    criterion = nn.BCEWithLogitsLoss()
    classifier = MLP(input_layer_size=len(FEATURES),
                     n_hidden_layers=config['mlp_n_hidden_layers'],
                     hidden_layers_size=config['mlp_hidden_layers_size'],
                     dropout_input_layer=config['mlp_dropout_input_layer'],
                     dropout_hidden_layers=config['mlp_dropout_hidden_layers'])
    optimizer = optim.Adam(params=classifier.parameters(),
                           lr=config['mlp_learning_rate'])
    return PyTorch(steps=[scaler], classifier=classifier, optimizer=optimizer, criterion=criterion,
                   features=FEATURES, target='target', soft_target='soft_target',
                   max_epochs=config['mlp_n_epochs'], batch_size=config['mlp_batch_size'], fading_factor=1)


def create_nb_model(config):
    classifier = GaussianNB()
    return NaiveBayes(steps=[], classifier=classifier,
                      features=FEATURES, target='target', soft_target='soft_target',
                      n_updates=config['nb_n_updates'], fading_factor=1)


def create_rf_model(config):
    classifier = RandomForestClassifier(
        n_estimators=0, criterion=config['rf_criterion'], max_depth=config['rf_max_depth'], min_samples_leaf=config['rf_min_samples_leaf'], max_features=config['rf_max_features'], min_impurity_decrease=config['rf_min_impurity_decrease'], warm_start=True, bootstrap=False)
    return RandomForest(steps=[], classifier=classifier,
                        features=FEATURES, target='target', soft_target='soft_target',
                        n_trees=config['rf_n_estimators'], fading_factor=1)


def create_lr_model(config):
    scaler = StandardScaler()
    classifier = SGDClassifier(loss='log', penalty='elasticnet',
                               alpha=config['lr_alpha'], l1_ratio=config['lr_l1_ratio'], shuffle=False)
    return LogisticRegression(n_epochs=config['lr_n_epochs'], steps=[scaler], classifier=classifier,
                              features=FEATURES, target='target', soft_target='soft_target',
                              batch_size=config['lr_batch_size'], fading_factor=1)


def create_svm_model(config):
    scaler = StandardScaler()
    classifier = SGDClassifier(loss='hinge', penalty='elasticnet',
                               alpha=config['svm_alpha'], l1_ratio=config['svm_l1_ratio'], shuffle=False)
    return LogisticRegression(n_epochs=config['svm_n_epochs'], steps=[scaler], classifier=classifier,
                              features=FEATURES, target='target', soft_target='soft_target',
                              batch_size=config['svm_batch_size'], fading_factor=1)


class Model(metaclass=ABCMeta):
    @abstractmethod
    def train(self, df_train, **kwargs):
        pass

    @abstractmethod
    def predict_proba(self, df_features, **kwargs):
        pass

    @abstractmethod
    def save(self):
        pass

    @abstractmethod
    def load(self):
        pass

    @property
    @abstractmethod
    def n_iterations(self):
        pass


class Classifier(Model):
    @abstractmethod
    def predict(self, df_features, **kwargs):
        pass


class Threshold(Classifier):
    def __init__(self, model):
        self.model = model

    def train(self, df_train, **kwargs):
        for metrics in self.model.train(df_train, **kwargs):
            yield _track_performance(metrics=metrics,
                                     classifier=self, df_train=df_train, **kwargs)

    def predict_proba(self, df_features, **kwargs):
        prediction = self.model.predict_proba(df_features, **kwargs)
        if kwargs.pop('track_time', 0):
            prediction = _track_time(prediction)
        return prediction

    def save(self):
        self.model.save()

    def load(self):
        self.model.load()

    @property
    def n_iterations(self):
        return self.model.n_iterations


class ScoreFixed(Threshold):
    def __init__(self, model, score=.5):
        super().__init__(model=model)
        self.score = score

    def predict(self, df_features, **kwargs):
        prediction = self.predict_proba(df_features=df_features, **kwargs)
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
            df_threshold, **kwargs)['probability'] if df_threshold is not None else None
        prediction = self.predict_proba(df_features=df_features, **kwargs)
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
        threshold_probabilities = self.predict_proba(df_threshold, **kwargs)[
            'probability']
        prediction = self.predict_proba(df_features=df_features)
        threshold = _tune_threshold(val_probabilities=threshold_probabilities,
                                    test_probabilities=prediction['probability'], normal_proportion=normal_proportion)
        threshold = threshold.values
        prediction['prediction'] = (
            prediction['probability'] >= threshold).round().astype('int')
        return prediction


class ORB(Classifier):
    def __init__(self, classifier, max_sample_size, th, l0, l1, m):
        self.classifier = classifier
        self.max_sample_size = max_sample_size
        self.th = th
        self.l0 = l0
        self.l1 = l1
        self.m = m

    def train(self, df_train, **kwargs):
        df_ma = kwargs.pop('df_ma', None)
        ma = self.th
        for i in range(self.classifier.n_iterations):
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
            new_kwargs['n_iterations'] = 1
            new_kwargs['max_sample_size'] = self.max_sample_size
            for metrics in self.classifier.train(df_train, **new_kwargs):
                yield _track_orb(metrics=metrics, ma=ma, obf0=obf0, obf1=obf1, **kwargs)
            df_output = self.classifier.predict(df_ma)
            ma = df_output['prediction'].mean()

    def predict(self, df_features, **kwargs):
        return self.classifier.predict(df_features, **kwargs)

    def predict_proba(self, df_features, **kwargs):
        return self.classifier.predict_proba(df_features, **kwargs)

    def save(self):
        self.classifier.save()

    def load(self):
        self.classifier.load()

    @property
    def n_iterations(self):
        return self.classifier.n_iterations


class PyTorch(Model):
    DIR = pathlib.Path('models')
    FILENAME = DIR / 'steps.cpt'

    def __init__(self, steps, classifier, optimizer, criterion, features, target, soft_target, max_epochs, batch_size, fading_factor):
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
        self.trained = False

    @property
    def n_iterations(self):
        return self.max_epochs

    def train(self, df_train, **kwargs):
        try:
            sampled_train_dataloader = _prepare_dataloaders(
                df_train, self.features, self.target, self.soft_target, self.batch_size, self.fading_factor, self.steps, **kwargs)
        except ValueError as e:
            logger.warning(e)
            return

        if torch.cuda.is_available():
            self.classifier = self.classifier.cuda()

        n_iterations = kwargs.pop('n_iterations', self.n_iterations)
        for epoch in range(n_iterations):
            self.classifier.train()
            for inputs, targets in sampled_train_dataloader:
                if torch.cuda.is_available():
                    inputs, targets = inputs.cuda(), targets.cuda()

                outputs = self.classifier(inputs.float())
                loss = self.criterion(outputs.view(
                    outputs.shape[0]), targets.float())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            self.trained = True
            yield

    def predict_proba(self, df_features, **kwargs):
        if self.trained:
            X = df_features[self.features].values
            X = _steps_transform(self.steps, X)
            y = np.zeros(len(X))
            dataloader = _dataloader(X, y)

            if torch.cuda.is_available():
                self.classifier = self.classifier.cuda()

            probabilities = []
            with torch.no_grad():
                self.classifier.eval()
                for inputs, targets in dataloader:
                    if torch.cuda.is_available():
                        inputs, targets = inputs.cuda(), targets.cuda()

                    outputs = self.classifier.forward_proba(inputs.float())
                    probabilities.append(outputs.detach().cpu().numpy())
            probabilities = np.concatenate(probabilities)
        else:
            probabilities = np.zeros(len(df_features))

        probability = df_features.copy()
        probability['probability'] = probabilities
        return probability

    def load(self):
        state = joblib.load(PyTorch.FILENAME)
        self.steps = state['steps']
        self.trained = state['trained']
        self.classifier.load()

    def save(self):
        mkdir(PyTorch.DIR)
        state = {
            'steps': self.steps,
            'trained': self.trained,
        }
        joblib.dump(state, PyTorch.FILENAME)
        self.classifier.save()


def _prepare_dataloaders(df_train, features, target, soft_target, batch_size, fading_factor, steps, **kwargs):
    X_train = df_train[features].values
    y_train = df_train[target].values
    soft_y_train = df_train[soft_target].values
    classes = np.unique(y_train)
    if len(classes) != 2:
        raise ValueError('It is expected two classes to train.')

    X_train = _steps_fit_transform(steps, X_train, y_train)

    sampled_train_dataloader = _dataloader(
        X_train, soft_y_train, batch_size=batch_size, sampler=_sampler(y_train, fading_factor, **kwargs))

    return sampled_train_dataloader


def _tensor(X, y):
    return torch.from_numpy(X), torch.from_numpy(y)


def _dataloader(X, y, batch_size=32, sampler=None):
    X, y = _tensor(X, y)
    dataset = data.TensorDataset(X, y)
    return data.DataLoader(dataset, batch_size=batch_size, sampler=sampler)


def _sampler(y, fading_factor, **kwargs):
    weights = kwargs.pop('weights', [1, 1])
    normal_indices = np.flatnonzero(y == 0)
    bug_indices = np.flatnonzero(y == 1)
    age_weights = np.zeros(len(y))
    # normal commit ages
    age_weights[normal_indices] = _fading_weights(
        size=len(normal_indices), fading_factor=fading_factor, total=weights[0])
    # bug commit doesn't age
    age_weights[bug_indices] = _fading_weights(
        size=len(bug_indices), fading_factor=fading_factor, total=weights[1])
    max_sample_size = kwargs.pop('max_sample_size', None)
    num_samples = len(y) if max_sample_size is None else min(
        len(y), max_sample_size)
    return data.WeightedRandomSampler(weights=age_weights, num_samples=num_samples, replacement=True)


def _fading_weights(size, fading_factor, total):
    fading_weights = reversed(range(size))
    fading_weights = [fading_factor**x for x in fading_weights]
    fading_weights = np.array(fading_weights)
    return (total * fading_weights) / np.sum(fading_weights)


def _steps_fit_transform(steps, X, y):
    for step in steps:
        X = step.fit_transform(X, y)
    return X


def _steps_transform(steps, X):
    for step in steps:
        try:
            X = step.transform(X)
        except NotFittedError:
            logger.warning('Step {} not fitted.'.format(step))
    return X


class Scikit(Model):

    def __init__(self, steps, classifier, features, target, soft_target, fading_factor, batch_size):
        super().__init__()
        self.steps = steps
        self.classifier = classifier
        self.features = features
        self.target = target
        self.soft_target = soft_target
        self.batch_size = batch_size
        self.fading_factor = fading_factor
        self.trained = False

    def train(self, df_train, **kwargs):
        batch_size = self.batch_size if self.batch_size is not None else len(
            df_train)
        try:
            sampled_train_dataloader = _prepare_dataloaders(
                df_train, self.features, self.target, self.soft_target, batch_size, self.fading_factor, self.steps, **kwargs)
        except ValueError as e:
            logger.warning(e)
            return

        n_iterations = kwargs.pop('n_iterations', self.n_iterations)
        sampled_classes = set()
        for i in range(n_iterations):
            for inputs, targets in sampled_train_dataloader:
                inputs, targets = inputs.numpy(), targets.numpy()
                sampled_classes.update(targets)
                self.train_iteration(inputs=inputs, targets=targets)

            if len(sampled_classes) == 2:
                self.trained = True
            yield

    @abstractmethod
    def train_iteration(self, inputs, targets):
        pass

    def predict_proba(self, df_features, **kwargs):
        if self.trained:
            X = df_features[self.features].values
            X = _steps_transform(self.steps, X)

            try:
                probabilities = self.classifier.predict_proba(X)
                probabilities = probabilities[:, 1]
            except AttributeError:
                probabilities = self.classifier.predict(X)
        else:
            probabilities = np.zeros(len(df_features))

        probability = df_features.copy()
        probability['probability'] = probabilities
        return probability

    def load(self):
        state = joblib.load(PyTorch.FILENAME)
        self.steps = state['steps']
        self.classifier = state['classifier']
        self.val_loss = state['val_loss']

    def save(self):
        mkdir(PyTorch.DIR)
        state = {'steps': self.steps,
                 'classifier': self.classifier,
                 'val_loss': self.val_loss, }
        joblib.dump(state, PyTorch.FILENAME)


class NaiveBayes(Scikit):

    def __init__(self, steps, classifier, features, target, soft_target, fading_factor, n_updates):
        super().__init__(steps=steps, classifier=classifier, features=features, target=target,
                         soft_target=soft_target, fading_factor=fading_factor, batch_size=None)
        self.n_updates = n_updates

    @property
    def n_iterations(self):
        return self.n_updates

    def train_iteration(self, inputs, targets):
        self.classifier.partial_fit(
            inputs, targets, classes=[0, 1])


class RandomForest(Scikit):

    def __init__(self, steps, classifier, features, target, soft_target, fading_factor, n_trees):
        super().__init__(steps=steps, classifier=classifier, features=features, target=target,
                         soft_target=soft_target, fading_factor=fading_factor, batch_size=None)
        self.n_trees = n_trees

    @property
    def n_iterations(self):
        return self.n_trees

    def train_iteration(self, inputs, targets):
        if len(np.unique(targets)) != 2:
            return
        self.classifier.n_estimators += 1
        self.classifier.fit(
            inputs, targets)

    def predict_proba(self, df_features, **kwargs):
        prediction = super().predict_proba(df_features, **kwargs)
        if kwargs.pop('track_rf', 0):
            prediction = _track_rf(prediction, self)
        return prediction


class LogisticRegression(Scikit):

    def __init__(self, steps, classifier, features, target, soft_target, fading_factor, n_epochs, batch_size):
        super().__init__(steps=steps, classifier=classifier, features=features, target=target,
                         soft_target=soft_target, fading_factor=fading_factor, batch_size=batch_size)
        self.n_epochs = n_epochs

    @property
    def n_iterations(self):
        return self.n_epochs

    def train_iteration(self, inputs, targets):
        self.classifier.partial_fit(
            inputs, targets, classes=[0, 1])


class Ensemble(Model):
    def __init__(self, models):
        super().__init__()
        self.models = models

    def n_iterations(self):
        return self.models[0].n_iterations

    def train(self, df_train, **kwargs):
        for model in self.models:
            model.train(df_train, **kwargs)

    def predict_proba(self, df_features, **kwargs):
        probability = df_features
        for index, model in enumerate(self.models):
            probability = model.predict_proba(probability, **kwargs)
            probability = probability.rename({
                'probability': 'probability{}'.format(index),
            },
                axis='columns')
        return _combine(probability)

    def save(self):
        for model in self.models:
            model.save()

    def load(self):
        for model in self.models:
            model.load()


def _combine(prediction):
    prediction = prediction.copy()
    probability_cols = [
        col for col in prediction.columns if 'probability' in col]
    prediction['probability'] = prediction[probability_cols].mean(
        axis='columns')
    return prediction


def _track_performance(metrics, classifier, df_train, **kwargs):
    df_val = kwargs.pop('df_val', None)
    if df_val is not None:
        train_prediction = classifier.predict(df_train)
        val_prediction = classifier.predict(df_val)
        train_loss = met.loss(train_prediction)
        train_gmean = met.gmean(
            train_prediction)
        val_gmean = met.gmean(
            val_prediction)
        metrics = _prepare_metrics(metrics)
        metrics.update({
            'train_loss': train_loss,
            'train_gmean': train_gmean,
            'val_gmean': val_gmean,
        })
        return metrics


def _track_orb(metrics, ma, obf0, obf1, **kwargs):
    df_val = kwargs.pop('df_val', None)
    if df_val is not None:
        metrics = _prepare_metrics(metrics)
        metrics.update({
            'ma': ma,
            'obf0': obf0,
            'obf1': obf1,
        })
        return metrics


def _prepare_metrics(metrics):
    return {} if metrics is None else dict(metrics)


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
        return [func(estimator) for estimator in rf.classifier.estimators_]
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