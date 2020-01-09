import numpy as np
import torch
import torch.utils.data as data
from sklearn.model_selection import train_test_split
from jitsdp import metrics

import logging
logger = logging.getLogger(__name__)


class Pipeline:

    def __init__(self, steps, classifier, optimizer, criterion, max_epochs, fading_factor, val_size=0.0):
        self.steps = steps
        self.classifier = classifier
        self.optimizer = optimizer
        self.criterion = criterion
        self.max_epochs = max_epochs
        self.fading_factor = fading_factor
        self.val_size = val_size

    def train(self, X, y):
        if self.has_validation():
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=self.val_size, shuffle=False)
            val_dataloader = self.__dataloader(X_val, y_val)
        else:
            X_train, y_train = X, y

        X_train = self.__steps_fit_transform(X_train, y_train)

        sampled_train_dataloader = self.__dataloader(
            X_train, y_train, batch_size=512, sampler=self.__sampler(y_train))
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
                loss = self.criterion(outputs.squeeze(), targets.float())
                train_loss += loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            train_loss = train_loss / len(sampled_train_dataloader)
            train_gmean = metrics.classifier_gmean(self.classifier, train_dataloader)
            val_gmean = None
            if self.has_validation():
                val_gmean = metrics.classifier_gmean(self.classifier, val_dataloader)
                # Best classifier
                if self.classifier.val_gmean is None or val_gmean > self.classifier.val_gmean:
                    self.classifier.epoch = epoch
                    self.classifier.val_gmean = val_gmean
                    self.classifier.save()

            logger.debug('Epoch: {}, Train loss: {}, Train g-mean: {}, Val g-mean: {}'.format(epoch,
                                                                                              train_loss, train_gmean, val_gmean))
        # Last classifier
        self.classifier.epoch = epoch
        self.classifier.val_gmean = val_gmean
        if not self.has_validation():
            self.classifier.save()

    def predict(self, X):
        X = self.__steps_transform(X)
        y = np.zeros(len(X))        
        dataloader = self.__dataloader(X, y)
        y_hat = []
        with torch.no_grad():
            self.classifier.eval()
            for inputs, targets in dataloader:
                if torch.cuda.is_available():
                    inputs, targets = inputs.cuda(), targets.cuda()

                outputs = self.classifier(inputs.float())
                predictions = torch.round(outputs).int()
                predictions = predictions.view(predictions.shape[0])
                y_hat.append(predictions.detach().cpu().numpy())

        return np.concatenate(y_hat)

    def __tensor(self, X, y):
        return torch.from_numpy(X), torch.from_numpy(y)

    def __dataloader(self, X, y, batch_size=32, sampler=None):
        X, y = self.__tensor(X, y)
        dataset = data.TensorDataset(X, y)
        return data.DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    def __sampler(self, y):
        n_samples = len(y)
        fading_weights = self.__fading_weights(n_samples)

        total = np.sum(fading_weights)
        bug = np.sum(fading_weights * y)
        normal = total - bug
        class_weights = total / [normal, bug]
        class_weights = class_weights[y]

        weights = fading_weights * class_weights
        return data.WeightedRandomSampler(weights=weights, num_samples=n_samples, replacement=True)

    def __fading_weights(self, size):
        fading_weights = reversed(range(size))
        fading_weights = [self.fading_factor**x for x in fading_weights]
        return np.array(fading_weights)

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
        self.classifier.load()

    def save(self):
        self.classifier.save()
