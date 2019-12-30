import numpy as np
import torch
import torch.utils.data as data
from sklearn.model_selection import train_test_split
from jitsdp import metrics


class Pipeline:

    def __init__(self, steps, classifier, optimizer, criterion, max_epochs, fading_factor):
        self.steps = steps
        self.classifier = classifier
        self.optimizer = optimizer
        self.criterion = criterion
        self.max_epochs = max_epochs
        self.fading_factor = fading_factor

    def train(self, X, y):
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.1, shuffle=False)
        sampled_train_dataloader = self.__dataloader(
            X_train, y_train, batch_size=512, sampler=self.__sampler(y_train))
        train_dataloader = self.__dataloader(X_train, y_train)
        val_dataloader = self.__dataloader(X_val, y_val)

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
            train_gmean = metrics.gmean(self.classifier, train_dataloader)
            val_gmean = metrics.gmean(self.classifier, val_dataloader)
            print('Epoch: {}, Train loss: {}, Train g-mean: {}, Val g-mean: {}'.format(epoch,
                                                                                       train_loss, train_gmean, val_gmean))

            if self.classifier.val_gmean is None or val_gmean > self.classifier.val_gmean:
                self.classifier.epoch = epoch
                self.classifier.val_gmean = val_gmean
                self.classifier.save()

        self.classifier.epoch = epoch
        self.classifier.val_gmean = val_gmean

    def evaluate(self, X, y):
        dataloader = self.__dataloader(X, y)
        return metrics.gmean_recalls(self.classifier, dataloader)

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

    @property
    def epoch(self):
        return self.classifier.epoch

    def load(self):
        self.classifier.load()
