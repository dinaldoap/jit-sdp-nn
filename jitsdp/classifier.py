from jitsdp.utils import mkdir

import torch
from torch import nn
import pathlib


class Classifier(nn.Module):
    DIR = pathlib.Path('models')
    FILENAME = DIR / 'classifier.cpt'

    def __init__(self, input_size, hidden_size, drop_prob, epoch=None, val_loss=None):
        super(Classifier, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.drop_prob = drop_prob
        self.epoch = epoch
        self.val_loss = val_loss
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fcout = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fcout(x))
        return x

    def save(self):
        mkdir(Classifier.DIR)
        checkpoint = {
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'drop_prob': self.drop_prob,
            'val_loss': self.val_loss,
            'epoch': self.epoch,
            'state_dict': self.state_dict()
        }
        with open(Classifier.FILENAME, 'wb') as f:
            torch.save(checkpoint, f)

    def load(self):
        with open(Classifier.FILENAME, 'rb') as f:
            checkpoint = torch.load(f)
            self.input_size = checkpoint['input_size']
            self.hidden_size = checkpoint['hidden_size']
            self.drop_prob = checkpoint['drop_prob']
            self.epoch = checkpoint['epoch']
            self.val_loss = checkpoint['val_loss']
            self.load_state_dict(checkpoint['state_dict'])


if __name__ == '__main__':
    classifier = Classifier(
        input_size=1, hidden_size=1, drop_prob=0.5)
    print(classifier)
    classifier.save()
    classifier.load()
    print(classifier)
