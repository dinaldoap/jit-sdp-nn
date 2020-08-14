# coding=utf-8
from jitsdp.utils import mkdir

import torch
from torch import nn
import pathlib


class MLP(nn.Module):
    DIR = pathlib.Path('models')
    FILENAME = DIR / 'classifier.cpt'

    def __init__(self, input_size, n_hidden_layers, hidden_layers_size, drop_prob_input, drop_prob_hidden, val_loss=None):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.hidden_layers_size = hidden_layers_size
        self.drop_prob_input = drop_prob_input
        self.drop_prob_hidden = drop_prob_hidden
        self.val_loss = val_loss
        self.fcs = nn.ModuleList(
            [nn.Linear(input_size, hidden_layers_size), nn.ReLU(), nn.Dropout(drop_prob_hidden)])
        for i in range(n_hidden_layers - 1):
            self.fcs.extend([nn.Linear(hidden_layers_size, hidden_layers_size),
                             nn.ReLU(), nn.Dropout(drop_prob_hidden)])
        self.fcout = nn.Linear(hidden_layers_size, 1)
        self.dropout_input = nn.Dropout(drop_prob_input)

    def forward(self, x):
        x = self.dropout_input(x)
        for func in self.fcs:
            x = func(x)
        return self.fcout(x)

    def forward_proba(self, x):
        x = self.forward(x)
        return torch.sigmoid(x)

    def save(self):
        mkdir(MLP.DIR)
        checkpoint = {
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'drop_prob_input': self.drop_prob_input,
            'drop_prob_hidden': self.drop_prob_hidden,
            'val_loss': self.val_loss,
            'state_dict': self.state_dict()
        }
        with open(MLP.FILENAME, 'wb') as f:
            torch.save(checkpoint, f)

    def load(self):
        with open(MLP.FILENAME, 'rb') as f:
            checkpoint = torch.load(f)
            self.input_size = checkpoint['input_size']
            self.hidden_size = checkpoint['hidden_size']
            self.drop_prob_input = checkpoint['drop_prob_input']
            self.drop_prob_hidden = checkpoint['drop_prob_hidden']
            self.val_loss = checkpoint['val_loss']
            self.load_state_dict(checkpoint['state_dict'])
