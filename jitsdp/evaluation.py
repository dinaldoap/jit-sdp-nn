from jitsdp import metrics
from jitsdp.classifier import Classifier
from jitsdp.pipeline import Pipeline
from jitsdp.plot import plot_recalls_gmean
from jitsdp.data import make_stream, FEATURES

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import random
from scipy.stats import mstats
import math

import logging


logging.getLogger('').handlers = []
logging.basicConfig(filename='logs/mlp.log', filemode='w', level=logging.DEBUG)

df_prequential = make_stream('https://raw.githubusercontent.com/dinaldoap/jit-sdp-data/master/brackets.csv')

def create_pipeline():
    scaler = StandardScaler()
    criterion = nn.BCELoss()
    classifier = Classifier(input_size=len(FEATURES), hidden_size=len(FEATURES) // 2, drop_prob=0.2)
    optimizer = optim.Adam(params=classifier.parameters(), lr=0.003)
    return Pipeline(steps=[scaler], classifier=classifier, optimizer=optimizer, criterion=criterion,
                    max_epochs=50, batch_size=512, fading_factor=1)



def evaluate(label, targets, predictions):
  gmean, recalls = metrics.gmean_recalls(targets, predictions)
  print('{} g-mean: {}, recalls: {}'.format(label, gmean, recalls))

def evaluate_train_test(seq, targets_train, predictions_train, targets_test, predictions_test, targets_unlabeled, predictions_unlabeled):
  print('Sequential: {}'.format(seq))
  evaluate('Train', targets_train, predictions_train)
  evaluate('Test', targets_test, predictions_test)    
  evaluate('Unlabeled', targets_unlabeled, predictions_unlabeled)
  train_label_total, train_label_normal, train_label_bug = metrics.proportions(targets_train)
  train_pred_total, train_pred_normal, train_pred_bug = metrics.proportions(predictions_train)
  test_label_total, test_label_normal, test_label_bug = metrics.proportions(targets_test)
  test_pred_total, test_pred_normal, test_pred_bug = metrics.proportions(predictions_test)
  unlabeled_total, unlabeled_normal, unlabeled_bug = metrics.proportions(predictions_unlabeled)
  print('Train label total: {}, normal: {:.2f}%, bug: {:.2f}%'.format(train_label_total, train_label_normal, train_label_bug))
  print('Train pred total: {}, normal: {:.2f}%, bug: {:.2f}%'.format(train_pred_total, train_pred_normal, train_pred_bug))
  print('Test label total: {}, normal: {:.2f}%, bug: {:.2f}%'.format(test_label_total, test_label_normal, test_label_bug))
  print('Test pred total: {}, normal: {:.2f}%, bug: {:.2f}%'.format(test_pred_total, test_pred_normal, test_pred_bug))
  print('Unlabeled pred total: {}, normal: {:.2f}%, bug: {:.2f}%'.format(unlabeled_total, unlabeled_normal, unlabeled_bug))



# split dataset in chunks for testing and iterate over them (chunk from current to current + interval or end)
# the previous chunks are used for training (chunks from start to current)
seconds_by_day = 24 * 60 * 60
verification_latency = 90 * seconds_by_day # seconds
interval = 500 # commits
end = len(df_prequential) # last commit
n_chunks = math.ceil(end / interval)
end = n_chunks * interval # last chunk end
#start = end - (n_chunks - 1) * interval # start test with second chunk
start = end - interval # use last two chunks to test

pipeline = create_pipeline()
pipeline.save()
predictions = [[None] * start]
for current in range(start, end, interval):
#for current in range(start, start+1, interval):
    df_train = df_prequential[:current].copy()
    df_test = df_prequential[current:min(current + interval, end)].copy()
    # check if fix has been done (bug) or verification latency has passed (normal), otherwise exclude commit
    train_timestamp = df_train['timestamp'].max()
    df_train['target'] = df_train.apply(lambda row: 1 if row.timestamp_fix <= train_timestamp else (0 if row.timestamp <= train_timestamp - verification_latency else None), axis='columns')
    df_unlabeled = df_train[pd.isnull(df_train['target'])]
    df_train = df_train.dropna(subset=['target'])
    df_train['target'] = df_train['target'].astype('int')
    # convert to numpy array
    X_train = df_train[FEATURES].values
    y_train = df_train['target'].values
    X_test = df_test[FEATURES].values
    y_test = df_test['target'].values
    X_unlabeled = df_unlabeled[FEATURES].values
    y_unlabeled = np.zeros(len(X_unlabeled), dtype=np.int64)
    # train and evaluate
    pipeline = create_pipeline()
    #pipeline.load()
    pipeline.train(X_train, y_train)
    pipeline.save()
    predictions_test = pipeline.predict(X_test)
    predictions.append(predictions_test)

predictions = np.concatenate(predictions)
results = df_prequential[['timestep', 'target']].copy()
results['prediction'] = predictions



prequential_recalls = metrics.prequential_recalls_gmean(results, .99)
plot_recalls_gmean(prequential_recalls)