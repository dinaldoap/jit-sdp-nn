from jitsdp import metrics
from jitsdp.classifier import Classifier
from jitsdp.constants import DIR
from jitsdp.pipeline import Pipeline
from jitsdp.plot import plot_recalls_gmean
from jitsdp.data import FEATURES, make_stream, save_results, load_results

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


def create_pipeline(config):
    scaler = StandardScaler()
    criterion = nn.BCELoss()
    classifier = Classifier(input_size=len(FEATURES),
                            hidden_size=len(FEATURES) // 2, drop_prob=0.2)
    optimizer = optim.Adam(params=classifier.parameters(), lr=0.003)
    return Pipeline(steps=[scaler], classifier=classifier, optimizer=optimizer, criterion=criterion,
                    features=FEATURES, target='target',
                    max_epochs=config['epochs'], batch_size=512, fading_factor=1, zero_fraction=.6)


def evaluate(label, targets, predictions):
    gmean, recalls = metrics.gmean_recalls(targets, predictions)
    print('{} g-mean: {}, recalls: {}'.format(label, gmean, recalls))


def evaluate_train_test(seq, targets_train, predictions_train, targets_test, predictions_test, targets_unlabeled, predictions_unlabeled):
    print('Sequential: {}'.format(seq))
    evaluate('Train', targets_train, predictions_train)
    evaluate('Test', targets_test, predictions_test)
    evaluate('Unlabeled', targets_unlabeled, predictions_unlabeled)
    train_label_total, train_label_normal, train_label_bug = metrics.proportions(
        targets_train)
    train_pred_total, train_pred_normal, train_pred_bug = metrics.proportions(
        predictions_train)
    test_label_total, test_label_normal, test_label_bug = metrics.proportions(
        targets_test)
    test_pred_total, test_pred_normal, test_pred_bug = metrics.proportions(
        predictions_test)
    unlabeled_total, unlabeled_normal, unlabeled_bug = metrics.proportions(
        predictions_unlabeled)
    print('Train label total: {}, normal: {:.2f}%, bug: {:.2f}%'.format(
        train_label_total, train_label_normal, train_label_bug))
    print('Train pred total: {}, normal: {:.2f}%, bug: {:.2f}%'.format(
        train_pred_total, train_pred_normal, train_pred_bug))
    print('Test label total: {}, normal: {:.2f}%, bug: {:.2f}%'.format(
        test_label_total, test_label_normal, test_label_bug))
    print('Test pred total: {}, normal: {:.2f}%, bug: {:.2f}%'.format(
        test_pred_total, test_pred_normal, test_pred_bug))
    print('Unlabeled pred total: {}, normal: {:.2f}%, bug: {:.2f}%'.format(
        unlabeled_total, unlabeled_normal, unlabeled_bug))


def run(config):
    df_prequential = make_stream(
        'https://raw.githubusercontent.com/dinaldoap/jit-sdp-data/master/{}.csv'.format(config['dataset']))
    # split test partition in folds and iterate over them (fold from current to current + fold_size or end)
    # the previous commits until current are used for training
    seconds_by_day = 24 * 60 * 60
    verification_latency = 90 * seconds_by_day  # seconds
    fold_size = config['fold_size']  # commits
    start = max(config['start'], fold_size) # start with data for training (minimum of one fold)
    interval = len(df_prequential) - start  # last commit
    n_folds = math.ceil(interval / fold_size) # number of folds rounded up
    n_folds = max(math.ceil(n_folds * config['folds']), 1) # use a fraction of folds (minimum of one)
    end = start + n_folds * fold_size  # last fold end

    pipeline = create_pipeline(config)
    pipeline.save()
    first_target_prediction = df_prequential[:start].copy()
    first_target_prediction['prediction'] = [None] * start
    target_prediction = [first_target_prediction]
    for current in range(start, end, fold_size):
        df_train = df_prequential[:current].copy()
        df_test = df_prequential[current:min(current + fold_size, end)].copy()
        # check if fix has been done (bug) or verification latency has passed (normal), otherwise is unlabeled
        train_timestamp = df_train['timestamp'].max()
        df_train['target'] = df_train.apply(lambda row: 1 if row.timestamp_fix <= train_timestamp else (
            0 if row.timestamp <= train_timestamp - verification_latency else None), axis='columns')
        df_unlabeled = df_train[pd.isnull(df_train['target'])]
        df_train = df_train.dropna(subset=['target'])
        df_train['target'] = df_train['target'].astype('int')
        # convert to numpy array
        df_unlabeled['target'] = np.zeros(len(df_unlabeled), dtype=np.int64)
        # train and predict
        pipeline = create_pipeline(config)
        # pipeline.load()
        pipeline.train(df_train, df_unlabeled)
        pipeline.save()
        target_prediction_test = pipeline.predict(df_test)
        target_prediction.append(target_prediction_test)

    target_prediction = pd.concat(target_prediction, sort=False)
    results = metrics.prequential_recalls_gmean(target_prediction, .99)
    save_results(results=results, dir=DIR / config['dataset'])
    report(config)

def report(config):
    results = load_results(dir=DIR / config['dataset'])
    plot_recalls_gmean(results, config=config, dir=DIR)
