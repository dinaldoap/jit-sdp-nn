from jitsdp import metrics as met
from jitsdp.data import make_stream, save_results, DATASETS, FEATURES
from jitsdp.orb import ORB
from jitsdp.pipeline import set_seed
from jitsdp.report import report
from jitsdp.utils import int_or_none, unique_dir, setup_and_run

import argparse
from datetime import datetime
import logging
import mlflow
import pathlib
import pandas as pd
import numpy as np
import sys
from skmultiflow.data import DataStream


def main():
    parser = argparse.ArgumentParser(
        description='Baseline: experiment execution')
    parser.add_argument('--experiment-name',   type=str,
                        help='Experiment name (default: None). None means default behavior of MLflow', default=None)
    parser.add_argument('--start',   type=int,
                        help='First commit to be used for testing (default: 0).',    default=0)
    parser.add_argument('--end',   type=int_or_none,
                        help='Last commit to be used for testing (default: None). None means all commits.',  default=5000)
    parser.add_argument('--pool-size',   type=int,
                        help='Number of processes used to run the experiment in parallel (default: 1).', default=1)
    parser.add_argument('--orb-decay-factor',   type=float,
                        help='Decay factor for calculating class proportions in training data (default: .99).',  default=.99)
    parser.add_argument('--orb-ma-window-size',   type=int,
                        help='The number of predictions or instances used for calculating moving average (default: 100).',  default=100)
    parser.add_argument('--orb-th',   type=float,
                        help='Expected value for the moving average of the model\'s output (default: .4).',  default=.99)
    parser.add_argument('--orb-l0',   type=float,
                        help='No description (default: 10.).',  default=10.)
    parser.add_argument('--orb-l1',   type=float,
                        help='No description (default: 12.).',  default=12.)
    parser.add_argument('--orb-m',   type=float,
                        help='No description (default: 1.5).',  default=1.5)
    parser.add_argument('--orb-rd',   type=int,
                        help='Whether must turn ORB rate-driven (default: 0).',
                        default=0, choices=[0, 1])
    parser.add_argument('--orb-rd-max-wait',   type=int,
                        help='The number of instances the model is trained before fully updating the moving average window (default: 300).',
                        default=300)
    parser.add_argument('--cross-project',   type=int,
                        help='Whether must use cross-project data (default: 0).', default=0, choices=[0, 1])
    parser.add_argument('--noise',   type=int,
                        help='Whether must keep noisy instances (default: 0).',
                        default=0, choices=[0, 1])
    parser.add_argument('--order',   type=int,
                        help='Whether must keep the order of the events (default: 0).',
                        default=0, choices=[0, 1])
    parser.add_argument('--seeds',   type=int,
                        help='Seeds of random state (default: [0]).',    default=[0], nargs='+')
    parser.add_argument('--datasets',   type=str, help='Datasets to run the experiment. (default: [\'brackets\']).',
                        default=['brackets'], choices=['brackets', 'camel', 'fabric8', 'jgroups', 'neutron', 'tomcat', 'broadleaf', 'nova', 'npm', 'spring-integration'], nargs='+')
    parser.add_argument('--models',   type=str,
                        help='Which models must use in the ensemble (default: [\'hts\']).', default=['hts'], choices=['hts'], nargs='+')
    parser.add_argument('--hts-n-estimators',   type=int,
                        help='The number of hoeffding trees (default: 1).',  default=1)
    parser.add_argument('--track-time',   type=int,
                        help='Whether must track time. (default: 0).',  default=0)
    parser.add_argument('--track-ihf',   type=int,
                        help='Whether must track iHF state (default: 0)',  default=0)
    parser.add_argument('--track-orb',   type=int,
                        help='Whether must track ORB state (default: 0)',  default=0)
    setup_and_run(parser, 'baseline', run)


def run(config):
    mlflow.log_params(config)
    set_seed(config)
    dataset = config['dataset']
    # stream with commit order
    df_commit = make_stream(dataset)
    # stream with labeling order
    df_test = df_commit.copy()
    end = len(df_test) if config['end'] is None else config['end']
    df_test = df_test[config['start']:end]
    df_train = df_commit.copy()
    df_train = df_train[:end]
    df_train = extract_events(df_train)
    if not config['noise']:
        df_train = remove_noise(df_train)
    if not config['order']:
        df_train = balance_events(df_train)

    test_steps = calculate_steps(
        df_test['timestamp'], df_train['timestamp_event'], right=False)
    train_steps = calculate_steps(
        df_train['timestamp_event'], df_test['timestamp'], right=True)
    train_steps = train_steps.to_list()

    train_stream = DataStream(df_train[FEATURES], y=df_train[['target']])
    model = ORB(features=FEATURES,
                decay_factor=config['orb_decay_factor'],
                ma_window_size=config['orb_ma_window_size'],
                th=config['orb_th'],
                l0=config['orb_l0'],
                l1=config['orb_l1'],
                m=config['orb_m'],
                n_estimators=config['hts_n_estimators'],
                )
    target_prediction = None
    train_first = len(test_steps) < len(train_steps)
    current_test = 0
    for test_index, test_step in test_steps.items():
        # train
        if train_first:
            train_step = train_steps.pop(0)
            X_train, y_train = train_stream.next_sample(train_step)
            model.train(
                X_train, y_train, rd=config['orb_rd'], rd_max_wait=config['orb_rd_max_wait'], track_orb=config['track_orb'])
        else:
            train_first = True
        # test
        df_batch_test = df_test[current_test:current_test + test_step]
        current_test += test_step
        target_prediction_test = model.predict(
            df_batch_test, rd=config['orb_rd'], track_time=config['track_time'], track_ihf=config['track_ihf'])
        target_prediction = pd.concat(
            [target_prediction, target_prediction_test])

    target_prediction = target_prediction.reset_index(drop=True)

    results = met.prequential_metrics(target_prediction, .99)
    save_results(results=results, dir=unique_dir(config))
    report(config)


def extract_events(df_commit):
    seconds_by_day = 24 * 60 * 60
    # seconds
    verification_latency = 90 * seconds_by_day
    # cleaned
    df_clean = df_commit[df_commit['target'] == 0]
    df_cleaned = df_clean.copy()
    df_cleaned['timestamp_event'] = df_cleaned['timestamp'] + \
        verification_latency
    # bugged
    df_bug = df_commit[df_commit['target'] == 1]
    df_bugged = df_bug.copy()
    df_bugged['timestamp_event'] = df_bugged['timestamp_fix'].astype(int)
    # bug cleaned
    df_bug_cleaned = df_bug.copy()
    waited_time = df_bug_cleaned['timestamp_fix'] - df_bug_cleaned['timestamp']
    df_bug_cleaned = df_bug_cleaned[waited_time >= verification_latency]
    df_bug_cleaned['target'] = 0
    df_bug_cleaned['timestamp_event'] = df_bug_cleaned['timestamp'] + \
        verification_latency
    # events
    df_events = pd.concat([df_cleaned, df_bugged, df_bug_cleaned])
    df_events = df_events.sort_values('timestamp_event')
    df_events = df_events[['timestamp_event'] + FEATURES + ['target']]
    return df_events


def remove_noise(df_events):
    grouped_target = df_events.groupby(FEATURES)['target']
    cumsum = grouped_target.cumsum()
    cumcount = grouped_target.cumcount()
    previous_clean = 3
    noise = cumcount - cumsum >= previous_clean
    noise = noise & (df_events['target'] == 1)
    return df_events[~noise]


def balance_events(df_events):
    cumsum = df_events['target'].cumsum()
    cumcount = np.array(range(len(df_events))) + 1
    cumprop = cumsum / cumcount
    timestamp_event_balance = df_events.loc[cumprop == .5, 'timestamp_event'].min(
    )
    # order balanced
    df_balanced = df_events[df_events['timestamp_event']
                            <= timestamp_event_balance]
    df_clean = df_balanced[df_balanced['target'] == 0].copy()
    df_bug = df_balanced[df_balanced['target'] == 1].copy()
    # assert balance
    assert len(df_clean) == len(df_bug)
    df_bug['timestamp_event'] = df_clean['timestamp_event'].values
    df_balanced = pd.concat([df_clean, df_bug])
    df_balanced = df_balanced.sort_values('timestamp_event', kind='mergesort')
    # order kept
    df_kept = df_events[df_events['timestamp_event'] > timestamp_event_balance]
    return pd.concat([df_balanced, df_kept])


def calculate_steps(data, bins, right):
    min_max = pd.concat([data[:1] - int(right), data[-1:] + int(not right)])
    internal_bins = bins[(min_max.min() < bins) & (bins < min_max.max())]
    full_bins = pd.concat([min_max[:1], internal_bins, min_max[-1:]])
    full_bins = full_bins.drop_duplicates()
    steps = pd.cut(data, bins=full_bins, right=right, include_lowest=True)
    steps = steps.value_counts(sort=False)
    steps = steps[steps > 0]
    return steps