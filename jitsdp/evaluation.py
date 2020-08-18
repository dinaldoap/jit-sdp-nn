# coding=utf-8
from jitsdp import metrics as met
from jitsdp.constants import DIR
from jitsdp.data import make_stream, save_results, load_results, DATASETS
from jitsdp.pipeline import create_pipeline, set_seed
from jitsdp.plot import plot_recalls_gmean, plot_proportions

import math
import mlflow
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


def run(config):
    mlflow.log_params(config)
    set_seed(config)
    dataset = config['dataset']
    df_prequential = make_stream(dataset)
    # split test partition in folds and iterate over them (fold from current to current + fold_size or end)
    # the previous commits until current are used for training
    seconds_by_day = 24 * 60 * 60
    # seconds
    verification_latency = config['borb_waiting_time'] * seconds_by_day
    # each fold has pull_request_size commits
    fold_size = config['borb_pull_request_size']
    start = config['start']
    end = len(df_prequential) if config['end'] is None else config['end']
    assert start < end, 'start must be lesser than end.'
    interval = end - start
    n_folds = math.ceil(interval / fold_size)  # number of folds rounded up
    end = start + n_folds * fold_size  # last fold end

    if config['f_val'] > 0:
        folds_by_step = max(n_folds // 4, 1)
        step = folds_by_step * fold_size
        n_steps = min(3, n_folds)
        start = end - n_steps * step - fold_size
    else:
        step = fold_size

    if config['cross_project']:
        others = list(DATASETS)
        others.remove(dataset)
        df_others = [make_stream(dataset) for dataset in others]
        df_others = pd.concat(df_others)
    else:
        # empty df with same schema
        df_others = df_prequential.head(0).copy()

    pipeline = create_pipeline(config)
    if config['incremental']:
        pipeline.save()
    target_prediction = None
    update_step = 0
    for current in range(start, end, step):
        df_train = df_prequential[:current].copy()
        df_test = df_prequential[current:min(current + step, end)].copy()
        df_train, df_tail = __prepare_tail_data(df_train, config)

        df_train = __prepare_train_data(
            df_train, df_others, verification_latency, config)

        df_train, df_val = __prepare_val_data(df_train, config)
        # train and predict
        pipeline = create_pipeline(config)
        if config['incremental']:
            pipeline.load()
        for metrics in pipeline.train(df_train, df_ma=df_tail, df_val=df_val):
            if __has_metrics(metrics):
                mlflow.log_metrics(metrics=metrics, step=update_step)
            update_step += 1
        if config['incremental']:
            pipeline.save()
        target_prediction_test = pipeline.predict(
            df_test, df_threshold=df_tail, df_proportion=df_train, track_rf=config['track_rf'], track_time=config['track_time'])
        target_prediction = pd.concat(
            [target_prediction, target_prediction_test])

    target_prediction = target_prediction.reset_index(drop=True)
    results = met.prequential_metrics(target_prediction, .99)
    save_results(results=results, dir=__unique_dir(config))
    report(config)


def report(config):
    dir = __unique_dir(config)
    results = load_results(dir=dir)
    plot_recalls_gmean(results, config=config, dir=dir)
    plot_proportions(results, config=config, dir=dir)
    metrics = ['r0', 'r1', 'r0-r1', 'gmean', 't1', 's1', 'p1']
    metrics = {'avg_{}'.format(
        metric): results[metric].mean() for metric in metrics}
    mlflow.log_metrics(metrics)
    mlflow.log_artifacts(local_dir=dir)


def __unique_dir(config):
    return DIR / '{}_{}_{}'.format(config['seed'], config['dataset'], config['model'])


def __verification_latency_label(train_timestamp, commit_timestamp, verification_latency):
    return .5 - .5 * (train_timestamp - commit_timestamp) / verification_latency


def __prepare_tail_data(df_train, config):
    if config['threshold'] in [1, 2] or config['borb']:
        # most recent commits  (labeled or not)
        tail_size = min(len(df_train), config['borb_ma_window_size'])
        df_tail = df_train[-tail_size:]
    else:
        df_tail = None
    return df_train, df_tail


def __prepare_train_data(df_train, df_others, verification_latency, config):
    train_timestamp = df_train['timestamp'].max()
    df_train_others = df_others[df_others['timestamp']
                                <= train_timestamp].copy()
    df_train = pd.concat([df_train, df_train_others])

    # add invalid label as a safe-guard
    df_train['soft_target'] = -1.
    if df_train.empty:
        return df_train
    # check if fix has been done (bug) or verification latency has passed (normal), otherwise is unlabeled
    indices_1 = df_train['timestamp_fix'] <= train_timestamp
    indices_0 = ~indices_1 & (
        df_train['timestamp'] <= train_timestamp - verification_latency)
    indices_vl = ~indices_1 & ~indices_0
    df_train.loc[indices_1, 'soft_target'] = 1.
    df_train.loc[indices_0, 'soft_target'] = 0.
    if config['uncertainty']:
        df_train.loc[indices_vl, 'soft_target'] = df_train[indices_vl].apply(lambda row: __verification_latency_label(
            train_timestamp, row.timestamp, verification_latency), axis='columns')
    else:
        df_train.loc[indices_vl, 'soft_target'] = np.nan

    df_train = df_train.dropna(subset=['soft_target'])
    df_train['target'] = df_train['soft_target'] > .5
    return df_train


def __prepare_val_data(df_train, config):
    f_val = config['f_val']
    if f_val > .0 and len(df_train) > 1:
        train_index, val_index = next(StratifiedShuffleSplit(
            n_splits=1, test_size=f_val).split(df_train, df_train['target']))
        df_train, df_val = df_train.iloc[train_index], df_train.iloc[val_index]
    else:
        df_val = None

    return df_train, df_val


def __has_metrics(metrics):
    return metrics is not None and len(metrics) > 0
