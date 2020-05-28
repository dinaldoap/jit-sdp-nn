from jitsdp import metrics as met
from jitsdp.constants import DIR
from jitsdp.data import make_stream, save_results, load_results
from jitsdp.pipeline import create_pipeline, set_seed
from jitsdp.plot import plot_recalls_gmean, plot_proportions

import math
import mlflow
import pandas as pd


def run(config):
    mlflow.log_params(config)
    set_seed(config)
    df_prequential = make_stream(
        'https://raw.githubusercontent.com/dinaldoap/jit-sdp-data/master/{}.csv'.format(config['dataset']))
    # split test partition in folds and iterate over them (fold from current to current + fold_size or end)
    # the previous commits until current are used for training
    seconds_by_day = 24 * 60 * 60
    verification_latency = 90 * seconds_by_day  # seconds
    fold_size = config['fold_size']  # commits
    # start with data for training (minimum of one fold)
    start = max(config['start'], fold_size)
    interval = len(df_prequential) - start  # last commit
    n_folds = math.ceil(interval / fold_size)  # number of folds rounded up
    # use a fraction of folds (minimum of one)
    n_folds = max(math.ceil(n_folds * config['f_folds']), 1)
    end = start + n_folds * fold_size  # last fold end

    if config['f_val'] > 0:
        step = (n_folds // 4) * fold_size
    else:
        step = fold_size

    pipeline = create_pipeline(config)
    if config['incremental']:
        pipeline.save()
    target_prediction = None
    train_step = 0
    for current in range(start, end, step):
        df_train = df_prequential[:current].copy()
        df_test = df_prequential[current:min(current + step, end)].copy()
        # check if fix has been done (bug) or verification latency has passed (normal), otherwise is unlabeled
        train_timestamp = df_train['timestamp'].max()
        df_train['soft_target'] = df_train.apply(lambda row: 1 if row.timestamp_fix <= train_timestamp
                                                 else 0 if row.timestamp <= train_timestamp - verification_latency
                                                 else __verification_latency_label(train_timestamp, row.timestamp, verification_latency, config), axis='columns')
        if config['threshold'] in [1, 2] or config['orb']:
            val_size = min(int(len(df_train) * .1), 100)
            df_val = df_train[-val_size:]
            df_train = df_train[:-val_size]
        else:
            df_val = None

        df_train = df_train.dropna(subset=['soft_target'])
        df_train['target'] = df_train['soft_target'] > .5
        # train and predict
        pipeline = create_pipeline(config)
        if config['incremental']:
            pipeline.load()
        for metrics in pipeline.train(df_train, df_ma=df_val):
            mlflow.log_metrics(metrics=metrics, step=train_step)
            train_step += 1
        if config['incremental']:
            pipeline.save()
        target_prediction_test = pipeline.predict(
            df_test, df_threshold=df_val, df_proportion=df_train)
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


def __verification_latency_label(train_timestamp, commit_timestamp, verification_latency, config):
    if config['uncertainty']:
        return .5 - .5 * (train_timestamp - commit_timestamp) / verification_latency

    return None
