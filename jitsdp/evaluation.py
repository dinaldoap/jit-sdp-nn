from jitsdp import metrics as met
from jitsdp.constants import DIR
from jitsdp.data import make_stream, save_results, load_results
from jitsdp.pipeline import create_pipeline, set_seed
from jitsdp.plot import plot_recalls_gmean, plot_proportions

import math
import mlflow
import pandas as pd


def run(config):
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
    n_folds = max(math.ceil(n_folds * config['folds']), 1)
    end = start + n_folds * fold_size  # last fold end

    pipeline = create_pipeline(config)
    # pipeline.save()
    target_prediction = []
    for current in range(start, end, fold_size):
        df_train = df_prequential[:current].copy()
        df_test = df_prequential[current:min(current + fold_size, end)].copy()
        # check if fix has been done (bug) or verification latency has passed (normal), otherwise is unlabeled
        train_timestamp = df_train['timestamp'].max()
        df_train['soft_target'] = df_train.apply(lambda row: 1 if row.timestamp_fix <= train_timestamp
                                                 else 0 if row.timestamp <= train_timestamp - verification_latency
                                                 else .5 - .5 * (train_timestamp - row.timestamp) / verification_latency, axis='columns')
        df_train['target'] = df_train['soft_target'] > .5
        # train and predict
        pipeline = create_pipeline(config)
        # pipeline.load()
        pipeline.train(df_train)
        # pipeline.save()
        if config['balance']:
            val_size = min(int(len(df_train) * .1), 100)
            df_val = df_train[-val_size:]
            df_train = df_train[:-val_size]
            target_prediction_test = pipeline.predict(
                df_test, df_val)
        else:
            target_prediction_test = pipeline.predict(df_test)
        target_prediction.append(target_prediction_test)

    target_prediction = pd.concat(target_prediction, sort=False)
    target_prediction = target_prediction.reset_index(drop=True)
    results = met.prequential_metrics(target_prediction, .99)
    save_results(results=results, dir=DIR / config['dataset'])
    report(config)


def report(config):
    subdir = DIR / config['dataset']
    results = load_results(dir=subdir)
    plot_recalls_gmean(results, config=config, dir=DIR)
    plot_proportions(results, config=config, dir=DIR)
    metrics = ['r0', 'r1', 'r0-r1', 'gmean', 'p0', 'p1']
    metrics = {'avg_{}'.format(
        metric): results[metric].mean() for metric in metrics}
    mlflow.log_params(config)
    mlflow.log_metrics(metrics)
    mlflow.log_artifacts(local_dir=subdir)
