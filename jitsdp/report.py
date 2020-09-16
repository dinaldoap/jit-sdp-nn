# coding=utf-8
from jitsdp.plot import plot_recalls_gmean, plot_proportions, plot_boxplot
from jitsdp.data import load_results
from jitsdp.utils import unique_dir, dir_to_path

import numpy as np
import pandas as pd
import mlflow


def report(config):
    dir = unique_dir(config)
    results = load_results(dir=dir)
    plot_recalls_gmean(results, config=config, dir=dir)
    plot_proportions(results, config=config, dir=dir)
    metrics = ['r0', 'r1', 'r0-r1', 'gmean', 't1', 's1', 'p1']
    metrics = {'avg_{}'.format(
        metric): results[metric].mean() for metric in metrics}
    mlflow.log_metrics(metrics)
    mlflow.log_artifacts(local_dir=dir)


def generate(config):
    n_datasets = 10
    n_cross_projects = 1
    n_models = 2
    n_configs = 1
    n_seeds = 1
    expected_max_results = n_models * n_cross_projects * \
        n_configs * n_datasets * n_seeds
    df_testing = mlflow.search_runs(
        experiment_ids='1', max_results=2 * expected_max_results)
    if not config['no_validation']:
        assert expected_max_results == len(df_testing)
        assert np.all(df_testing['status'] == 'FINISHED')
    df_testing.columns = remove_columns_prefix(df_testing.columns)
    df_testing = df_testing.sort_values(by='dataset')
    # plotting
    plot_boxplot(df_testing, dir_to_path('logs'))


def remove_columns_prefix(cols):
    new_cols = []
    for col in cols:
        new_col = col.split('.')
        if len(new_col) > 1:
            new_col = '.'.join(new_col[1:])
        else:
            new_col = '.'.join(new_col[:])
        new_cols.append(new_col)
    return new_cols


if __name__ == '__main__':
    generate({'no_validation': 1})
