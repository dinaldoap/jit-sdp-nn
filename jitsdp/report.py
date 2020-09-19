# coding=utf-8
from jitsdp.plot import plot_recalls_gmean, plot_proportions, plot_boxplot
from jitsdp.data import load_results
from jitsdp.utils import unique_dir, dir_to_path
from jitsdp.testing import get_best_configs, remove_columns_prefix

import numpy as np
import pandas as pd
import mlflow
import sys


def report(config):
    dir = unique_dir(config)
    results = load_results(dir=dir)
    plot_recalls_gmean(results, config=config, dir=dir)
    plot_proportions(results, config=config, dir=dir)
    metrics = ['r0', 'r1', 'r0-r1', 'g-mean', 'c1', 't1', 's1', 'p1']
    metrics = {metric: results[metric].mean() for metric in metrics}
    mlflow.log_metrics(metrics)
    mlflow.log_artifacts(local_dir=dir)


def generate(config):
    df_best_configs, config_cols = get_best_configs(config)
    # replace nan by -1 to allow join
    df_best_configs = df_best_configs.fillna(-1)
    df_best_configs = df_best_configs[config_cols].set_index(config_cols)
    df_testing = mlflow.search_runs(
        experiment_ids='1', max_results=sys.maxsize)
    # replace nan by -1 to allow join
    df_testing = df_testing.fillna(-1)
    df_testing.columns = remove_columns_prefix(df_testing.columns)
    df_testing = df_testing.join(df_best_configs, on=config_cols, how='inner')
    if not config['no_validation']:
        n_datasets = 10
        n_cross_projects = 2
        n_models = 8
        n_configs = 1
        n_seeds = 5
        expected_max_results = n_models * n_cross_projects * \
            n_configs * n_datasets * n_seeds
        assert expected_max_results == len(df_testing)
        assert np.all(df_testing['status'] == 'FINISHED')
    df_testing = df_testing.sort_values(by='dataset')
    # plotting
    plot_boxplot(df_testing, dir_to_path('logs'))


if __name__ == '__main__':
    generate({'no_validation': 1, 'start': 0, 'end': 20})
