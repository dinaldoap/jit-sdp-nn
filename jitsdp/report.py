# coding=utf-8
from jitsdp.plot import plot_recalls_gmean, plot_proportions, plot_boxplot
from jitsdp.data import load_results
from jitsdp.utils import unique_dir, dir_to_path
from jitsdp import testing

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


def add_arguments(parser, dirname):
    testing.add_arguments(parser, dirname)
    parser.add_argument('--testing-experiment-name',   type=str,
                        help='Experiment name used for testing (default: testing).', default='testing')


def generate(config):
    df_best_configs, config_cols = testing.get_best_configs(config)
    # replace nan by -1 to allow join
    df_best_configs = df_best_configs.fillna(-1)
    df_best_configs = df_best_configs[config_cols].set_index(config_cols)
    testing_experiment_name = config['testing_experiment_name']
    testing_experiment_id = mlflow.get_experiment_by_name(
        testing_experiment_name).experiment_id
    df_testing = mlflow.search_runs(
        experiment_ids=testing_experiment_id, max_results=sys.maxsize)
    # replace nan by -1 to allow join
    df_testing = df_testing.fillna(-1)
    df_testing.columns = testing.remove_columns_prefix(df_testing.columns)
    df_testing = df_testing.join(df_best_configs, on=config_cols, how='inner')
    if not config['no_validation']:
        n_datasets = 10
        n_cross_projects = config['cross_project'] + 1
        n_models = 8
        n_configs = 1
        n_seeds = 30
        expected_n_runs = n_models * n_cross_projects * \
            n_configs * n_datasets * n_seeds
        n_runs = len(df_tuning)
        assert expected_n_runs == n_runs, ' Number of runs in experiment {}: {}. Expected: {}.'.format(
            testing_experiment_name,  n_runs, expected_n_runs)
        assert np.all(df_testing['status'] == 'FINISHED')
    df_testing = df_testing.sort_values(by='dataset')
    # plotting
    plot_boxplot(df_testing, dir_to_path(config['filename']))
