# coding=utf-8
from jitsdp.utils import filename_to_path

import mlflow
import numpy as np
import pandas as pd
import re


def generate(config):
    # print_data(df_tuning)
    df_best_configs, _ = get_best_configs(config)
    # print_data(df_best_configs)
    commands = tuning_to_testing(df_best_configs['run.command'])
    file_ = filename_to_path(config['filename'])
    with open(file_, mode='w') as out:
        for command in commands:
            out.write(command)
            out.write('\n')


def get_best_configs(config):
    n_datasets = 10
    n_cross_projects = 2
    n_models = 8
    n_configs = config['end'] - config['start']
    n_seeds = 5
    expected_n_runs = n_models * n_cross_projects * \
        n_configs * n_datasets * n_seeds
    experiment_name = config['experiment_name']
    experiment_id = mlflow.get_experiment_by_name(
        experiment_name).experiment_id
    df_tuning = mlflow.search_runs(
        experiment_ids=experiment_id, max_results=2 * expected_n_runs)
    if not config['no_validation']:
        n_runs = len(df_tuning)
        assert expected_n_runs == n_runs, ' Number of runs in experiment {}: {}. Expected: {}.'.format(
            experiment_name,  n_runs, expected_n_runs)
        assert np.all(df_tuning['status'] == 'FINISHED')
    config_cols = remove_columns_prefix(config_columns(df_tuning.columns))
    df_tuning.columns = remove_columns_prefix(df_tuning.columns)
    df_best_configs = df_tuning.groupby(by=config_cols, as_index=False, dropna=False).agg({
        'g-mean': 'mean', 'run.command': 'first'})
    df_best_configs = df_best_configs.sort_values(
        by='g-mean', ascending=False, kind='mergesort')
    df_best_configs = df_best_configs.drop_duplicates(
        subset=['rate_driven', 'meta_model', 'model', 'dataset'])
    df_best_configs = df_best_configs.sort_values(
        by=['dataset', 'model'], ascending=True, kind='mergesort')
    return df_best_configs, config_cols


def tuning_to_testing(commands):
    seeds = range(30)
    for command in commands:
        for seed in seeds:
            new_command = command.replace('end', 'start')
            new_command = re.sub(
                r'seed \d+', 'seed {}'.format(seed), new_command)
            new_command = new_command + \
                ' --end None --experiment-name testing --track-time 1 --track-forest 1'
            yield new_command


def config_columns(cols):
    exclusions = ['seed', 'start', 'end',
                  'experiment_name', 'track_time', 'track_forest']
    exclusions = set(['params.{}'.format(name) for name in exclusions])
    return [col for col in cols if col.startswith('params') and not col in exclusions]


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


def print_data(df):
    print(len(df))
    print(df.columns)
    print(df.head(1))
