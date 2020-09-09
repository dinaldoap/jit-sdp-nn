import mlflow
import numpy as np
import pandas as pd
import re


def generate(config):
    # TODO: generate tuning and testing scripts in the current folder
    # print_data(df_tuning)
    n_datasets = 10
    n_cross_projects = 2
    n_models = 8
    n_configs = config['end'] - config['start']
    n_seeds = 5
    expected_max_results = n_models * n_datasets * n_configs * n_seeds
    df_tuning = mlflow.search_runs(experiment_ids=0, max_results= 2 * expected_max_results)
    assert expected_max_results == len(df_tuning)
    assert np.all(df_tuning['status'] == 'FINISHED')
    config_cols = config_columns(df_tuning.columns)
    df_best_configs = df_tuning.groupby(by=config_cols, as_index=False, dropna=False).agg({
        'metrics.avg_gmean': ['mean', 'std'], 'tags.run.command': 'first'})
    df_best_configs.columns = remove_columns_prefix(df_best_configs.columns)
    df_best_configs = df_best_configs.sort_values(
        by='avg_gmean.mean', ascending=False, kind='mergesort')
    df_best_configs = df_best_configs.drop_duplicates(
        subset=['rate_driven', 'meta_model', 'model', 'dataset'])
    # print_data(df_best_configs)
    commands = tuning_to_testing(df_best_configs['run.command.first'])

    with open('jitsdp/dist/testing.sh', mode='w') as out:
        for command in commands:
            out.write(command)
            out.write('\n')


def tuning_to_testing(commands):
    seeds = range(30)
    for command in commands:
        for seed in seeds:
            new_command = command.replace('end', 'start')
            new_command = re.sub(
                r'seed \d+', 'seed {}'.format(seed), new_command)
            new_command = new_command + ' --experiment-name testing --track-time 1 --track-forest 1'
            yield new_command


def config_columns(cols):
    return [col for col in cols if col.startswith('params') and not col.endswith('seed')]


def remove_columns_prefix(cols):
    new_cols = []
    for col in cols:
        first_level = col[0]
        first_level = first_level.split('.')
        first_level = '.'.join(first_level[1:])
        second_level = col[1]
        second_level = second_level if len(
            second_level) == 0 else '.{}'.format(second_level)
        new_col = '{}{}'.format(first_level, second_level)
        new_cols.append(new_col)
    return new_cols


def print_data(df):
    print(len(df))
    print(df.columns)
    print(df.head(1))