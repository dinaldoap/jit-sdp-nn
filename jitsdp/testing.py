import numpy as np
import pandas as pd
import re


def best_configs():
    df_tuning = pd.read_csv('data/tuning.csv')
    # print_data(df_tuning)
    n_datasets, n_cross_projects, n_models, n_configs, n_seeds = 10, 2, 6, 2, 5
    assert n_models * n_datasets * n_seeds * n_configs == len(df_tuning)
    assert np.all(df_tuning[df_tuning['status'] == 'FINISHED'])
    config_cols = config_columns(df_tuning.columns)
    df_best_configs = df_tuning.groupby(by=config_cols, as_index=False, dropna=False).agg({
        'metrics.avg_gmean': ['mean', 'std'], 'tags.run.command': 'first'})
    df_best_configs.columns = remove_columns_prefix(df_best_configs.columns)
    df_best_configs = df_best_configs.sort_values(
        by='avg_gmean.mean', ascending=False, kind='mergesort')
    df_best_configs = df_best_configs.drop_duplicates(
        subset=['rate_driven', 'meta_model', 'model', 'dataset'])
    print_data(df_best_configs)
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
                'seed \d+', 'seed {}'.format(seed), new_command)
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


if __name__ == '__main__':
    best_configs()
