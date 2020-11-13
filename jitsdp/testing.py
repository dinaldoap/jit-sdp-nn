# coding=utf-8
from jitsdp.tuning import add_shared_arguments
from jitsdp import tuning
from jitsdp.utils import filename_to_path
from jitsdp.data import load_runs

import mlflow
import numpy as np
import pandas as pd
import re
import sys


def add_arguments(parser, filename):
    add_shared_arguments(parser, filename)
    parser.add_argument('--testing-start', type=int,
                        help='First commit to be used for testing.', required=True)


def add_shared_arguments(parser, filename):
    tuning.add_shared_arguments(parser, filename)
    parser.add_argument('--tuning-experiment-name',   type=str,
                        help='Experiment name used for tuning (default: Default).', default='Default')
    parser.add_argument('--no-validation',
                        help='Disable validations of the data the flows from hyperparameter tuning to testing.', action='store_true')


def generate(config):
    # print_data(df_tuning)
    df_best_configs, _ = get_best_configs(config)
    # print_data(df_best_configs)
    commands = tuning_to_testing(
        df_best_configs['run.command'], config['testing_start'])
    file_ = filename_to_path(config['filename'])
    with open(file_, mode='w') as out:
        for command in commands:
            out.write(command)
            out.write('\n')


def get_best_configs(config):
    df_best_configs, config_cols = configs_results(config)
    df_best_configs = df_best_configs.sort_values(
        by='g-mean', ascending=False, kind='mergesort')
    df_best_configs = df_best_configs.drop_duplicates(
        subset=['meta_model', 'model', 'cross_project', 'dataset'])
    df_best_configs = df_best_configs.sort_values(
        by=['dataset', 'meta_model', 'model', 'cross_project'], ascending=True, kind='mergesort')
    return df_best_configs, config_cols


def configs_results(config):
    tuning_experiment_name = config['tuning_experiment_name']
    tuning_experiment_id = mlflow.get_experiment_by_name(
        tuning_experiment_name).experiment_id
    df_tuning = load_runs(tuning_experiment_id)
    df_tuning = valid_data(config, df_tuning, single_config=False, n_seeds=3)
    config_cols = remove_columns_prefix(config_columns(df_tuning.columns))
    df_tuning.columns = remove_columns_prefix(df_tuning.columns)
    df_configs_results = df_tuning.groupby(by=config_cols, as_index=False, dropna=False).agg({
        'g-mean': 'mean', 'run.command': 'first'})
    return df_configs_results, config_cols


def valid_data(config, df_runs, single_config, n_seeds):
    if not config['no_validation']:
        n_datasets = 10
        n_cross_projects = len(config['cross_project'])
        n_models = 6
        n_configs = 1 if single_config else config['end'] - config['start']
        expected_n_runs = n_models * n_cross_projects * \
            n_configs * n_datasets * n_seeds
        n_runs = len(df_runs)
        assert expected_n_runs == n_runs, ' Number of runs: {}. Expected: {}.'.format(
            n_runs, expected_n_runs)
        assert np.all(df_runs['status'] == 'FINISHED')
    else:
        df_runs = df_runs[df_runs['status'] == 'FINISHED']
    return df_runs


def tuning_to_testing(commands, testing_start):
    seeds = [
        126553321124052187622850717793961581415, 14787688002703798423351339219352185274,
        193056263972238819939972572120722936383, 114338109874184942262013079161379184987,
        187271577167371661491931762450134721284, 54698102517174123462444542558024060414,
        27646170786402952858031977871644291788, 72381234526383921728120594777263261368,
        29349119487831557257107505317979405654, 76727776937254199199525190949144326949,
        329750211155635991608625780527685436923, 6550265051387167741700808336407247928,
        154295897404741793897478976647482581107, 317811204935223348093468100252472859938,
        246161674499922800864018660497418925492, 60330052066067520304003611920231920926,
        232180032819347367777866605512888510283, 56941727144663183188449674638672970413,
        143525101485785175188371633342791328500, 615958507449322123931599253290210168,
        27126682959748074983895968893025267066, 151751458357273121450769716956495337029,
        296590805974379211532355156730832227482, 306142036899271559955646908007986804643,
        277456020953988706779517483877489883610, 68400791372814003886493895832540654659,
        37963028988999854340935760070049051603, 171444486591363246524155040296406447243,
        130476781972747964311769411495909616655, 45201615096789524889009985710706651144,
    ]
    for seed in seeds:
        for command in commands:
            new_command = re.sub(
                r'end \d+', 'start {}'.format(testing_start), command)
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
