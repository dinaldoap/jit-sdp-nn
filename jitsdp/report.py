# coding=utf-8
from jitsdp.plot import plot_recalls_gmean, plot_proportions, plot_boxplot, plot_efficiency_curves, plot_critical_distance
from jitsdp.data import load_results
from jitsdp.utils import unique_dir, dir_to_path
from jitsdp import testing

import numpy as np
import pandas as pd
import mlflow
from scipy.stats import friedmanchisquare
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
    efficiency_curves(config)
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
        n_runs = len(df_testing)
        assert expected_n_runs == n_runs, ' Number of runs in experiment {}: {}. Expected: {}.'.format(
            testing_experiment_name,  n_runs, expected_n_runs)
        assert np.all(df_testing['status'] == 'FINISHED')
    df_testing = df_testing.sort_values(
        by=['dataset', 'meta_model', 'model', 'rate_driven', 'cross_project'])
    df_testing['name'] = df_testing.apply(lambda row: name(
        row, config['cross_project']), axis='columns')
    # plotting
    plot_boxplot(df_testing, dir_to_path(config['filename']))
    statistical_analysis(config, df_testing)


def name(row, cross_project):
    meta_model = row['meta_model']
    rate_driven = 'rd' if row['rate_driven'] == '1' else 'nrd'
    model = row['model']
    if cross_project:
        train_data = '-cp' if row['cross_project'] == '1' else '-wp'
    else:
        train_data = ''
    return '{}$_{{{}}}$-{}{}'.format(meta_model.upper(), rate_driven, model.upper(), train_data.upper())


def efficiency_curves(config):
    df_configs_results, _ = testing.configs_results(config)
    df_configs_results = df_configs_results[[
        'meta_model', 'rate_driven', 'model', 'cross_project', 'dataset', 'g-mean']]
    df_efficiency_curve = df_configs_results.groupby(
        by=['meta_model', 'rate_driven', 'model', 'cross_project', 'dataset']).apply(efficiency_curve)
    df_efficiency_curve = df_efficiency_curve.reset_index()
    df_efficiency_curve['name'] = df_efficiency_curve.apply(lambda row: name(
        row, config['cross_project']), axis='columns')
    plot_efficiency_curves(df_efficiency_curve,
                           dir_to_path(config['filename']))


def efficiency_curve(df_results):
    df_results = df_results.copy()
    total_trials = len(df_results)
    maximums_by_experiment_size = []
    for experiment_size in [1, 2, 4, 8, 16, 32]:
        df_results['experiment'] = np.array(
            range(total_trials)) // experiment_size
        maximums = df_results.groupby('experiment')['g-mean'].max()
        maximums_by_experiment_size.extend(
            [{'experiment_size': experiment_size, 'g-mean': maximum} for maximum in maximums])
    df_efficiency_curve = pd.DataFrame(maximums_by_experiment_size)
    df_efficiency_curve = df_efficiency_curve.set_index('experiment_size')
    return df_efficiency_curve


def statistical_analysis(config, df_testing):
    df_testing = df_testing.groupby(['dataset', 'meta_model', 'model', 'rate_driven',
                                     'cross_project'], as_index=False).agg({'name': 'first', 'g-mean': 'mean'})
    df_inferential = pd.pivot_table(
        df_testing, columns='name', values='g-mean', index='dataset')
    measurements = [df_inferential[column]
                    for column in df_inferential.columns]
    test_stat, p_value = friedmanchisquare(*measurements)
    dir = dir_to_path(config['filename'])
    with open(dir / 'p-value.txt', 'w') as f:
        f.write('p-value: {}'.format(p_value))

    avg_rank = df_inferential.rank(axis='columns', ascending=False)
    avg_rank = avg_rank.mean()
    plot_critical_distance(avg_rank, df_inferential,
                           dir)
