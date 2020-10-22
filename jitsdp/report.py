# coding=utf-8
from jitsdp.plot import plot_recalls_gmean, plot_proportions, plot_boxplot, plot_efficiency_curves, plot_critical_distance
from jitsdp.data import load_results, load_runs, save_results
from jitsdp.utils import unique_dir, dir_to_path
from jitsdp import testing

from collections import namedtuple
import numpy as np
import pandas as pd
import mlflow
from scipy.stats import friedmanchisquare, wilcoxon
import sys


def report(config, results):
    # metrics
    metrics = ['r0', 'r1', 'r0-r1', 'g-mean',
               'tr1', 'te1', 'pr1', 'th-ma', 'th-pr1']
    metrics = {metric: results[metric].mean() for metric in metrics}
    mlflow.log_metrics(metrics)
    # artifacts
    dir = unique_dir(config)
    save_results(results=results, dir=dir)
    plot_recalls_gmean(results, config=config, dir=dir)
    plot_proportions(results, config=config, dir=dir)
    mlflow.log_artifacts(local_dir=dir)


def add_arguments(parser, dirname):
    testing.add_arguments(parser, dirname)
    parser.add_argument('--testing-experiment-name',   type=str,
                        help='Experiment name used for testing (default: testing).', default='testing')


def generate(config):
    efficiency_curves(config)
    df_testing = best_configs_testing(config)
    # plotting
    Metric = namedtuple('Metric', ['column', 'name', 'ascending'])
    metrics = [
        Metric('g-mean', 'g-mean', False),
        Metric('r0-r1', '|r0-r1|', True),
        Metric('th-ma', '|th-ma|', True),
        Metric('th-pr1', '|th-pr1|', True),
    ]
    plot_boxplot(df_testing, metrics, dir_to_path(config['filename']))
    statistical_analysis(config, df_testing, metrics)
    table(config, df_testing, metrics)


def best_configs_testing(config):
    df_best_configs, config_cols = testing.get_best_configs(config)
    # replace nan by -1 to allow join
    df_best_configs = df_best_configs.fillna(-1)
    df_best_configs = df_best_configs[config_cols].set_index(config_cols)
    testing_experiment_name = config['testing_experiment_name']
    testing_experiment_id = mlflow.get_experiment_by_name(
        testing_experiment_name).experiment_id
    df_testing = load_runs(testing_experiment_id)
    # replace nan by -1 to allow join
    df_testing = df_testing.fillna(-1)
    df_testing.columns = testing.remove_columns_prefix(df_testing.columns)
    df_testing = df_testing.join(df_best_configs, on=config_cols, how='inner')
    df_testing = testing.valid_data(
        config, df_testing, single_config=True, n_seeds=30)
    df_testing = df_testing.sort_values(
        by=['dataset', 'meta_model', 'model', 'rate_driven', 'cross_project'])
    df_testing['name'] = df_testing.apply(lambda row: name(
        row, config['cross_project']), axis='columns')
    return df_testing


def name(row, cross_project):
    meta_model = row['meta_model']
    model = row['model']
    if cross_project:
        train_data = '-cp' if row['cross_project'] == '1' else '-wp'
    else:
        train_data = ''
    return '{}-{}{}'.format(meta_model.upper(), model.upper(), train_data.upper())


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
    total_trials = len(df_results)
    maximums_by_experiment_size = []
    np.random.seed(0)
    for experiment_size in [1, 2, 4, 8, 16, 32]:
        maximums = []
        for i in range(1000):
            sample_indices = np.random.choice(total_trials, experiment_size)
            maximum = df_results['g-mean'].iloc[sample_indices].max()
            maximums.append(maximum)
        maximums_by_experiment_size.extend(
            [{'experiment_size': experiment_size, 'g-mean': maximum} for maximum in maximums])
    df_efficiency_curve = pd.DataFrame(maximums_by_experiment_size)
    df_efficiency_curve = df_efficiency_curve.set_index('experiment_size')
    return df_efficiency_curve


def statistical_analysis(config, df_testing, metrics):
    for metric in metrics:
        df_inferential = df_testing.groupby(['dataset', 'meta_model', 'model', 'rate_driven',
                                             'cross_project'], as_index=False).agg({'name': 'first', metric.column: 'mean'})
        df_inferential = pd.pivot_table(
            df_inferential, columns='name', values=metric.column, index='dataset')
        measurements = [df_inferential[column]
                        for column in df_inferential.columns]
        _, friedman_p_value = friedmanchisquare(*measurements)
        dir = dir_to_path(config['filename'])
        with open(dir / '{}.txt'.format(metric.column), 'w') as f:
            f.write('Friedman p-value: {}\n'.format(friedman_p_value))
            if not config['cross_project']:
                _, wilcoxon_p_value = wilcoxon(
                    df_inferential['BORB-IHF'], df_inferential['ORB-OHT'], alternative='less' if metric.ascending else 'greater')
                f.write('Wilcoxon p-value: {}'.format(wilcoxon_p_value))

        avg_rank = df_inferential.rank(
            axis='columns', ascending=metric.ascending)
        avg_rank = avg_rank.mean()
        plot_critical_distance(avg_rank, df_inferential, metric,
                               dir)


def table(config, df_testing, metrics):
    metric_columns = {metric.column: ['mean', 'std'] for metric in metrics}
    df_table = df_testing.groupby(by=['dataset', 'name']).agg(metric_columns)
    metric_names = {metric.column: metric.name for metric in metrics}
    df_table = df_table.rename(metric_names, axis='columns')
    dir = dir_to_path(config['filename'])
    df_table.to_csv(dir / 'table.csv')
