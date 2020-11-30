# coding=utf-8
from jitsdp.plot import plot_recalls_gmean, plot_proportions, plot_boxplot, plot_tuning_convergence, plot_critical_distance
from jitsdp.data import DATASETS, load_results, load_runs, make_stream, save_results
from jitsdp.utils import unique_dir, dir_to_path, split_proposal_baseline
from jitsdp import testing

from collections import namedtuple
import itertools
import numpy as np
import pandas as pd
import mlflow
from scipy.stats import friedmanchisquare, wilcoxon, spearmanr
from statsmodels.stats.multitest import multipletests


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
    testing.add_shared_arguments(parser, dirname)
    parser.add_argument('--testing-experiment-name',   type=str,
                        help='Experiment name used for testing (default: testing).', default='testing')


def generate(config):
    tuning_convergence(config)
    df_testing = best_configs_testing(config)
    # plotting
    Metric = namedtuple('Metric', ['column', 'name', 'ascending', 'baseline'])
    metrics = [
        Metric('g-mean', 'g-mean', False, True),
        Metric('r0-r1', '|$r_0-r_1$|', True, True),
        Metric('r0', '$r_0$', False, True),
        Metric('r1', '$r_1$', False, True),
        Metric('th-ma', '|$fr_1-ir_1$|', True, False),
        Metric('th-pr1', '|$fr_1-pr_1$|', True, False),
    ]
    plots(config, df_testing, metrics)
    statistical_analysis(config, df_testing, metrics)
    table(config, df_testing, metrics)
    datasets_statistics(config)


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
        by=['dataset', 'meta_model', 'model', 'cross_project'])
    df_testing['classifier'] = df_testing.apply(lambda row: format_classifier(
        row, config['cross_project']), axis='columns')
    return df_testing


def format_classifier(row, cross_project):
    meta_model = row['meta_model']
    model = row['model']
    if len(cross_project) > 1:
        train_data = '-cp' if row['cross_project'] == '1' else '-wp'
    else:
        train_data = ''
    return '{}-{}{}'.format(meta_model.upper(), model.upper(), train_data.upper())


def tuning_convergence(config):
    df_configs_results, _ = testing.configs_results(config)
    df_configs_results = df_configs_results[[
        'meta_model', 'model', 'cross_project', 'dataset', 'g-mean']]
    df_tuning_convergence = df_configs_results.groupby(
        by=['meta_model', 'model', 'cross_project', 'dataset']).apply(tuning_convergence_by_dataset)
    df_tuning_convergence = df_tuning_convergence.reset_index()
    df_tuning_convergence['classifier'] = df_tuning_convergence.apply(lambda row: format_classifier(
        row, config['cross_project']), axis='columns')
    plot_tuning_convergence(df_tuning_convergence,
                            dir_to_path(config['filename']))


def tuning_convergence_by_dataset(df_results):
    total_trials = len(df_results)
    maximums_by_experiment_size = []
    rng = np.random.default_rng(268737018669781749321160927763689789779)
    for experiment_size in [1, 2, 4, 8, 16, 32, 64, 128]:
        maximums = []
        for i in range(1000):
            sample_indices = rng.choice(total_trials, experiment_size)
            maximum = df_results['g-mean'].iloc[sample_indices].max()
            maximums.append(maximum)
        maximums_by_experiment_size.extend(
            [{'experiment_size': experiment_size, 'g-mean': maximum} for maximum in maximums])
    df_tuning_convergence = pd.DataFrame(maximums_by_experiment_size)
    df_tuning_convergence = df_tuning_convergence.set_index('experiment_size')
    return df_tuning_convergence


def plots(config, df_testing: pd.DataFrame, metrics):
    for metric in metrics:
        plot_data = filter_baseline(df_testing, metric)
        plot_boxplot(plot_data, metric, dir_to_path(config['filename']))


def statistical_analysis(config, df_testing: pd.DataFrame, metrics):
    config_cols = ['dataset', 'meta_model', 'model', 'cross_project']
    agg_cols = {metric.column: 'mean' for metric in metrics}
    agg_cols.update({'classifier': 'first'})
    df_metrics = df_testing.groupby(config_cols, as_index=False).agg(agg_cols)
    dir = dir_to_path(config['filename'])
    for metric in metrics:
        df_inferential = filter_baseline(df_metrics, metric)
        df_inferential = pd.pivot_table(
            df_inferential, columns='classifier', values=metric.column, index='dataset')
        df_inferential = df_inferential * (-1 if metric.ascending else 1)
        with open(dir / '{}.txt'.format(metric.column), 'w') as f:
            write_friedman(df_inferential, f)
            safe_write_wilcoxon(config, df_inferential, metric.baseline, f)

        avg_rank = df_inferential.rank(
            axis='columns', ascending=False)
        avg_rank = avg_rank.mean()
        plot_critical_distance(avg_rank, df_inferential, metric,
                               dir)

    with open(dir / 'correlations.txt', 'w') as f:
        df_correlation = df_testing[~df_testing['classifier'].str.contains(
            'ORB-OHT')].copy()
        agg_rate_cols = {
            'th-pr1': 'mean',
            'th-ma': 'mean',
            'g-mean': 'mean'
        }
        for col in agg_rate_cols.keys():
            df_correlation[col] = df_correlation[col].astype(float)
        df_correlation = df_correlation.groupby(
            config_cols, as_index=False).agg(agg_rate_cols)
        quantiles = df_correlation['th-pr1'].quantile(q=[.7, .8, .9])
        f.write('quantiles |th-pr1|:\n')
        quantiles.to_string(f)
        f.write('\n')
        max_distance = .05
        _, wilcoxon_p_value = wilcoxon(
            df_correlation['th-pr1'], [max_distance] * len(df_correlation), alternative='less')
        f.write(
            '|th-pr1| < {}, wilcoxon p-value : {}, reject: {}\n'.format(max_distance, wilcoxon_p_value, wilcoxon_p_value < .05))
        correlation, p_value = spearmanr(
            df_correlation['th-ma'], df_correlation['g-mean'])
        f.write('corr(|th-ma|, g-mean): {}, p-value: {}\n'.format(correlation, p_value))


def filter_baseline(df_testing, metric):
    _, baseline_name = split_proposal_baseline(
        df_testing['classifier'].unique())
    if metric.baseline:
        return df_testing
    else:
        return df_testing[df_testing['classifier'] != baseline_name[0]]


def write_friedman(df_inferential, f):
    measurements = [df_inferential[column]
                    for column in df_inferential.columns]
    _, friedman_p_value = friedmanchisquare(*measurements)
    f.write('Friedman p-value: {}\n'.format(friedman_p_value))


def safe_write_wilcoxon(config, df_inferential, baseline, f):
    if len(config['cross_project']) == 1:
        try:
            write_wilcoxon(df_inferential, baseline, f)
        except ValueError as e:
            f.write(repr(e))


def write_wilcoxon(df_inferential: pd.DataFrame, baseline, f):
    proposal_names, baseline_name = split_proposal_baseline(
        df_inferential.columns)
    if baseline:
        pairs = list(itertools.product(proposal_names, baseline_name))
    else:
        pairs = list(itertools.combinations(proposal_names, 2))
    p_values = []
    r_pluses = []
    for name0, name1 in pairs:
        _, wilcoxon_p_value = wilcoxon(
            df_inferential[name0], df_inferential[name1], alternative='two-sided')
        r_plus, _ = wilcoxon(
            df_inferential[name0], df_inferential[name1], alternative='greater')
        p_values.append(wilcoxon_p_value)
        r_pluses.append(r_plus)

    reject, p_values, _, _ = multipletests(
        pvals=p_values, alpha=.05, method='hs', is_sorted=False, returnsorted=False)
    count = len(df_inferential)
    middle_rank_sum = ((count * (count + 1)) / 2) / 2
    for i, (name0, name1) in enumerate(pairs):
        if reject[i] and r_pluses[i] != middle_rank_sum:
            winner = name0 if r_pluses[i] > middle_rank_sum else name1
        else:
            winner = 'None'
        f.write(
            '{} = {}, wilcoxon p-value: {}, reject: {}, winner: {}\n'.format(name0, name1, p_values[i], reject[i], winner))


def table(config, df_testing: pd.DataFrame, metrics):
    metric_columns = {metric.column: ['mean', 'std'] for metric in metrics}
    df_table = df_testing.groupby(
        by=['dataset', 'classifier']).agg(metric_columns)
    df_table = df_table.round(4)
    df_table = df_table.apply(
        lambda row: format_metric(metrics, row), axis='columns')
    dir = dir_to_path(config['filename'])
    df_table.to_csv(dir / 'table.csv')


def format_metric(metrics, row):
    data = []
    index = []
    for metric in metrics:
        data.append('{:06.2%} ({:06.2%})'.format(
            row[(metric.column, 'mean')], row[(metric.column, 'std')]))
        index.append(metric.name)
    return pd.Series(data, index=index)


def datasets_statistics(config):
    language = {
        'fabric8': 'Java',
        'jgroups': 'Java',
        'camel': 'Java',
        'tomcat': 'Java',
        'brackets': 'JavaScript',
        'neutron': 'Python',
        'spring-integration': 'Java',
        'broadleaf': 'Java',
        'nova': 'Python',
        'npm': 'JavaScript',
    }
    rows = []
    columns = pd.MultiIndex.from_tuples([('dataset', ''), ('code changes', ''), ('defect-inducing proportions', 'entire dataset'),
                                         ('defect-inducing proportions', 'validation segment'), ('defect-inducing proportions', 'testing segment'), ('language', '')])
    for dataset in sorted(DATASETS):
        row = []
        df_dataset = make_stream(dataset)
        row.append(dataset)
        row.append(len(df_dataset))
        percentual_format = '{:03.0%}'
        row.append(percentual_format.format(df_dataset['target'].mean()))
        row.append(percentual_format.format(
            df_dataset.loc[:5000, 'target'].mean()))
        row.append(percentual_format.format(
            df_dataset.loc[5000:, 'target'].mean()))
        row.append(language[dataset])
        rows.append(row)
    dir = dir_to_path(config['filename'])
    df_datasets = pd.DataFrame(rows, columns=columns)
    df_datasets.to_csv(dir / 'datasets.csv', index=False)
