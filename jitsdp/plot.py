# coding=utf-8
from jitsdp.constants import DIR
from jitsdp.utils import mkdir, split_proposal_baseline

import matplotlib.pyplot as plt
import Orange as og
import seaborn as sns


def setup(font_scale=None):
    plt.figure(figsize=(14, 9))
    if font_scale is None:
        sns.set()
    else:
        sns.set(font_scale=font_scale)


def plot_oversampling_boosting_factors(data, x_metric, value_metrics, grid_metric, dir):
    setup()
    cols_to_names = {metric.column: metric.name for metric in [
        x_metric] + value_metrics + [grid_metric]}
    data = data.rename(cols_to_names, axis='columns')
    value_metric_names = [metric.name for metric in value_metrics]
    data = data.melt(id_vars=[x_metric.name, grid_metric.name],
                     value_vars=value_metric_names,
                     var_name='factor',
                     value_name='value')
    sns.relplot(x=x_metric.name, y='value',
                hue='factor', data=data,
                kind='line', aspect=2,
                row=grid_metric.name)
    plt.savefig(dir / 'oversampling_boosting_factors.png', bbox_inches='tight')
    plt.clf()


def plot_recalls_gmean(data, config, dir=DIR):
    __plot_metrics(data=data, config=config, dir=dir, metrics=[
                   'r0', 'r1', 'r0-r1', 'g-mean'], filename='recalls_gmean.png')


def plot_proportions(data, config, dir=DIR):
    __plot_metrics(data=data, config=config, dir=dir, metrics=[
                   'tr1', 'te1', 'pr1'], filename='proportions.png')
    __plot_metrics(data=data, config=config, dir=dir, metrics=[
                   'th-ma'], filename='distance_induced_rate.png')
    __plot_metrics(data=data, config=config, dir=dir, metrics=[
                   'th-pr1'], filename='distance_actual_rate.png')


def __plot_metrics(data, config, dir, metrics, filename):
    assert len(metrics) <= 4, 'Only support four or less metrics.'
    avgs = [data[metric].mean() for metric in metrics]
    data = data.melt(id_vars='timestep',
                     value_vars=metrics,
                     var_name='metric',
                     value_name='value')
    setup()
    ax = sns.lineplot(x='timestep', y='value',
                      hue='metric', data=data)
    styles = ['--', '-.', ':', '-']
    for i, metric in enumerate(metrics):
        ax.axhline(avgs[i], ls=styles[i], c='black',
                   label='avg({})={:.2f}'.format(metric, avgs[i]))
    ax.set_title(config['dataset'])
    plt.legend()
    mkdir(dir)
    plt.savefig(dir / filename)
    plt.clf()


def plot_streams(data, metrics, dir, filename):
    __plot_metrics_grid(data=data, dir=dir, metrics=metrics,
                        filename=filename, col='classifier', row='dataset')


def __plot_metrics_grid(data, dir, metrics, filename, col, row):
    assert len(metrics) <= 5, 'Only support four or less metrics.'
    setup(font_scale=2.5)
    cols_to_names = {metric.column: metric.name for metric in metrics}
    data = data.rename(cols_to_names, axis='columns')
    metric_names = [metric.name for metric in metrics]
    data = data.melt(id_vars=['timestep', col, row],
                     value_vars=metric_names,
                     var_name='metric',
                     value_name='value')

    sns.relplot(x='timestep', y='value',
                hue='metric', data=data,
                kind='line', aspect=2.5,
                col=col, row=row,
                facet_kws={'sharex': False})
    plt.savefig(dir / filename, bbox_inches='tight')
    plt.clf()


def plot_boxplot(data, metric, dir):
    setup()
    ax = sns.barplot(data=data, x='dataset', y=metric.column, hue='classifier')
    ax.set_title('{}'.format(metric.name))
    plt.savefig(dir / '{}.png'.format(metric.column))
    plt.clf()


def plot_tuning_convergence(data, dir):
    setup()
    ax = sns.catplot(x="experiment_size", y="g-mean",
                     hue="classifier", col="dataset",
                     data=data, kind="boxen",
                     k_depth='proportion', outlier_prop=0.05,
                     showfliers=False, col_wrap=3)
    ax.set_axis_labels(x_var='number of random configurations')
    plt.savefig(dir / 'tuning_convergence.png', bbox_inches='tight')
    plt.clf()


def plot_critical_distance(avg_rank, data, metric, dir):
    if metric.baseline:
        test = 'bonferroni-dunn'
        _, baseline_name = split_proposal_baseline(data.columns)
        cdmethod = data.columns.get_loc(baseline_name[0])
    else:
        test = 'nemenyi'
        cdmethod = None
    try:
        cd = og.evaluation.compute_CD(
            avranks=avg_rank, n=len(data), alpha='0.05', test=test)
    except IndexError:
        cd = None
    og.evaluation.graph_ranks(
        avranks=avg_rank, names=data.columns, cd=cd, cdmethod=cdmethod)
    plt.savefig(dir / '{}_cd.png'.format(metric.column), bbox_inches='tight')
    plt.clf()


def plot_fix_delay(data, dir):
    setup()
    data['fix_delay'] = data['fix_delay'] + 1
    ax = sns.boxenplot(x='dataset', y='fix_delay', data=data)
    ax.set(ylabel='days (log scale)')
    medians = data.groupby('dataset').agg({'fix_delay': 'median'})
    medians = medians['fix_delay']
    min_median = round(medians.min())
    max_median = round(medians.max())
    color = 'black'
    alpha = .3
    ax.axhline(min_median, ls='--', c=color, alpha=alpha,
               label='{} and {} days (log scale)'.format(min_median, max_median))
    ax.axhline(max_median, ls='--', c=color, alpha=alpha)
    plt.legend()
    plt.yscale('log')
    plt.savefig(dir / 'fix_delay.png')
    plt.clf()
