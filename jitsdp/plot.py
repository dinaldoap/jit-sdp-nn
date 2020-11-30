# coding=utf-8
from jitsdp.constants import DIR
from jitsdp.utils import mkdir, split_proposal_baseline

import matplotlib.pyplot as plt
import Orange as og
import seaborn as sns


def setup():
    plt.figure(figsize=(14, 9))
    sns.set()


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


def plot_boxplot(data, metric, dir):
    setup()
    ax = sns.barplot(data=data, x='dataset', y=metric.column, hue='classifier')
    ax.set_title('{}'.format(metric.name))
    plt.savefig(dir / '{}.png'.format(metric.column))
    plt.clf()


def plot_tuning_convergence(data, dir):
    setup()
    ax = sns.catplot(x="experiment_size", y="g-mean",
                     hue="name", col="dataset",
                     data=data, kind="boxen",
                     k_depth='proportion', outlier_prop=0.05,
                     showfliers=False, col_wrap=3)
    ax.set_axis_labels(x_var='experiment size')
    plt.savefig(dir / 'tuning_convergence.png')
    plt.clf()


def plot_critical_distance(avg_rank, data, metric, dir):
    if metric.baseline:
        test = 'bonferroni-dunn'
        _, baseline_name = split_proposal_baseline(data.columns)
        cdmethod = data.columns.get_loc(baseline_name[0])
    else:
        test = 'nemenyi'
        cdmethod = None

    cd = og.evaluation.compute_CD(
        avranks=avg_rank, n=len(data), alpha='0.05', test=test)
    og.evaluation.graph_ranks(
        avranks=avg_rank, names=data.columns, cd=cd, cdmethod=cdmethod)
    plt.savefig(dir / '{}_cd.png'.format(metric.column), bbox_inches='tight')
    plt.clf()
