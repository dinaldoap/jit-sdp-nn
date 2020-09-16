# coding=utf-8
from jitsdp.constants import DIR
from jitsdp.utils import mkdir

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(rc={'figure.figsize': (14, 9)})


def plot_recalls_gmean(data, config, dir=DIR):
    __plot_metrics(data=data, config=config, dir=dir, metrics=[
                   'r0', 'r1', 'r0-r1', 'gmean'], filename='recalls_gmean.png')


def plot_proportions(data, config, dir=DIR):
    __plot_metrics(data=data, config=config, dir=dir, metrics=[
                   't1', 's1', 'p1'], filename='proportions.png')


def __plot_metrics(data, config, dir, metrics, filename):
    assert len(metrics) <= 4, 'Only support three or less metrics.'
    avgs = [data[metric].mean() for metric in metrics]
    data = data.melt(id_vars='timestep',
                     value_vars=metrics,
                     var_name='metric',
                     value_name='value')
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


def plot_boxplot(data, dir):
    metrics = {
        'avg_gmean': 'g-mean',
        'avg_r0-r1': '|r0-r1|'
    }
    for metric_id, metric_name in metrics.items():
        ax = sns.boxplot(data=data, x='dataset', y=metric_id, hue='model')
        ax.set_title('{}'.format(metric_name))
        plt.savefig(dir / '{}.png'.format(metric_id))
        plt.clf()
