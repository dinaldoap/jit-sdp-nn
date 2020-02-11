from jitsdp.constants import DIR
from jitsdp.utils import mkdir

import seaborn as sns
import matplotlib.pyplot as plt

sns.set(rc={'figure.figsize': (12, 9)})


def plot_recalls_gmean(data, config, dir=DIR):
    __plot_metrics(data=data, config=config, dir=dir, metrics=[
                   'r0', 'r1', 'gmean'], filename='recalls_gmean.png')


def plot_proportions(data, config, dir=DIR):
    __plot_metrics(data=data, config=config, dir=dir, metrics=[
                   'p0', 'p1'], filename='proportions.png')


def __plot_metrics(data, config, dir, metrics, filename):
    assert len(metrics) <= 3, 'Only support three or less metrics.'
    avgs = [data[metric].mean() for metric in metrics]
    data = data.melt(id_vars='timestep',
                     value_vars=metrics,
                     var_name='metric',
                     value_name='value')
    ax = sns.lineplot(x='timestep', y='value',
                      hue='metric', data=data)
    styles = ['--', '-.', '-']
    for i, metric in enumerate(metrics):
        ax.axhline(avgs[i], ls=styles[i], c='black',
                   label='avg({})={:.2f}'.format(metric, avgs[i]))
    ax.set_title(config['dataset'])
    plt.legend()
    subdir = dir / config['dataset']
    mkdir(subdir)
    plt.savefig(subdir / filename)
    plt.clf()
