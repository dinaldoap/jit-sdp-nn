from jitsdp.constants import DIR
from jitsdp.utils import mkdir

import seaborn as sns
import matplotlib.pyplot as plt

sns.set(rc={'figure.figsize': (12, 9)})


def plot_recalls_gmean(data, config, dir=DIR):
    avg_gmean = data['gmean'].mean()
    data = data.melt(id_vars='timestep',
                     value_vars=['r0', 'r1', 'gmean'],
                     var_name='metric',
                     value_name='value')
    ax = sns.lineplot(x='timestep', y='value',
                      hue='metric', data=data)
    ax.axhline(avg_gmean, ls='--', label='avg(gmean)={:.2f}'.format(avg_gmean))
    ax.set_title(config['dataset'])
    plt.legend()
    subdir = dir / config['dataset']
    mkdir(subdir)
    plt.savefig(subdir / 'recalls_gmean.png')
    plt.clf()


def plot_proportions(data, config, dir=DIR):
    avg_p0 = data['p0'].mean()
    avg_p1 = data['p1'].mean()
    data = data.melt(id_vars='timestep',
                     value_vars=['p0', 'p1'],
                     var_name='metric',
                     value_name='value')
    ax = sns.lineplot(x='timestep', y='value',
                      hue='metric', data=data)
    ax.axhline(avg_p0, ls='--', c='black', label='avg(p0)={:.2f}'.format(avg_p0))
    ax.axhline(avg_p1, ls='-.', c='black', label='avg(p1)={:.2f}'.format(avg_p1))
    ax.set_title(config['dataset'])
    plt.legend()
    subdir = dir / config['dataset']
    mkdir(subdir)
    plt.savefig(subdir / 'proportions.png')
    plt.clf()
