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
