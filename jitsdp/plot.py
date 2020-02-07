import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

sns.set(rc={'figure.figsize': (12, 9)})

DIR = Path('logs')

def create_dir(dir):
    dir.mkdir(parents=True, exist_ok=True)

def plot_recalls_gmean(data, dir=DIR):
    avg_gmean = data['gmean'].mean()
    data = data.melt(id_vars='timestep',
                    value_vars=['r0', 'r1', 'gmean'],
                    var_name='metric',
                    value_name='value')
    ax = sns.lineplot(x='timestep', y='value',
                 hue='metric', data=data)
    ax.axhline(avg_gmean, ls='--', label='avg(gmean)')
    plt.legend()
    create_dir(dir)
    plt.savefig(dir / 'recalls_gmean.png')
