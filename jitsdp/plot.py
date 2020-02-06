import seaborn as sns
import matplotlib.pyplot as plt
import os

sns.set(rc={'figure.figsize': (12, 9)})


def plot_recalls_gmean(data, dir='logs'):
    data = data.melt(id_vars='timestep',
                    value_vars=['r0', 'r1', 'gmean'],
                    var_name='metric',
                    value_name='value')
    sns.lineplot(x='timestep', y='value',
                 hue='metric', data=data)
    plt.savefig(os.path.join(dir, 'recalls_gmean.png'))
