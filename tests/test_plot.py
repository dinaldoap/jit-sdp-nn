from jitsdp import plot

from constants import DIR

import numpy as np
import pandas as pd


def test_plot_recalls_gmean():
    data = pd.DataFrame({
        'timestep': [0, 1, 2],
        'r0':    [0., 1., 2.],
        'r1':    [1., 0., 8.],
        'gmean': [0., 0., 4.],
    })
    plot.plot_recalls_gmean(data, dir=DIR)
    assert (DIR / 'recalls_gmean.png').exists()
