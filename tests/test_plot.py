# coding=utf-8
from jitsdp import plot

from jitsdp.constants import DIR

import numpy as np
import pandas as pd


def test_plot_recalls_gmean():
    config = {
        'dataset': 'brackets'
    }
    data = pd.DataFrame({
        'timestep': [0, 1, 2],
        'r0':    [0., 1., 2.],
        'r1':    [1., 0., 8.],
        'r0-r1': [1., 1., 6.],
        'g-mean': [0., 0., 4.],
    })
    plot.plot_recalls_gmean(data=data, config=config, dir=DIR)
    assert (DIR / 'recalls_gmean.png').exists()


def test_plot_proportions():
    config = {
        'dataset': 'brackets'
    }
    data = pd.DataFrame({
        'timestep': [0, 1, 2],
        'tr1':    [1., .5, 0.],
        'te1':    [1., .5, 1.],
        'pr1':    [0., .5, 0.],
        'th-ma':    [.1, .2, .1],
        'th-pr1':    [.1, .2, .2],
    })
    plot.plot_proportions(data=data, config=config, dir=DIR)
    assert (DIR / 'proportions.png').exists()
    assert (DIR / 'rate_driven_val.png').exists()
    assert (DIR / 'rate_driven_test.png').exists()
