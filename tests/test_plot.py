from jitsdp import plot

import numpy as np
import pandas as pd
import os

def test_plot_recalls_gmean(tmp_path):
    data = pd.DataFrame({
        'timestep': [0, 1, 2],
        'r0':    [0., 1., 2.],
        'r1':    [1., 0., 8.],
        'gmean': [0., 0., 4.],
    })
    plot.plot_recalls_gmean(data, dir=tmp_path)
    assert (tmp_path / 'recalls_gmean.png').exists()