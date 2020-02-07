from jitsdp.data import save_data, load_data

from constants import DIR

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal


def test_save_load():
    expected = pd.DataFrame({
        'col': [1, 2, 3]
    })
    save_data(expected, DIR)
    saved_loaded = load_data(DIR)
    assert_frame_equal(expected, saved_loaded)
