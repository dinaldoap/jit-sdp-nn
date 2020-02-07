from jitsdp.data import save_results, load_results

from constants import DIR

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal


def test_save_load_results():
    expected = pd.DataFrame({
        'col': [1, 2, 3]
    })
    save_results(expected, DIR)
    saved_loaded = load_results(DIR)
    assert_frame_equal(expected, saved_loaded)
