__all__ = [
    "copy_input_target_catalog",
]


import os

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal


def copy_input_target_catalog(filepath_master, filepath_copy):
    df_master = pd.read_csv(filepath_master, index_col=0)

    if os.path.exists(filepath_copy):
        df_copy = pd.read_csv(filepath_copy, index_col=0)

        try:
            assert_frame_equal(df_master, df_copy, check_like=True)
        except AssertionError:
            print(
                f"AssertionError: existing file {filepath_copy} "
                f"does not match active file {filepath_master}"
            )

    else:
        os.makedirs(os.path.dirname(filepath_copy), exist_ok=True)
        df_master.to_csv(filepath_copy)


def save_omc_ephemeris(
    filename, index, ttime_obs, ttime_mod, out_prob, out_class, verbose=True
):
    data_out = np.vstack([index, ttime_obs, ttime_mod, out_prob, out_class]).swapaxes(
        0, 1
    )

    np.savetxt(
        filename,
        data_out,
        fmt=("%1d", "%.8f", "%.8f", "%.8f", "%1d"),
        delimiter="\t",
    )

    if verbose:
        print(f"successfully wrote omc ephemeris to {filename}")
