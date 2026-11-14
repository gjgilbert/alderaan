__all__ = [
    "copy_input_target_catalog",
    "dynesty_results_to_fits",
]


import os

import numpy as np
import pandas as pd
from astropy.io import fits
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


def dynesty_results_to_fits(results, mission, target, run_id, results_dir):
    """
    results : dynesty.DynamicNestedSampling.results
    mission : str
    target : str
    run_id : str
    results_dir : str
    """
    npl = (results.samples.shape[1] - 2) // 5

    # package nested samples
    samples_keys = []

    for n in range(npl):
        samples_keys += f"C0_{n} C1_{n} ROR_{n} IMPACT_{n} DUR14_{n}".split()

    samples_keys += ["LD_Q1", "LD_Q2"]
    samples_keys += ["LN_WT", "LN_LIKE", "LN_Z"]

    samples_data = np.vstack(
        [results.samples.T, results.logwt, results.logl, results.logz]
    ).T
    samples_df = pd.DataFrame(samples_data, columns=samples_keys)

    # primary HDU
    primary_hdu = fits.PrimaryHDU()
    primary_hdu.header["MISSION"] = mission
    primary_hdu.header["TARGET"] = target
    primary_hdu.header["RUN_ID"] = run_id
    primary_hdu.header["NPL"] = npl

    # samples HDU
    samples_hdu = fits.BinTableHDU(
        data=samples_df.to_records(index=False), name="SAMPLES"
    )

    samples_hdu.header["NITER"] = results.niter
    samples_hdu.header["NBATCH"] = len(results.batch_nlive)
    for i, nlive in enumerate(results.batch_nlive):
        samples_hdu.header[f"NLIVE{i}"] = nlive
    samples_hdu.header["EFF"] = results.eff

    # build HDU List
    hdul = fits.HDUList([primary_hdu, samples_hdu])

    # add transit times to HDU List
    for n in range(npl):
        ttimes_file = os.path.join(
            results_dir,
            f"{target}_{str(n).zfill(2)}_quick.ttvs",
        )
        ttimes_keys = "INDEX TTIME MODEL OUT_PROB OUT_FLAG".split()
        ttimes_data = np.loadtxt(ttimes_file)

        ttimes_df = pd.DataFrame(ttimes_data, columns=ttimes_keys)
        ttimes_df.INDEX = ttimes_df.INDEX.astype("int")
        ttimes_df.OUT_FLAG = ttimes_df.OUT_FLAG.astype("int")

        ttimes_hdu = fits.BinTableHDU(
            data=ttimes_df.to_records(index=False),
            name=f"TTIMES_{str(n).zfill(2)}",
        )

        hdul.append(ttimes_hdu)

    return hdul
