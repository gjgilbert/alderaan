__all__ = ['resolve_config_path',
           'copy_input_target_catalog',
           'parse_koi_catalog',
           'parse_holczer16_catalog',
           'save_omc_ephemeris',
           'save_dynesty_results_to_fits',
          ]


import os
import sys
from pathlib import Path

from astropy.io import fits
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from alderaan.ephemeris import Ephemeris


def resolve_config_path(path_str, base_path):
    return os.path.join(str(Path(path_str.format(base_path=base_path)).resolve()),'')


def copy_input_target_catalog(filepath_master, filepath_copy):
    df_master = pd.read_csv(filepath_master, index_col=0)
    
    if os.path.exists(filepath_copy):
        df_copy = pd.read_csv(filepath_copy, index_col=0)

        try:
            assert_frame_equal(df_master, df_copy, check_like=True)
        except AssertionError:
            print(f"AssertionError: existing file {filepath_copy} does not match active file {filepath_master}")

    else:
        os.makedirs(os.path.dirname(filepath_copy), exist_ok=True)
        df_master.to_csv(filepath_copy)


def parse_koi_catalog(filepath, koi_id):
    """Reads a Kepler Object of Interest Catalog and performs consistency checks
    
    Expected columns are:
        [kic_id, koi_id, npl, period, epoch, depth, duration, impact, ld_u1, ld_u2]

    Args:
        filepath (str) : path to csv file
        koi_id (str) : KOI identification number in the format, e.g., K01234

    Returns:
        pd.DataFrame: catalog of target star/planet properties
    """
    # read catalog from csv file
    catalog = pd.read_csv(filepath, index_col=0)
    catalog = catalog.loc[catalog.koi_id == koi_id]

    # sort by ascending period
    catalog = catalog.sort_values(by='period').reset_index(drop=True)

    # check for consistency in multi-planet systems
    if not all(kic_id == catalog.kic_id.to_numpy()[0] for kic_id in catalog.kic_id):
        raise ValueError("There are inconsistencies with KIC in the csv input file")

    if not all(npl == catalog.npl.to_numpy()[0] for npl in catalog.npl):
        raise ValueError("There are inconsistencies with NPL in the csv input file")

    if not all(
        ld_u1 == catalog.limbdark_1.to_numpy()[0] for ld_u1 in catalog.limbdark_1
    ):
        raise ValueError("There are inconsistencies with LD_U1 in the csv input file")

    if not all(
        ld_u2 == catalog.limbdark_2.to_numpy()[0] for ld_u2 in catalog.limbdark_2
    ):
        raise ValueError("There are inconsistencies with LD_U2 in the csv input file")

    # check for NaN valued transit parameters
    if np.any(
        np.isnan(np.array(catalog["period epoch depth duration impact".split()]))
    ):
        raise ValueError("NaN values found in input catalog")
    
    return catalog


def parse_holczer16_catalog(filepath, koi_id, num_planets):
    """Reads transit time table from Holczer+2016 into a list of Ephemeris objects

    Automatically corrects for zero-point offsets between catalogs
      - Holczer+2016 used BJD - 2454900
      - Kepler Project used BJKD = BJD - 2454833

    Args:
        filepath (str) : path to Holczer+2016 table
        koi_id (str) : KOI identification number in the format, e.g., K01234
        num_planets : total number of planets in the system

    Returns:
        list : list of (0,num_planets) Ephemeris objects
    """
    data = np.loadtxt(filepath, usecols=[0,1,2,3,4], dtype=str)
    ephemerides = []

    planet_id = data[:,0]
    index = np.array(data[:,1], dtype=int)
    ttime = np.array(data[:,2], dtype=float) + np.array(data[:,3],dtype=float)/24/60 + 67
    error = np.array(data[:,4], dtype=float)/24/60

    for n in range(num_planets):
        use = planet_id == f"{int(koi_id[1:])}.0{1+n}"
        if np.sum(use) > 0:
            ephemerides.append(Ephemeris(index=index[use], ttime=ttime[use], error=error[use]))

    return ephemerides


def save_omc_ephemeris(filepath, omc, verbose=True):
    if omc.quality is not None:
        q = omc.quality
    else:
        q = np.ones(len(omc.ttime), dtype=bool)

    _static_ephemeris = omc._static_epoch + omc._static_period * omc.index[q]

    data_out = np.vstack(
        [omc.index[q],
            omc.yobs[q] + _static_ephemeris,
            omc.ymod[q] + _static_ephemeris,
            omc.out_prob[q],
            omc.out_class[q],
        ]
    ).swapaxes(0,1)

    np.savetxt(
        filepath,
        data_out,
        fmt=("%1d", "%.8f", "%.8f", "%.8f", "%1d"),
        delimiter="\t",
    )

    if verbose:
        print(f"successfully wrote omc ephemeris to {filepath}")


def save_detrended_litecurve(filepath, litecurve, target):
    """
    Docstring
    """
    assert np.all(litecurve.obsmode == litecurve.obsmode[0])

    hdul = [fits.PrimaryHDU()]
    hdul[0].header["TARGET"] = target
    hdul[0].header["OBSMODE"] = litecurve.obsmode[0]

    for k, v in litecurve.__dict__.items():
        if isinstance(v, np.ndarray):
            if np.issubdtype(v.dtype, np.bool_):
                v = v.astype(np.int16)
            if np.issubdtype(v.dtype, np.number):
                hdul.append(fits.ImageHDU(v, name=k.upper()))

    hdul = fits.HDUList(hdul)
    hdul.writeto(filepath, overwrite=True)
    

def save_dynesty_results(output_dir, results, mission, target, run_id):
    """
    Docstring
    """
    npl = (results.samples.shape[1] - 2) // 5

    # package nested samples
    samples_keys = []

    for n in range(npl):
        samples_keys += "C0_{0} C1_{0} ROR_{0} IMPACT_{0} DUR14_{0}".format(n).split()

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
    try:
        for n in range(npl):
            ttimes_file = os.path.join(
                output_dir,
                f"results/{run_id}/{target}/{target}_{str(n).zfill(2)}_quick.ttvs",
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
   
    except FileNotFoundError as e:
        print(e)

    return hdul
