__all__ = ['resolve_config_path',
           'parse_koi_catalog',
           'parse_holczer16_catalog',
           'copy_input_target_catalog',
           'dynesty_results_to_fits',
           'fetch_toi_table',
           'parse_toi_catalog',
          ]


import os
import sys
import urllib.request
from astropy.io import fits
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from alderaan.ephemeris import Ephemeris


def resolve_config_path(path_str, base_path):
    return os.path.join(str(Path(path_str.format(base_path=base_path)).resolve()),'')


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

def fetch_k2_table(catalog_dir, force_refresh=False):
    """Downloads the full K2 table from the NASA Exoplanet Archive TAP API.

    The table is cached as a local CSV file. If a cached file already exists
    in catalog_dir and force_refresh=False, the download is skipped.

    Args:
        catalog_dir (str) : directory to store the cached CSV
        force_refresh (bool) : if True, re-download even if cached file exists

    Returns:
        str : path to the cached CSV file
    """
    os.makedirs(catalog_dir, exist_ok=True)
    cached_path = os.path.join(catalog_dir, "k2_conf_cand.csv")

    if os.path.exists(cached_path) and not force_refresh:
        print(f"Using cached K2 table: {cached_path}")
        return cached_path

    tap_url = (
        "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?"
        "query=select+*+from+k2pandc&format=csv"
    )
    print(f"Downloading K2 table from Exoplanet Archive...")
    urllib.request.urlretrieve(tap_url, cached_path)
    print(f"K2 table saved to: {cached_path}")

    return cached_path


def fetch_toi_table(catalog_dir, force_refresh=False):
    """Downloads the full TOI table from the NASA Exoplanet Archive TAP API.

    The table is cached as a local CSV file. If a cached file already exists
    in catalog_dir and force_refresh=False, the download is skipped.

    Args:
        catalog_dir (str) : directory to store the cached CSV
        force_refresh (bool) : if True, re-download even if cached file exists

    Returns:
        str : path to the cached CSV file
    """
    os.makedirs(catalog_dir, exist_ok=True)
    cached_path = os.path.join(catalog_dir, "toi_catalog.csv")

    if os.path.exists(cached_path) and not force_refresh:
        print(f"Using cached TOI table: {cached_path}")
        return cached_path

    tap_url = (
        "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?"
        "query=select+*+from+TOI&format=csv"
    )
    print(f"Downloading TOI table from Exoplanet Archive...")
    urllib.request.urlretrieve(tap_url, cached_path)
    print(f"TOI table saved to: {cached_path}")

    return cached_path


def parse_k2_catalog(filepath, epic_id):
    """Reads a K2 catalog CSV and returns a DataFrame matching ALDERAAN format.

    Analogous to parse_koi_catalog() but for K2 Objects of Interest.

    Args:
        filepath (str) : path to K2 CSV file (from fetch_k2_table)
        epic_id (str) : system-level EPIC identifier, e.g. "EPIC 201912552"

    Returns:
        pd.DataFrame : catalog with columns:
            [tic_id, toi_id, npl, period, epoch, depth, duration,
             impact, limbdark_1, limbdark_2, Rstar, Mstar, Teff]

    # NOTE: K2 exoplanet archive catalog does not have limb darkening. Default for now is [0.4, 0.2].
    """

    # Parse the system number from epic_id string (e.g., "EPIC-201912552" -> 201912552)
    system_num = int(epic_id.split("-")[1])

    # Read the full K2 table
    k2_table = pd.read_csv(filepath, comment='#')

    # Drop rows with missing 'epic_hostname' values
    k2_table = k2_table[k2_table['epic_hostname'].notna()]

    # Strip "EPIC " prefix from 'epic_candname' column
    k2_table['epic_hostname'] = k2_table['epic_hostname'].str.replace('EPIC ', '', regex=False)

    # Filter to planet candidates in this system
    # The 'epic_hostname' column contains values like 201912552, 201912553, etc.
    system_mask = k2_table['epic_hostname'].apply(lambda x: int(x) == system_num)
    system = k2_table.loc[system_mask].copy()

    if len(system) == 0:
        raise ValueError(f"No K2 entries found for {epic_id} (system number {system_num})")

    # Build the output DataFrame
    npl = len(system)

    catalog = pd.DataFrame()
    catalog['epic_id'] = system['epic_hostname'].values
    catalog['epic_cand_name'] = system['epic_candname'].values
    catalog['npl'] = npl
    catalog['period'] = system['pl_orbper'].values
    catalog['epoch'] = system['pl_tranmid'].values - 2454833.0  # BJD -> BKJD
    catalog['depth'] = (system['pl_ratror'].values**2)*1E6          # ppm
    catalog['duration'] = system['pl_trandur'].values        # hours

    # Limb darkening: default to [0.4, 0.2]
    catalog['limbdark_1'] = 0.4
    catalog['limbdark_2'] = 0.2

    # Impact parameter: default to 0.5
    catalog['impact'] = 0.5

    # Stellar parameters (may have NaNs)
    catalog['Rstar'] = system['st_rad'].values if 'st_rad' in system.columns else np.nan
    catalog['Mstar'] = system['st_mass'].values if 'st_mass' in system.columns else np.nan
    catalog['Teff'] = system['st_teff'].values if 'st_teff' in system.columns else np.nan

    # Sort by ascending period
    catalog = catalog.sort_values(by='period').reset_index(drop=True)

    # Consistency checks (same as parse_koi_catalog)
    if not all(epic == catalog.epic_id.to_numpy()[0] for epic in catalog.epic_id):
        raise ValueError("There are inconsistencies with EPIC_ID in the K2 catalog")

    if not all(
        ld_u1 == catalog.limbdark_1.to_numpy()[0] for ld_u1 in catalog.limbdark_1
    ):
        raise ValueError("There are inconsistencies with LD_U1 in the K2 catalog")

    if not all(
        ld_u2 == catalog.limbdark_2.to_numpy()[0] for ld_u2 in catalog.limbdark_2
    ):
        raise ValueError("There are inconsistencies with LD_U2 in the K2 catalog")

    # Check for NaN valued transit parameters
    if np.any(
        np.isnan(np.array(catalog["period epoch depth duration impact".split()]))
    ):
        raise ValueError("NaN values found in K2 catalog for required transit parameters")

    return catalog


def parse_toi_catalog(filepath, toi_id):
    """Reads a TOI catalog CSV and returns a DataFrame matching ALDERAAN format.

    Analogous to parse_koi_catalog() but for TESS Objects of Interest.

    Args:
        filepath (str) : path to TOI CSV file (from fetch_toi_table)
        toi_id (str) : system-level TOI identifier, e.g. "TOI-5145"

    Returns:
        pd.DataFrame : catalog with columns:
            [tic_id, toi_id, npl, period, epoch, depth, duration,
             impact, limbdark_1, limbdark_2, Rstar, Mstar, Teff]
    """
    # Parse the system number from toi_id string (e.g., "TOI-5145" -> 5145)
    system_num = int(toi_id.split("-")[1])

    # Read the full TOI table
    toi_table = pd.read_csv(filepath, comment='#')

    # Filter to planet candidates in this system
    # The 'toi' column contains values like 5145.01, 5145.02, etc.
    system_mask = toi_table['toi'].apply(lambda x: int(x) == system_num)
    system = toi_table.loc[system_mask].copy()

    if len(system) == 0:
        raise ValueError(f"No TOI entries found for {toi_id} (system number {system_num})")

    # Build the output DataFrame
    npl = len(system)

    catalog = pd.DataFrame()
    catalog['tic_id'] = system['tid'].values
    catalog['toi_id'] = [toi_id] * npl
    catalog['toi_pl'] = system['toi'].values
    catalog['npl'] = npl
    catalog['period'] = system['pl_orbper'].values
    catalog['epoch'] = system['pl_tranmid'].values - 2457000.0  # BJD -> BTJD
    catalog['depth'] = system['pl_trandep'].values          # ppm
    catalog['duration'] = system['pl_trandurh'].values        # hours

    # Impact parameter: not in TOI table, default to 0.5
    catalog['impact'] = 0.5

    # Limb darkening: default to [0.4, 0.2]
    catalog['limbdark_1'] = 0.4
    catalog['limbdark_2'] = 0.2

    # Stellar parameters (may have NaNs)
    catalog['Rstar'] = system['st_rad'].values if 'st_rad' in system.columns else np.nan
    catalog['Mstar'] = system['st_mass'].values if 'st_mass' in system.columns else np.nan
    catalog['Teff'] = system['st_teff'].values if 'st_teff' in system.columns else np.nan

    # Sort by ascending period
    catalog = catalog.sort_values(by='period').reset_index(drop=True)

    # Consistency checks (same as parse_koi_catalog)
    if not all(tic == catalog.tic_id.to_numpy()[0] for tic in catalog.tic_id):
        raise ValueError("There are inconsistencies with TIC_ID in the TOI catalog")

    if not all(
        ld_u1 == catalog.limbdark_1.to_numpy()[0] for ld_u1 in catalog.limbdark_1
    ):
        raise ValueError("There are inconsistencies with LD_U1 in the TOI catalog")

    if not all(
        ld_u2 == catalog.limbdark_2.to_numpy()[0] for ld_u2 in catalog.limbdark_2
    ):
        raise ValueError("There are inconsistencies with LD_U2 in the TOI catalog")

    # Check for NaN valued transit parameters
    if np.any(
        np.isnan(np.array(catalog["period epoch depth duration impact".split()]))
    ):
        raise ValueError("NaN values found in TOI catalog for required transit parameters")

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


def save_omc_ephemeris(filename, omc, verbose=True):
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
        filename,
        data_out,
        fmt=("%1d", "%.8f", "%.8f", "%.8f", "%1d"),
        delimiter="\t",
    )

    if verbose:
        print(f"successfully wrote omc ephemeris to {filename}")

def dynesty_results_to_fits(results, context):
    """
    results : dynesty.DynamicNestedSampling.results
    context : alderaan.utils.pipeline.PipelineContext
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
    primary_hdu.header["MISSION"] = context.mission
    primary_hdu.header["TARGET"] = context.target
    primary_hdu.header["RUN_ID"] = context.run_id
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
    hduL = fits.HDUList([primary_hdu, samples_hdu])

    # add transit times to HDU List
    for n in range(npl):
        ttimes_file = os.path.join(
            context.results_dir,
            f"{context.target}_{str(n).zfill(2)}_quick.ttvs",
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

        hduL.append(ttimes_hdu)

    return hduL