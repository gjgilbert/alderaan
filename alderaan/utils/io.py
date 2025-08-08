__all__ = ['expand_config_path', 
           'parse_koi_catalog',
           'parse_holczer16_catalog',
          ]


import os
import sys
from pathlib import Path

base_path = Path(__file__).resolve().parents[2]
if str(base_path) not in sys.path:
    sys.path.insert(0, str(base_path))

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from alderaan.schema.ephemeris import Ephemeris


def expand_config_path(path_str):
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
