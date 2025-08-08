__all__ = [
    "parse_catalog",
    "read_mast_files",
    "cleanup_lkc",
    "LightKurve_to_LiteCurve",
    "transit_parameters_to_dataframe",
    "load_detrended_lightcurve",
    "dynesty_results_to_fits",
]


import os
from copy import deepcopy

from astropy.io import fits
import lightkurve as lk
import numpy as np
import pandas as pd

from .LiteCurve import LiteCurve


def parse_catalog(file, mission, target):
    """
    Parse catalog data for a single star given an input csv file

    Parameters
    ----------
        file : str
            path to input csv file with star and planet properties
        mission : str
            can be 'Kepler' or 'Kepler-Validation'
        target : str
            target id, e.g. K00137

    Returns
    -------
        catalog : pd.DataFrame
            dataframe with planet and star properties, sorted by ascending period

    """
    catalog = pd.read_csv(file, index_col=0)
    catalog = catalog.loc[catalog.koi_id == target]

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

    # sort by ascending period
    catalog = catalog.sort_values(by="period").reset_index(drop=True)

    return catalog


def read_mast_files(mast_files, kic_id, obsmode, exclude=None):
    """
    Docstring
    """
    if exclude is None:
        exclude = []

    rawdata_list = []
    for i, mf in enumerate(mast_files):
        with fits.open(mf) as hdu_list:
            if hdu_list[0].header["OBSMODE"] == obsmode and ~np.isin(
                hdu_list[0].header["QUARTER"], exclude
            ):
                rawdata_list.append(lk.read(mf))

    lkcol = cleanup_lkc(lk.LightCurveCollection(rawdata_list), kic_id)

    return lkcol


def cleanup_lkc(lk_collection, kic):
    """
    Join each quarter in a lk.LightCurveCollection into a single lk.LightCurve
    Performs only the minimal detrending step remove_nans()

    Parameters
    ----------
        lk_collection : lk.LightCurveCollection
            lk.LightCurveCollection() with (possibly) multiple entries per quarter
        kic : int
            Kepler Input Catalogue (KIC) number for target

    Returns
    -------
        lkc : lk.LightCurveCollection
            lk.LightCurveCollection() with only one entry per quarter
    """
    lk_col = deepcopy(lk_collection)

    quarters = []
    for lkc in lk_col:
        quarters.append(lkc.quarter)

    data_out = []
    for q in np.unique(quarters):
        lkc_list = []
        cadno = []

        for lkc in lk_col:
            if (lkc.quarter == q) * (lkc.targetid == kic):
                lkc_list.append(lkc)
                cadno.append(lkc.cadenceno.min())

        order = np.argsort(cadno)
        lkc_list = [lkc_list[j] for j in order]

        # the operation "stitch" converts a LightCurveCollection to a single LightCurve
        lkc = lk.LightCurveCollection(lkc_list).stitch().remove_nans()

        data_out.append(lkc)

    return lk.LightCurveCollection(data_out)


def LightKurve_to_LiteCurve(lklc):
    return LiteCurve(
        time=np.array(lklc.time.value, dtype="float"),
        flux=np.array(lklc.flux.value, dtype="float"),
        error=np.array(lklc.flux_err.value, dtype="float"),
        cadno=np.array(lklc.cadenceno.value, dtype="int"),
        quarter=lklc.quarter * np.ones(len(lklc.time), dtype="int"),
        season=(lklc.quarter % 4) * np.ones(len(lklc.time), dtype="int"),
        quality=lklc.quality.value,
    )


def transit_parameters_to_dataframe(koi_id, kic_id, planets, limbdark):
    npl = len(planets)

    data = {}
    keys = "koi_id kic_id npl period epoch depth duration impact limbdark_1 limbdark_2".split()

    for k in keys:
        data[k] = [None] * npl

    for n, p in enumerate(planets):
        data["koi_id"][n] = koi_id
        data["kic_id"][n] = kic_id
        data["npl"][n] = npl
        data["period"][n] = p.period
        data["epoch"][n] = p.epoch
        data["depth"][n] = p.depth * 1e6
        data["duration"][n] = p.duration * 24
        data["impact"][n] = p.impact
        data["limbdark_1"][n] = limbdark[0]
        data["limbdark_2"][n] = limbdark[1]

    return pd.DataFrame(data)


def load_detrended_lightcurve(filename):
    """
    Load a fits file previously generated by LiteCurve.to_fits()

    Parameters
    ----------
        filename : string

    Returns
    -------
        litecurve : LiteCurve() object

    """
    litecurve = LiteCurve()

    with fits.open(filename) as hdulist:
        litecurve.time = np.array(hdulist["TIME"].data, dtype="float64")
        litecurve.flux = np.array(hdulist["FLUX"].data, dtype="float64")
        litecurve.error = np.array(hdulist["ERROR"].data, dtype="float64")
        litecurve.cadno = np.array(hdulist["CADNO"].data, dtype="int")
        litecurve.quarter = np.array(hdulist["QUARTER"].data, dtype="int")

    return litecurve


def dynesty_results_to_fits(results, project_dir, mission, target, run_id):
    """
    results : dynesty.DynamicNestedSampling.results
    target : (str) name of target, e.g. 'K00137'
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
    hduL = fits.HDUList([primary_hdu, samples_hdu])

    # add transit times to HDU List
    for n in range(npl):
        ttimes_file = os.path.join(
            project_dir,
            f"Results/{run_id}/{target}/{target}_{str(n).zfill(2)}_quick.ttvs",
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
