"""Kepler-specific catalog parsing and target loading"""

__all__ = [
    "parse_koi_catalog",
    "parse_holczer16_catalog",
    "match_holczer16_transit_times",
    "load_kepler_data",
]

import logging
import os
import time

import numpy as np
import pandas as pd

from alderaan.data_models.litecurve import LiteCurve
from alderaan.data_models.planet import Planet
from alderaan.data_models.star import Star
from alderaan.data_models.transit_times import TransitTimes
from alderaan.utils.stats import robust_polyfit

logger = logging.getLogger(__name__)


def parse_koi_catalog(filepath, koi_id):
    """
    Read a Kepler Object of Interest catalog and perform consistency checks

    Expected columns include
        [kic_id, koi_id, npl, period, epoch, depth, duration, impact,
         limbdark_1, limbdark_2]

    Parameters
    ----------
    filepath : str
        path to csv file
    koi_id : str
        KOI identification number in the format, e.g., K01234

    Returns
    -------
    catalog : pd.DataFrame
        target star/planet properties sorted by ascending period
        (depth as fraction, duration in days)
    """
    # read catalog from csv file
    catalog = pd.read_csv(filepath, index_col=0)
    catalog = catalog.loc[catalog.koi_id == koi_id]

    if len(catalog) == 0:
        raise ValueError(f"{koi_id} not found in {filepath}")

    # sort by ascending period
    catalog = catalog.sort_values(by="period").reset_index(drop=True)

    # check for consistency in multi-planet systems
    if not all(kic_id == catalog.kic_id.to_numpy()[0] for kic_id in catalog.kic_id):
        raise ValueError(f"{koi_id}: inconsistent kic_id across rows in {filepath}")

    if not all(npl == catalog.npl.to_numpy()[0] for npl in catalog.npl):
        raise ValueError(f"{koi_id}: inconsistent npl across rows in {filepath}")

    if catalog.npl[0] != len(catalog):
        raise ValueError(
            f"{koi_id}: npl = {catalog.npl[0]} but {len(catalog)} rows in {filepath}"
        )

    if not all(
        ld_u1 == catalog.limbdark_1.to_numpy()[0] for ld_u1 in catalog.limbdark_1
    ):
        raise ValueError(f"{koi_id}: inconsistent limbdark_1 across rows in {filepath}")

    if not all(
        ld_u2 == catalog.limbdark_2.to_numpy()[0] for ld_u2 in catalog.limbdark_2
    ):
        raise ValueError(f"{koi_id}: inconsistent limbdark_2 across rows in {filepath}")

    if np.any(np.diff(catalog.period) == 0):
        raise ValueError(f"{koi_id}: duplicate periods in {filepath}")

    # check for NaN valued transit parameters
    params = catalog["period epoch depth duration impact".split()]
    if params.isna().any().any():
        raise ValueError(
            f"{koi_id}: NaN values in columns "
            f"{list(params.columns[params.isna().any()])} of {filepath}"
        )

    # convert units
    catalog["depth"] = catalog["depth"] * 1e-6  # ppm --> fraction
    catalog["duration"] = catalog["duration"] / 24.0  # hrs --> days

    return catalog


def parse_holczer16_catalog(filepath, koi_id, num_planets):
    """
    Read the Holczer+2016 transit time table for one KOI

    Automatically corrects for zero-point offsets between catalogs
      - Holczer+2016 used BJD - 2454900
      - Kepler Project used BJKD = BJD - 2454833

    Parameters
    ----------
    filepath : str
        path to Holczer+2016 table
    koi_id : str
        KOI identification number in the format, e.g., K01234
    num_planets : int
        total number of planets in the system

    Returns
    -------
    transit_times : list of (index, ttime, error) tuples
        one tuple per planet found in the table (0 to num_planets tuples)
    """
    data = np.loadtxt(filepath, usecols=[0, 1, 2, 3, 4], dtype=str)
    transit_times = []

    planet_id = data[:, 0]
    index = np.array(data[:, 1], dtype=int)
    ttime = (
        np.array(data[:, 2], dtype=float)
        + np.array(data[:, 3], dtype=float) / 24 / 60
        + 67
    )
    error = np.array(data[:, 4], dtype=float) / 24 / 60

    for n in range(num_planets):
        use = planet_id == f"{int(koi_id[1:])}.{n + 1:02d}"
        if np.sum(use) > 0:
            transit_times.append((index[use], ttime[use], error[use]))

    return transit_times


def match_holczer16_transit_times(planets, transit_times, target):
    """
    Assign Holczer+2016 transit times to planets by matching periods

    A table matches a planet if its linear-fit period is within
    rtol=0.01, atol=(planet duration) of the planet's catalog period

    Parameters
    ----------
    planets : list of Planet
    transit_times : list of (index, ttime, error) tuples
        output of parse_holczer16_catalog
    target : str

    Returns
    -------
    planets : list of Planet
        matched planets carry Holczer+2016 transit times

    Raises
    ------
    ValueError
        if a planet matches more than one table, or a table matches more
        than one planet
    """
    periods = np.array(
        [robust_polyfit(index, ttime, 1)[0] for index, ttime, _ in transit_times]
    )

    used = set()
    for n, p in enumerate(planets):
        match = np.isclose(periods, p.period, rtol=0.01, atol=p.duration)

        if np.sum(match) > 1:
            raise ValueError(f"planet {n} matches {np.sum(match)} Holczer+2016 tables")

        if np.sum(match) == 1:
            k = int(np.flatnonzero(match)[0])
            if k in used:
                raise ValueError(f"Holczer+2016 table {k} matches multiple planets")

            index, ttime, error = transit_times[k]
            logger.info(
                "planet %d: matched Holczer+2016 table %d (%d transits; "
                "P = %.6f d catalog, %.6f d fit)",
                n,
                k,
                len(ttime),
                p.period,
                periods[k],
            )
            try:
                p.set_transit_times(
                    TransitTimes(
                        "Kepler",
                        target,
                        index - index[0],
                        ttime_obs=ttime,
                        error_obs=error,
                    )
                )
            except ValueError as e:
                raise ValueError(f"planet {n}, Holczer+2016 table {k}: {e}") from e
            used.add(k)
        else:
            logger.info(
                "planet %d: no Holczer+2016 match (P = %.6f d); "
                "using KOI linear ephemeris",
                n,
                p.period,
            )

    unused = sorted(set(range(len(transit_times))) - used)
    if unused:
        logger.warning(
            "Holczer+2016 tables %s matched no KOI planet (P = %s d)",
            unused,
            ", ".join(f"{periods[k]:.6f}" for k in unused),
        )

    return planets


def load_kepler_data(config):
    """
    Load the star, planets, and long cadence litecurve for one Kepler target

    Parameters
    ----------
    config : dict
        pipeline configuration built by `alderaan run`

    Returns
    -------
    star : Star
    planets : list of Planet
        ephemerides initialized from the KOI catalog and Holczer+2016
    litecurve : LiteCurve
        raw litecurve, all visits concatenated
    """
    t0 = time.perf_counter()
    target = config["RUN"]["target"]
    visits = config["RUN"].get("visits")
    logger.info("start load_kepler_data(target=%r, visits=%r)", target, visits)
    input_dir = config["PATHS"]["input_dir"]
    catalog_dir = config["PATHS"]["catalog_dir"]
    koi_filepath = os.path.join(catalog_dir, config["CATALOGS"]["koi_catalog"])
    ttv_filepath = os.path.join(catalog_dir, config["CATALOGS"]["ttv_catalog"])

    # load KOI catalog
    catalog = parse_koi_catalog(koi_filepath, target)
    logger.debug("read KOI catalog %s", koi_filepath)

    npl = int(catalog.npl[0])
    star = Star(
        catalog.koi_id[0],
        int(catalog.kic_id[0]),
        catalog.limbdark_1[0],
        catalog.limbdark_2[0],
    )

    # load lightcurves
    litecurve = LiteCurve.from_kplr_pdcsap(
        input_dir, target, star.kic_id, "long cadence", visits=visits
    )

    t_min = litecurve.time.min()
    t_max = litecurve.time.max()
    if t_min < 0:
        raise ValueError(
            "Lightcurve has negative timestamps...this will cause problems"
        )

    per_visit = litecurve.split_visits()
    for lc in per_visit:
        assert len(np.unique(lc.visit)) == 1, "expected one quarter per litecurve"
        assert len(np.unique(lc.obsmode)) == 1, "expected one obsmode per litecurve"

    loaded = [int(lc.visit[0]) for lc in per_visit]
    logger.info(
        "loaded %d long cadence quarters (%s), %d cadences, t = %.1f to %.1f BKJD",
        len(loaded),
        ", ".join(str(v) for v in loaded),
        len(litecurve.time),
        t_min,
        t_max,
    )
    if visits is not None:
        missing = sorted(set(visits) - set(loaded))
        if missing:
            logger.warning("requested quarters %s not found in %s", missing, input_dir)

    # initialize planets
    planets = [None] * npl
    for n in range(npl):
        planets[n] = Planet(
            catalog.period[n],
            catalog.epoch[n],
            catalog.depth[n],
            catalog.duration[n],
            catalog.impact[n],
        )
        planets[n].set_transit_times(
            TransitTimes.from_linear_ephemeris(
                "Kepler",
                target,
                catalog.period[n],
                catalog.epoch[n],
                t_min,
                t_max,
            )
        )

    logger.info(
        "KOI %s / KIC %d: %d planets, limb darkening u1 = %.3f, u2 = %.3f",
        star.koi_id,
        star.kic_id,
        npl,
        star.u1,
        star.u2,
    )
    for n, p in enumerate(planets):
        logger.info(
            "planet %d (KOI catalog): P = %.6f d, epoch = %.4f BKJD, "
            "depth = %.0f ppm, T14 = %.2f hr, b = %.2f",
            n,
            p.period,
            p.epoch,
            p.depth * 1e6,
            p.duration * 24,
            p.impact,
        )

    # load Holczer+2016 transit times
    holczer_transit_times = parse_holczer16_catalog(ttv_filepath, star.koi_id, npl)

    logger.info(
        "%d transit time tables found in Holczer+2016 (%s)",
        len(holczer_transit_times),
        ttv_filepath,
    )

    planets = match_holczer16_transit_times(planets, holczer_transit_times, target)

    # clip planet transit times to the litecurve time range and refit ephemerides
    for n, p in enumerate(planets):
        n_before = len(p.transit_times.ttime_obs)
        p.set_transit_times(p.transit_times.clip(t_min, t_max))
        p.set_static_period_and_epoch()

        logger.info(
            "planet %d: %d of %d transits within litecurve span; "
            "P = %.6f d, epoch = %.4f BKJD",
            n,
            len(planets[n].transit_times.ttime_obs),
            n_before,
            planets[n].period,
            planets[n].epoch,
        )
        if len(planets[n].transit_times.ttime_obs) < 3:
            logger.warning(
                "planet %d: only %d transits within litecurve span",
                n,
                len(planets[n].transit_times.ttime_obs),
            )

    logger.info("done load_kepler_data in %.1f s", time.perf_counter() - t0)
    return star, planets, litecurve
