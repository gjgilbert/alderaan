"""Kepler-specific catalog parsing and target loading"""

__all__ = [
    "parse_koi_catalog",
    "parse_holczer16_catalog",
    "match_holczer16_transit_times",
    "load_kepler_data",
]

import logging
import os

import numpy as np
import pandas as pd

from alderaan.data_models import LiteCurve, Planet, Star, TransitTimes
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
    for col in ("kic_id", "npl", "limbdark_1", "limbdark_2"):
        if (catalog[col] != catalog[col][0]).any():
            raise ValueError(f"{koi_id}: inconsistent {col} across rows in {filepath}")

    if catalog.npl[0] != len(catalog):
        raise ValueError(
            f"{koi_id}: npl = {catalog.npl[0]} but {len(catalog)} rows in {filepath}"
        )

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


def parse_holczer16_catalog(filepath, koi_id):
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

    Returns
    -------
    transit_times : list of (index, ttime, error) tuples
        one tuple per planet of this KOI found in the table, in planet-number
        order (e.g. 1234.01, 1234.02, ...)
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

    koi_prefix = f"{int(koi_id[1:])}."
    koi_planets = [pid for pid in np.unique(planet_id) if pid.startswith(koi_prefix)]
    for pid in koi_planets:
        use = planet_id == pid
        transit_times.append((index[use], ttime[use], error[use]))

    return transit_times


def match_holczer16_transit_times(planets, transit_times, t_min, t_max):
    """
    Assign Holczer+2016 transit times to planets by matching periods

    A table matches a planet if its linear-fit period is within
    rtol=0.01, atol=(planet duration) of the planet's catalog period

    Parameters
    ----------
    planets : list of Planet
    transit_times : list of (index, ttime, error) tuples
        output of parse_holczer16_catalog
    t_min, t_max : float
        litecurve time span

    Returns
    -------
    planets : list of Planet
        matched planets carry every transit of the Holczer+2016 linear
        ephemeris within [t_min, t_max), with Holczer+2016 times as ttime_obs
        (NaN where Holczer+2016 has no measurement)

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
            try:
                # every transit of the Holczer linear ephemeris within [t_min, t_max);
                # epochs Holczer did not measure keep ttime_obs = NaN
                h_period, h_epoch = robust_polyfit(index, ttime, 1)
                tt = TransitTimes.from_linear_ephemeris(
                    p.mission, p.target, h_period, h_epoch, t_min, t_max
                )
                k_h = np.round((ttime - tt.ttime_mod[0]) / h_period).astype(int)
                use = (k_h >= 0) & (k_h < len(tt.index))
                ttime_obs = np.full(len(tt.index), np.nan)
                error_obs = np.full(len(tt.index), np.nan)
                ttime_obs[k_h[use]] = ttime[use]
                error_obs[k_h[use]] = error[use]
                p.set_transit_times(
                    TransitTimes(
                        p.mission,
                        p.target,
                        tt.index,
                        ttime_obs=ttime_obs,
                        error_obs=error_obs,
                        ttime_mod=np.where(
                            np.isfinite(ttime_obs), ttime_obs, tt.ttime_mod
                        ),
                    )
                )
            except ValueError as e:
                raise ValueError(f"planet {n}, Holczer+2016 table {k}: {e}") from e
            used.add(k)

    unused = sorted(set(range(len(transit_times))) - used)
    if unused:
        logger.warning(
            "Holczer+2016 tables %s matched no KOI planet (P = %s d)",
            unused,
            ", ".join(f"{periods[k]:.6f}" for k in unused),
        )

    return planets


def load_kepler_data(target, input_dir, catalog_dir, visits=None):
    """
    Load the star, planets, and long cadence litecurve for one Kepler target

    Parameters
    ----------
    target : str
        KOI identifier, e.g. K00137
    input_dir : str
        directory of MAST photometry downloads
    catalog_dir : str
        directory holding the KOI and Holczer+2016 catalogs
    visits : list of int, optional
        quarters to load; default loads all

    Returns
    -------
    star : Star
    planets : list of Planet
        ephemerides initialized from the KOI catalog and Holczer+2016
    litecurve : LiteCurve
        raw litecurve, all visits concatenated
    """
    koi_filepath = os.path.join(catalog_dir, "kepler_dr25_gaia_dr2_crossmatch.csv")
    ttv_filepath = os.path.join(catalog_dir, "holczer_2016_kepler_ttvs.txt")

    # load KOI catalog
    catalog = parse_koi_catalog(koi_filepath, target)

    npl = int(catalog.npl[0])
    star = Star(
        "Kepler",
        target,
        catalog.koi_id[0],
        int(catalog.kic_id[0]),
        catalog.limbdark_1[0],
        catalog.limbdark_2[0],
    )

    # load lightcurves
    litecurve = LiteCurve.from_kplr_pdcsap(
        input_dir, target, star.kic_id, "long cadence", visits=visits
    )

    # the photometry span, fixed on the litecurve (also used by TransitModel)
    t_min, t_max = litecurve.t_start, litecurve.t_end
    if t_min < 0:
        raise ValueError(
            "Lightcurve has negative timestamps...this will cause problems"
        )

    loaded = [int(v) for v in np.unique(litecurve.visit)]
    if visits is not None:
        missing = sorted(set(visits) - set(loaded))
        if missing:
            raise FileNotFoundError(
                f"requested quarters {missing} not found in {input_dir}"
            )

    # initialize planets
    planets = []
    for row in catalog.itertuples():
        p = Planet(
            "Kepler",
            target,
            row.period,
            row.epoch,
            row.depth,
            row.duration,
            row.impact,
        )
        p.set_transit_times(
            TransitTimes.from_linear_ephemeris(
                "Kepler", target, row.period, row.epoch, t_min, t_max
            )
        )
        planets.append(p)

    # load Holczer+2016 transit times
    holczer_transit_times = parse_holczer16_catalog(ttv_filepath, star.koi_id)
    planets = match_holczer16_transit_times(
        planets, holczer_transit_times, t_min, t_max
    )

    # clip planet transit times to the litecurve time range and refit ephemerides
    for n, p in enumerate(planets):
        p.set_transit_times(p.transit_times.clip(t_min, t_max))
        p.set_static_period_and_epoch()
        n_after = len(p.transit_times.ttime_obs)
        if n_after < 3:
            logger.warning(
                "planet %d: only %d transits within litecurve span", n, n_after
            )
        logger.info(
            "planet %d (catalog): P = %.6f d, epoch = %.5f BKJD, depth = %.0f ppm, "
            "T14 = %.2f hr, b = %.2f; %d transits, %d with Holczer+2016 times",
            n,
            p.period,
            p.epoch,
            p.depth * 1e6,
            p.duration * 24,
            p.impact,
            n_after,
            np.sum(np.isfinite(p.transit_times.error_obs)),
        )

    logger.info("star: u1 = %.3f, u2 = %.3f (catalog)", star.u1, star.u2)
    logger.info(
        "loaded %s: %d planets, quarters %s, %d cadences",
        target,
        npl,
        ",".join(str(v) for v in loaded),
        len(litecurve.time),
    )
    return star, planets, litecurve
