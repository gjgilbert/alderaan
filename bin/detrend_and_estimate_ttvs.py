#!/usr/bin/env python
# coding: utf-8

# # Detrend and Estimate TTVs


import os
import sys
import glob
import shutil
import warnings
from copy import deepcopy
from datetime import datetime
from timeit import default_timer as timer

import argparse
import astropy
from astropy.io import fits
from astropy.stats import mad_std
from astropy.timeseries import LombScargle
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import numpy.polynomial.polynomial as poly
import pandas as pd
import lightkurve as lk
from scipy import stats
from scipy.ndimage import median_filter

import pymc3 as pm
import pymc3_ext as pmx
import exoplanet as exo
import aesara_theano_fallback.tensor as T
from aesara_theano_fallback import aesara as theano


# #### Flush buffer and silence extraneous warnings


# flush buffer to avoid mixed outputs from progressbar
sys.stdout.flush()

# turn off FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)

# supress UnitsWarnings (this code doesn't use astropy units)
warnings.filterwarnings(
    action="ignore", category=astropy.units.UnitsWarning, module="astropy"
)


# #### Initialize timer


print("")
print("+" * shutil.get_terminal_size().columns)
print("ALDERAAN Detrending and TTV Estimation")
print(f"Initialized {datetime.now().strftime('%d-%b-%Y at %H:%M:%S')}")
print("+" * shutil.get_terminal_size().columns)
print("")

# start program timer
global_start_time = timer()


# #### Parse inputs


try:
    parser = argparse.ArgumentParser(
        description="Inputs for ALDERAAN transit fiting pipeline"
    )
    parser.add_argument(
        "--mission",
        default=None,
        type=str,
        required=True,
        help="Mission name; can be 'Kepler' or 'Kepler-Validation'",
    )
    parser.add_argument(
        "--target",
        default=None,
        type=str,
        required=True,
        help="Target name; format should be K00000",
    )
    parser.add_argument(
        "--project_dir",
        default=None,
        type=str,
        required=True,
        help="Project directory for saving outputs",
    )
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="Data directory for accessing MAST lightcurves",
    )
    parser.add_argument(
        "--catalog",
        default=None,
        type=str,
        required=True,
        help="CSV file containing input planetary parameters",
    )
    parser.add_argument(
        "--run_id", default=None, type=str, required=True, help="run identifier"
    )
    parser.add_argument(
        "--verbose",
        default=False,
        type=bool,
        required=False,
        help="'True' to enable verbose logging",
    )
    parser.add_argument(
        "--iplot",
        default=False,
        type=bool,
        required=False,
        help="'True' to enable interactive matplotlib backend; default 'agg'",
    )
    parser.add_argument(
        "--use_sc",
        default=False,
        type=bool,
        required=False,
        help="'True' to use short cadence data where available",
    )

    args = parser.parse_args()
    MISSION = args.mission
    TARGET = args.target
    PROJECT_DIR = args.project_dir
    DATA_DIR = args.data_dir
    CATALOG = args.catalog
    RUN_ID = args.run_id
    VERBOSE = args.verbose
    IPLOT = args.iplot
    USE_SC = args.use_sc

except SystemExit:
    warnings.warn("No arguments were parsed from the command line")


print("")
print(f"   MISSION : {MISSION}")
print(f"   TARGET  : {TARGET}")
print(f"   RUN ID  : {RUN_ID}")
print("")
print(f"   Project directory : {PROJECT_DIR}")
print(f"   Data directory    : {DATA_DIR}")
print(f"   Input catalog     : {CATALOG}")
print("")
print(f"   theano cache : {theano.config.compiledir}")
print("")


# #### Build directory structure


# directories in which to place pipeline outputs for this run
RESULTS_DIR = os.path.join(PROJECT_DIR, "Results", RUN_ID, TARGET)
FIGURE_DIR = os.path.join(PROJECT_DIR, "Figures", RUN_ID, TARGET)

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

sys.path.append(PROJECT_DIR)


# #### Import ALDERAAN routines


import alderaan.io as io
import alderaan.detrend as detrend
import alderaan.omc as omc
from alderaan.astro import (
    get_transit_depth,
    predict_tc_error,
    make_transit_mask,
    set_oversample_factor,
)
from alderaan.constants import lcit, scit
from alderaan.utils import boxcar_smooth, bin_data, LS_estimator
from alderaan.validate import remove_known_transits, inject_synthetic_transits
from alderaan.Planet import Planet


# #### Set matplotlib backends


if not IPLOT:
    mpl.use("agg")

if np.any(np.array(["agg", "png", "svg", "pdf", "ps"]) == mpl.get_backend()):
    warnings.warn("Selected matplotlib backend does not support interactive plotting")
    IPLOT = False


# MAIN SCRIPT BEGINS HERE
def main():
    # # ################
    # # --- DATA I/O ---
    # # ################

    print("\nLoading data...\n")

    # ## Read in planet and star properties

    catalog = io.parse_catalog(
        os.path.join(PROJECT_DIR, f"Catalogs/{CATALOG}"), MISSION, TARGET
    )

    KOI_ID = catalog.koi_id.to_numpy()[0]
    KIC_ID = catalog.kic_id.to_numpy()[0]

    NPL = catalog.npl.to_numpy()[0]

    PERIODS = catalog.period.to_numpy()
    EPOCHS = catalog.epoch.to_numpy()
    DEPTHS = catalog.depth.to_numpy() * 1e-6
    DURS = catalog.duration.to_numpy() / 24
    IMPACTS = catalog.impact.to_numpy()

    U1 = catalog.limbdark_1.to_numpy()[0]
    U2 = catalog.limbdark_2.to_numpy()[0]

    if np.any(np.diff(PERIODS) <= 0):
        raise ValueError("Planets should be ordered by ascending period")

    # ## Read in pre-downloaded lightcurve data

    mast_files = glob.glob(DATA_DIR + f"kplr{KIC_ID:09d}*.fits")
    mast_files.sort()

    # short cadence data
    if USE_SC:
        sc_raw_data = io.read_mast_files(mast_files, KIC_ID, "short cadence")
        sc_quarters = sc_raw_data.quarter
    else:
        sc_raw_data = []
        sc_quarters = []

    sc_data = []
    for i, scrd in enumerate(sc_raw_data):
        sc_data.append(io.LightKurve_to_LiteCurve(scrd))

    # long cadence data
    lc_raw_data = io.read_mast_files(
        mast_files, KIC_ID, "long cadence", exclude=sc_quarters
    )
    lc_quarters = lc_raw_data.quarter

    lc_data = []
    for i, lcrd in enumerate(lc_raw_data):
        lc_data.append(io.LightKurve_to_LiteCurve(lcrd))

    # # ####################
    # # --- PRELIMINARIES ---
    # # ####################

    # ## Establish time baseline

    print("Establishing observation baseline")

    time_min = []
    time_max = []

    for i, scd in enumerate(sc_data):
        time_min.append(scd.time.min())
        time_max.append(scd.time.max())

    for i, lcd in enumerate(lc_data):
        time_min.append(lcd.time.min())
        time_max.append(lcd.time.max())

    TIME_START = np.min(time_min)
    TIME_END = np.max(time_max)

    if TIME_START < 0:
        raise ValueError("START TIME is negative...this will cause problems")

    # put epochs in range (TIME_START, TIME_START + PERIOD)
    for npl in range(NPL):
        if EPOCHS[npl] < TIME_START:
            adj = 1 + (TIME_START - EPOCHS[npl]) // PERIODS[npl]
            EPOCHS[npl] += adj * PERIODS[npl]

        if EPOCHS[npl] > (TIME_START + PERIODS[npl]):
            adj = (EPOCHS[npl] - TIME_START) // PERIODS[npl]
            EPOCHS[npl] -= adj * PERIODS[npl]

    # ## Initialize Planet objects

    print(f"Initializing {NPL} Planet objects")

    planets = []
    for npl in range(NPL):
        p = Planet()

        # put in some basic transit parameters
        p.period = PERIODS[npl]
        p.epoch = EPOCHS[npl]
        p.depth = DEPTHS[npl]
        p.duration = DURS[npl]
        p.impact = IMPACTS[npl]

        if p.impact > 1 - np.sqrt(p.depth):
            p.impact = (1 - np.sqrt(p.depth)) ** 2

        # estimate transit times from linear ephemeris
        p.tts = np.arange(p.epoch, TIME_END, p.period)

        # make transit indexes
        p.index = np.array(np.round((p.tts - p.epoch) / p.period), dtype="int")

        # add to list
        planets.append(p)

    # ## Set oversample factor

    oversample = set_oversample_factor(
        PERIODS, DEPTHS, DURS, lc_data[0].flux, lc_data[0].error
    )

    if oversample >= 30:
        raise ValueError("attempting to set oversample factor greater than 30")

    print(f"Setting oversample = {oversample}")

    # # ############################
    # # ----- TRANSIT TIME SETUP -----
    # # ############################

    print("\nBuilding initial TTV model...\n")

    h16file = os.path.join(PROJECT_DIR, "Catalogs/holczer_2016_kepler_ttvs.txt")
    holczer = omc.load_holczer_ttvs(h16file, NPL, KOI_ID)
    holczer = omc.clean_holczer_ttvs(holczer, TIME_START, TIME_END)
    planets = omc.match_holczer_ttvs(planets, holczer)

    for n, p in enumerate(planets):
        if np.isfinite(holczer["period"][n]):
            xtime = holczer["tts"][n]
            yomc = (
                holczer["tts"][n]
                - holczer["epoch"][n]
                - holczer["period"][n] * holczer["inds"][n]
            )
            out = holczer["out"][n]

            xtime_interp = holczer["full_tts"][n]
            yomc_interp = (
                holczer["full_tts"][n]
                - holczer["epoch"][n]
                - holczer["period"][n] * holczer["full_inds"][n]
            )

            plt.figure(figsize=(12, 4))
            plt.plot(xtime[~out], yomc[~out] * 24 * 60, "o", c="grey", label="Holczer")
            plt.plot(xtime[out], yomc[out] * 24 * 60, "rx")
            plt.plot(xtime_interp, yomc_interp * 24 * 60, "k+", label="Interpolation")
            plt.xlabel("Time [BJKD]", fontsize=20)
            plt.ylabel("O-C [min]", fontsize=20)
            plt.legend(fontsize=12)

            figpath = os.path.join(FIGURE_DIR, TARGET + f"_ttvs_holczer_{npl:02d}.png")
            plt.savefig(figpath, bbox_inches="tight")

            if IPLOT:
                plt.show()
            else:
                plt.close()

    # # ##################################
    # # ----- INJECTION-AND-RECOVERY -----
    # # ##################################

    if MISSION == "Kepler-Validation":
        # remove known transits of real planets
        if len(lc_data) > 0:
            lc_data = remove_known_transits(
                planets, lc_data, [U1, U2], lcit, oversample
            )

        if len(sc_data) > 0:
            sc_data = remove_known_transits(planets, sc_data, [U1, U2], scit, 1)

        # load synthetic catalog
        path = os.path.join(PROJECT_DIR, f"Simulations/{RUN_ID}/{RUN_ID}.csv")
        catalog = io.parse_catalog(path, MISSION, TARGET)

        KOI_ID = catalog.koi_id.to_numpy()[0]
        KIC_ID = catalog.kic_id.to_numpy()[0]

        NPL = catalog.npl.to_numpy()[0]

        PERIODS = catalog.period.to_numpy()
        EPOCHS = catalog.epoch.to_numpy()
        DEPTHS = catalog.depth.to_numpy() * 1e-6
        DURS = catalog.duration.to_numpy() / 24
        IMPACTS = catalog.impact.to_numpy()

        U1 = catalog.limbdark_1.to_numpy()[0]
        U2 = catalog.limbdark_2.to_numpy()[0]

        if np.any(np.diff(PERIODS) <= 0):
            raise ValueError("Planets should be ordered by ascending period")

        # put epochs in range (TIME_START, TIME_START + PERIOD)
        for npl in range(NPL):
            if EPOCHS[npl] < TIME_START:
                adj = 1 + (TIME_START - EPOCHS[npl]) // PERIODS[npl]
                EPOCHS[npl] += adj * PERIODS[npl]

            if EPOCHS[npl] > (TIME_START + PERIODS[npl]):
                adj = (EPOCHS[npl] - TIME_START) // PERIODS[npl]
                EPOCHS[npl] -= adj * PERIODS[npl]

        # initalize synthetic planet objects
        planets = []
        for npl in range(NPL):
            p = Planet()

            # put in some basic transit parameters
            p.period = PERIODS[npl]
            p.epoch = EPOCHS[npl]
            p.depth = DEPTHS[npl] * 1e6
            p.duration = DURS[npl] * 24
            p.impact = IMPACTS[npl]

            # load true transit times
            tts_file = os.path.join(
                PROJECT_DIR, f"Simulations/{RUN_ID}/S{TARGET[1:]}_{npl}.tts"
            )
            true_tts = np.loadtxt(tts_file).swapaxes(0, 1)

            p.tts = true_tts[1]
            p.index = np.array(true_tts[0], dtype="int")

            planets.append(p)

        # inject synthetic transits
        if len(lc_data) > 0:
            lc_data = inject_synthetic_transits(
                planets, lc_data, [U1, U2], lcit, oversample
            )

        if len(sc_data) > 0:
            sc_data = inject_synthetic_transits(planets, sc_data, [U1, U2], scit, 1)

        lc_raw_sim_data = deepcopy(lc_data)
        sc_raw_sim_data = deepcopy(sc_data)

    # # #########################
    # # ----- 1ST DETRENDING -----
    # # #########################

    print("\nDetrending lightcurves (1st pass)...\n")

    # ## Detrend the lightcurves

    # remove bad cadences
    for i, lcd in enumerate(lc_data):
        lcd = detrend.remove_bad_cadences(
            lcd, planets, rel_masksize=1.5, min_masksize=1.0 / 24
        )

    for i, scd in enumerate(sc_data):
        scd = detrend.remove_bad_cadences(
            scd, planets, rel_masksize=1.5, min_masksize=1.0 / 24
        )

    # track stellar and instrumental oscillations
    oscillation_period_by_quarter = np.ones(18) * np.nan

    for i, lcd in enumerate(lc_data):
        oscillation_period_by_quarter[
            lcd.quarter[0]
        ] = detrend.estimate_oscillation_period(lcd, min_period=1.0)

    for i, scd in enumerate(sc_data):
        oscillation_period_by_quarter[
            scd.quarter[0]
        ] = detrend.estimate_oscillation_period(scd, min_period=1.0)

    oscillation_period_by_season = np.zeros((4, 2))

    for i in range(4):
        oscillation_period_by_season[i, 0] = np.nanmedian(
            oscillation_period_by_quarter[i::4]
        )
        oscillation_period_by_season[i, 1] = mad_std(
            oscillation_period_by_quarter[i::4], ignore_nan=True
        )

    # detrend long cadence data
    break_tolerance = np.max([int(DURS.min() / lcit * 5 / 2), 13])
    min_per = 1.0

    for i, lcd in enumerate(lc_data):
        print(f"QUARTER {lcd.quarter[0]}")

        nom_per = oscillation_period_by_season[lcd.quarter[0] % 4][0]

        lcd = detrend.flatten_litecurve(
            lcd, break_tolerance, min_per, nominal_period=nom_per, verbose=VERBOSE
        )

    if len(lc_data) > 0:
        lc = detrend.stitch(lc_data)
    else:
        lc = None

    # detrend short cadence data
    break_tolerance = np.max([int(DURS.min() / scit * 5 / 2), 91])
    min_per = 1.0

    for i, scd in enumerate(sc_data):
        print(f"QUARTER {scd.quarter[0]}")

        nom_per = oscillation_period_by_season[scd.quarter[0] % 4][0]

        scd = detrend.flatten_litecurve(
            scd, break_tolerance, min_per, nominal_period=nom_per, verbose=VERBOSE
        )

    if len(sc_data) > 0:
        sc = detrend.stitch(sc_data)
    else:
        sc = None

    # ## Make wide masks that track each planet individually
    # #### These masks have width 2.5 transit durations, which is probably wider than the masks used for detrending

    if sc is not None:
        sc_mask = np.zeros((NPL, len(sc.time)), dtype="bool")
        for n, p in enumerate(planets):
            sc_mask[n] = make_transit_mask(
                sc.time, p.tts, np.max([2 / 24, 2.5 * p.duration])
            )

        sc.mask = sc_mask.sum(axis=0) > 0

    else:
        sc_mask = None

    if lc is not None:
        lc_mask = np.zeros((NPL, len(lc.time)), dtype="bool")
        for n, p in enumerate(planets):
            lc_mask[n] = make_transit_mask(
                lc.time, p.tts, np.max([2 / 24, 2.5 * p.duration])
            )

        lc.mask = lc_mask.sum(axis=0) > 0

    else:
        lc_mask = None

    # ## Flag high quality transits (quality = 1)
    # #### Good transits must have  at least 50% photometry coverage in/near transit

    # number of photometric points expected in transit
    count_expect_lc = np.zeros(NPL, dtype="int")
    count_expect_sc = np.zeros(NPL, dtype="int")

    for n, p in enumerate(planets):
        count_expect_lc[n] = np.max([1, int(np.floor(p.duration / lcit))])
        count_expect_sc[n] = np.max([1, int(np.floor(p.duration / scit))])
        quality = np.zeros(len(p.tts), dtype="bool")

        for i, t0 in enumerate(p.tts):
            if sc is not None:
                in_sc = np.abs(sc.time - t0) / p.duration < 0.5
                near_sc = np.abs(sc.time - t0) / p.duration < 1.5

                qual_in = np.sum(in_sc) > 0.5 * count_expect_sc[n]
                qual_near = np.sum(near_sc) > 1.5 * count_expect_sc[n]

                quality[i] += qual_in * qual_near

            if lc is not None:
                in_lc = np.abs(lc.time - t0) / p.duration < 0.5
                near_lc = np.abs(lc.time - t0) / p.duration < 1.5

                qual_in = np.sum(in_lc) > 0.5 * count_expect_lc[n]
                qual_near = np.sum(near_lc) > 1.5 * count_expect_lc[n]

                quality[i] += qual_in * qual_near

        p.quality = np.copy(quality)

    # ## Flag overlapping transits

    dur_max = np.max(DURS)
    overlap = []

    for i in range(NPL):
        overlap.append(np.zeros(len(planets[i].tts), dtype="bool"))

        for j in range(NPL):
            if i != j:
                for ttj in planets[j].tts:
                    overlap[i] += np.abs(planets[i].tts - ttj) / dur_max < 1.5

        planets[i].overlap = np.copy(overlap[i])

    # ## Count up transits and calculate initial fixed transit times

    num_transits = np.zeros(NPL, dtype="int")
    transit_inds = []
    fixed_tts = []

    for npl, p in enumerate(planets):
        transit_inds.append(np.array((p.index - p.index.min())[p.quality], dtype="int"))
        fixed_tts.append(np.copy(p.tts)[p.quality])

        num_transits[npl] = len(transit_inds[npl])
        transit_inds[npl] -= transit_inds[npl].min()

    # ## Grab data near transits

    # go quarter-by-quarter
    all_time = [None] * 18
    all_flux = [None] * 18
    all_error = [None] * 18
    all_dtype = ["none"] * 18

    for q in range(18):
        if sc is not None:
            if np.isin(q, sc.quarter):
                use = (sc.mask) * (sc.quarter == q)

                if np.sum(use) > np.min(count_expect_sc):
                    all_time[q] = sc.time[use]
                    all_flux[q] = sc.flux[use]
                    all_error[q] = sc.error[use]
                    all_dtype[q] = "short"

                else:
                    all_dtype[q] = "short_no_transits"

        if lc is not None:
            if np.isin(q, lc.quarter):
                use = (lc.mask) * (lc.quarter == q)

                if np.sum(use) > np.min(count_expect_lc):
                    all_time[q] = lc.time[use]
                    all_flux[q] = lc.flux[use]
                    all_error[q] = lc.error[use]
                    all_dtype[q] = "long"

                else:
                    all_dtype[q] = "long_no_transits"

    all_dtype = np.array(all_dtype)

    # track which quarters have coverage
    quarters = np.arange(18)[(all_dtype == "short") + (all_dtype == "long")]

    # track mean and variance of each quarter
    mean_by_quarter = np.ones(18) * np.nan
    var_by_quarter = np.ones(18) * np.nan

    for q in range(18):
        if sc is not None:
            if np.isin(q, sc.quarter):
                mean_by_quarter[q] = np.mean(sc.flux[sc.quarter == q])
                var_by_quarter[q] = np.var(sc.flux[sc.quarter == q])

        if lc is not None:
            if np.isin(q, lc.quarter):
                mean_by_quarter[q] = np.mean(lc.flux[lc.quarter == q])
                var_by_quarter[q] = np.var(lc.flux[lc.quarter == q])

    # broadcast oversample factor
    long_cadence_oversample = np.copy(oversample)

    oversample = np.zeros(18, dtype="int")
    oversample[all_dtype == "short"] = 1
    oversample[all_dtype == "long"] = long_cadence_oversample

    # broadcast exposure times
    texp = np.zeros(18)
    texp[all_dtype == "short"] = scit
    texp[all_dtype == "long"] = lcit

    # ## Pull basic transit parameters

    periods = np.zeros(NPL)
    epochs = np.zeros(NPL)
    depths = np.zeros(NPL)
    durs = np.zeros(NPL)
    impacts = np.zeros(NPL)

    for npl, p in enumerate(planets):
        periods[npl] = p.period
        epochs[npl] = p.epoch
        depths[npl] = p.depth
        durs[npl] = p.duration
        impacts[npl] = p.impact

    # ## Define Legendre polynomials

    # Legendre polynomials for better orthogonality; "x" is in the range (-1,1)
    t = [None] * NPL
    x = [None] * NPL
    Leg0 = [None] * NPL
    Leg1 = [None] * NPL

    # this assumes a baseline in the range (TIME_START,TIME_END)
    for n, p in enumerate(planets):
        t[n] = p.epoch + transit_inds[n] * p.period
        x[n] = 2 * (t[n] - TIME_START) / (TIME_END - TIME_START) - 1

        Leg0[n] = np.ones_like(x[n])
        Leg1[n] = x[n].copy()

    print("")
    print("cumulative runtime = ", int(timer() - global_start_time), "s")
    print("")

    # # ############################
    # # ----- LIGHTCURVE FITTING -----
    # # ############################

    # ## Fit transit SHAPE model

    print("\nFitting transit SHAPE model...\n")

    with pm.Model() as shape_model:
        # planetary parameters
        log_r = pm.Uniform(
            "log_r",
            lower=np.log(1e-5),
            upper=np.log(0.99),
            shape=NPL,
            testval=np.log(np.sqrt(depths)),
        )
        r = pm.Deterministic("r", T.exp(log_r))
        b = pm.Uniform("b", lower=0.0, upper=1.0, shape=NPL)

        log_dur = pm.Normal("log_dur", mu=np.log(durs), sd=5.0, shape=NPL)
        dur = pm.Deterministic("dur", T.exp(log_dur))

        # polynomial TTV parameters
        C0 = pm.Normal("C0", mu=0.0, sd=durs / 2, shape=NPL)
        C1 = pm.Normal("C1", mu=0.0, sd=durs / 2, shape=NPL)

        transit_times = []
        for npl in range(NPL):
            transit_times.append(
                pm.Deterministic(
                    f"tts_{npl}",
                    fixed_tts[npl] + C0[npl] * Leg0[npl] + C1[npl] * Leg1[npl],
                )
            )

        # set up stellar model and planetary orbit
        starrystar = exo.LimbDarkLightCurve([U1, U2])
        orbit = exo.orbits.TTVOrbit(
            transit_times=transit_times,
            transit_inds=transit_inds,
            b=b,
            ror=r,
            duration=dur,
        )

        # track period and epoch
        pm.Deterministic("P", orbit.period)
        pm.Deterministic("T0", orbit.t0)

        # nuissance parameters
        mbq = mean_by_quarter[(all_dtype == "short") + (all_dtype == "long")]
        vbq = var_by_quarter[(all_dtype == "short") + (all_dtype == "long")]

        flux0 = pm.Normal("flux0", mu=mbq, sd=mbq / 10, shape=len(quarters))
        log_jit = pm.Normal("log_jit", mu=np.log(vbq), sd=10, shape=len(quarters))

        # now evaluate the model for each quarter
        light_curves = [None] * len(quarters)
        model_flux = [None] * len(quarters)
        flux_err = [None] * len(quarters)
        obs = [None] * len(quarters)

        for j, q in enumerate(quarters):
            # calculate light curves
            light_curves[j] = starrystar.get_light_curve(
                orbit=orbit, r=r, t=all_time[q], oversample=oversample[q], texp=texp[q]
            )

            model_flux[j] = pm.math.sum(light_curves[j], axis=-1) + flux0[j] * T.ones(
                len(all_time[q])
            )
            flux_err[j] = T.sqrt(
                np.mean(all_error[q]) ** 2 + T.exp(log_jit[j])
            ) / np.sqrt(2)

            obs[j] = pm.Normal(
                f"obs_{j}", mu=model_flux[j], sd=flux_err[j], observed=all_flux[q]
            )

    with shape_model:
        shape_map = shape_model.test_point
        shape_map = pmx.optimize(
            start=shape_map, vars=[flux0, log_jit], progress=VERBOSE
        )
        shape_map = pmx.optimize(start=shape_map, vars=[b, r, dur], progress=VERBOSE)
        shape_map = pmx.optimize(start=shape_map, vars=[C0, C1], progress=VERBOSE)
        shape_map = pmx.optimize(start=shape_map, progress=VERBOSE)

    # grab transit times and ephemeris
    shape_transit_times = []
    shape_linear_ephemeris = []

    for npl, p in enumerate(planets):
        shape_transit_times.append(shape_map[f"tts_{npl}"])
        shape_linear_ephemeris.append(
            shape_map["P"][npl] * transit_inds[npl] + shape_map["T0"][npl]
        )

    # update parameter values
    periods = np.atleast_1d(shape_map["P"])
    epochs = np.atleast_1d(shape_map["T0"])
    depths = np.atleast_1d(get_transit_depth(shape_map["r"], shape_map["b"]))
    durs = np.atleast_1d(shape_map["dur"])
    impacts = np.atleast_1d(shape_map["b"])
    rors = np.atleast_1d(shape_map["r"])

    for npl, p in enumerate(planets):
        p.period = periods[npl]
        p.epoch = epochs[npl]
        p.depth = depths[npl]
        p.duration = durs[npl]
        p.impact = impacts[npl]

    print("")
    print("cumulative runtime = ", int(timer() - global_start_time), "s")
    print("")

    # ## Fit SLIDE TTVs

    print("\nFitting TTVs..\n")

    # get list of quarters with observations
    if lc is not None:
        lc_quarters = np.unique(lc.quarter)
    else:
        lc_quarters = np.array([], dtype="int")

    if sc is not None:
        sc_quarters = np.unique(sc.quarter)
    else:
        sc_quarters = np.array([], dtype="int")

    quarters = np.sort(np.hstack([lc_quarters, sc_quarters]))
    seasons = np.sort(np.unique(quarters % 4))

    # get list of threshold times between quarters
    thresh = np.zeros(len(quarters) + 1)
    thresh[0] = TIME_START

    for j, q in enumerate(quarters):
        if np.isin(q, sc_quarters):
            thresh[j + 1] = sc.time[sc.quarter == q].max()
        if np.isin(q, lc_quarters):
            thresh[j + 1] = lc.time[lc.quarter == q].max()

    thresh[0] -= 1.0
    thresh[-1] += 1.0

    # track individual transits
    transit_quarter = [None] * NPL

    for npl in range(NPL):
        tts = shape_transit_times[npl]
        transit_quarter[npl] = np.zeros(len(tts), dtype="int")

        for j, q in enumerate(quarters):
            transit_quarter[npl][(tts >= thresh[j]) * (tts < thresh[j + 1])] = q

    slide_transit_times = []
    slide_error = []

    for npl, p in enumerate(planets):
        print("\nPLANET", npl)

        slide_transit_times.append([])
        slide_error.append([])

        # create template transit
        starrystar = exo.LimbDarkLightCurve([U1, U2])
        orbit = exo.orbits.KeplerianOrbit(
            t0=0, period=p.period, b=p.impact, ror=rors[npl], duration=p.duration
        )

        slide_offset = 1.0
        template_time = np.arange(0, (0.02 + p.duration) * (slide_offset + 1.6), scit)
        template_time = np.hstack([-template_time[:-1][::-1], template_time])
        template_flux = (
            1.0
            + starrystar.get_light_curve(
                orbit=orbit, r=rors[npl], t=template_time, oversample=1
            )
            .sum(axis=-1)
            .eval()
        )

        # empty lists to hold new transit time and uncertainties
        tts = -99 * np.ones_like(shape_transit_times[npl])
        err = -99 * np.ones_like(shape_transit_times[npl])

        for i, t0 in enumerate(shape_transit_times[npl]):
            # print(i, np.round(t0,2))
            if ~p.overlap[p.quality][i]:
                # identify quarter
                q = transit_quarter[npl][i]

                # set exposure time and oversample factor
                if all_dtype[q] == "long":
                    exptime = lcit
                    texp_offsets = np.linspace(
                        -exptime / 2.0, exptime / 2.0, oversample[q]
                    )
                elif all_dtype[q] == "short":
                    exptime = scit
                    texp_offsets = np.array([0.0])
                else:
                    raise ValueError("data cadence expected to be 'long' or 'short'")

                # grab data near each non-overlapping transit
                use = np.abs(all_time[q] - t0) / p.duration < 2.5
                mask = np.abs(all_time[q] - t0) / p.duration < 1.0

                t_ = all_time[q][use]
                f_ = all_flux[q][use]
                m_ = mask[use]

                t_supersample = (texp_offsets + t_.reshape(t_.size, 1)).flatten()

                # remove any residual out-of-transit trend
                try:
                    trend = poly.polyval(t_, poly.polyfit(t_[~m_], f_[~m_], 1))
                except TypeError:
                    trend = np.ones_like(f_)

                f_ /= trend
                e_ = np.ones_like(f_) * np.std(f_[~m_])

                # slide along transit time vector and calculate chisq
                gridstep = scit / 1.618 / 3
                tc_vector = np.arange(0, p.duration * slide_offset, gridstep)
                tc_vector = t0 + np.hstack([-tc_vector[:-1][::-1], tc_vector])
                chisq_vector = np.zeros_like(tc_vector)

                for j, tc in enumerate(tc_vector):
                    y_ = np.interp(t_supersample - tc, template_time, template_flux)
                    y_ = bin_data(t_supersample, y_, exptime, bin_centers=t_)[1]

                    chisq_vector[j] = np.sum((f_ - y_) ** 2 / e_**2)

                # grab points near minimum chisq
                delta_chisq = 1.0

                loop = True
                while loop:
                    # incrememnt delta_chisq and find minimum
                    delta_chisq *= 2
                    min_chisq = chisq_vector.min()

                    # grab the points near minimum
                    tcfit = tc_vector[chisq_vector < min_chisq + delta_chisq]
                    x2fit = chisq_vector[chisq_vector < min_chisq + delta_chisq]

                    # eliminate points far from the local minimum
                    spacing = np.median(tcfit[1:] - tcfit[:-1])
                    faraway = (
                        np.abs(tcfit - np.median(tcfit)) / spacing > 1 + len(tcfit) / 2
                    )

                    tcfit = tcfit[~faraway]
                    x2fit = x2fit[~faraway]

                    # check for stopping conditions
                    if len(x2fit) >= 7:
                        loop = False

                    if delta_chisq >= 16:
                        loop = False

                # fit a parabola around the minimum (need at least 3 pts)
                if len(tcfit) < 3:
                    # print("TOO FEW POINTS")
                    tts[i] = np.nan
                    err[i] = np.nan

                else:
                    quad_coeffs = np.polyfit(tcfit, x2fit, 2)
                    quadfit = np.polyval(quad_coeffs, tcfit)
                    qtc_min = -quad_coeffs[1] / (2 * quad_coeffs[0])
                    qx2_min = np.polyval(quad_coeffs, qtc_min)
                    qtc_err = np.sqrt(1 / quad_coeffs[0])

                    # transit time and scaled error
                    tts[i] = np.mean([qtc_min, np.median(tcfit)])
                    err[i] = qtc_err * (1 + np.std(x2fit - quadfit))

                    # check that the fit is well-conditioned (ie. a negative t**2 coefficient)
                    if quad_coeffs[0] <= 0.0:
                        # print("INVERTED PARABOLA")
                        tts[i] = np.nan
                        err[i] = np.nan

                    # check that the recovered transit time is within the expected range
                    if (tts[i] < tcfit.min()) or (tts[i] > tcfit.max()):
                        # print("T0 OUT OF BOUNDS")
                        tts[i] = np.nan
                        err[i] = np.nan

                # show plots
                do_plots = False
                if do_plots:
                    fig, ax = plt.subplots(1, 2, figsize=(10, 3))

                    if ~np.isnan(tts[i]):
                        tc = tts[i]
                    else:
                        tc = t0

                    ax[0].plot(t_ - tc, f_, "ko")
                    ax[0].plot((t_ - tc)[m_], f_[m_], "o", c=f"C{npl}")
                    ax[0].plot(template_time, template_flux, c=f"C{npl}", lw=2)

                    ax[1].plot(tcfit, x2fit, "ko")
                    ax[1].plot(tcfit, quadfit, c=f"C{npl}", lw=3)
                    ax[1].axvline(tc, color="k", ls="--", lw=2)

                    if IPLOT:
                        plt.show()
                    else:
                        plt.close()

            else:
                # print("OVERLAPPING TRANSITS")
                tts[i] = np.nan
                err[i] = np.nan

        slide_transit_times[npl] = np.copy(tts)
        slide_error[npl] = np.copy(err)

    # flag transits for which the slide method failed
    for npl, p in enumerate(planets):
        bad = np.isnan(slide_transit_times[npl]) + np.isnan(slide_error[npl])
        bad += slide_error[npl] > 8 * np.nanmedian(slide_error[npl])

        slide_transit_times[npl][bad] = shape_transit_times[npl][bad]
        slide_error[npl][bad] = np.nan

    refit = []

    for npl in range(NPL):
        refit.append(np.isnan(slide_error[npl]))

        # if every slide fit worked, randomly select a pair of transits for refitting
        # this is easier than tracking the edge cases -- we'll use the slide ttvs in the final vector anyway
        if np.all(~refit[npl]):
            refit[npl][np.random.randint(len(refit[npl]), size=2)] = True

    print("")
    print("cumulative runtime = ", int(timer() - global_start_time), "s")
    print("")

    # ## Fit MAP INDEPENDENT TTVs
    #
    # #### Only refit transit times for which the slide method failed

    if sc is not None:
        sc_map_mask = np.zeros((NPL, len(sc.time)), dtype="bool")
        for npl, p in enumerate(planets):
            tts = slide_transit_times[npl][refit[npl]]
            sc_map_mask[npl] = make_transit_mask(
                sc.time, tts, np.max([2 / 24, 2.5 * p.duration])
            )

        sc_map_mask = sc_map_mask.sum(axis=0) > 0

    else:
        sc_map_mask = None

    if lc is not None:
        lc_map_mask = np.zeros((NPL, len(lc.time)), dtype="bool")
        for npl, p in enumerate(planets):
            tts = slide_transit_times[npl][refit[npl]]
            lc_map_mask[npl] = make_transit_mask(
                lc.time, tts, np.max([2 / 24, 2.5 * p.duration])
            )

        lc_map_mask = lc_map_mask.sum(axis=0) > 0

    else:
        lc_map_mask = None

    # grab data near transits for each quarter
    map_time = [None] * 18
    map_flux = [None] * 18
    map_error = [None] * 18
    map_dtype = ["none"] * 18

    for q in range(18):
        if sc is not None:
            if np.isin(q, sc.quarter):
                use = (sc_map_mask) * (sc.quarter == q)

                if np.sum(use) > 45:
                    map_time[q] = sc.time[use]
                    map_flux[q] = sc.flux[use]
                    map_error[q] = sc.error[use]
                    map_dtype[q] = "short"

                else:
                    map_dtype[q] = "short_no_transits"

        if lc is not None:
            if np.isin(q, lc.quarter):
                use = (lc_map_mask) * (lc.quarter == q)

                if np.sum(use) > 5:
                    map_time[q] = lc.time[use]
                    map_flux[q] = lc.flux[use]
                    map_error[q] = lc.error[use]
                    map_dtype[q] = "long"

                else:
                    map_dtype[q] = "long_no_transits"

    map_quarters = np.arange(18)[
        (np.array(map_dtype) == "short") + (np.array(map_dtype) == "long")
    ]

    with pm.Model() as indep_model:
        # transit times
        tt_offset = []
        map_tts = []
        map_inds = []

        for npl in range(NPL):
            use = np.copy(refit[npl])

            tt_offset.append(
                pm.Normal(f"tt_offset_{npl}", mu=0, sd=1, shape=np.sum(use))
            )

            map_tts.append(
                pm.Deterministic(
                    f"tts_{npl}",
                    shape_transit_times[npl][use] + tt_offset[npl] * durs[npl] / 3,
                )
            )

            map_inds.append(transit_inds[npl][use])

        # set up stellar model and planetary orbit
        starrystar = exo.LimbDarkLightCurve([U1, U2])
        orbit = exo.orbits.TTVOrbit(
            transit_times=map_tts,
            transit_inds=map_inds,
            period=periods,
            b=impacts,
            ror=rors,
            duration=durs,
        )

        # nuissance parameters
        flux0 = pm.Normal(
            "flux0",
            mu=np.mean(good_flux),
            sd=np.std(good_flux),
            shape=len(map_quarters),
        )
        log_jit = pm.Normal(
            "log_jit", mu=np.log(np.var(good_flux) / 10), sd=10, shape=len(map_quarters)
        )

        # now evaluate the model for each quarter
        light_curves = [None] * len(map_quarters)
        model_flux = [None] * len(map_quarters)
        flux_err = [None] * len(map_quarters)
        obs = [None] * len(map_quarters)

        for j, q in enumerate(map_quarters):
            # calculate light curves
            light_curves[j] = starrystar.get_light_curve(
                orbit=orbit,
                r=rors,
                t=map_time[q],
                oversample=oversample[q],
                texp=texp[q],
            )

            model_flux[j] = pm.math.sum(light_curves[j], axis=-1) + flux0[j] * T.ones(
                len(map_time[q])
            )
            flux_err[j] = T.sqrt(
                np.mean(map_error[q]) ** 2 + T.exp(log_jit[j])
            ) / np.sqrt(2)

            obs[j] = pm.Normal(
                f"obs_{j}", mu=model_flux[j], sd=flux_err[j], observed=map_flux[q]
            )

    with indep_model:
        indep_map = indep_model.test_point
        indep_map = pmx.optimize(
            start=indep_map, vars=[flux0, log_jit], progress=VERBOSE
        )

        for npl in range(NPL):
            indep_map = pmx.optimize(
                start=indep_map, vars=[tt_offset[npl]], progress=VERBOSE
            )

        indep_map = pmx.optimize(start=indep_map, progress=VERBOSE)

    indep_transit_times = []
    indep_error = []
    indep_linear_ephemeris = []
    full_indep_linear_ephemeris = []

    for npl, p in enumerate(planets):
        indep_transit_times.append(np.copy(slide_transit_times[npl]))
        indep_error.append(np.copy(slide_error[npl]))

        replace = np.isnan(slide_error[npl])

        if np.any(replace):
            indep_transit_times[npl][replace] = indep_map[f"tts_{npl}"]

        pfit = poly.polyfit(transit_inds[npl], indep_transit_times[npl], 1)

        indep_linear_ephemeris.append(poly.polyval(transit_inds[npl], pfit))
        full_indep_linear_ephemeris.append(poly.polyval(p.index, pfit))

        if np.all(replace):
            indep_error[npl][replace] = mad_std(
                indep_transit_times[npl] - indep_linear_ephemeris[npl]
            )
        elif np.any(replace):
            indep_error[npl][replace] = mad_std(
                indep_transit_times[npl][~replace]
                - indep_linear_ephemeris[npl][~replace]
            )

    for npl, p in enumerate(planets):
        print(f"\nPLANET {npl}")

        # grab data
        xtime = indep_linear_ephemeris[npl]
        yomc = indep_transit_times[npl] - indep_linear_ephemeris[npl]

        yerr_expected = predict_tc_error(
            np.sqrt(p.depth), p.impact, p.duration, lcit, np.median(lc.error)
        )
        yerr_measured = indep_error[npl]

        yerr = np.sqrt(yerr_expected**2 + yerr_measured**2)

        print("  expected uncertainty: {0:.1f} min".format(yerr_expected * 24 * 60))
        print(
            "  measured uncertainty: {0:.1f} min".format(
                np.median(yerr_measured * 24 * 60)
            )
        )
        print("  adopted uncertainty:  {0:.1f} min".format(np.median(yerr * 24 * 60)))
        print("  dispersion:           {0:.1f} min".format(mad_std(yomc) * 24 * 60))

    print("")
    print("cumulative runtime = ", int(timer() - global_start_time), "s")
    print("")

    # # ###############################
    # # ----- OMC MODEL SELECTION -----
    # # ###############################

    print("\nIdentifying best OMC model...\n")

    # ## Search for periodic signals

    print("...searching for periodic signals")

    indep_freqs = []
    indep_faps = []

    for npl, p in enumerate(planets):
        # grab data
        xtime = indep_linear_ephemeris[npl]
        yomc = indep_transit_times[npl] - indep_linear_ephemeris[npl]

        # estimate measurement uncertainty
        yerr_expected = predict_tc_error(
            np.sqrt(p.depth), p.impact, p.duration, lcit, np.median(lc.error)
        )
        yerr_measured = indep_error[npl]

        # flag outliers
        ymed = boxcar_smooth(median_filter(yomc, size=5, mode="mirror"), winsize=5)
        out = np.abs(yomc - ymed) / mad_std(yomc - ymed) > 5.0

        # search for a periodic component
        peakfreq = np.nan
        peakfap = 1.0

        if NPL == 1:
            fap = 0.1
        elif NPL > 1:
            fap = 0.99
        else:
            raise ValueError("NPL must be >= 1")

        if np.sum(~out) > 8:
            try:
                xf, yf, freqs, faps = LS_estimator(xtime[~out], yomc[~out], fap=fap)

                if len(freqs) > 0:
                    if freqs[0] > xf.min():
                        peakfreq = freqs[0]
                        peakfap = faps[0]

            except Exception:
                pass

        indep_freqs.append(peakfreq)
        indep_faps.append(peakfap)

    omc_freqs = []
    omc_faps = []

    # for single planet systems, use the direct LS output
    if NPL == 1:
        if np.isnan(indep_freqs[0]):
            omc_freqs.append(None)
            omc_faps.append(None)
        else:
            omc_freqs.append(indep_freqs[0])
            omc_faps.append(indep_faps[0])

    # for multiplanet systems, check if any statistically marginal frequencies match between planets
    elif NPL > 1:
        for i in range(NPL):
            # save any low FAP frequencies
            if indep_faps[i] < 0.1:
                omc_freqs.append(indep_freqs[i])
                omc_faps.append(indep_faps[i])

            # check if the LS frequency is close to that of any other planet
            else:
                close = False

                df_min = 1 / (
                    indep_linear_ephemeris[i].max() - indep_linear_ephemeris[i].min()
                )

                for j in range(i + 1, NPL):
                    # delta-freq (LS) between two planets
                    df_ij = np.abs(indep_freqs[i] - indep_freqs[j])

                    if df_ij < df_min:
                        close = True

                if close:
                    omc_freqs.append(indep_freqs[i])
                    omc_faps.append(indep_faps[i])

                else:
                    omc_freqs.append(None)
                    omc_faps.append(None)

    omc_pers = []

    for npl in range(NPL):
        print("\nPLANET", npl)

        # roughly model OMC based on single frequency sinusoid (if found)
        if omc_freqs[npl] is not None:
            print("periodic signal found at P =", int(1 / omc_freqs[npl]), "d")

            # store frequency
            omc_pers.append(1 / omc_freqs[npl])

            # grab data and plot
            xtime = indep_linear_ephemeris[npl]
            yomc = indep_transit_times[npl] - indep_linear_ephemeris[npl]
            LS = LombScargle(xtime, yomc)

            plt.figure(figsize=(12, 3))
            plt.plot(xtime, yomc * 24 * 60, "o", c="lightgrey")
            plt.plot(
                xtime, LS.model(xtime, omc_freqs[npl]) * 24 * 60, c=f"C{npl}", lw=3
            )
            if IPLOT:
                plt.show()
            else:
                plt.close()

        else:
            print("no sigificant periodic component found")
            omc_pers.append(
                2
                * (
                    indep_linear_ephemeris[npl].max()
                    - indep_linear_ephemeris[npl].min()
                )
            )

    # ## Determine best OMC model

    print("...running model selection routine")

    quick_transit_times = []
    full_quick_transit_times = []

    outlier_prob = []
    outlier_class = []

    for npl, p in enumerate(planets):
        print(f"\nPLANET {npl}")

        # grab data
        xtime = indep_linear_ephemeris[npl]
        yomc = indep_transit_times[npl] - indep_linear_ephemeris[npl]

        # flag outliers
        if len(yomc) > 16:
            ymed = boxcar_smooth(median_filter(yomc, size=5, mode="mirror"), winsize=5)
        else:
            ymed = np.median(yomc)

        if len(yomc) > 4:
            out = np.abs(yomc - ymed) / mad_std(yomc - ymed) > 5.0
        else:
            out = np.zeros(len(yomc), dtype="bool")

        # estimate uncertainty
        yerr_expected = predict_tc_error(
            np.sqrt(p.depth), p.impact, p.duration, lcit, np.median(lc.error)
        )
        yerr_measured = indep_error[npl]

        yerr = np.sqrt(yerr_expected**2 + yerr_measured**2)
        ymax = np.sqrt(mad_std(yomc) ** 2 - np.median(yerr) ** 2)

        # compare various models
        aiclist = []
        biclist = []

        if np.sum(~out) >= 16:
            min_polyorder = -1
            max_polyorder = 3
        elif np.sum(~out) >= 8:
            min_polyorder = -1
            max_polyorder = 2
        elif np.sum(~out) >= 4:
            min_polyorder = 0
            max_polyorder = 2
        else:
            min_polyorder = 1
            max_polyorder = 1

        # don't use a GP on very noisy data
        if np.median(yerr) >= 0.5 * mad_std(yomc):
            min_polyorder = np.max([0, min_polyorder])

        for polyorder in range(min_polyorder, max_polyorder + 1):
            if polyorder == -1:
                omc_model = omc.matern32_model(
                    xtime[~out], yomc[~out], yerr[~out], ymax, xtime
                )
            elif polyorder == 0:
                omc_model = omc.sin_model(
                    xtime[~out], yomc[~out], yerr[~out], omc_pers[npl], xtime
                )
            elif polyorder >= 1:
                omc_model = omc.poly_model(
                    xtime[~out], yomc[~out], yerr[~out], polyorder, xtime
                )
            else:
                raise ValueError("polyorder must be >= -1")

            with omc_model:
                omc_map = omc_model.test_point
                omc_map = pmx.optimize(start=omc_map, progress=VERBOSE)
                omc_trace = pmx.sample(
                    tune=8000,
                    draws=2000,
                    chains=2,
                    target_accept=0.95,
                    start=omc_map,
                    progressbar=VERBOSE,
                )

            omc_trend = np.nanmedian(omc_trace["pred"], 0)
            residuals = yomc - omc_trend

            plt.figure(figsize=(12, 3))
            plt.errorbar(
                xtime,
                yomc * 24 * 60,
                yerr=yerr * 24 * 60,
                fmt="o",
                c="lightgrey",
                zorder=-1,
            )
            plt.plot(xtime[out], yomc[out] * 24 * 60, "rx")
            plt.plot(xtime, omc_trend * 24 * 60, c=f"C{npl}", lw=2)
            plt.xlabel("Time [BJKD]", fontsize=16)
            plt.ylabel("O-C [min]", fontsize=16)
            if IPLOT:
                plt.show()
            else:
                plt.close()

            # calculate AIC & BIC
            n = len(yomc)

            if polyorder == -1:
                k = np.nanmedian(omc_trace["dof"])
                print(f"keff : {k:.1f}")
            elif polyorder == 0:
                k = 3
            else:
                k = polyorder + 1

            chisq = np.sum((yomc[~out] - omc_trend[~out]) ** 2 / yerr[~out] ** 2)
            lnlike = -chisq

            aic = 2 * k - 2 * lnlike
            bic = k * np.log(n) - 2 * lnlike

            aiclist.append(aic)
            biclist.append(bic)

            print("AIC:", np.round(aic, 1))
            print("BIC:", np.round(bic, 1))

        # choose the best model and recompute
        polyorder_aic = np.arange(min_polyorder, max_polyorder + 1)[np.argmin(aiclist)]
        polyorder_bic = np.arange(min_polyorder, max_polyorder + 1)[np.argmin(biclist)]
        polyorder = np.max([polyorder_aic, polyorder_bic])

        xt_predict = full_indep_linear_ephemeris[npl]

        if polyorder == -1:
            omc_model = omc.matern32_model(
                xtime[~out], yomc[~out], yerr[~out], ymax, xt_predict
            )
        elif polyorder == 0:
            omc_model = omc.sin_model(
                xtime[~out], yomc[~out], yerr[~out], omc_pers[npl], xt_predict
            )
        elif polyorder >= 1:
            omc_model = omc.poly_model(
                xtime[~out], yomc[~out], yerr[~out], polyorder, xt_predict
            )

        with omc_model:
            omc_map = omc_model.test_point
            omc_map = pmx.optimize(start=omc_map, progress=VERBOSE)
            omc_trace = pmx.sample(
                tune=8000,
                draws=2000,
                chains=2,
                target_accept=0.95,
                start=omc_map,
                progressbar=VERBOSE,
            )

        # flag outliers with K-means clustering
        omc_trend = np.nanmedian(omc_trace["pred"], 0)
        residuals = yomc - omc_trend[np.isin(xt_predict, xtime)]
        mix_model = omc.mix_model(residuals)

        with mix_model:
            mix_trace = pmx.sample(
                tune=8000, draws=2000, chains=1, target_accept=0.95, progressbar=VERBOSE
            )

        loc = np.nanmedian(mix_trace["mu"], axis=0)
        scales = np.nanmedian(1 / np.sqrt(mix_trace["tau"]), axis=0)

        fg_prob, bad = omc.flag_outliers(residuals, loc, scales)

        while np.sum(bad) / len(bad) > 0.3:
            thresh = np.max(fg_prob[bad])
            bad = fg_prob < thresh

        num_bad = np.sum(bad)
        perc_bad = 100 * num_bad / len(bad)

        err_ = np.median(yerr_measured[~bad]) * 24 * 60
        rms_ = mad_std(residuals[~bad]) * 24 * 60

        print(
            f"{num_bad:.0f} outliers found out of {len(bad):.0f} transit times ({perc_bad:.1f}%)"
        )
        print(f"measured error: {err_:.1f} min")
        print(f"residual RMS: {rms_:.1f} min")

        # save the final results
        full_omc_trend = np.nanmedian(omc_trace["pred"], 0)

        full_quick_transit_times.append(
            full_indep_linear_ephemeris[npl] + full_omc_trend
        )
        quick_transit_times.append(full_quick_transit_times[npl][transit_inds[npl]])

        outlier_prob.append(1 - fg_prob)
        outlier_class.append(bad)

        plt.figure(figsize=(12, 4))
        plt.scatter(
            xtime, yomc * 24 * 60, c=1 - fg_prob, cmap="viridis", label="MAP TTVs"
        )
        plt.plot(xtime[bad], yomc[bad] * 24 * 60, "rx")
        plt.plot(
            full_indep_linear_ephemeris[npl],
            full_omc_trend * 24 * 60,
            "k",
            label="Quick model",
        )
        plt.xlabel("Time [BJKD]", fontsize=20)
        plt.ylabel("O-C [min]", fontsize=20)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.legend(fontsize=14, loc="upper right")
        plt.title(TARGET, fontsize=20)
        plt.text(
            xtime.min(),
            yomc.min() * 24 * 60,
            f"measured error = {err_:.1f} min",
            fontsize=16,
            ha="left",
            backgroundcolor="w",
        )
        plt.text(
            xtime.max(),
            yomc.min() * 24 * 60,
            f"residual RMS = {rms_:.1f} min",
            fontsize=16,
            ha="right",
            backgroundcolor="w",
        )
        plt.savefig(
            os.path.join(FIGURE_DIR, TARGET + f"_ttvs_quick_{npl:02d}.png"),
            bbox_inches="tight",
        )
        if IPLOT:
            plt.show()
        else:
            plt.close()

    # ## Estimate TTV scatter w/ uncertainty buffer

    ttv_scatter = np.zeros(NPL)
    ttv_buffer = np.zeros(NPL)

    for npl in range(NPL):
        # estimate TTV scatter
        ttv_scatter[npl] = mad_std(indep_transit_times[npl] - quick_transit_times[npl])

        # based on scatter in independent times, set threshold so not even one outlier is expected
        N = len(transit_inds[npl])
        eta = np.max([3.0, stats.norm.interval((N - 1) / N)[1]])

        ttv_buffer[npl] = eta * ttv_scatter[npl] + lcit

    # ## Update TTVs

    for npl, p in enumerate(planets):
        # update transit time info in Planet objects
        epoch, period = poly.polyfit(p.index, full_quick_transit_times[npl], 1)

        p.epoch = np.copy(epoch)
        p.period = np.copy(period)
        p.tts = np.copy(full_quick_transit_times[npl])

    print("")
    print("cumulative runtime = ", int(timer() - global_start_time), "s")
    print("")

    # ## Flag outliers based on transit model

    print("\nFlagging outliers based on transit model...\n")

    res = [None] * len(quarters)
    bad = [None] * len(quarters)

    for j, q in enumerate(quarters):
        print(f"QUARTER {q}")

        if (all_dtype[q] != "long") and (all_dtype[q] != "short"):
            pass

        else:
            if all_dtype[q] == "long":
                use = lc.quarter == q
                t_ = lc.time[use]
                f_ = lc.flux[use]
            elif all_dtype[q] == "short":
                use = sc.quarter == q
                t_ = sc.time[use]
                f_ = sc.flux[use]
            else:
                raise ValueError("cadence data type must be 'short' or 'long'")

            # grab transit times for each planet
            wp = []
            tts = []
            inds = []

            for npl in range(NPL):
                qtt = quick_transit_times[npl]
                use = (qtt > t_.min()) * (qtt < t_.max())

                if np.sum(use) > 0:
                    wp.append(npl)
                    tts.append(qtt[use])
                    inds.append(transit_inds[npl][use] - transit_inds[npl][use][0])

            if len(tts) == 0:
                all_dtype[q] = all_dtype[q] + "_no_transits"
            else:
                # set up model
                starrystar = exo.LimbDarkLightCurve([U1, U2])

                orbit = exo.orbits.TTVOrbit(
                    transit_times=tts,
                    transit_inds=inds,
                    period=list(periods[wp]),
                    b=impacts[wp],
                    ror=rors[wp],
                    duration=durs[wp],
                )

                # calculate light curves
                light_curves = starrystar.get_light_curve(
                    orbit=orbit,
                    r=rors[wp],
                    t=t_,
                    oversample=oversample[q],
                    texp=texp[q],
                )

                model_flux = 1.0 + pm.math.sum(light_curves, axis=-1).eval()

                # flag outliers
                N = len(f_)
                eta = np.max([3.0, stats.norm.interval((N - 1) / N)[1]])

                res[j] = f_ - model_flux
                bad[j] = np.abs(res[j] - np.median(res[j])) / (mad_std(res[j])) > eta

                loop = 0
                count = np.sum(bad[j])
                while loop < 5:
                    bad[j] = (
                        np.abs(res[j] - np.median(res[j][~bad[j]]))
                        / (mad_std(res[j][~bad[j]]))
                        > eta
                    )

                    if np.sum(bad[j]) == count:
                        loop = 5
                    else:
                        loop += 1

            if res[j] is not None:
                print(" outliers rejected:", np.sum(bad[j]))

                plt.figure(figsize=(20, 3))
                plt.plot(t_, res[j], "k.")
                plt.plot(t_[bad[j]], res[j][bad[j]], "x", c="C1", ms=20)

                for n in range(len(tts)):
                    plt.plot(tts[n], np.zeros_like(tts[n]), "|", c="C0", ms=100, mew=3)

                plt.ylim(np.percentile(res[j], 0.15), np.percentile(res[j], 99.85))
                if IPLOT:
                    plt.show()
                else:
                    plt.close()

    good_cadno_lc = []
    good_cadno_sc = []

    for j, q in enumerate(quarters):
        if all_dtype[q] == "long":
            use = lc.quarter == q
            good_cadno_lc.append(lc.cadno[use][~bad[j]])

        if all_dtype[q] == "short":
            use = sc.quarter == q
            good_cadno_sc.append(sc.cadno[use][~bad[j]])

    if len(good_cadno_lc) > 0:
        good_cadno_lc = np.hstack(good_cadno_lc)

    if len(good_cadno_sc) > 0:
        good_cadno_sc = np.hstack(good_cadno_sc)

    # # #########################
    # # ----- 2ND DETRENDING -----
    # # #########################

    print("\nResetting to raw MAST data and performing 2nd DETRENDING...\n")

    # ## Reset to raw input data

    if MISSION == "Kepler":
        # reset LONG CADENCE data
        if lc is not None:
            lc_data = io.cleanup_lkfc(lc_raw_collection, KIC)

        # make sure there is at least one transit in the long cadence data
        # this shouldn't be an issue for real KOIs, but can happen for simulated data
        if np.sum(np.array(all_dtype) == "long") == 0:
            lc_data = []

        lc_quarters = []
        for i, lcd in enumerate(lc_data):
            lc_quarters.append(lcd.quarter)

        # reset SHORT CADENCE data
        if sc is not None:
            sc_data = io.cleanup_lkfc(sc_raw_collection, KIC)

        # make sure there is at least one transit in the short cadence data
        # this shouldn't be an issue for real KOIs, but can happen for simulated data
        if np.sum(np.array(all_dtype) == "short") == 0:
            sc_data = []

        sc_quarters = []
        for i, scd in enumerate(sc_data):
            sc_quarters.append(scd.quarter)

        # convert LightKurves to LiteCurves
        sc_lite = []
        lc_lite = []

        for i, scd in enumerate(sc_data):
            sc_lite.append(io.LightKurve_to_LiteCurve(scd))

        for i, lcd in enumerate(lc_data):
            lc_lite.append(io.LightKurve_to_LiteCurve(lcd))

    elif MISSION == "Simulated":
        sc_lite = deepcopy(sc_raw_sim_data)
        lc_lite = deepcopy(lc_raw_sim_data)

    # ## Remove flagged cadences

    sc_data = []
    for i, scl in enumerate(sc_lite):
        qmask = np.isin(scl.cadno, good_cadno_sc)

        if np.sum(qmask) / len(qmask) > 0.1:
            sc_data.append(scl.remove_flagged_cadences(qmask))

    lc_data = []
    for i, lcl in enumerate(lc_lite):
        qmask = np.isin(lcl.cadno, good_cadno_lc)

        if np.sum(qmask) / len(qmask) > 0.1:
            lc_data.append(lcl.remove_flagged_cadences(qmask))

    # ## Detrend the lightcurves

    # array to hold dominant oscillation period for each quarter
    oscillation_period_by_quarter = np.ones(18) * np.nan

    # long cadence data
    min_period = 1.0

    for i, lcd in enumerate(lc_data):
        qmask = lk.KeplerQualityFlags.create_quality_mask(
            lcd.quality, bitmask="default"
        )
        lcd.remove_flagged_cadences(qmask)

        # make transit mask
        lcd.mask = np.zeros(len(lcd.time), dtype="bool")
        for npl, p in enumerate(planets):
            lcd.mask += make_transit_mask(
                lcd.time, p.tts, np.max([1 / 24, 0.5 * p.duration + ttv_buffer[npl]])
            )

        lcd.clip_outliers(kernel_size=13, sigma_upper=5, sigma_lower=5, mask=lcd.mask)
        lcd.clip_outliers(kernel_size=13, sigma_upper=5, sigma_lower=1000, mask=None)

        # identify primary oscillation period
        ls_estimate = LombScargle(lcd.time, lcd.flux)
        xf, yf = ls_estimate.autopower(
            minimum_frequency=1 / (lcd.time.max() - lcd.time.min()),
            maximum_frequency=1 / min_period,
        )

        peak_freq = xf[np.argmax(yf)]
        peak_per = np.max([1.0 / peak_freq, 1.001 * min_period])

        oscillation_period_by_quarter[lcd.quarter[0]] = peak_per

    # short cadence data
    min_period = 1.0

    for i, scd in enumerate(sc_data):
        qmask = lk.KeplerQualityFlags.create_quality_mask(
            scd.quality, bitmask="default"
        )
        scd.remove_flagged_cadences(qmask)

        # make transit mask
        scd.mask = np.zeros(len(scd.time), dtype="bool")
        for npl, p in enumerate(planets):
            scd.mask += make_transit_mask(
                scd.time, p.tts, np.max([1 / 24, 0.5 * p.duration + ttv_buffer[npl]])
            )

        scd.clip_outliers(kernel_size=13, sigma_upper=5, sigma_lower=5, mask=scd.mask)
        scd.clip_outliers(kernel_size=13, sigma_upper=5, sigma_lower=1000, mask=None)

        # identify primary oscillation period
        ls_estimate = LombScargle(scd.time, scd.flux)
        xf, yf = ls_estimate.autopower(
            minimum_frequency=1 / (scd.time.max() - scd.time.min()),
            maximum_frequency=1 / min_period,
        )

        peak_freq = xf[np.argmax(yf)]
        peak_per = np.max([1.0 / peak_freq, 1.001 * min_period])

        oscillation_period_by_quarter[scd.quarter[0]] = peak_per

    # seasonal approach assumes both stellar and instrumental effects are present
    oscillation_period_by_season = np.zeros((4, 2))

    for i in range(4):
        oscillation_period_by_season[i, 0] = np.nanmedian(
            oscillation_period_by_quarter[i::4]
        )
        oscillation_period_by_season[i, 1] = mad_std(
            oscillation_period_by_quarter[i::4], ignore_nan=True
        )

    # detrend long cadence data
    break_tolerance = np.max([int(DURS.min() / lcit * 5 / 2), 13])
    min_period = 1.0

    for i, lcd in enumerate(lc_data):
        print(f"QUARTER {lcd.quarter[0]}")

        nom_per = oscillation_period_by_season[lcd.quarter[0] % 4][0]

        try:
            lcd = detrend.flatten_with_gp(
                lcd,
                break_tolerance,
                min_period,
                nominal_period=nom_per,
                verbose=VERBOSE,
            )
        except LinAlgError:
            warnings.warn(
                "Initial detrending failed...attempting to refit without exponential ramp component"
            )
            try:
                lcd = detrend.flatten_with_gp(
                    lcd,
                    break_tolerance,
                    min_period,
                    nominal_period=nom_per,
                    correct_ramp=False,
                    verbose=VERBOSE,
                )
            except LinAlgError:
                warnings.warn(
                    "Detrending with RotationTerm failed...attempting to detrend with SHOTerm"
                )
                lcd = detrend.flatten_with_gp(
                    lcd,
                    break_tolerance,
                    min_period,
                    nominal_period=nom_per,
                    kterm="SHOTerm",
                    correct_ramp=False,
                    verbose=VERBOSE,
                )

    if len(lc_data) > 0:
        lc = detrend.stitch(lc_data)
    else:
        lc = None

    # detrend short cadence data
    break_tolerance = np.max([int(DURS.min() / scit * 5 / 2), 91])
    min_period = 1.0

    for i, scd in enumerate(sc_data):
        print(f"QUARTER {scd.quarter[0]}")

        nom_per = oscillation_period_by_season[scd.quarter[0] % 4][0]

        try:
            scd = detrend.flatten_with_gp(
                scd,
                break_tolerance,
                min_period,
                nominal_period=nom_per,
                verbose=VERBOSE,
            )
        except LinAlgError:
            warnings.warn(
                "Initial detrending failed...attempting to refit without exponential ramp component"
            )
            try:
                scd = detrend.flatten_with_gp(
                    scd,
                    break_tolerance,
                    min_period,
                    nominal_period=nom_per,
                    correct_ramp=False,
                    verbose=VERBOSE,
                )
            except LinAlgError:
                warnings.warn(
                    "Detrending with RotationTerm failed...attempting to detrend with SHOTerm"
                )
                scd = detrend.flatten_with_gp(
                    scd,
                    break_tolerance,
                    min_period,
                    nominal_period=nom_per,
                    kterm="SHOTerm",
                    correct_ramp=False,
                    verbose=VERBOSE,
                )

    if len(sc_data) > 0:
        sc = detrend.stitch(sc_data)
    else:
        sc = None

    # ## Do a final round of outliers rejection

    print("\nFlagging remaining outliers...\n")

    res = [None] * len(quarters)
    bad = [None] * len(quarters)

    for j, q in enumerate(quarters):
        print(f"QUARTER {q}")

        if (all_dtype[q] != "long") and (all_dtype[q] != "short"):
            pass

        else:
            if all_dtype[q] == "long":
                use = lc.quarter == q
                t_ = lc.time[use]
                f_ = lc.flux[use]
            elif all_dtype[q] == "short":
                use = sc.quarter == q
                t_ = sc.time[use]
                f_ = sc.flux[use]
            else:
                raise ValueError("cadence data type must be 'short' or 'long'")

            # grab transit times for each planet
            wp = []
            tts = []
            inds = []

            for npl in range(NPL):
                qtt = quick_transit_times[npl]
                use = (qtt > t_.min()) * (qtt < t_.max())

                if np.sum(use) > 0:
                    wp.append(npl)
                    tts.append(qtt[use])
                    inds.append(transit_inds[npl][use] - transit_inds[npl][use][0])

            if len(tts) > 0:
                # set up model
                starrystar = exo.LimbDarkLightCurve([U1, U2])

                orbit = exo.orbits.TTVOrbit(
                    transit_times=tts,
                    transit_inds=inds,
                    period=list(periods[wp]),
                    b=impacts[wp],
                    ror=rors[wp],
                    duration=durs[wp],
                )

                # calculate light curves
                light_curves = starrystar.get_light_curve(
                    orbit=orbit,
                    r=rors[wp],
                    t=t_,
                    oversample=oversample[q],
                    texp=texp[q],
                )

                model_flux = 1.0 + pm.math.sum(light_curves, axis=-1).eval()

                # flag outliers
                N = len(f_)
                eta = np.max([3.0, stats.norm.interval((N - 1) / N)[1]])

                res[j] = f_ - model_flux
                bad[j] = np.abs(res[j] - np.median(res[j])) / (mad_std(res[j])) > eta

                loop = 0
                count = np.sum(bad[j])
                while loop < 5:
                    bad[j] = (
                        np.abs(res[j] - np.median(res[j][~bad[j]]))
                        / (mad_std(res[j][~bad[j]]))
                        > eta
                    )

                    if np.sum(bad[j]) == count:
                        loop = 5
                    else:
                        loop += 1

            if res[j] is not None:
                print(" outliers rejected:", np.sum(bad[j]))

                plt.figure(figsize=(20, 3))
                plt.plot(t_, res[j], "k.")
                plt.plot(t_[bad[j]], res[j][bad[j]], "x", c="C1", ms=20)

                for n in range(len(tts)):
                    plt.plot(tts[n], np.zeros_like(tts[n]), "|", c="C0", ms=100, mew=3)

                plt.ylim(np.percentile(res[j], 0.15), np.percentile(res[j], 99.85))
                if IPLOT:
                    plt.show()
                else:
                    plt.close()

    good_cadno_lc = []
    good_cadno_sc = []

    for j, q in enumerate(quarters):
        if all_dtype[q] == "long":
            use = lc.quarter == q
            good_cadno_lc.append(lc.cadno[use][~bad[j]])

        if all_dtype[q] == "short":
            use = sc.quarter == q
            good_cadno_sc.append(sc.cadno[use][~bad[j]])

    if len(good_cadno_lc) > 0:
        good_cadno_lc = np.hstack(good_cadno_lc)

    if len(good_cadno_sc) > 0:
        good_cadno_sc = np.hstack(good_cadno_sc)

    if lc is not None:
        qmask = np.isin(lc.cadno, good_cadno_lc)
        lc = lc.remove_flagged_cadences(qmask)

    if sc is not None:
        qmask = np.isin(sc.cadno, good_cadno_sc)
        sc = sc.remove_flagged_cadences(qmask)

    # # ##############################################
    # # ----- MAKE PLOTS, OUTPUT DATA, & CLEAN UP -----
    # # ##############################################

    # ## Make individual mask for where each planet transits
    # #### These masks have width 1.5 transit durations, which may be wider than the masks used for detrending

    if sc is not None:
        sc_mask = np.zeros((NPL, len(sc.time)), dtype="bool")
        for npl, p in enumerate(planets):
            sc_mask[npl] = make_transit_mask(
                sc.time, p.tts, np.max([3 / 24, 1.5 * p.duration])
            )

        sc.mask = sc_mask.sum(axis=0) > 0

    else:
        sc_mask = None

    if lc is not None:
        lc_mask = np.zeros((NPL, len(lc.time)), dtype="bool")
        for npl, p in enumerate(planets):
            lc_mask[npl] = make_transit_mask(
                lc.time, p.tts, np.max([3 / 24, 1.5 * p.duration])
            )

        lc.mask = lc_mask.sum(axis=0) > 0

    else:
        lc_mask = None

    # ## Flag high quality transits (quality = 1)
    # #### Good transits must have  at least 50% photometry coverage in/near transit

    for npl, p in enumerate(planets):
        count_expect_lc = np.max([1, int(np.floor(p.duration / lcit))])
        count_expect_sc = np.max([15, int(np.floor(p.duration / scit))])

        quality = np.zeros(len(p.tts), dtype="bool")

        for i, t0 in enumerate(p.tts):
            if sc is not None:
                in_sc = np.abs(sc.time - t0) / p.duration < 0.5
                near_sc = np.abs(sc.time - t0) / p.duration < 1.5

                qual_in = np.sum(in_sc) > 0.5 * count_expect_sc
                qual_near = np.sum(near_sc) > 1.5 * count_expect_sc

                quality[i] += qual_in * qual_near

            if lc is not None:
                in_lc = np.abs(lc.time - t0) / p.duration < 0.5
                near_lc = np.abs(lc.time - t0) / p.duration < 1.5

                qual_in = np.sum(in_lc) > 0.5 * count_expect_lc
                qual_near = np.sum(near_lc) > 1.5 * count_expect_lc

                quality[i] += qual_in * qual_near

        p.quality = np.copy(quality)

    # ## Flag which transits overlap (overlap = 1)

    # identify overlapping transits
    overlap = []

    for i in range(NPL):
        overlap.append(np.zeros(len(planets[i].tts), dtype="bool"))

        for j in range(NPL):
            if i != j:
                for ttj in planets[j].tts:
                    overlap[i] += np.abs(planets[i].tts - ttj) / durs.max() < 1.5

        planets[i].overlap = np.copy(overlap[i])

    # ## Make phase-folded transit plots

    print("\nMaking phase-folded transit plots...\n")

    for npl, p in enumerate(planets):
        tts = p.tts[p.quality * ~p.overlap]

        if len(tts) == 0:
            print(
                f"No non-overlapping high quality transits found for planet {npl} (P = {p.period:.1f} d)"
            )

        else:
            t_folded = []
            f_folded = []

            # grab the data
            for t0 in tts:
                if sc is not None:
                    use = np.abs(sc.time - t0) / p.duration < 1.5

                    if np.sum(use) > 0:
                        t_folded.append(sc.time[use] - t0)
                        f_folded.append(sc.flux[use])

                if lc is not None:
                    use = np.abs(lc.time - t0) / p.duration < 1.5

                    if np.sum(use) > 0:
                        t_folded.append(lc.time[use] - t0)
                        f_folded.append(lc.flux[use])

            # sort the data
            t_folded = np.hstack(t_folded)
            f_folded = np.hstack(f_folded)

            order = np.argsort(t_folded)
            t_folded = t_folded[order]
            f_folded = f_folded[order]

            # bin the data
            t_binned, f_binned = bin_data(t_folded, f_folded, p.duration / 11)

            # set undersampling factor and plotting limits
            inds = np.arange(len(t_folded), dtype="int")
            inds = np.random.choice(inds, size=np.min([3000, len(inds)]), replace=False)

            ymin = 1 - 3 * np.std(f_folded) - p.depth
            ymax = 1 + 3 * np.std(f_folded)

            # plot the data
            plt.figure(figsize=(12, 4))
            plt.plot(t_folded[inds] * 24, f_folded[inds], ".", c="lightgrey")
            plt.plot(
                t_binned * 24,
                f_binned,
                "o",
                ms=8,
                color=f"C{npl}",
                label=f"{TARGET}-{npl}",
            )
            plt.xlim(t_folded.min() * 24, t_folded.max() * 24)
            plt.ylim(ymin, ymax)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.xlabel("Time from mid-transit [hrs]", fontsize=20)
            plt.ylabel("Flux", fontsize=20)
            plt.legend(fontsize=20, loc="upper right", framealpha=1)
            plt.savefig(
                os.path.join(FIGURE_DIR, TARGET + f"_folded_transit_{npl:02d}.png"),
                bbox_inches="tight",
            )
            if IPLOT:
                plt.show()
            else:
                plt.close()

    # ## Save transit times

    for npl, p in enumerate(planets):
        keep = np.isin(quick_transit_times[npl], p.tts[p.quality])

        data_out = np.vstack(
            [
                transit_inds[npl][keep],
                indep_transit_times[npl][keep],
                quick_transit_times[npl][keep],
                outlier_prob[npl][keep],
                outlier_class[npl][keep],
            ]
        ).swapaxes(0, 1)

        fname_out = os.path.join(RESULTS_DIR, f"{TARGET}_{npl:02d}_quick.ttvs")
        np.savetxt(
            fname_out,
            data_out,
            fmt=("%1d", "%.8f", "%.8f", "%.8f", "%1d"),
            delimiter="\t",
        )

    # ## Save detrended lightcurves

    print("\nSaving detrended lightcurves...\n")

    if lc is not None:
        filename = os.path.join(RESULTS_DIR, f"{TARGET}_lc_detrended.fits")
        lc.to_fits(TARGET, filename, cadence="LONG")
    else:
        print("No long cadence data")

    if sc is not None:
        filename = os.path.join(RESULTS_DIR, f"{TARGET}_sc_detrended.fits")
        sc.to_fits(TARGET, filename, cadence="SHORT")
    else:
        print("No short cadence data")

    # ## Exit program

    print("")
    print("+" * shutil.get_terminal_size().columns)
    print(
        f"Automated lightcurve detrending complete {datetime.now().strftime('%d-%b-%Y at %H:%M:%S')}"
    )
    print("Total runtime = %.1f min" % ((timer() - global_start_time) / 60))
    print("+" * shutil.get_terminal_size().columns)


if __name__ == "__main__":
    main()
