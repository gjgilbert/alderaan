#######################################
# - Fit Transit Shape Independently - #
#######################################

# This script fits a transit lightcurve model for a single Kepler planet
# Overlapping transits in multiplanet systems are removed
# Transits are modeled using PyMC3/HMC following the umbrella sampling routine outlined in Gilbert 2022
# The code outputs a posterior MCMC chain for a *single* umbrella window

import os
import sys
import glob
import shutil
import warnings
from datetime import datetime
from timeit import default_timer as timer

print("")
print("+" * shutil.get_terminal_size().columns)
print("ALDERAAN Transit Fitting (single planet)")
print("Initialized {0}".format(datetime.now().strftime("%d-%b-%Y at %H:%M:%S")))
print("+" * shutil.get_terminal_size().columns)
print("")

# start program timer
global_start_time = timer()

# MCMC parameters are always manually set w/in the code
Nchain = 2
Ntune = 20000
Ndraw = 10000
target_accept = 0.95

# parse inputs
import argparse
import matplotlib as mpl

parser = argparse.ArgumentParser(
    description="Inputs for ALDERAAN transit fiting pipeline"
)
parser.add_argument(
    "--mission",
    default=None,
    type=str,
    required=True,
    help="Mission name; can be 'Kepler' or 'Simulated'",
)
parser.add_argument(
    "--target",
    default=None,
    type=str,
    required=True,
    help="Target name; format should be K00000 or S00000",
)
parser.add_argument(
    "--planet_no",
    default=None,
    type=int,
    required=True,
    help="ALDERAAN zero-indexed planet identifier (i.e. *NOT* KOI_ID)",
)
parser.add_argument(
    "--root_dir",
    default=None,
    type=str,
    required=True,
    help="Root directory for system",
)
parser.add_argument(
    "--project_dir",
    default=None,
    type=str,
    required=True,
    help="Project directory for accessing lightcurve data and saving outputs",
)
parser.add_argument(
    "--catalog",
    default=None,
    type=str,
    required=True,
    help="CSV file containing input planetary parameters",
)
parser.add_argument(
    "--umbrella",
    default=None,
    type=str,
    required=True,
    help="Umbrella can be 'N', 'T', or 'G'",
)
parser.add_argument(
    "--interactive",
    default=False,
    type=bool,
    required=False,
    help="'True' to enable interactive plotting; by default matplotlib backend will be set to 'Agg'",
)

args = parser.parse_args()
MISSION = args.mission
TARGET = args.target
PLANET_NO = args.planet_no
ROOT_DIR = args.root_dir
PROJECT_DIR = ROOT_DIR + args.project_dir
CATALOG = PROJECT_DIR + "Catalogs/" + args.catalog
UMBRELLA = args.umbrella

# set plotting backend
if args.interactive == False:
    mpl.use("agg")

# set environment variables
sys.path.append(PROJECT_DIR)

# build directory structure
FIGURE_DIR = PROJECT_DIR + "Figures/" + TARGET + "/"
TRACE_DIR = PROJECT_DIR + "Traces/" + TARGET + "/"
QUICK_TTV_DIR = PROJECT_DIR + "QuickTTVs/" + TARGET + "/"
DLC_DIR = PROJECT_DIR + "Detrended_lightcurves/" + TARGET + "/"
NOISE_DIR = PROJECT_DIR + "Noise_models/" + TARGET + "/"

# check if all the output directories exist and if not, create them
if os.path.exists(FIGURE_DIR) == False:
    os.mkdir(FIGURE_DIR)

if os.path.exists(TRACE_DIR) == False:
    os.mkdir(TRACE_DIR)

if os.path.exists(QUICK_TTV_DIR) == False:
    os.mkdir(QUICK_TTV_DIR)

if os.path.exists(DLC_DIR) == False:
    os.mkdir(DLC_DIR)

if os.path.exists(NOISE_DIR) == False:
    os.mkdir(NOISE_DIR)


# import packages
import astropy.stats
from astropy.io import fits
import json
import matplotlib.pyplot as plt
import numpy as np
import numpy.polynomial.polynomial as poly
import pandas as pd
from scipy import stats

import pymc3 as pm
import pymc3_ext as pmx
import exoplanet as exo
import aesara_theano_fallback.tensor as T
from aesara_theano_fallback import aesara as theano
from celerite2.theano import GaussianProcess
from celerite2.theano import terms as GPterms

from alderaan.constants import *
import alderaan.io as io
from alderaan.detrend import make_transitmask
from alderaan.LiteCurve import LiteCurve
from alderaan.Planet import Planet


# flush buffer to avoid mixed outputs from progressbar
sys.stdout.flush()

# turn off FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)

# check for interactive matplotlib backends
if np.any(np.array(["agg", "png", "svg", "pdf", "ps"]) == mpl.get_backend()):
    iplot = False
else:
    iplot = True

# echo theano cache directory
print("theano cache: {0}\n".format(theano.config.compiledir))


################
# - DATA I/O - #
################

print("\nLoading data...\n")

# Read in planet and star properties from csv file
target_dict = pd.read_csv(CATALOG)

# set KOI_ID global variable
if MISSION == "Kepler":
    KOI_ID = TARGET
elif MISSION == "Simulated":
    KOI_ID = "K" + TARGET[1:]
else:
    raise ValueError("MISSION must be 'Kepler' or 'Simulated'")

# pull relevant quantities and establish GLOBAL variables
use = np.array(target_dict["koi_id"]) == KOI_ID

KIC = np.array(target_dict["kic_id"], dtype="int")[use]
NPL = np.array(target_dict["npl"], dtype="int")[use]

PERIODS = np.array(target_dict["period"], dtype="float")[use]
DEPTHS = np.array(target_dict["depth"], dtype="float")[use] * 1e-6  # [ppm] --> []
DURS = np.array(target_dict["duration"], dtype="float")[use] / 24  # [hrs] --> [days]

# do some consistency checks
if all(k == KIC[0] for k in KIC):
    KIC = KIC[0]
else:
    raise ValueError("There are inconsistencies with KIC in the csv input file")

if all(n == NPL[0] for n in NPL):
    NPL = NPL[0]
else:
    raise ValueError("There are inconsistencies with NPL in the csv input file")

# sort planet parameters by period
order = np.argsort(PERIODS)

PERIODS = PERIODS[order]
DEPTHS = DEPTHS[order]
DURS = DURS[order]


# Read in filtered lightcurves
# These can be generated by running the script "analyze_autocorrelated_noise.py"

try:
    lc = io.load_detrended_lightcurve(DLC_DIR + TARGET + "_lc_filtered.fits")
    lc.season = lc.quarter % 4
except:
    lc = None

try:
    sc = io.load_detrended_lightcurve(DLC_DIR + TARGET + "_sc_filtered.fits")
    sc.season = sc.quarter % 4
except:
    sc = None


# Read in quick transit times
# These can be generated by running the script "detrend_and_estimate_ttvs.py"

epochs = np.zeros(NPL)
periods = np.zeros(NPL)
ephemeris = [None] * NPL

transit_inds = []
indep_transit_times = []
quick_transit_times = []

for npl in range(NPL):
    fname_in = QUICK_TTV_DIR + TARGET + "_{:02d}".format(npl) + "_quick.ttvs"
    data_in = np.genfromtxt(fname_in)

    transit_inds.append(np.array(data_in[:, 0], dtype="int"))
    indep_transit_times.append(np.array(data_in[:, 1], dtype="float"))
    quick_transit_times.append(np.array(data_in[:, 2], dtype="float"))

    # do a quick fit to get a linear ephemeris
    pfit = poly.polyfit(transit_inds[npl], quick_transit_times[npl], 1)

    epochs[npl] = pfit[0]
    periods[npl] = pfit[1]
    ephemeris[npl] = poly.polyval(transit_inds[npl], pfit)

# make sure transit_inds are zero-indexed
for npl in range(NPL):
    transit_inds[npl] = np.array(transit_inds[npl] - transit_inds[npl][0], dtype="int")


# Make OMC plots
fig, axes = plt.subplots(NPL, figsize=(12, 3 * NPL))
if NPL == 1:
    axes = [axes]

for npl in range(NPL):
    xtime = ephemeris[npl]
    yomc_i = (indep_transit_times[npl] - ephemeris[npl]) * 24 * 60
    yomc_q = (quick_transit_times[npl] - ephemeris[npl]) * 24 * 60

    axes[npl].plot(xtime, yomc_i, "o", c="lightgrey")
    axes[npl].plot(xtime, yomc_q, lw=2, c="C{0}".format(npl))
    axes[npl].set_ylabel("O-C [min]", fontsize=20)
axes[NPL - 1].set_xlabel("Time [BJKD]", fontsize=20)

if iplot:
    plt.show()
else:
    plt.close()


#####################
# - PRELIMINARIES - #
#####################

print("\nRunning preliminaries...\n")

# Establish time baseline
time_min = []
time_max = []

if sc is not None:
    time_min.append(sc.time.min())
    time_max.append(sc.time.max())

if lc is not None:
    time_min.append(lc.time.min())
    time_max.append(lc.time.max())

TIME_START = np.min(time_min)
TIME_END = np.max(time_max)


# Put epochs in range (TIME_START, TIME_START + PERIOD)
for npl in range(NPL):
    if epochs[npl] < TIME_START:
        adj = 1 + (TIME_START - epochs[npl]) // periods[npl]
        epochs[npl] += adj * periods[npl]

    if epochs[npl] > (TIME_START + periods[npl]):
        adj = (epochs[npl] - TIME_START) // periods[npl]
        epochs[npl] -= adj * periods[npl]


# Identify and remove overlapping transits
overlap = []

for i in range(NPL):
    overlap.append(np.zeros(len(ephemeris[i]), dtype="bool"))

    for j in range(NPL):
        if i != j:
            for tt in ephemeris[j]:
                overlap[i] += np.abs(ephemeris[i] - tt) < (DURS[i] + DURS[j] + lcit)

ephemeris = [ephemeris[npl][~overlap[npl]] for npl in range(NPL)]
transit_inds = [transit_inds[npl][~overlap[npl]] for npl in range(NPL)]
quick_transit_times = [quick_transit_times[npl][~overlap[npl]] for npl in range(NPL)]


# Track which quarter each transit falls in
if lc is not None:
    lc_quarters = np.unique(lc.quarter)
else:
    lc_quarters = np.array([])

if sc is not None:
    sc_quarters = np.unique(sc.quarter)
else:
    sc_quarters = np.array([])

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
    tts = ephemeris[npl]
    transit_quarter[npl] = np.zeros(len(tts), dtype="int")

    for j, q in enumerate(quarters):
        transit_quarter[npl][(tts >= thresh[j]) * (tts < thresh[j + 1])] = q


# Make transit masks
if sc is not None:
    sc_mask = np.zeros((NPL, len(sc.time)), dtype="bool")
    for npl in range(NPL):
        sc_mask[npl] = make_transitmask(sc.time, quick_transit_times[npl], masksize=1.5)

if lc is not None:
    lc_mask = np.zeros((NPL, len(lc.time)), dtype="bool")
    for npl in range(NPL):
        lc_mask[npl] = make_transitmask(lc.time, quick_transit_times[npl], masksize=1.5)


# Grab data near transits for each quarter
all_time = [None] * 18
all_flux = [None] * 18
all_error = [None] * 18
all_mask = [None] * 18
all_dtype = ["none"] * 18

for q in range(18):
    if sc is not None:
        if np.isin(q, sc.quarter):
            use = (sc_mask.sum(0) != 0) * (sc.quarter == q)

            if np.sum(use) > 45:
                all_time[q] = sc.time[use]
                all_flux[q] = sc.flux[use]
                all_error[q] = sc.error[use]
                all_mask[q] = sc_mask[:, use]
                all_dtype[q] = "short"
            else:
                all_dtype[q] = "short_no_transits"

    if lc is not None:
        if np.isin(q, lc.quarter):
            use = (lc_mask.sum(0) != 0) * (lc.quarter == q)

            if np.sum(use) > 5:
                all_time[q] = lc.time[use]
                all_flux[q] = lc.flux[use]
                all_error[q] = lc.error[use]
                all_mask[q] = lc_mask[:, use]
                all_dtype[q] = "long"
            else:
                all_dtype[q] = "long_no_transits"


# set oversampling factors and expoure times
oversample = np.zeros(18, dtype="int")
texp = np.zeros(18)

oversample[np.array(all_dtype) == "short"] = 1
oversample[np.array(all_dtype) == "long"] = 15

texp[np.array(all_dtype) == "short"] = scit
texp[np.array(all_dtype) == "long"] = lcit


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


# Use Legendre polynomials over transit times for better orthogonality; "x" is in the range (-1,1)
# The current version of the code only uses 1st order polynomials, but 2nd and 3rd are retained for posterity
Leg0 = []
Leg1 = []
Leg2 = []
Leg3 = []
t = []

# this assumes a baseline in the range (TIME_START,TIME_END)
for npl in range(NPL):
    t.append(ephemeris[npl])
    x = 2 * (t[npl] - TIME_START) / (TIME_END - TIME_START) - 1

    Leg0.append(np.ones_like(x))
    Leg1.append(x.copy())
    Leg2.append(0.5 * (3 * x**2 - 1))
    Leg3.append(0.5 * (5 * x**3 - 3 * x))


# Read in GP noise model priors from analyze_autocorrelated_noise.py
gp_percs = []

for z in range(4):
    try:
        fname_in = NOISE_DIR + TARGET + "_shoterm_gp_priors_{0}.txt".format(z)

        with open(fname_in) as infile:
            gp_percs.append(json.load(infile))

    except:
        gp_percs.append(None)

# convert the percentile priors into Gaussians
gp_priors = []

for z in range(4):
    if gp_percs[z] is not None:
        gpz = {}

        for k in gp_percs[z].keys():
            if k != "percentiles":
                perc = np.array(gp_percs[z]["percentiles"])

                med = np.array(gp_percs[z][k])[perc == 50.0][0]
                err1 = np.array(gp_percs[z][k])[perc == 84.135][0]
                err2 = np.array(gp_percs[z][k])[perc == 15.865][0]

                dev = np.sqrt((err1 - med) ** 2 / 2 + (err2 - med) ** 2 / 2)

                gpz[k] = (med, dev)

        gp_priors.append(gpz)

    else:
        # these are dummy values that effectively create a zero-amplitude kernel
        gpz = {}
        gpz["logw0"] = [np.log(2 * pi / (7 * DURS.max()))]
        gpz["logSw4"] = [-100.0]
        gpz["logQ"] = [np.log(1 / np.sqrt(2))]

        gp_priors.append(gpz)

# calculate a few convenience quantities
for z in range(4):
    gpz = gp_priors[z]

    logS = gpz["logSw4"][0] - 4 * gpz["logw0"][0]

    if len(gpz["logSw4"]) == 1:
        gp_priors[z]["logS"] = [logS]

    if len(gpz["logSw4"]) == 2:
        logS_var = gpz["logSw4"][1] ** 2 + 16 * gpz["logw0"][1] ** 2
        gp_priors[z]["logS"] = np.array([logS, np.sqrt(logS_var)])


##########################
# - LIGHTCURVE FITTING - #
##########################

print("\nModeling Lightcurve...\n")

for npl in range(NPL):
    if npl == PLANET_NO:
        print("PLANET", npl)

        with pm.Model() as shape_model:
            # identify which quarters and seasons have data
            which_quarters = np.unique(transit_quarter[npl])
            which_seasons = np.unique(which_quarters % 4)

            # limb darkening
            u = exo.distributions.QuadLimbDark("LD_U", testval=[0.40, 0.25])

            # draw (r,b) parameters
            rmin, rmax = 1e-5, 0.99

            if UMBRELLA == "N":
                # draw (log_r, b) --> (r,b)
                log_r = pm.Uniform(
                    "LN_ROR",
                    lower=np.log(rmin),
                    upper=np.log(rmax),
                    testval=np.log(np.sqrt(DEPTHS[npl])),
                )
                r = pm.Deterministic("ROR", T.exp(log_r))
                b = pm.Uniform("IMPACT", lower=0, upper=1 - r)
                g = pm.Deterministic("GAMMA", (1 - b) / r)

                # this adjustment term makes samples uniform in the (r,b) plane
                adj = pm.Potential("adj", T.log(1 - r) + T.log(r))

                # umbrella bias
                norm = 1 / rmin - 1.5
                psi = pm.Potential(
                    "PSI", T.log(T.switch(T.lt(g, 2), g - 1, 1.0)) / norm
                )

            elif UMBRELLA == "T":
                # draw (log_r, gamma) --> (r,b)
                log_r = pm.Uniform(
                    "LN_ROR",
                    lower=np.log(rmin),
                    upper=np.log(rmax),
                    testval=np.log(np.sqrt(DEPTHS[npl])),
                )
                r = pm.Deterministic("ROR", T.exp(log_r))
                g = pm.Uniform(
                    "GAMMA", lower=0, upper=T.switch(r < 0.5, 2, 1 / r), testval=1.0
                )
                b = pm.Deterministic("IMPACT", 1 - g * r)

                # Jacobian for (r,b) --> (r,gamma)
                jac = pm.Potential("jac", T.log(1 / r))

                # this adjustment term makes samples uniform in the (r,b) plane
                adj = pm.Potential(
                    "adj", 2 * T.log(r) + T.switch(r < 0.5, T.log(2 * r), 0.0)
                )

                # umbrella bias
                norm = 1.0
                psi = pm.Potential("PSI", T.log(T.switch(T.lt(g, 1), g, 2 - g)) / norm)

            elif UMBRELLA == "G":
                # draw (log_lambda, gamma) --> (r,b)
                g = pm.Uniform("GAMMA", lower=-0.99, upper=1.0, testval=0.0)
                log_lam = pm.Uniform(
                    "LN_LAM",
                    lower=np.log((g + 1) * rmin**2),
                    upper=np.log((g + 1) * rmax**2),
                )
                lam = pm.Deterministic("LAM", T.exp(log_lam))

                r = pm.Deterministic("ROR", pm.math.sqrt(lam / (g + 1)))
                log_r = pm.Deterministic("LN_ROR", T.log(r))
                b = pm.Deterministic("IMPACT", 1 - g * r)

                # Jacobian for (r,b) --> (lambda,gamma)
                jac = pm.Potential("jac", T.log(2 + 2 * g))

                # this adjustment term makes samples uniform in the (r,b) plane
                adj = pm.Potential("adj", -T.log(2 + 2 * g) + 2 * T.log(r))

                # umbrella bias
                norm = 1.0
                psi = pm.Potential(
                    "PSI", T.log(T.switch(T.lt(g, 0), 1 + g, 1 - g)) / norm
                )

            else:
                raise ValueError("Umbrella must be 'N', 'T', or 'G'")

            # enforce log-uniform prior on r
            r_marginal = pm.Potential("r_marginal", -T.log(1 + r) - T.log(r))

            # draw transit duration
            log_dur = pm.Uniform(
                "LN_DUR14",
                lower=np.log(scit),
                upper=np.log(3 * DURS[npl]),
                testval=np.log(DURS[npl]),
            )
            dur = pm.Deterministic("DUR14", T.exp(log_dur))

            # polynomial TTV parameters (coefficients for Legendre polynomials)
            C0 = pm.Normal(
                "C0", mu=0, sd=3 * np.std(quick_transit_times[npl] - ephemeris[npl])
            )
            C1 = pm.Normal(
                "C1", mu=0, sd=3 * np.std(quick_transit_times[npl] - ephemeris[npl])
            )

            # transit times
            transit_times = pm.Deterministic(
                "TTIMES", quick_transit_times[npl] + C0 * Leg0[npl] + C1 * Leg1[npl]
            )

            # set up limb darkened star and planetary orbit
            starrystar = exo.LimbDarkLightCurve(u)
            orbit = exo.orbits.TTVOrbit(
                transit_times=[transit_times],
                transit_inds=[transit_inds[npl]],
                b=b,
                duration=dur,
                ror=r,
            )

            # track period, epoch, and stellar density
            P = pm.Deterministic("PERIOD", orbit.period)
            T0 = pm.Deterministic("T0", orbit.t0)

            # build the GP kernel using a different noise model for each season
            logS = [None] * 4
            logw0 = [None] * 4
            logQ = [None] * 4
            kernel = [None] * 4

            for z in which_seasons:
                gpz = gp_priors[z]

                # logS[z]   = pm.Normal('GP_LNS_{0}'.format(z), mu=gpz['logS'][0], sd=gpz['logS'][1])
                # logw0[z]  = pm.Normal('GP_LNOM_{0}'.format(z), mu=gpz['logw0'][0], sd=gpz['logw0'][1])
                logS[z] = gpz["logS"][0]
                logw0[z] = gpz["logw0"][0]
                logQ[z] = gpz["logQ"][0]
                kernel[z] = GPterms.SHOTerm(
                    S0=T.exp(logS[z]), w0=T.exp(logw0[z]), Q=T.exp(logQ[z])
                )

            # mean flux and jitter for each quarter
            nq = len(which_quarters)
            mbq = mean_by_quarter[which_quarters]
            vbq = var_by_quarter[which_quarters]

            flux0 = pm.Normal("FLUXZPT", mu=mbq, sd=np.sqrt(vbq), shape=nq)
            log_jit = pm.Normal(
                "LN_JIT", mu=np.log(vbq / 10), sd=5.0 * np.ones(nq), shape=nq
            )

            # now evaluate the model
            light_curves = [None] * nq
            model_flux = [None] * nq
            flux_err = [None] * nq
            gp = [None] * nq

            for j, q in enumerate(which_quarters):
                # grab time and flux
                t_ = all_time[q][all_mask[q][npl]]
                f_ = all_flux[q][all_mask[q][npl]]

                # calculate light curves
                light_curves[j] = starrystar.get_light_curve(
                    orbit=orbit, r=r, t=t_, oversample=oversample[q], texp=texp[q]
                )

                model_flux[j] = pm.math.sum(light_curves[j], axis=-1) + flux0[
                    j
                ] * T.ones(len(t_))
                flux_err[j] = T.sqrt(
                    np.mean(all_error[q]) ** 2 + T.exp(log_jit[j])
                ) / np.sqrt(2)

                # here's the GP (w/ kernel by season)
                gp[j] = GaussianProcess(
                    kernel[q % 4],
                    t=t_,
                    diag=flux_err[j] ** 2 * T.ones(len(t_)),
                    mean=model_flux[j],
                )

                gp[j].marginal("gp_{0}".format(j), observed=f_)

        with shape_model:
            shape_map = shape_model.test_point
            shape_map = pmx.optimize(start=shape_map, vars=[flux0, log_jit])
            shape_map = pmx.optimize(start=shape_map, vars=[C0, C1])
            shape_map = pmx.optimize(start=shape_map, vars=[r, b, dur])

        with shape_model:
            trace = pmx.sample(
                tune=Ntune,
                draws=Ndraw,
                start=shape_map,
                chains=Nchain,
                target_accept=target_accept,
                return_inferencedata=True,
            )

            summary = pm.summary(trace)


# Write MCMC trace to .fits file
print("\nWriting MCMC trace to .fits\n")

# set up header info
summary_info = [
    # ['STATISTIC', 'DESCRIPTION'],
    ["MEAN", "Mean of the posterior"],
    ["SD", "Standard deviation of the posterior"],
    ["HDI_03", "Highest density interval of posterior at 3%"],
    ["HDI_97", "Highest density interval of posterior at 97%"],
    ["MCSE_MU", "Markov Chain Standard Error of posterior mean"],
    ["MCSE_SD", "Markov Chain Standard Error of posterior std dev"],
    ["ESS_BULK", "Effective sample size for bulk posterior"],
    ["ESS_TAIL", "Effective sample size for tail posterior"],
    ["R_HAT", "Estimate of rank normalized R-hat statistic"],
]

data_info = [
    # ['VARIABLE', 'UNITS',         'DESCRIPTION',   'PRIOR'],
    ["PERIOD", "days", "Orbital period", "Normal"],
    ["T0", "days", "Mid-point of first transit", "Normal"],
    [
        "LN_ROR",
        "dimensionless",
        "Natural log of planet-to-star radius ratio",
        "Uniform",
    ],
    ["LN_DUR14", "days", "Natural log of transit duration: 1st-4th contact", "Uniform"],
    ["IMPACT", "dimensionless", "Transit impact parameter", "Uniform"],
    [
        "LD_U1",
        "dimensionless",
        "1st quadratic stellar limb darkening coefficient",
        "Kipping (2013)",
    ],
    [
        "LD_U2",
        "dimensionless",
        "2nd quadratic stellar limb darkening coefficient",
        "Kipping (2013)",
    ],
    ["FLUXZPT", "dimensionless", "Mean flux offset", "Normal"],
    ["LN_JIT", "dimensionless", "Photometric jitter, added in quadrature", "Normal"],
    ["PSI", "dimensionless", "Umbrella bias", "Gilbert (2022)"],
    ["GAMMA", "dimensionless", "Grazing coordinate", "Gilbert (2022)"],
    ["LN_LIKE", "dimensionless", "Natural log-likelihood of the model fit", ""],
    ["QUALITY", "bool", "Data quality flag indicating divergent samples", ""],
]

SUMMARY_DTYPE = [
    ("PARAM_NAME", "U10"),
    ("MEAN", "<f8"),
    ("SD", "<f8"),
    ("HDI_03", "<f8"),
    ("HDI_97", "<f8"),
    ("MCSE_MU", "<f8"),
    ("MCSE_SD", "<f8"),
    ("ESS_BULK", "<f8"),
    ("ESS_TAIL", "<f8"),
    ("R_HAT", "<f8"),
]


# Build trace dataframe
trace_df = pd.DataFrame()

# split 2-dim LD_U into separate columns
for k in list(trace.posterior.keys()):
    if k == "LD_U":
        trace_df["LD_U1"] = list(trace.posterior["LD_U"][:, :, 0].data.reshape(-1))
        trace_df["LD_U2"] = list(trace.posterior["LD_U"][:, :, 1].data.reshape(-1))

    elif k == "FLUXZPT":
        for j, q in enumerate(which_quarters):
            trace_df["FLUXZPT_{0}".format(str(q).zfill(2))] = list(
                trace.posterior["FLUXZPT"][:, :, j].data.reshape(-1)
            )

    elif k == "LN_JIT":
        for j, q in enumerate(which_quarters):
            trace_df["LN_JIT_{0}".format(str(q).zfill(2))] = list(
                trace.posterior["LN_JIT"][:, :, j].data.reshape(-1)
            )

    elif k == "TTIMES":
        pass

    else:
        trace_df[k] = trace.posterior[k].data.reshape(-1)

# pull likelihood
ln_like = np.zeros((Nchain, Ndraw), dtype="float")
for dv in trace.log_likelihood.data_vars:
    ln_like += np.squeeze(trace.log_likelihood[dv].data)

trace_df["LN_LIKE"] = ln_like.reshape(-1)

# pull quality flags
trace_df["QUALITY"] = trace["sample_stats"]["diverging"].data.reshape(-1).astype(int)

# to avoid redundancy, prefer LN_ROR over ROR
if "ROR" in list(summary.index) and "LN_ROR" in list(summary.index):
    trace_df.drop("ROR", axis=1, inplace=True)

# to avoid redundancy, prefer LN_DUR14 over DUR14
if "DUR14" in list(summary.index) and "LN_DUR14" in list(summary.index):
    trace_df.drop("DUR14", axis=1, inplace=True)


# Build summary dataframe
summary_df = pd.DataFrame(columns=summary.keys())


def mapper_fluxzpt(index):
    return index[:7] + "_{0}".format(str(index[8:-1].zfill(2)))


def mapper_ln_jit(index):
    return index[:6] + "_{0}".format(str(index[7:-1].zfill(2)))


def mapper_ld_u(index):
    return index[:4] + "{0}".format(int(index[5]) + 1)


for idx in summary.index:
    if np.isin(idx, trace_df.keys()):
        summary_df = pd.concat([summary_df, summary[summary.index == idx]])

    elif idx[:7] == "FLUXZPT":
        summary_df = pd.concat([summary_df, summary[summary.index == idx]])
        summary_df.rename(index={idx: mapper_fluxzpt(idx)}, inplace=True)

    elif idx[:6] == "LN_JIT":
        summary_df = pd.concat([summary_df, summary[summary.index == idx]])
        summary_df.rename(index={idx: mapper_ln_jit(idx)}, inplace=True)

    elif idx[:4] == "LD_U":
        summary_df = pd.concat([summary_df, summary[summary.index == idx]])
        summary_df.rename(index={idx: mapper_ld_u(idx)}, inplace=True)

    elif idx[:6] == "PERIOD":
        summary_df = pd.concat([summary_df, summary[summary.index == idx]])
        summary_df.rename(index={idx: "PERIOD"}, inplace=True)

    elif idx[:2] == "T0":
        summary_df = pd.concat([summary_df, summary[summary.index == idx]])
        summary_df.rename(index={idx: "T0"}, inplace=True)


# Build primary HDU and header
timestamp = datetime.strptime(
    trace.posterior.attrs["created_at"], "%Y-%m-%dT%H:%M:%S.%f"
)

hdu = fits.PrimaryHDU()
hdu.header["TARGET"] = (TARGET + "-" + str(PLANET_NO).zfill(2), "Target name")
hdu.header["PHOTOSRC"] = (MISSION, "Source of photometry")
hdu.header["UMBRELLA"] = (UMBRELLA, "see Gilbert 2022")
hdu.header["DATETIME"] = (
    timestamp.strftime("%Y-%m-%d %H:%M:%S"),
    "Date and time of analysis",
)
hdu.header["COMMENT"] = "ALDERAAN single planet transit fit"

f_name = PROJECT_DIR + "Traces/{0}/{0}-{1}_{2}_indep.fits".format(
    TARGET, str(PLANET_NO).zfill(2), UMBRELLA
)
hdu.writeto(f_name, overwrite=True)

# store data from dataframes into a single FITS file
with fits.open(f_name) as hduL:
    # Store trace data and summary data to HDUs for each model
    rec_data = trace_df.to_records(index=False)
    hdu_data = fits.BinTableHDU(data=rec_data, name="POSTERIOR")

    rec_summ = summary_df.to_records().astype(SUMMARY_DTYPE)
    hdu_summ = fits.BinTableHDU(data=rec_summ, name="MCMC_STATS")

    hduL.append(hdu_data)
    hduL.append(hdu_summ)

    # posterior data
    data_head = hduL["POSTERIOR"].header
    data_head["RUNTIME"] = (
        np.round(trace.posterior.attrs["sampling_time"], 2),
        "Duration of model sampling (seconds)",
    )
    data_head["NTUNE"] = (Ntune, "Number of tuning steps")
    data_head["NDRAW"] = (Ndraw, "Number of sampler draws")
    data_head["NCHAIN"] = (Nchain, "Number of sampler chains")

    for i, dinf in enumerate(data_info):
        data_head[dinf[0]] = (dinf[1], dinf[2])

    # MCMC sampler statistics
    summ_head = hduL["MCMC_STATS"].header

    for i, sinf in enumerate(summary_info):
        summ_head[sinf[0]] = ("", sinf[1])

    hduL.writeto(f_name, overwrite=True)


# Exit program

print("")
print("+" * shutil.get_terminal_size().columns)
print(
    "Exoplanet recovery complete {0}".format(
        datetime.now().strftime("%d-%b-%Y at %H:%M:%S")
    )
)
print("Total runtime = %.1f min" % ((timer() - global_start_time) / 60))
print("+" * shutil.get_terminal_size().columns)
