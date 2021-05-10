#!/usr/bin/env python
# coding: utf-8

# # Detrend and Estimate Noise

# In[ ]:


import numpy as np
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt

import numpy.polynomial.polynomial as poly
import scipy.optimize as op
import scipy.signal as sig
from   scipy import stats
from   scipy import fftpack
from   scipy import ndimage
from   scipy.interpolate import UnivariateSpline
import astropy
from   astropy.io import fits as pyfits
from   astropy.timeseries import LombScargle
from   sklearn.cluster import KMeans

import csv
import sys
import os
import importlib as imp
import glob
from   timeit import default_timer as timer
import warnings
import progressbar
import argparse
import json
from   copy import deepcopy

import lightkurve as lk
import exoplanet as exo
import theano.tensor as T
import theano
import pymc3 as pm
import corner

from alderaan.constants import *
from alderaan.utils import *
from alderaan.Planet import *
from alderaan.LiteCurve import *
import alderaan.io as io
import alderaan.detrend as detrend
import alderaan.noise as noise

# flush buffer to avoid mixed outputs from progressbar
sys.stdout.flush()

# turn off FutureWarnings
warnings.filterwarnings('ignore', category=FutureWarning)

# start program timer
global_start_time = timer()

# LCIT and SCIT in [days]
lcit = LCIT/60/24
scit = SCIT/3600/24


# # Manually set I/O parameters
# #### User should manually set MISSION, TARGET, PRIMARY_DIR,  and CSV_FILE
# #### Note that if using DR25, you will need to manually correct epochs from BJD to BJKD with an offset of 2454833.0 days; the cumulative exoplanet archive catalog has already converted epochs to BJKD

# In[ ]:



# here's where we parse the inputs
try:
    parser = argparse.ArgumentParser(description="Inputs for ALDERAAN transit fiting pipeline")
    parser.add_argument("--mission", default=None, type=str, required=True,                         help="Mission name")
    parser.add_argument("--target", default=None, type=str, required=True,                         help="Target name; see ALDERAAN documentation for acceptable formats")
    parser.add_argument("--primary_dir", default=None, type=str, required=True,                         help="Primary directory path for accessing lightcurve data and saving outputs")
    parser.add_argument("--csv_file", default=None, type=str, required=True,                         help="Path to .csv file containing input planetary parameters")


    args = parser.parse_args()
    MISSION      = args.mission
    TARGET       = args.target
    PRIMARY_DIR  = args.primary_dir
    CSV_FILE     = args.csv_file    
    
except:
    pass


# # Make sure the necessary paths exist

# In[ ]:


# directory in which to find lightcurve data
if MISSION == 'Kepler': DOWNLOAD_DIR = PRIMARY_DIR + 'MAST_downloads/'
if MISSION == 'Simulated': DOWNLOAD_DIR = PRIMARY_DIR + 'Simulations/'

# directories in which to place pipeline outputs
FIGURE_DIR    = PRIMARY_DIR + 'Figures/' + TARGET + '/'
TRACE_DIR     = PRIMARY_DIR + 'Traces/' + TARGET + '/'
QUICK_TTV_DIR = PRIMARY_DIR + 'QuickTTVs/' + TARGET + '/'
DLC_DIR       = PRIMARY_DIR + 'Detrended_lightcurves/' + TARGET + '/'
NOISE_DIR     = PRIMARY_DIR + 'Noise_models/' + TARGET + '/'


# check if all the paths exist and create them if not
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


# # Read in planet and stellar parameters

# In[ ]:


# Read in the data from csv file
print('Reading in data from csv file')

# read in a csv file containing info on targets
csv_keys, csv_values = io.read_csv_file(CSV_FILE)

# put these csv data into a dictionary
target_dict = {}
for k in csv_keys: 
    target_dict[k] = io.get_csv_data(k, csv_keys, csv_values)

    
if MISSION == 'Kepler':
    KOI_ID = TARGET
    
elif MISSION == 'Simulated':
    KOI_ID = "K" + TARGET[1:]

else:
    raise ValueError("MISSION must be 'Kepler' or 'Simulated'")
    
    
# pull relevant quantities and establish GLOBAL variables
use = np.array(target_dict['koi_id']) == KOI_ID

KIC = np.array(target_dict['kic_id'], dtype='int')[use]
NPL = np.array(target_dict['npl'], dtype='int')[use]

RSTAR = np.array(target_dict['rstar'],  dtype='float')[use]
RSTAR_ERR1 = np.array(target_dict['rstar_err1'],  dtype='float')[use]
RSTAR_ERR2 = np.array(target_dict['rstar_err2'],  dtype='float')[use]

MSTAR  = np.array(target_dict['mstar'], dtype='float')[use]
MSTAR_ERR1 = np.array(target_dict['mstar_err1'],  dtype='float')[use]
MSTAR_ERR2 = np.array(target_dict['mstar_err2'],  dtype='float')[use]

U1 = np.array(target_dict['limbdark_1'], dtype='float')[use]
U2 = np.array(target_dict['limbdark_2'], dtype='float')[use]

PERIODS = np.array(target_dict['period'], dtype='float')[use]
EPOCHS  = np.array(target_dict['epoch'],  dtype='float')[use]
DEPTHS  = np.array(target_dict['depth'], dtype='float')[use]*1e-6          # [ppm] --> []
DURS    = np.array(target_dict['duration'], dtype='float')[use]/24         # [hrs] --> [days]
IMPACTS = np.array(target_dict['impact'], dtype='float')[use]


# In[ ]:


# do some consistency checks
if all(k == KIC[0] for k in KIC): KIC = KIC[0]
else: raise ValueError('There are inconsistencies with KIC in the csv input file')

if all(n == NPL[0] for n in NPL): NPL = NPL[0]
else: raise ValueError('There are inconsistencies with NPL in the csv input file')

if all(r == RSTAR[0] for r in RSTAR): RSTAR = RSTAR[0]
else: raise ValueError('There are inconsistencies with RSTAR in the csv input file')

if all(r == RSTAR_ERR1[0] for r in RSTAR_ERR1): RSTAR_ERR1 = RSTAR_ERR1[0]
else: raise ValueError('There are inconsistencies with RSTAR_ERR1 in the csv input file')
    
if all(r == RSTAR_ERR2[0] for r in RSTAR_ERR2): RSTAR_ERR2 = RSTAR_ERR2[0]
else: raise ValueError('There are inconsistencies with RSTAR_ERR2 in the csv input file')  
    
if all(m == MSTAR[0] for m in MSTAR): MSTAR = MSTAR[0]
else: raise ValueError('There are inconsistencies with MSTAR in the csv input file')

if all(m == MSTAR_ERR1[0] for m in MSTAR_ERR1): MSTAR_ERR1 = MSTAR_ERR1[0]
else: raise ValueError('There are inconsistencies with MSTAR_ERR1 in the csv input file')
    
if all(m == MSTAR_ERR2[0] for m in MSTAR_ERR2): MSTAR_ERR2 = MSTAR_ERR2[0]
else: raise ValueError('There are inconsistencies with MSTAR_ERR2 in the csv input file')
    
if all(u == U1[0] for u in U1): U1 = U1[0]
else: raise ValueError('There are inconsistencies with U1 in the csv input file')

if all(u == U2[0] for u in U2): U2 = U2[0]
else: raise ValueError('There are inconsistencies with U2 in the csv input file')


# In[ ]:


# combline stellar radius/mass uncertainties
MSTAR_ERR = np.sqrt(MSTAR_ERR1**2 + MSTAR_ERR2**2)/np.sqrt(2)
RSTAR_ERR = np.sqrt(RSTAR_ERR1**2 + RSTAR_ERR2**2)/np.sqrt(2)

# limb darkening coefficients
UCOEFFS = [U1, U2]


# # Read in CDPP values

# In[ ]:


temp = []

with open(PRIMARY_DIR + "Catalogs/keplerstellar_cdpp.txt", "r") as file:
    temp = [line.rstrip('\n').split(",") for line in file]
    
    
keys = temp[0]
vals = temp[1:]

data = {}

for i, k in enumerate(keys):
    data[k] = []
    
    for j, v in enumerate(vals):
        data[k].append(v[i])

for k in data.keys():
    data[k] = np.array(data[k])
    
data["cdpp03"] = data.pop("rrmscdpp03p0")
data["cdpp06"] = data.pop("rrmscdpp06p0")
data["cdpp09"] = data.pop("rrmscdpp09p0")
data["cdpp12"] = data.pop("rrmscdpp12p0")
data["cdpp15"] = data.pop("rrmscdpp15p0")

for k in data.keys():
    empty = data[k] == ""
    
    data[k][empty] = "-99"
    
    try:
        data[k] = np.array(data[k], dtype="float")
        data[k][empty] = np.nan
    except:
        pass

data["kepid"] = np.array(data["kepid"], dtype="int")


use = data["kepid"] == KIC

c03 = np.nanmean(data["cdpp03"][use])
c06 = np.nanmean(data["cdpp06"][use])
c09 = np.nanmean(data["cdpp09"][use])
c12 = np.nanmean(data["cdpp12"][use])
c15 = np.nanmean(data["cdpp15"][use])

CDPP = np.array([c03, c06, c09, c12, c15])


# # Read in Holczer+ 2016 TTVs

# In[ ]:


HOLCZER_FILE = PRIMARY_DIR + "Catalogs/holczer_2016_kepler_ttvs.txt"

if MISSION == "Kepler":
    holczer_data = np.loadtxt(HOLCZER_FILE, usecols=[0,1,2,3])

    holczer_inds = []
    holczer_tts  = []
    holczer_pers = []

    for npl in range(NPL):
        koi = int(TARGET[1:]) + 0.01*(1+npl)
        use = np.isclose(holczer_data[:,0], koi, rtol=1e-10, atol=1e-10)
        
        # Holczer uses BJD -24548900; BJKD = BJD - 2454833
        if np.sum(use) > 0:
            holczer_inds.append(np.array(holczer_data[use,1], dtype="int"))
            holczer_tts.append(holczer_data[use,2] + holczer_data[use,3]/24/60 + 67)
            holczer_pers.append(np.median(holczer_tts[npl][1:] - holczer_tts[npl][:-1]))
            
        else:
            holczer_inds.append(None)
            holczer_tts.append(None)
            holczer_pers.append(np.nan)
            
    holczer_pers = np.asarray(holczer_pers)


# # Read in pre-downloaded lightcurve data
# #### Kepler data can be retrieved by running the script "download_from_MAST.py"
# #### Simulated data can be produced by running the script "simulate_lightcurve.py"

# In[ ]:


# short cadence
try:
    if MISSION == 'Kepler':
        sc_path  = glob.glob(DOWNLOAD_DIR + 'mastDownload/Kepler/kplr' + '{0:09d}'.format(KIC) + '*_sc*/')[0]
        sc_files = glob.glob(sc_path + '*')

        sc_rawdata_list = []
        for i, scf in enumerate(sc_files):
            oscfi = lk.search.open(sc_files[i])
            sc_rawdata_list.append(oscfi)

        sc_rawdata = lk.LightCurveFileCollection(sc_rawdata_list)
        sc_data = detrend.cleanup_lkfc(sc_rawdata, KIC)

        
    elif MISSION == 'Simulated':
        sc_path = DOWNLOAD_DIR + 'Lightcurves/Kepler/simkplr' + '{0:09d}'.format(KIC) + '_sc/'
        sc_files = glob.glob(sc_path + '*')

        sc_rawdata_list = []
        for i, scf in enumerate(sc_files):
            sc_rawdata_list.append(io.load_sim_fits(scf))


        quarters = []
        for i, scrd in enumerate(sc_rawdata_list):
            quarters.append(scrd.quarter)

        order = np.argsort(quarters)

        sc_data_list = []
        for j, q in enumerate(quarters):
            sc_data_list.append(sc_rawdata_list[order[j]])

        sc_rawdata = lk.LightCurveFileCollection(sc_rawdata_list)
        sc_data = lk.LightCurveCollection(sc_data_list)
        
        
except:
    sc_data = []

    
sc_quarters = []
for i, scd in enumerate(sc_data):
    sc_quarters.append(scd.quarter)    


# In[ ]:


# long cadence
if MISSION == 'Kepler':
    lc_path  = glob.glob(DOWNLOAD_DIR + 'mastDownload/Kepler/kplr' + '{0:09d}'.format(KIC) + '*_lc*/')[0]
    lc_files = glob.glob(lc_path + '*')

    lc_rawdata_list = []
    for i, lcf in enumerate(lc_files):
        olcfi = lk.search.open(lc_files[i])

        if ~np.isin(olcfi.quarter, lc_quarters):
            lc_rawdata_list.append(olcfi)

    lc_rawdata = lk.LightCurveFileCollection(lc_rawdata_list)
    lc_data = detrend.cleanup_lkfc(lc_rawdata, KIC)
    
    
elif MISSION == 'Simulated':
    lc_path = DOWNLOAD_DIR + 'Lightcurves/Kepler/simkplr' + '{0:09d}'.format(KIC) + '_lc/'
    lc_files = glob.glob(lc_path + '*')
    
    lc_rawdata_list = []
    for i, lcf in enumerate(lc_files):
        lc_rawdata_list.append(io.load_sim_fits(lcf))
        
    
    quarters = []
    for i, lcrd in enumerate(lc_rawdata_list):
        quarters.append(lcrd.quarter)
        
    order = np.argsort(quarters)
    
    lc_data_list = []
    for j, q in enumerate(quarters):
        lc_data_list.append(lc_rawdata_list[order[j]])

    lc_rawdata = lk.LightCurveFileCollection(lc_rawdata_list)
    lc_data = lk.LightCurveCollection(lc_data_list)
    
    
lc_quarters = []
for i, lcd in enumerate(lc_data):
    lc_quarters.append(lcd.quarter)


# In[ ]:


# determine the time baseline
time_min = []
time_max = []

try:
    for i, scd in enumerate(sc_data):
        time_min.append(scd.time.min())
        time_max.append(scd.time.max())
        
except:
    pass


try:
    for i, lcd in enumerate(lc_data):
        time_min.append(lcd.time.min())
        time_max.append(lcd.time.max())
        
except:
    pass
    
    
TIME_START = np.min(time_min)
TIME_END   = np.max(time_max)

if TIME_START < 0:
    raise ValueError("START TIME [BKJD] is negative...this will cause problems")


# put epochs in range (TIME_START, TIME_START + PERIOD)
for npl in range(NPL):
    if EPOCHS[npl] < TIME_START:
        adj = 1 + (TIME_START - EPOCHS[npl])//PERIODS[npl]
        EPOCHS[npl] += adj*PERIODS[npl]        
        
    if EPOCHS[npl] > (TIME_START + PERIODS[npl]):
        adj = (EPOCHS[npl] - TIME_START)//PERIODS[npl]
        EPOCHS[npl] -= adj*PERIODS[npl]


# # Initialize Planet objects

# In[ ]:


# initialize Planet objects
print('Initializing %d Planet objects' %NPL)

planets = []
for npl in range(NPL):
    p = Planet()
    
    # put in some basic transit parameters
    p.epoch    = EPOCHS[npl]
    p.period   = PERIODS[npl]
    p.depth    = DEPTHS[npl]
    p.duration = DURS[npl]
    p.impact   = IMPACTS[npl]
    
    if p.impact > 1 - np.sqrt(p.depth):
        p.impact = (1 - np.sqrt(p.depth))**2
        
    # estimate transit times from linear ephemeris
    p.tts = np.arange(p.epoch, TIME_END, p.period)

    # make transit indexes
    p.index = np.array(np.round((p.tts-p.epoch)/p.period),dtype='int')
    
    # add to list
    planets.append(p)


# put planets in order by period
order = np.argsort(PERIODS)

sorted_planets = []
for npl in range(NPL):
    sorted_planets.append(planets[order[npl]])

planets = np.copy(sorted_planets)


# # For simulated data, initilize with true TTVs + noise
# #### Synthetic "Holczer" TTVs are approximated as ground truth + Student-t2 noise

# In[ ]:


if MISSION == "Simulated":
    holczer_inds = []
    holczer_tts  = []
    holczer_pers = []
    
    for npl, p in enumerate(planets):
        # read in the "ground truth" TTVs
        fname_in = DOWNLOAD_DIR + "TTVs/" + TARGET + "_0{0}_sim_ttvs.txt".format(npl)
        data_in  = np.loadtxt(fname_in).swapaxes(0,1)
    
        inds = np.array(data_in[0], dtype="int")
        tts_true  = np.array(data_in[1], dtype="float")
        
        # add some noise and reject transits without photometry cover
        if len(tts_true) > 20:
            tts_noisy = tts_true + stats.t.rvs(df=2, size=len(tts_true))*p.duration/3
        else:
            tts_noisy = tts_true + np.random.normal(size=len(tts_true))*p.duration/3
        
        keep = np.zeros(len(tts_noisy), dtype="bool")
        
        for i, t0 in enumerate(tts_noisy):
            for j, scd in enumerate(sc_data):
                if np.min(np.abs(scd.time - t0)) < p.duration:
                    keep[i] = True
            for j, lcd in enumerate(lc_data):
                if np.min(np.abs(lcd.time - t0)) < p.duration:
                    keep[i] = True
        
        holczer_inds.append(inds[keep])
        holczer_tts.append(tts_noisy[keep])
        holczer_pers.append(p.period)
        
        
    holczer_pers = np.array(holczer_pers)


# # Smooth and interpolate Holczer TTVs where they exist

# In[ ]:


for npl in range(NPL):
    print("\nPLANET", npl)
    try:
        # fit a linear ephemeris 
        pfit  = poly.polyfit(holczer_inds[npl], holczer_tts[npl], 1)
        ephem = poly.polyval(holczer_inds[npl], pfit)
        
        
        # put fitted epoch in range (TIME_START, TIME_START + PERIOD)
        hepoch, hper = pfit

        if hepoch < TIME_START:
            adj = 1 + (TIME_START - hepoch)//hper
            hepoch += adj*hper       

        if hepoch > (TIME_START + hper):
            adj = (hepoch - TIME_START)//hper
            hepoch -= adj*hper      

        hephem = np.arange(hepoch, TIME_END, hper)        
        hinds  = np.array(np.round((hephem-hepoch)/hper),dtype='int')
        
        
        # calculate OMC and flag outliers
        xtime = np.copy(holczer_tts[npl])
        yomc  = (holczer_tts[npl] - ephem)

        ymed = ndimage.median_filter(yomc, size=5, mode="mirror")
        out  = np.abs(yomc-ymed)/astropy.stats.mad_std(yomc-ymed) > 3.0
                
        # set up a GP using a Matern-3/2 kernel
        with pm.Model() as holczer_model:

            # build the kernel 
            log_sigma = pm.Normal("log_sigma", mu=np.log(np.std(yomc)), sd=10)
            log_rho = pm.Normal("log_rho", mu=np.log(xtime[1]-xtime[0]), sd=10)

            kernel = exo.gp.terms.Matern32Term(log_sigma=log_sigma, log_rho=log_rho)

            # nusiance parameters
            mean = pm.Normal("mean", mu=np.mean(yomc), sd=10)
            logvar = pm.Normal("logvar", mu=np.log(np.var(yomc)), sd=10)

            # here's the GP
            gp = exo.gp.GP(kernel, xtime[~out], T.exp(logvar)*T.ones(len(xtime[~out])))

            # add custom potential (log-prob fxn) with the GP likelihood
            pm.Potential("obs", gp.log_likelihood(yomc[~out] - mean))

            # track GP prediction
            gp_pred = pm.Deterministic("gp_pred", gp.predict(hephem))


        # find the MAP solution
        with holczer_model:
            holczer_map = exo.optimize(start=holczer_model.test_point)
            
        htts = hephem + holczer_map["mean"] + holczer_map["gp_pred"]

        holczer_inds[npl] = np.copy(hinds)
        holczer_tts[npl] = np.copy(htts)
            
            
        # plot the results
        plt.figure(figsize=(12,4))
        plt.plot(xtime[~out], yomc[~out]*24*60, 'o', c="grey", label="Holczer")
        plt.plot(xtime[out], yomc[out]*24*60, "rx")
        plt.plot(hephem, (htts-hephem)*24*60, "k+", label="Interpolation")
        plt.xlabel("Time [BJKD]", fontsize=20)
        plt.ylabel("O-C [min]", fontsize=20)
        plt.legend(fontsize=12)
        plt.close()
            
            
    except:
        pass


# In[ ]:


# check if Holczer TTVs exist, and if so, replace the linear ephemeris
holczer_transit_times = []

for npl, p in enumerate(planets):
    match = np.isclose(holczer_pers, p.period, rtol=0.1, atol=DURS.max())
    
    if np.sum(match) > 1:
        raise ValueError("Something has gone wrong matching periods between DR25 and Holczer+ 2016")
        
    if np.sum(match) == 1:
        loc = np.squeeze(np.where(match))
    
        hinds = holczer_inds[loc]
        htts  = holczer_tts[loc]
        
        for i, t0 in enumerate(p.tts):
            for j, tH in enumerate(htts):
                if np.abs(t0-tH)/p.period < 0.25:
                    p.tts[i] = tH
                    
        holczer_transit_times.append(np.copy(p.tts))
        
        
    else:
        holczer_transit_times.append(None)


# In[ ]:


# plot the OMC TTVs
fig, axes = plt.subplots(NPL, figsize=(12,8))
if NPL == 1: axes = [axes]

for npl, p in enumerate(planets):
    xtime = poly.polyval(p.index, poly.polyfit(p.index, p.tts, 1))
    yomc  = (p.tts - xtime)*24*60
    
    axes[npl].plot(xtime, yomc, '.', c='C{0}'.format(npl))
    axes[npl].set_ylabel('O-C [min]', fontsize=20)
axes[NPL-1].set_xlabel('Time [BJKD]', fontsize=20)
plt.savefig(FIGURE_DIR + TARGET + '_ttvs_initial.pdf', bbox_inches='tight')
plt.close()


# In[ ]:


FULL_FIXED_EPHEMERIS = []
FULL_FIXED_INDS = []

for npl, p in enumerate(planets):
    FULL_FIXED_EPHEMERIS.append(poly.polyval(p.index, poly.polyfit(p.index, p.tts, 1)))
    FULL_FIXED_INDS.append(np.copy(p.index - p.index.min()))


# # Detrend the lightcurves

# In[ ]:


# clean up the LONG CADENCE data
for i, lcq in enumerate(lc_data):
    lcq = detrend.remove_flagged_cadences(lcq)
    
    mask = np.zeros(len(lcq.time), dtype="bool")
    for npl, p in enumerate(planets):
        mask += detrend.make_transitmask(lcq.time, p.tts, p.duration, masksize=1.5)
    
    lcq = detrend.clip_outliers(lcq, kernel_size=5, sigma_upper=5.0, sigma_lower=5.0, mask=mask)
    lcq = detrend.clip_outliers(lcq, kernel_size=5, sigma_upper=5.0, sigma_lower=100.0)
    
    lc_data[i] = lcq

# broadcast quarter and channel integers into arrays (for convenient use after stitching)
for i, lcq in enumerate(lc_data):
    lcq.quarter = lcq.quarter*np.ones(len(lcq.time))
    lcq.channel = lcq.channel*np.ones(len(lcq.time))

    
if len(lc_data) > 0:
    # combine quarters into a single LiteCurve
    lc = detrend.stitch_lkc(lc_data)
    
    # make a mask where planets transit
    lc_mask = np.zeros((NPL,len(lc.time)),dtype='bool')
    for npl, p in enumerate(planets):
        lc_mask[npl] = detrend.make_transitmask(lc.time, p.tts, p.duration, masksize=1.5)

    lc.mask = deepcopy(lc_mask)
    
    # plot the non-detrended data
    plt.figure(figsize=(20,4))
    plt.plot(lc.time, lc.flux, c="k", lw=0.5)
    plt.xlim(lc.time.min(), lc.time.max())
    plt.xlabel('Time [BKJD]', fontsize=20)
    plt.ylabel('Relative Flux', fontsize=20)
    plt.close()
    
else:
    lc = None


# In[ ]:


if lc is not None:
    # set break tolerance and nominal minimum oscillation period
    break_tolerance = np.max([int(DURS.min()/(LCIT/60/24)*5/2), 13])
    min_period = 5*DURS.max()
    
    lc = detrend.flatten_with_gp(lc, break_tolerance=break_tolerance, min_period=min_period)

    # determine seasons
    lc.season = lc.quarter % 4

    # plot detrended data
    plt.figure(figsize=(16,4))
    plt.plot(lc.time, lc.flux, c='k', lw=0.5)
    plt.xlim(lc.time.min(), lc.time.max())
    plt.xlabel('Time [BKJD]', fontsize=20)
    plt.ylabel('Relative Flux', fontsize=20)
    plt.savefig(FIGURE_DIR + TARGET + '_long_cadence_flux.pdf', bbox_inches='tight')
    plt.close()


# In[ ]:


# clean up the SHORT CADENCE data
for i, scq in enumerate(sc_data):
    scq = detrend.remove_flagged_cadences(scq)
    
    mask = np.zeros(len(scq.time), dtype="bool")
    for npl, p in enumerate(planets):
        mask += detrend.make_transitmask(scq.time, p.tts, p.duration, masksize=1.5)
    
    scq = detrend.clip_outliers(scq, kernel_size=5, sigma_upper=5.0, sigma_lower=5.0, mask=mask)
    scq = detrend.clip_outliers(scq, kernel_size=5, sigma_upper=5.0, sigma_lower=100.0)
    
    sc_data[i] = scq


# broadcast quarter and channel integers into arrays (for convenient use after stitching)
for i, scq in enumerate(sc_data):
    scq.quarter = scq.quarter*np.ones(len(scq.time))
    scq.channel = scq.channel*np.ones(len(scq.time))

    
if len(sc_data) > 0:
    # combine quarters into a single LiteCurve
    sc = detrend.stitch_lkc(sc_data)
    
    # make a mask where planets transit
    sc_mask = np.zeros((NPL,len(sc.time)),dtype='bool')
    for npl, p in enumerate(planets):
        sc_mask[npl] = detrend.make_transitmask(sc.time, p.tts, p.duration, masksize=1.5)

    sc.mask = deepcopy(sc_mask)
    
    # plot the non-detrended data
    plt.figure(figsize=(20,4))
    plt.plot(sc.time, sc.flux, c="k", lw=0.5)
    plt.xlim(sc.time.min(), sc.time.max())
    plt.xlabel('Time [BKJD]', fontsize=20)
    plt.ylabel('Relative Flux', fontsize=20)
    plt.close()
    
else:
    sc = None


# In[ ]:


if sc is not None:    
    # detrend
    break_tolerance = np.max([int(DURS.min()/(SCIT/3600/24)*5/2), 91])
    min_period = 5*DURS.max()
    
    sc = detrend.flatten_with_gp(sc, break_tolerance=break_tolerance, min_period=min_period)

    # determine seasons
    sc.season = sc.quarter % 4

    # plot detrended data
    plt.figure(figsize=(16,4))
    plt.plot(sc.time, sc.flux, c='k', lw=0.5)
    plt.xlim(sc.time.min(), sc.time.max())
    plt.xlabel('Time [BKJD]', fontsize=20)
    plt.ylabel('Relative Flux', fontsize=20)
    plt.savefig(FIGURE_DIR + TARGET + '_short_cadence_flux.pdf', bbox_inches='tight')
    plt.close()


# # Make individual mask for where each planet transits
# ### These masks have width 2.5 transit durations, which may be wider than the masks used for detrending

# In[ ]:


print('Making transit masks')
try:
    sc_mask = np.zeros((NPL,len(sc.time)),dtype='bool')
    for npl, p in enumerate(planets):
        sc_mask[npl] = detrend.make_transitmask(sc.time, p.tts, p.duration, masksize=2.5)

    sc.mask = sc_mask

except:
    sc_mask = None

    
try:
    lc_mask = np.zeros((NPL,len(lc.time)),dtype='bool')
    for npl, p in enumerate(planets):
        lc_mask[npl] = detrend.make_transitmask(lc.time, p.tts, p.duration, masksize=2.5)
    
    lc.mask = lc_mask

except:
    lc_mask = None


# # Flag high quality transits (quality = 1)
# 
# ### Good transits must have  at least 50% photometry coverage in/near transit

# In[ ]:


for npl, p in enumerate(planets):
    count_expect_lc = int(np.ceil(p.duration/lcit))
    count_expect_sc = int(np.ceil(p.duration/scit))
        
    quality = []
    
    for i, t0 in enumerate(p.tts):
        
        try:
            in_sc = np.abs(sc.time - t0)/p.duration < 0.5
            in_lc = np.abs(lc.time - t0)/p.duration < 0.5
            
            near_sc = np.abs(sc.time - t0)/p.duration < 1.5
            near_lc = np.abs(lc.time - t0)/p.duration < 1.5
            
            qual_in = (np.sum(in_sc) > 0.5*count_expect_sc) + (np.sum(in_lc) > 0.5*count_expect_lc)
            qual_near = (np.sum(near_sc) > 1.5*count_expect_sc) + (np.sum(near_lc) > 1.5*count_expect_lc)
            
            quality.append(qual_in*qual_near)
                                    
                        
        except:
            in_lc = np.abs(lc.time - t0)/p.duration < 0.5
            near_lc = np.abs(lc.time - t0)/p.duration < 1.5
            
            qual_in = (np.sum(in_lc) > 0.5*count_expect_lc)
            qual_near = (np.sum(near_lc) > 1.5*count_expect_lc)
            
            quality.append(qual_in*qual_near)
            
    p.quality = np.copy(quality)


# # Flag which transits overlap (overlap = 1)

# In[ ]:


# identify overlapping transits
dur_max = np.max(DURS)
overlap = []

for i in range(NPL):
    overlap.append(np.zeros(len(planets[i].tts), dtype='bool'))
    
    for j in range(NPL):
        if i != j:
            for ttj in planets[j].tts:
                overlap[i] += np.abs(planets[i].tts - ttj)/dur_max < 1.5
                
    planets[i].overlap = np.copy(overlap[i])


# # Count up transits and calculate initial fixed transit times

# In[ ]:


num_transits  = np.zeros(NPL)
transit_inds  = []
fixed_tts = []

for npl, p in enumerate(planets):
    transit_inds.append(np.array((p.index - p.index.min())[p.quality], dtype="int"))
    fixed_tts.append(np.copy(p.tts)[p.quality])
    
    num_transits[npl] = len(transit_inds[npl])
    transit_inds[npl] -= transit_inds[npl].min()


# # Grab the relevant data and starting transit parameters

# In[ ]:


# grab data near transits for each quarter
all_time = []
all_flux = []
all_error = []
all_dtype = []

lc_flux = []
sc_flux = []

if sc is not None:
    for q in range(18):
        if np.isin(q, sc.quarter)*np.isin(q, lc.quarter):
            raise ValueError("Double counting data in both short and long cadence")


        elif np.isin(q, sc.quarter):
            use = (sc.mask.sum(axis=0) > 0)*(sc.quarter == q)

            if np.sum(use) > 45:
                all_time.append(sc.time[use])
                all_flux.append(sc.flux[use])
                all_error.append(sc.error[use])
                all_dtype.append('short')

                sc_flux.append(sc.flux[use])
                
            else:
                all_time.append(None)
                all_flux.append(None)
                all_error.append(None)
                all_dtype.append('short_no_transits')


        elif np.isin(q, lc.quarter):
            use = (lc.mask.sum(axis=0) > 0)*(lc.quarter == q)
            
            if np.sum(use) > 5:
                all_time.append(lc.time[use])
                all_flux.append(lc.flux[use])
                all_error.append(lc.error[use])
                all_dtype.append('long')

                lc_flux.append(lc.flux[use])
                
            else:
                all_time.append(None)
                all_flux.append(None)
                all_error.append(None)
                all_dtype.append('long_no_transits')


        else:
            all_time.append(None)
            all_flux.append(None)
            all_error.append(None)
            all_dtype.append('none')
            
else:
    for q in range(18):
        if np.isin(q, lc.quarter):
            use = (lc.mask.sum(axis=0) > 0)*(lc.quarter == q)

            if np.sum(use) > 3:
                all_time.append(lc.time[use])
                all_flux.append(lc.flux[use])
                all_error.append(lc.error[use])
                all_dtype.append('long')

                lc_flux.append(lc.flux[use])
                
            else:
                all_time.append(None)
                all_flux.append(None)
                all_error.append(None)
                all_dtype.append('long_no_transits')


        else:
            all_time.append(None)
            all_flux.append(None)
            all_error.append(None)
            all_dtype.append('none')



# check which quarters have data and transits
good = (np.array(all_dtype) == 'short') + (np.array(all_dtype) == 'long')
quarters = np.arange(18)[good]
nq = len(quarters)


# make some linear flux arrays (for convenience use laster)
try: sc_flux_lin = np.hstack(sc_flux)
except: sc_flux_lin = np.array([])
    
try: lc_flux_lin = np.hstack(lc_flux)
except: lc_flux_lin = np.array([])
    
try:
    good_flux = np.hstack([sc_flux_lin, lc_flux_lin])
except:
    try:
        good_flux = np.hstack(sc_flux)
    except:
        good_flux = np.hstack(lc_flux)


# In[ ]:


# pull basic transit parameters
epochs    = np.zeros(NPL)
periods   = np.zeros(NPL)
depths    = np.zeros(NPL)
durations = np.zeros(NPL)
impacts   = np.zeros(NPL)

for npl, p in enumerate(planets):
    epochs[npl]    = p.epoch
    periods[npl]   = p.period
    depths[npl]    = p.depth
    durations[npl] = p.duration
    impacts[npl]   = p.impact

radii = np.sqrt(depths)*RSTAR


# In[ ]:


print('')
print('cumulative runtime = ', int(timer() - global_start_time), 's')
print('')


# # (1) Fit transit SHAPE model

# The TTV model when determining transit shape is built as perturbations from fixed (i.e. invariant) transit times. These transit times ("FIXED_TTS") are not varied during model fitting. If TTVs are available from Holczer+ 2016, these are used; if not, the fixed transit times determined from a linear ephemeris using Kepler pipeline epoch and period.
# 
# Long-term, secular TTVs are parameterized as Legendre polynomials as functions of dimensionless variable ("x") in the range (-1,1)
# 
# Unlike the transit time and index values attached to each Planet object, there may be gaps in the FIXED_TTS vector where there is no available photometric data.

# In[ ]:


print('\n(1) Fitting transit SHAPE model')


# In[ ]:


# use Legendre polynomials for better orthogonality; "x" is in the range (-1,1)
Leg0 = []
Leg1 = []
Leg2 = []
Leg3 = []
t = []

# this assumes a baseline in the range (TIME_START,TIME_END)
for npl, p in enumerate(planets):    
    t.append(p.epoch + transit_inds[npl]*p.period)
    x = 2*(t[npl]-TIME_START)/(TIME_END-TIME_START) - 1

    Leg0.append(np.ones_like(x))
    Leg1.append(x.copy())
    Leg2.append(0.5*(3*x**2 - 1))
    Leg3.append(0.5*(5*x**3 - 3*x))


# In[ ]:


with pm.Model() as shape_model:
    # stellar parameters (limb darkening using Kipping 2013)
    u = exo.distributions.QuadLimbDark('u', testval=np.array([U1,U2]))

    Rstar = pm.Bound(pm.Normal, lower=RSTAR-3*RSTAR_ERR, upper=RSTAR+3*RSTAR_ERR)('Rstar', mu=RSTAR, sd=RSTAR_ERR)
    Mstar = pm.Bound(pm.Normal, lower=MSTAR-3*MSTAR_ERR, upper=MSTAR+3*MSTAR_ERR)('Mstar', mu=MSTAR, sd=MSTAR_ERR)


    # planetary parameters (impact parameter using Espinoza 2018)
    logr = pm.Uniform('logr', lower=np.log(0.0003), upper=np.log(0.3), testval=np.log(radii), shape=NPL)
    rp   = pm.Deterministic('rp', T.exp(logr))
    
    b  = exo.distributions.ImpactParameter('b', ror=rp/Rstar, testval=impacts, shape=NPL)
    
    
    # polynomial TTV parameters    
    C0 = pm.Normal('C0', mu=0.0, sd=durations/2, shape=NPL)
    C1 = pm.Normal('C1', mu=0.0, sd=durations/2, shape=NPL)
    
    transit_times = []
    for npl in range(NPL):
        transit_times.append(pm.Deterministic('tts_{0}'.format(npl),                                               fixed_tts[npl] + C0[npl]*Leg0[npl] + C1[npl]*Leg1[npl]))
    
    
    # set up stellar model and planetary orbit
    exoSLC = exo.StarryLightCurve(u)
    orbit  = exo.orbits.TTVOrbit(transit_times=transit_times, transit_inds=transit_inds,                                  b=b, r_star=Rstar, m_star=Mstar)
    
    # track period and epoch
    T0 = pm.Deterministic('T0', orbit.t0)
    P  = pm.Deterministic('P', orbit.period)
    
    
    # nuissance parameters
    flux0 = pm.Normal('flux0', mu=1.0, sd=np.std(good_flux))

    if len(sc_flux_lin) > 1:
        logvar = pm.Normal('logvar', mu=np.log(np.var(sc_flux_lin)), sd=np.log(4))
    else:
        logvar = pm.Normal('logvar', mu=np.log(np.var(lc_flux_lin)*30), sd=np.log(4))
    
    
    # now evaluate the model for each quarter
    light_curves       = [None]*nq
    summed_light_curve = [None]*nq
    model_flux         = [None]*nq
    
    obs = [None]*nq
    
    
    for j, q in enumerate(quarters):
        # set oversampling factor
        if all_dtype[q] == 'short':
            oversample = 1
        elif all_dtype[q] == 'long':
            oversample = 15
            
        # calculate light curves
        light_curves[j] = exoSLC.get_light_curve(orbit=orbit, r=rp, t=all_time[q], oversample=oversample)
        summed_light_curve[j] = pm.math.sum(light_curves[j], axis=-1) + flux0*T.ones(len(all_time[q]))
        model_flux[j] = pm.Deterministic('model_flux_{0}'.format(j), summed_light_curve[j])
        

        if all_dtype[q] == 'short':
            obs[j] = pm.Normal("obs_{0}".format(j), 
                               mu=model_flux[j], 
                               sd=T.sqrt(T.exp(logvar)),
                               observed=all_flux[q])
        
        elif all_dtype[q] == 'long':
            obs[j] = pm.Normal("obs_{0}".format(j), 
                               mu=model_flux[j], 
                               sd=T.sqrt(T.exp(logvar)/30),
                               observed=all_flux[q])
            
        else:
            raise ValueError("Cadence data type must be 'short' or 'long'")


# In[ ]:


with shape_model:
    shape_map = exo.optimize(start=shape_model.test_point, vars=[flux0, logvar])
    shape_map = exo.optimize(start=shape_map, vars=[b])
    shape_map = exo.optimize(start=shape_map, vars=[u, Mstar])
    shape_map = exo.optimize(start=shape_map, vars=[C0, C1])
    shape_map = exo.optimize(start=shape_map)


# In[ ]:


# grab transit times and ephemeris
shape_transit_times = []
shape_ephemeris = []

for npl, p in enumerate(planets):
    shape_transit_times.append(shape_map['tts_{0}'.format(npl)])
    shape_ephemeris.append(shape_map['P'][npl]*transit_inds[npl] + shape_map['T0'][npl])

    
# plot the OMC TTVs
fig, axes = plt.subplots(NPL, figsize=(12,8))
if NPL == 1: axes = [axes]

for npl, p in enumerate(planets):
    xtime = shape_transit_times[npl]
    yomc  = (shape_transit_times[npl] - shape_ephemeris[npl])*24*60
    
    axes[npl].plot(xtime, yomc, '.', c='C{0}'.format(npl))
    axes[npl].set_ylabel('O-C [min]', fontsize=20)
axes[NPL-1].set_xlabel('Time [BJKD]', fontsize=20)
plt.close()


# In[ ]:


# update stellar parameter values
Rstar = shape_map['Rstar']
Mstar = shape_map['Mstar']
u = shape_map['u']


# update planet parameter values
periods = shape_map['P']
epochs  = shape_map['T0']

rp = shape_map['rp']
b  = shape_map['b']

sma  = get_sma(periods, Mstar)
durs = get_dur_tot(periods, rp, Rstar, b, sma)


for npl, p in enumerate(planets):
    p.epoch    = epochs[npl]
    p.period   = periods[npl]
    p.depth    = (rp[npl]/Rstar)**2
    p.duration = durs[npl]
    p.impact   = b[npl]


# In[ ]:


print('')
print('cumulative runtime = ', int(timer() - global_start_time), 's')
print('')


# # (2) Fit slide TTVs

# In[ ]:


print('\n(2) Fitting SLIDE TTVs')


# In[ ]:


slide_transit_times = []
slide_error = []

t_all = np.array(np.hstack(all_time), dtype="float")
f_all = np.array(np.hstack(all_flux), dtype="float")

for npl, p in enumerate(planets):
    print("\nPLANET", npl)
    
    slide_transit_times.append([])
    slide_error.append([])
    
    # create template transit
    exoSLC = exo.StarryLightCurve(u)
    orbit  = exo.orbits.KeplerianOrbit(t0=0, period=p.period, b=b[npl], r_star=Rstar, m_star=Mstar)

    gridstep     = scit/2
    slide_offset = 1.0
    delta_chisq  = 2.0

    template_time = np.arange(-(0.02+p.duration)*(slide_offset+1.6), (0.02+p.duration)*(slide_offset+1.6), gridstep)
    template_flux = 1.0 + exoSLC.get_light_curve(orbit=orbit, r=rp[npl], t=template_time).sum(axis=-1).eval()
    
        
    
    # empty lists to hold new transit time and uncertainties
    tts = -99*np.ones_like(shape_transit_times[npl])
    err = -99*np.ones_like(shape_transit_times[npl])
    
    for i, t0 in enumerate(shape_transit_times[npl]):
        #print(i, np.round(t0,2))
        if ~p.overlap[p.quality][i]:
        
            # grab flux near each non-overlapping transit
            use = np.abs(t_all - t0)/p.duration < 2.5
            mask = np.abs(t_all - t0)/p.duration < 1.0

            t_ = t_all[use]
            f_ = f_all[use]
            m_ = mask[use]
            
            try:
                trend = poly.polyval(t_, poly.polyfit(t_[~m_], f_[~m_], 1))
            
                f_ /= trend
                e_ = np.ones_like(f_)*np.std(f_[~m_])
                
            except:
                e_ = np.ones_like(f_)*np.std(f_)
            

            tc_vector = t0 + np.arange(-p.duration*slide_offset, p.duration*slide_offset, gridstep)
            chisq_vector = np.zeros_like(tc_vector)

            # slide along transit time vector and calculate chisq
            for j, tc in enumerate(tc_vector):
                y_ = np.interp(t_-tc, template_time, template_flux)
                chisq_vector[j] = np.sum((f_ - y_)**2/e_**2)

            chisq_vector = boxcar_smooth(chisq_vector, winsize=7)


            # grab points near minimum chisq
            delta_chisq = 1
            
            loop = True
            while loop:
                # incrememnt delta_chisq and find minimum
                delta_chisq += 1
                min_chisq = chisq_vector.min()
                
                # grab the points near minimum
                tcfit = tc_vector[chisq_vector < min_chisq+delta_chisq]
                x2fit = chisq_vector[chisq_vector < min_chisq+delta_chisq]

                # eliminate points far from the local minimum
                spacing = np.median(tcfit[1:]-tcfit[:-1])
                faraway = np.abs(tcfit-np.median(tcfit))/spacing > 1 + len(tcfit)/2
                
                tcfit = tcfit[~faraway]
                x2fit = x2fit[~faraway]
                
                # check for stopping conditions
                if len(x2fit) >= 3:
                    loop = False
                    
                if delta_chisq >= 9:
                    loop = False
                    
                    
            # fit a parabola around the minimum (need at least 3 pts)
            if len(tcfit) < 3:
                #print("TOO FEW POINTS")
                tts[i] = np.nan
                err[i] = np.nan

            else:
                quad_coeffs = np.polyfit(tcfit, x2fit, 2)
                quadfit = np.polyval(quad_coeffs, tcfit)
                qtc_min = -quad_coeffs[1]/(2*quad_coeffs[0])
                qx2_min = np.polyval(quad_coeffs, qtc_min)
                qtc_err = np.sqrt(1/quad_coeffs[0])

                # here's the fitted transit time
                tts[i] = np.mean([qtc_min,np.median(tcfit)])
                err[i] = qtc_err*1.0

                # check that the fit is well-conditioned (ie. a negative t**2 coefficient)
                if quad_coeffs[0] <= 0.0:
                    #print("INVERTED PARABOLA")
                    tts[i] = np.nan
                    err[i] = np.nan

                # check that the recovered transit time is within the expected range
                if (tts[i] < tcfit.min()) or (tts[i] > tcfit.max()):
                    #print("T0 OUT OF BOUNDS")
                    tts[i] = np.nan
                    err[i] = np.nan


            # show plots
            if ~np.isnan(tts[i]):
                do_plots = False
                    
                if do_plots:
                    fig, ax = plt.subplots(1,2, figsize=(10,3))

                    ax[0].plot(t_-tts[i], f_, "ko")
                    ax[0].plot((t_-tts[i])[m_], f_[m_], "o", c="C{0}".format(npl))
                    ax[0].plot(template_time, template_flux, c="C{0}".format(npl), lw=2)

                    ax[1].plot(tcfit, x2fit, "ko")
                    ax[1].plot(tcfit, quadfit, c="C{0}".format(npl), lw=3)
                    ax[1].axvline(tts[i], color="k", ls="--", lw=2)

                    plt.close()

        else:
            #print("OVERLAPPING TRANSITS")
            tts[i] = np.nan
            err[i] = np.nan
        
    slide_transit_times[npl] = np.copy(tts)
    slide_error[npl] = np.copy(err)


# In[ ]:


for npl, p in enumerate(planets):
    bad = np.isnan(slide_transit_times[npl]) + np.isnan(slide_error[npl])
    bad += slide_error[npl] > 8*np.nanmedian(slide_error[npl])
    
    slide_transit_times[npl][bad] = shape_transit_times[npl][bad]
    slide_error[npl][bad] = np.nan


# In[ ]:


# grab transit times and ephemeris
# plot the OMC TTVs
fig, axes = plt.subplots(NPL, figsize=(12,8))
if NPL == 1: axes = [axes]

for npl, p in enumerate(planets):
    ephem = poly.polyval(transit_inds[npl], poly.polyfit(transit_inds[npl], slide_transit_times[npl], 1))
    
    xtime = slide_transit_times[npl]
    yomc  = (slide_transit_times[npl] - ephem)*24*60
    yerr  = slide_error[npl]*24*60
    
    axes[npl].errorbar(xtime, yomc, yerr=yerr, fmt='.', color='C{0}'.format(npl))
    axes[npl].set_ylabel('O-C [min]', fontsize=20)
axes[NPL-1].set_xlabel('Time [BJKD]', fontsize=20)
plt.close()


# # (3) Fit MAP independent TTVs

# In[ ]:


print('\n(3) Fitting INDEPENDENT TTVs')


# In[ ]:


with pm.Model() as indep_model:
    # transit times
    tt_offset = []
    transit_times = []
    
    for npl in range(NPL):
        tt_offset.append(pm.Normal('tt_offset_{0}'.format(npl), mu=0, sd=1, shape=len(slide_transit_times[npl])))
        
        transit_times.append(pm.Deterministic('tts_{0}'.format(npl),
                                              slide_transit_times[npl] + tt_offset[npl]*durations[npl]/3))
    
    # set up stellar model and planetary orbit
    exoSLC = exo.StarryLightCurve(u)
    orbit  = exo.orbits.TTVOrbit(transit_times=transit_times, transit_inds=transit_inds,                                  b=b, r_star=Rstar, m_star=Mstar)
    
    # track period and epoch
    T0 = pm.Deterministic('T0', orbit.t0)
    P  = pm.Deterministic('P', orbit.period)
    
        
    # nuissance parameters
    flux0 = pm.Normal('flux0', mu=1.0, sd=np.std(good_flux))

    if len(sc_flux_lin) > 1:
        logvar = pm.Normal('logvar', mu=np.log(np.var(sc_flux_lin)), sd=np.log(4))
    else:
        logvar = pm.Normal('logvar', mu=np.log(np.var(lc_flux_lin)*30), sd=np.log(4))
    
    
    # now evaluate the model for each quarter
    light_curves       = [None]*nq
    summed_light_curve = [None]*nq
    model_flux         = [None]*nq
    
    obs = [None]*nq
    
    for j, q in enumerate(quarters):
        # set oversampling factor
        if all_dtype[q] == 'short':
            oversample = 1
        elif all_dtype[q] == 'long':
            oversample = 15
            
        # calculate light curves
        light_curves[j] = exoSLC.get_light_curve(orbit=orbit, r=rp, t=all_time[q], oversample=oversample)
        summed_light_curve[j] = pm.math.sum(light_curves[j], axis=-1) + flux0*T.ones(len(all_time[q]))
        model_flux[j] = pm.Deterministic('model_flux_{0}'.format(j), summed_light_curve[j])
        
        
        if all_dtype[q] == 'short':
            obs[j] = pm.Normal("obs_{0}".format(j), 
                               mu=model_flux[j], 
                               sd=T.sqrt(T.exp(logvar)),
                               observed=all_flux[q])
        
        elif all_dtype[q] == 'long':
            obs[j] = pm.Normal("obs_{0}".format(j), 
                               mu=model_flux[j], 
                               sd=T.sqrt(T.exp(logvar)/30),
                               observed=all_flux[q])
            
        else:
            raise ValueError("Cadence data type must be 'short' or 'long'")


# In[ ]:


with indep_model:
    indep_map = exo.optimize(start=indep_model.test_point, vars=[flux0, logvar])
    
    for npl in range(NPL):
        indep_map = exo.optimize(start=indep_map, vars=[tt_offset[npl]])
        
    indep_map = exo.optimize(start=indep_map)


# In[ ]:


indep_transit_times = []
indep_ephemeris = []
full_indep_ephemeris = []

indep_error = np.copy(slide_error)

for npl, p in enumerate(planets):
    # pull the MAP solution
    indep_transit_times.append(indep_map['tts_{0}'.format(npl)])
    indep_ephemeris.append(indep_map['P'][npl]*transit_inds[npl] + indep_map['T0'][npl])
    full_indep_ephemeris.append(indep_map['P'][npl]*p.index + indep_map['T0'][npl])

    # replace the MAP TTVs with SLIDE TTVs where applicable
    replace = ~np.isnan(slide_error[npl])
    indep_transit_times[npl][replace] = np.copy(slide_transit_times[npl][replace])
    
    indep_error[npl][~replace] = np.std(indep_transit_times[npl] - indep_ephemeris[npl])
    
    
fig, axes = plt.subplots(NPL, figsize=(12,8))
if NPL == 1: axes = [axes]

for npl, p in enumerate(planets):
    xtime = indep_transit_times[npl]
    yomc  = (indep_transit_times[npl] - indep_ephemeris[npl])*24*60
    yerr  = (indep_error[npl])*24*60
    
    axes[npl].errorbar(xtime, yomc, yerr=yerr, fmt='.', c='C{0}'.format(npl))
    axes[npl].set_ylabel('O-C [min]', fontsize=20)
axes[NPL-1].set_xlabel('Time [BJKD]', fontsize=20)
plt.savefig(FIGURE_DIR + TARGET + '_ttvs_indep.pdf', bbox_inches='tight')
plt.close()


# In[ ]:


print('')
print('cumulative runtime = ', int(timer() - global_start_time), 's')
print('')


# # Search for periodic signals in the OMC curves

# In[ ]:


print("Searching for periodic signals")


# In[ ]:


indep_freqs = []
indep_faps = []

for npl, p in enumerate(planets):
    # grab data
    xtime = indep_ephemeris[npl]
    yomc  = indep_transit_times[npl] - indep_ephemeris[npl]

    ymed = boxcar_smooth(ndimage.median_filter(yomc, size=5, mode="mirror"), winsize=5)
    out  = np.abs(yomc-ymed)/astropy.stats.mad_std(yomc-ymed) > 5.0
    
    
    # search for a periodic component
    peakfreq = np.nan
    peakfap = 1.0
    
    if NPL == 1: fap = 0.1
    elif NPL > 1: fap = 0.99
    
    if np.sum(~out) > 8:
        try:
            xf, yf, freqs, faps = LS_estimator(xtime[~out], yomc[~out], fap=fap)

            if len(freqs) > 0:
                if freqs[0] > xf.min():
                    peakfreq = freqs[0]
                    peakfap = faps[0]
        except:
            pass
    
    indep_freqs.append(peakfreq)
    indep_faps.append(peakfap)


# In[ ]:


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
            
            df_min = 1/(indep_ephemeris[i].max() - indep_ephemeris[i].min())
            
            for j in range(i+1, NPL):
                # delta-freq (LS) between two planets
                df_ij = np.abs(indep_freqs[i]-indep_freqs[j])
                
                if df_ij < df_min:
                    close = True
                    
            if close:
                omc_freqs.append(indep_freqs[i])
                omc_faps.append(indep_faps[i])
                
            else:
                omc_freqs.append(None)
                omc_faps.append(None)


# In[ ]:


omc_periodic_trend = []
full_omc_periodic_trend = []

for npl in range(NPL):
    print("\nPLANET", npl)
    
    # grab data
    xtime = indep_ephemeris[npl]
    yomc  = indep_transit_times[npl] - indep_ephemeris[npl]
    
    
    # roughly model OMC based on single frequency sinusoid (if found)
    if omc_freqs[npl] is not None:
        print("periodic component found at P =", int(1/omc_freqs[npl]), "d")
        
        LS = LombScargle(xtime, yomc)
        
        omc_periodic_trend.append(LS.model(xtime, omc_freqs[npl]))
        full_omc_periodic_trend.append(LS.model(full_indep_ephemeris[npl], omc_freqs[npl]))
        
    else:
        omc_periodic_trend.append(None)
        full_omc_periodic_trend.append(None)
        
        
    plt.figure(figsize=(12,4))
    plt.plot(xtime, yomc*24*60, "o", c="grey")
    if omc_periodic_trend[npl] is not None:
        plt.plot(xtime, omc_periodic_trend[npl]*24*60, c="C{0}".format(npl))
    plt.close()


# # Determine best OMC model

# In[ ]:


print("Determining best OMC model")


# In[ ]:


def build_omc_model(xtime, yomc, out, polyorder=1, fixed_trend=None, use_gp=False, xpredict=None, ftpredict=None):
    
    if fixed_trend is None:
        fixed_trend = np.zeros_like(xtime)
    
    with pm.Model() as model:
        
        # build mean trend
        C0 = pm.Normal("C0", mu=0, sd=10)
        C1 = 0.0
        C2 = 0.0
        C3 = 0.0

        if polyorder >= 1: C1 = pm.Normal("C1", mu=0, sd=10)
        if polyorder >= 2: C2 = pm.Normal("C2", mu=0, sd=10)
        if polyorder >= 3: C3 = pm.Normal("C3", mu=0, sd=10)
        if polyorder >= 4: raise ValueError("only configured for 3rd order polynomials")
            
        mean = pm.Deterministic("mean", fixed_trend[~out] + C0 + C1*xtime[~out] 
                                + C2*xtime[~out]**2 + C3*xtime[~out]**3)
        
        # nusiance parameters
        logjit = pm.Normal("logjit", mu=np.log(np.var(yomc)), sd=10)
        
        
        if use_gp:
            # build the kernel and gp
            log_sigma = pm.Normal("log_sigma", mu=np.log(np.std(yomc)), sd=10)
            log_rho = pm.Normal("log_rho", mu=np.log((xtime[-1]-xtime[0])/len(xtime)), sd=10)

            kernel = exo.gp.terms.Matern32Term(log_sigma=log_sigma, log_rho=log_rho)

            gp = exo.gp.GP(kernel, xtime[~out], T.exp(logjit)*T.ones(len(xtime[~out])))

            # here's the likelihood
            pm.Potential("obs", gp.log_likelihood(yomc[~out] - mean))

            # track GP prediction and trend
            trend = pm.Deterministic("trend", fixed_trend + gp.predict(xtime) 
                                     + C0 + C1*xtime + C2*xtime**2 + C3*xtime**3)
            
            if xpredict is not None:
                if ftpredict is None:
                    ftpredict = np.zeros_like(xpredict)
                
                ypredict = pm.Deterministic("ypredict", ftpredict + gp.predict(xpredict)
                                            + C0 + C1*xpredict + C2*xpredict**2 + C3*xpredict**3)
            
        
        else:
            # here's the likelihood
            pm.Normal("obs", mu=mean, sd=T.sqrt(T.exp(logjit)*T.ones(len(xtime[~out]))), observed=yomc[~out])
            
            #track the trend
            trend = pm.Deterministic("trend", fixed_trend + C0 + C1*xtime + C2*xtime**2 + C3*xtime**3)
            
            if xpredict is not None:
                ypredict = pm.Deterministic("ypredict", fixed_trend + C0 + C1*xpredict 
                                            + C2*xpredict**2 + C3*xpredict**3)
        
    
    return model


# In[ ]:


def build_mix_model(res):
    
    resnorm = res / np.std(res)
    resnorm -= np.mean(resnorm)
    
    with pm.Model() as model:
        
        # mixture parameters
        w = pm.Dirichlet("w", np.array([1.,1.]))
        mu = pm.Normal("mu", mu=0.0, sd=1.0, shape=1)
        tau = pm.Gamma("tau", 1.0, 1.0, shape=2)
        
        # here's the potential
        obs = pm.NormalMixture("obs", w, mu=mu*T.ones(2), tau=tau, observed=resnorm)
        
    return model


# In[ ]:


def flag_res_outliers(res, loc, scales):
    resnorm = residuals/np.std(residuals)
    resnorm -= np.mean(resnorm)
    
    order = np.argsort(scales)
    scales = scales[order]
    
    z_fg = stats.norm(loc=loc, scale=scales[0]).pdf(resnorm)
    z_bg = stats.norm(loc=loc, scale=scales[1]).pdf(resnorm)

    fg_prob = z_fg/(z_fg+z_bg)
    fg_prob = (fg_prob - fg_prob.min())/(fg_prob.max()-fg_prob.min())


    # use KMeans clustering to assign each point to the foreground or background
    km = KMeans(n_clusters=2)
    group = km.fit_predict(fg_prob.reshape(-1,1))
    centroids = np.array([np.mean(fg_prob[group==0]), np.mean(fg_prob[group==1])])

    bad = group == np.argmin(centroids)
        
    return fg_prob, bad


# In[ ]:


matern_ephemeris = []
matern_transit_times = []


for npl, p in enumerate(planets):
    print("\nPLANET", npl)
    
    # grab data
    xtime = indep_ephemeris[npl]
    yomc  = indep_transit_times[npl] - indep_ephemeris[npl]

    ymed = boxcar_smooth(ndimage.median_filter(yomc, size=5, mode="mirror"), winsize=5)
    out  = np.abs(yomc-ymed)/astropy.stats.mad_std(yomc-ymed) > 5.0
    
    
    # compare various models
    aiclist = []
    biclist = []
    outlist = []
    fgplist = []
    
    if np.sum(~out) >= 16: max_polyorder = 4
    elif np.sum(~out) >= 8: max_polyorder = 3
    else: max_polyorder = 2
    
    for polyorder in range(1, max_polyorder):
        
        # build the OMC model
        omc_model = build_omc_model(xtime, yomc, out, fixed_trend=omc_periodic_trend[npl], 
                                    polyorder=polyorder, use_gp=False)

        with omc_model:
            omc_map = exo.optimize(start=omc_model.test_point)

        with omc_model:
            omc_trace = pm.sample(tune=3000, draws=1000, start=omc_map, chains=2, 
                                  step=exo.get_dense_nuts_step(target_accept=0.9))

        
        # flag outliers via mixture model of the residuals
        omc_trend = np.nanmedian(omc_trace["trend"], 0)
        residuals = yomc - omc_trend
        
        mix_model = build_mix_model(residuals)
        
        with mix_model:
            mix_trace = pm.sample(tune=3000, draws=1000, start=mix_model.test_point, chains=1, 
                                  step=exo.get_dense_nuts_step(target_accept=0.9))

        loc = np.nanmedian(mix_trace["mu"], axis=0)
        scales = np.nanmedian(1/np.sqrt(mix_trace["tau"]), axis=0)

        fg_prob, bad = flag_res_outliers(residuals, loc, scales)
        
        outlist.append(bad)
        fgplist.append(fg_prob)
        
        print("polyorder:", polyorder)
        print("loc:", np.round(loc,3))
        print("scale:", np.round(scales,3))
        print("{0} outliers found out of {1} transit times ({2}%)".format(np.sum(bad), len(bad), 
                                                                          np.round(100.*np.sum(bad)/len(bad),1)))
        
        # calculate AIC & BIC
        n = len(yomc)
        k = polyorder + 3
        
        AIC = n*np.log(np.sum(residuals[~bad]**2)/np.sum(~bad)) + 2*k
        BIC = n*np.log(np.sum(residuals[~bad]**2)/np.sum(~bad)) + k*np.log(n)
        
        aiclist.append(AIC)
        biclist.append(BIC)
        
        print("AIC:", np.round(AIC,1))
        print("BIC:", np.round(BIC,1))
      
    
    # choose the best model and recompute
    bad = outlist[np.argmin(aiclist)]
    polyorder = 1 + np.argmin(aiclist)
    
    omc_model = build_omc_model(xtime, yomc, bad, polyorder=polyorder, use_gp=True,
                                fixed_trend = omc_periodic_trend[npl], 
                                xpredict = full_indep_ephemeris[npl],
                                ftpredict = full_omc_periodic_trend[npl])

    with omc_model:
        omc_map = exo.optimize(start=omc_model.test_point)

    with omc_model:
        omc_trace = pm.sample(tune=9000, draws=1000, start=omc_map, chains=2, 
                              step=exo.get_dense_nuts_step(target_accept=0.9))

    omc_trend = np.nanmedian(omc_trace["trend"], 0)


    # save the final results
    mtts = full_indep_ephemeris[npl] + np.nanmedian(omc_trace["ypredict"], 0)
    mephem = poly.polyval(p.index, poly.polyfit(p.index, mtts, 1))
                                                                          
    matern_transit_times.append(mtts)
    matern_ephemeris.append(mephem)  
    
     
    
    # plot the final trend and outliers
    fg_prob = fgplist[np.argmin(aiclist)]
    
    plt.figure(figsize=(12,4))
    plt.scatter(xtime, yomc*24*60, c=1-fg_prob, cmap="viridis", label="Quick TTVs")
    plt.plot(xtime[bad], yomc[bad]*24*60, "rx")
    plt.plot(full_indep_ephemeris[npl], np.nanmedian(omc_trace["ypredict"],0)*24*60, "k", label="Matern-3/2 fit")
    plt.xlabel("Time [BJKD]", fontsize=20)
    plt.ylabel("O-C [min]", fontsize=20)
    plt.legend(fontsize=12)
    plt.savefig(FIGURE_DIR + TARGET + '_ttvs_matern_{0}.pdf'.format(npl), bbox_inches='tight')
    plt.title(TARGET, fontsize=20)
    plt.close()


# In[ ]:


# fit exponential to CDPP values
tdur = np.array([3,6,9,12,15])

def cdpp_fxn(theta, x):
    alpha, beta, x0, tau = theta

    return alpha + beta*np.exp(-(x-x0)/tau)

def res_fxn(theta, x, y):
    return y - cdpp_fxn(theta, x)


theta_guess = [CDPP.min(), CDPP.max()-CDPP.min(), 3.0, 3.0]
theta_out, success = op.leastsq(res_fxn, theta_guess, args=(tdur, CDPP))
    
plt.figure()
plt.plot(tdur, CDPP, "ko")
plt.plot(tdur, cdpp_fxn(theta_guess, tdur), "r")
plt.plot(tdur, cdpp_fxn(theta_out, tdur), "b")
plt.close()


# In[ ]:


# take weighted average of independent and matern TTVs
quick_transit_times = []
quick_ephemeris = []

for npl, p in enumerate(planets):
    # determine weights based on transit signal-to-noise
    cdpp = cdpp_fxn(theta_out, p.duration*24)
    snr  = p.depth*1e6/cdpp
    w    = (snr/50)**2
    
    # grab transit times
    inds = transit_inds[npl]
    itt = indep_transit_times[npl]
    mtt = matern_transit_times[npl]

    qtt = (1.0*mtt[inds] + w*itt)/(1+w)
    
    quick_transit_times.append(np.copy(qtt))
    quick_ephemeris.append(poly.polyval(inds, poly.polyfit(inds, qtt, 1)))
    
    
fig, axes = plt.subplots(NPL, figsize=(12,8))
if NPL == 1: axes = [axes]

for npl, p in enumerate(planets):
    xtime = quick_transit_times[npl]
    yomc  = (quick_transit_times[npl] - quick_ephemeris[npl])*24*60
    
    axes[npl].plot(xtime, yomc, '.', c='C{0}'.format(npl))
    axes[npl].set_ylabel('O-C [min]', fontsize=20)
axes[NPL-1].set_xlabel('Time [BJKD]', fontsize=20)
plt.savefig(FIGURE_DIR + TARGET + '_ttvs_quick.pdf', bbox_inches='tight')
plt.close()


# In[ ]:


# Save Quick TTV and MAP TTV measurements to a text file
for npl in range(NPL):
    # Quick TTVs
    data_out  = np.vstack([transit_inds[npl], quick_transit_times[npl]]).swapaxes(0,1)
    fname_out = QUICK_TTV_DIR + TARGET + '_{:02d}'.format(npl) + '_quick_ttvs.txt'
    
    np.savetxt(fname_out, data_out, fmt=('%1d', '%.8f'), delimiter='\t')
    

    # MAP TTVs
    data_out  = np.vstack([transit_inds[npl], indep_transit_times[npl]]).swapaxes(0,1)
    fname_out = QUICK_TTV_DIR + TARGET + '_{:02d}'.format(npl) + '_map_ttvs.txt'
    
    np.savetxt(fname_out, data_out, fmt=('%1d', '%.8f'), delimiter='\t')    


# In[ ]:


# update transit times in Planet objects
for npl, p in enumerate(planets):
    p.tts = np.copy(matern_transit_times[npl])
    p.tts[transit_inds[npl]] = np.copy(quick_transit_times[npl])


# # Flag outliers based on transit model
# #### Cadences must be flagged as outliers from BOTH the Matern model and the Independent model to be rejected

# In[ ]:


print("Flagging remaining outliers\n")


# In[ ]:


res_i = []
res_m = []

for j, q in enumerate(quarters):
    print("QUARTER", q)
    
    # grab time and flux data
    if all_dtype[q] == "long":
        use = lc.quarter == q
        t_ = lc.time[use]
        f_ = lc.flux[use]
        
    elif all_dtype[q] == "short":
        use = sc.quarter == q
        t_ = sc.time[use]
        f_ = sc.flux[use]
        
    
    # grab transit times for each planet
    wp_i = []
    tts_i = []
    inds_i = []
    
    wp_m = []
    tts_m = []
    inds_m = []
    
    for npl in range(NPL):
        itt = indep_transit_times[npl]
        mtt = matern_transit_times[npl][transit_inds[npl]]
        
        use_i = (itt > t_.min())*(itt < t_.max())
        use_m = (mtt > t_.min())*(mtt < t_.max())
        
        if np.sum(use_i) > 0:
            wp_i.append(npl)
            tts_i.append(itt[use_i])
            inds_i.append(transit_inds[npl][use_i] - transit_inds[npl][use_i][0])
            
        if np.sum(use_m) > 0:
            wp_m.append(npl)
            tts_m.append(itt[use_m])
            inds_m.append(transit_inds[npl][use_m] - transit_inds[npl][use_m][0])
            

    
    # first check independent transit times
    wp = np.copy(wp_i)
    tts = np.copy(tts_i)
    inds = np.copy(inds_i)
    
    if len(tts) > 0:
        # set up model
        exoSLC = exo.StarryLightCurve(u)
        orbit  = exo.orbits.TTVOrbit(transit_times=tts, transit_inds=inds, period=list(periods[wp]), 
                                     b=b[wp], r_star=Rstar, m_star=Mstar)

        # set oversampling factor
        if all_dtype[q] == 'short':
            oversample = 1
        elif all_dtype[q] == 'long':
            oversample = 15

        # calculate light curves
        light_curves = exoSLC.get_light_curve(orbit=orbit, r=rp[wp], t=t_, oversample=oversample)
        model_flux = 1.0 + pm.math.sum(light_curves, axis=-1).eval()

    else:
        model_flux = np.ones_like(f_)*np.mean(f_)
    
    # calculate residuals
    res_i.append(f_ - model_flux)
    
    
    
    # then check matern transit times
    wp = np.copy(wp_m)
    tts = np.copy(tts_m)
    inds = np.copy(inds_m)
    
    if len(tts) > 0:
        # set up model
        exoSLC = exo.StarryLightCurve(u)
        orbit  = exo.orbits.TTVOrbit(transit_times=tts, transit_inds=inds, period=list(periods[wp]), 
                                     b=b[wp], r_star=Rstar, m_star=Mstar)

        # set oversampling factor
        if all_dtype[q] == 'short':
            oversample = 1
        elif all_dtype[q] == 'long':
            oversample = 15

        # calculate light curves
        light_curves = exoSLC.get_light_curve(orbit=orbit, r=rp[wp], t=t_, oversample=oversample)
        model_flux = 1.0 + pm.math.sum(light_curves, axis=-1).eval()

    else:
        model_flux = np.ones_like(f_)*np.mean(f_)
    
    # calculate residuals
    res_m.append(f_ - model_flux)


# In[ ]:


for j, q in enumerate(quarters):
    print("\nQUARTER", q)
    res = 0.5*(res_i[j] + res_m[j])
    x_ = np.arange(len(res))
    
    bad_i = np.abs(res_i[j] - np.mean(res_i[j]))/astropy.stats.mad_std(res_i[j]) > 5.0
    bad_m = np.abs(res_m[j] - np.mean(res_m[j]))/astropy.stats.mad_std(res_m[j]) > 5.0
    
    bad = bad_i * bad_m
    
    print("   outliers rejected:", np.sum(bad))
    print("   marginal outliers:", np.sum(bad_i*~bad_m)+np.sum(~bad_i*bad_m))
    
    plt.figure(figsize=(20,3))
    plt.plot(x_, res, "k", lw=0.5)
    plt.plot(x_[bad], res[bad], "rx")
    plt.xlim(x_.min(), x_.max())
    plt.close()


# In[ ]:


bad_lc = []
bad_sc = []

for q in range(18):
    if all_dtype[q] == "long_no_transits":
        bad = np.ones(np.sum(lc.quarter == q), dtype="bool")
        bad_lc = np.hstack([bad_lc, bad])
        
        
    if all_dtype[q] == "short_no_transits":
        bad = np.ones(np.sum(sc.quarter == q), dtype="bool")
        bad_sc = np.hstack([bad_sc, bad])    
    
    
    if (all_dtype[q] == "short") + (all_dtype[q] == "long"):
        j = np.where(quarters == q)[0][0]

        res = 0.5*(res_i[j] + res_m[j])
        x_ = np.arange(len(res))

        bad_i = np.abs(res_i[j] - np.mean(res_i[j]))/astropy.stats.mad_std(res_i[j]) > 5.0
        bad_m = np.abs(res_m[j] - np.mean(res_m[j]))/astropy.stats.mad_std(res_m[j]) > 5.0

        bad = bad_i * bad_m

        if all_dtype[q] == "short":
            bad_sc = np.hstack([bad_sc, bad])

        if all_dtype[q] == "long":
            bad_lc = np.hstack([bad_lc, bad])
        
        
bad_lc = np.array(bad_lc, dtype="bool")
bad_sc = np.array(bad_sc, dtype="bool")


# In[ ]:


if sc is not None:
    sc.time = sc.time[~bad_sc]
    sc.flux = sc.flux[~bad_sc]
    sc.error = sc.error[~bad_sc]
    sc.cadno = sc.cadno[~bad_sc]
    sc.mask  = None

    sc.channel = sc.channel[~bad_sc]
    sc.quarter = sc.quarter[~bad_sc]
    sc.season  = sc.season[~bad_sc]
    sc.centroid_col = sc.centroid_col[~bad_sc]
    sc.centroid_row = sc.centroid_row[~bad_sc]


if lc is not None:
    lc.time = lc.time[~bad_lc]
    lc.flux = lc.flux[~bad_lc]
    lc.error = lc.error[~bad_lc]
    lc.cadno = lc.cadno[~bad_lc]
    lc.mask  = None

    lc.channel = lc.channel[~bad_lc]
    lc.quarter = lc.quarter[~bad_lc]
    lc.season  = lc.season[~bad_lc]
    lc.centroid_col = lc.centroid_col[~bad_lc]
    lc.centroid_row = lc.centroid_row[~bad_lc]


# In[ ]:


if sc is not None:
    good_cadno_sc = np.copy(sc.cadno)
    
if lc is not None:
    good_cadno_lc = np.copy(lc.cadno)


# # Detrend again with better estimates of transit timing

# In[ ]:


# get estimate of ttv amplitude and a reasonable buffer
ttv_amps   = np.zeros(NPL)
ttv_buffer = np.zeros(NPL)

for npl in range(NPL):
    # estimate TTV amplitude
    ttv_amps[npl] = astropy.stats.mad_std(indep_transit_times[npl] - indep_ephemeris[npl])

    # based on scatter in independent times, set threshold so not even one outlier is expected
    N   = len(transit_inds[npl])
    eta = np.max([3., stats.norm.interval((N-1)/N)[1]])

    ttv_buffer[npl] = eta*ttv_amps[npl] + 0.5/24


# In[ ]:


# long cadence
if MISSION == 'Kepler':
    lc_path  = glob.glob(DOWNLOAD_DIR + 'mastDownload/Kepler/kplr' + '{0:09d}'.format(KIC) + '*_lc*/')[0]
    lc_files = glob.glob(lc_path + '*')

    lc_rawdata_list = []
    for i, lcf in enumerate(lc_files):
        olcfi = lk.search.open(lc_files[i])

        if ~np.isin(olcfi.quarter, sc_quarters):
            lc_rawdata_list.append(olcfi)

    lc_rawdata = lk.LightCurveFileCollection(lc_rawdata_list)
    lc_data = detrend.cleanup_lkfc(lc_rawdata, KIC)
    
    
elif MISSION == 'Simulated':
    lc_path = DOWNLOAD_DIR + 'Lightcurves/Kepler/simkplr' + '{0:09d}'.format(KIC) + '_lc/'
    lc_files = glob.glob(lc_path + '*')
    
    lc_rawdata_list = []
    for i, lcf in enumerate(lc_files):
        lc_rawdata_list.append(io.load_sim_fits(lcf))
        
    
    quarters = []
    for i, lcrd in enumerate(lc_rawdata_list):
        quarters.append(lcrd.quarter)
        
    order = np.argsort(quarters)
    
    lc_data_list = []
    for j, q in enumerate(quarters):
        lc_data_list.append(lc_rawdata_list[order[j]])

    lc_rawdata = lk.LightCurveFileCollection(lc_rawdata_list)
    lc_data = lk.LightCurveCollection(lc_data_list)

    
    
# make sure there is at least one transit in the long cadence data
# this shouldn't be an issue for real KOIs, but can happen for simulated data
if np.sum(np.array(all_dtype) == "long") == 0:
    lc_data = []
    
    
lc_quarters = []
for i, lcd in enumerate(lc_data):
    lc_quarters.append(lcd.quarter)


# In[ ]:


# short cadence
try:
    if MISSION == 'Kepler':
        sc_path  = glob.glob(DOWNLOAD_DIR + 'mastDownload/Kepler/kplr' + '{0:09d}'.format(KIC) + '*_sc*/')[0]
        sc_files = glob.glob(sc_path + '*')

        sc_rawdata_list = []
        for i, scf in enumerate(sc_files):
            oscfi = lk.search.open(sc_files[i])
            sc_rawdata_list.append(oscfi)

        sc_rawdata = lk.LightCurveFileCollection(sc_rawdata_list)
        sc_data = detrend.cleanup_lkfc(sc_rawdata, KIC)

        
    elif MISSION == 'Simulated':
        sc_path = DOWNLOAD_DIR + 'Lightcurves/Kepler/simkplr' + '{0:09d}'.format(KIC) + '_sc/'
        sc_files = glob.glob(sc_path + '*')

        sc_rawdata_list = []
        for i, scf in enumerate(sc_files):
            sc_rawdata_list.append(io.load_sim_fits(scf))


        quarters = []
        for i, scrd in enumerate(sc_rawdata_list):
            quarters.append(scrd.quarter)

        order = np.argsort(quarters)

        sc_data_list = []
        for j, q in enumerate(quarters):
            sc_data_list.append(sc_rawdata_list[order[j]])

        sc_rawdata = lk.LightCurveFileCollection(sc_rawdata_list)
        sc_data = lk.LightCurveCollection(sc_data_list)
        
        
except:
    sc_data = []
    

# make sure there is at least one transit in the short cadence data
# this shouldn't be an issue for real KOIs, but can happen for simulated data
if np.sum(np.array(all_dtype) == "short") == 0:
    sc_data = []
    
    
sc_quarters = []
for i, scd in enumerate(sc_data):
    sc_quarters.append(scd.quarter)    


# In[ ]:


# clean up the LONG CADENCE data
for i, lcq in enumerate(lc_data):
    lcq = detrend.remove_flagged_cadences(lcq)
    #lcq = detrend.clip_outliers(lcq, kernel_size=5, sigma_upper=5.0, sigma_lower=15.0)
    lc_data[i] = lcq

# broadcast quarter and channel integers into arrays (for convenient use after stitching)
for i, lcq in enumerate(lc_data):
    lcq.quarter = lcq.quarter*np.ones(len(lcq.time))
    lcq.channel = lcq.channel*np.ones(len(lcq.time))

    
if len(lc_data) > 0:
    # combine quarters into a single LiteCurve
    lc = detrend.stitch_lkc(lc_data)
    
    # remove cadences flagged by first fitting iteration
    keep = np.isin(lc.cadno, good_cadno_lc)
    
    lc.time = lc.time[keep]
    lc.flux = lc.flux[keep]
    lc.error = lc.error[keep]
    lc.cadno = lc.cadno[keep]
    lc.quarter = lc.quarter[keep]
    lc.channel = lc.channel[keep]
    lc.centroid_col = lc.centroid_col[keep]
    lc.centroid_row = lc.centroid_row[keep]    
    
    # make a mask where planets transit
    lc_mask = np.zeros((NPL,len(lc.time)),dtype='bool')
    for npl, p in enumerate(planets):
        masksize = np.min([1.5, 0.5 + ttv_buffer[npl]/p.duration])

        lc_mask[npl] = detrend.make_transitmask(lc.time, p.tts, p.duration, masksize=masksize)

    lc.mask = deepcopy(lc_mask)
    
    # plot the non-detrended data
    plt.figure(figsize=(20,4))
    plt.plot(lc.time, lc.flux, c="k", lw=0.5)
    plt.xlim(lc.time.min(), lc.time.max())
    plt.xlabel('Time [BKJD]', fontsize=20)
    plt.ylabel('Relative Flux', fontsize=20)
    plt.close()
    
else:
    lc = None


# In[ ]:


if lc is not None:    
    # detrend
    break_tolerance = np.max([int(DURS.min()/(LCIT/60/24)*5/2), 13])
    min_period = 5*DURS.max()
    
    lc = detrend.flatten_with_gp(lc, break_tolerance=break_tolerance, min_period=min_period)

    # determine seasons
    lc.season = lc.quarter % 4

    # plot detrended data
    plt.figure(figsize=(16,4))
    plt.plot(lc.time, lc.flux, c='k', lw=0.5)
    plt.xlim(lc.time.min(), lc.time.max())
    plt.xlabel('Time [BKJD]', fontsize=20)
    plt.ylabel('Relative Flux', fontsize=20)
    plt.savefig(FIGURE_DIR + TARGET + '_long_cadence_flux.pdf', bbox_inches='tight')
    plt.close()


# In[ ]:


# clean up the SHORT CADENCE data
for i, scq in enumerate(sc_data):
    scq = detrend.remove_flagged_cadences(scq)
    #scq = detrend.clip_outliers(scq, kernel_size=13, sigma_upper=5.0, sigma_lower=15.0)
    sc_data[i] = scq

# broadcast quarter and channel integers into arrays (for convenient use after stitching)
for i, scq in enumerate(sc_data):
    scq.quarter = scq.quarter*np.ones(len(scq.time))
    scq.channel = scq.channel*np.ones(len(scq.time))

    
if len(sc_data) > 0:
    # combine quarters into a single LiteCurve
    sc = detrend.stitch_lkc(sc_data)
    
    # remove cadences flagged by first fitting iteration
    keep = np.isin(sc.cadno, good_cadno_sc)
    
    sc.time = sc.time[keep]
    sc.flux = sc.flux[keep]
    sc.error = sc.error[keep]
    sc.cadno = sc.cadno[keep]
    sc.quarter = sc.quarter[keep]
    sc.channel = sc.channel[keep]
    sc.centroid_col = sc.centroid_col[keep]
    sc.centroid_row = sc.centroid_row[keep]
    
    # make a mask where planets transit
    sc_mask = np.zeros((NPL,len(sc.time)),dtype='bool')
    for npl, p in enumerate(planets):
        masksize = np.min([1.5, 0.5 + ttv_buffer[npl]/p.duration])
        
        sc_mask[npl] = detrend.make_transitmask(sc.time, p.tts, p.duration, masksize=masksize)

    sc.mask = deepcopy(sc_mask)
    
    # plot the non-detrended data
    plt.figure(figsize=(20,4))
    plt.plot(sc.time, sc.flux, c="k", lw=0.5)
    plt.xlim(sc.time.min(), sc.time.max())
    plt.xlabel('Time [BKJD]', fontsize=20)
    plt.ylabel('Relative Flux', fontsize=20)
    plt.close()
    
else:
    sc = None


# In[ ]:


if sc is not None:    
    # detrend
    break_tolerance = np.max([int(DURS.min()/(SCIT/3600/24)*5/2), 91])
    min_period = 5*DURS.max()
    
    sc = detrend.flatten_with_gp(sc, break_tolerance=break_tolerance, min_period=min_period)

    # determine seasons
    sc.season = sc.quarter % 4

    # plot detrended data
    plt.figure(figsize=(16,4))
    plt.plot(sc.time, sc.flux, c='k', lw=0.5)
    plt.xlim(sc.time.min(), sc.time.max())
    plt.xlabel('Time [BKJD]', fontsize=20)
    plt.ylabel('Relative Flux', fontsize=20)
    plt.savefig(FIGURE_DIR + TARGET + '_short_cadence_flux.pdf', bbox_inches='tight')
    plt.close()


# # Make individual mask for where each planet transits
# ### These masks have width 1.5 transit durations, which may be wider than the masks used for detrending

# In[ ]:


print('Making transit masks')
try:
    sc_mask = np.zeros((NPL,len(sc.time)),dtype='bool')
    for npl, p in enumerate(planets):
        sc_mask[npl] = detrend.make_transitmask(sc.time, p.tts, p.duration, masksize=1.5)

    sc.mask = sc_mask

except:
    sc_mask = None

    
try:
    lc_mask = np.zeros((NPL,len(lc.time)),dtype='bool')
    for npl, p in enumerate(planets):
        lc_mask[npl] = detrend.make_transitmask(lc.time, p.tts, p.duration, masksize=1.5)
    
    lc.mask = lc_mask

except:
    lc_mask = None


# # Flag high quality transits (quality = 1)
# 
# ### Good transits must have  at least 50% photometry coverage in/near transit

# In[ ]:


for npl, p in enumerate(planets):
    count_expect_lc = int(np.ceil(p.duration/lcit))
    count_expect_sc = int(np.ceil(p.duration/scit))
    
    quality = []
    
    # cut out the stamps:
    for i, t0 in enumerate(p.tts):
        
        try:
            in_sc = np.abs(sc.time - t0)/p.duration < 0.5
            in_lc = np.abs(lc.time - t0)/p.duration < 0.5
            
            near_sc = np.abs(sc.time - t0)/p.duration < 1.5
            near_lc = np.abs(lc.time - t0)/p.duration < 1.5
            
            qual_in = (np.sum(in_sc) > 0.5*count_expect_sc) + (np.sum(in_lc) > 0.5*count_expect_lc)
            qual_near = (np.sum(near_sc) > 1.5*count_expect_sc) + (np.sum(near_lc) > 1.5*count_expect_lc)
            
            quality.append(qual_in*qual_near)
                                    
                        
        except:
            in_lc = np.abs(lc.time - t0)/p.duration < 0.5
            near_lc = np.abs(lc.time - t0)/p.duration < 1.5
            
            qual_in = (np.sum(in_lc) > 0.5*count_expect_lc)
            qual_near = (np.sum(near_lc) > 1.5*count_expect_lc)
            
            quality.append(qual_in*qual_near)
            
    p.quality = np.copy(quality)


# In[ ]:


print('')
print('cumulative runtime = ', int(timer() - global_start_time), 's')
print('')


# # Make phase-folded transit plots

# In[ ]:


for npl, p in enumerate(planets):
    tts = p.tts[p.quality*~p.overlap]
    
    
    if len(tts) == 0:
        print("No non-overlapping high quality transits found for planet {0} (P = {1} d)".format(npl, p.period))
    
    
    else:
        t_ = []
        f_ = []


        for t0 in tts:
            try:
                neartransit_sc = np.abs(sc.time - t0)/p.duration < 1.5
            except:
                neartransit_sc = None

            try:
                neartransit_lc = np.abs(lc.time - t0)/p.duration < 1.5
            except:
                neartransit_lc = None

            if (neartransit_lc is not None) and (np.sum(neartransit_lc) > 0):
                t_.append(lc.time[neartransit_lc] - t0)
                f_.append(lc.flux[neartransit_lc])

            if (neartransit_sc is not None) and (np.sum(neartransit_sc) > 0):
                undersample = 6
                t_.append(sc.time[neartransit_sc][::undersample] - t0)
                f_.append(sc.flux[neartransit_sc][::undersample])

        t_ = np.hstack(t_)
        f_ = np.hstack(f_)

        order = np.argsort(t_)
        t_ = t_[order]
        f_ = f_[order]

        f_bin = bin_data(t_, f_, LCIT/60/24/2)
        t_bin = bin_data(t_, t_, LCIT/60/24/2)

        plt.figure(figsize=(12,4))
        plt.plot(t_*24, f_, ".", c="lightgrey")
        plt.plot(t_bin*24, f_bin, "o", ms=5, color="C{0}".format(npl))
        plt.savefig(FIGURE_DIR + TARGET + '_{0:02d}_folded_transit_.pdf'.format(npl), bbox_inches='tight')
        plt.close()


# # Generate and model empirical autocorrelation function (ACF)

# In[ ]:


# generating figures inside imported modules creates issues with UChicago Midway RCC cluster
# it's easier to just define the function here in the main script

def plot_acf(xcor, acf_emp, acf_mod, xf, yf, freqs, target_name, season):
    """
    Docstring
    """
    fig = plt.figure(figsize=(20,5))

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.8)

    ax = plt.subplot2grid(shape=(5,10), loc=(0,0), rowspan=3, colspan=7)
    ax.plot(xcor*24, acf_emp, color='lightgrey')
    ax.plot(xcor*24, acf_mod, c='red')
    ax.set_xlim(xcor.min()*24,xcor.max()*24)
    ax.set_xticks(np.arange(0,xcor.max()*24,2))
    ax.set_xticklabels([])
    ax.set_ylim(acf_emp.min()*1.1, acf_emp.max()*1.1)
    ax.set_ylabel('ACF', fontsize=20)
    ax.text(xcor.max()*24-0.15, acf_emp.max(), '%s, SEASON %d' %(target_name, season), va='top', ha='right', fontsize=20)


    ax = plt.subplot2grid(shape=(5,10), loc=(0,7), rowspan=5, colspan=3)
    ax.plot(xf/24/3600*1e3, yf, color='k', lw=0.5)
    for f in freqs:
        ax.axvline(f/24/3600*1e3, color='red', zorder=0, lw=3, alpha=0.3) 
    ax.set_xlim(xf.min()/24/3600*1e3, xf.max()/24/3600*1e3)
    ax.set_ylim(yf.min(),1.2*yf.max())
    ax.set_ylabel('Power', fontsize=20)
    ax.set_yticks([])
    ax.set_xlabel('Frequency [mHz]', fontsize=20)

    for i, sf in enumerate(np.sort(freqs)[::-1]):
        ax.text(xf.min()/24/3600*1e3+0.1, yf.max()*(1.1-0.1*i), '%.2f min' %(24*60/sf), fontsize=16)


    ax = plt.subplot2grid(shape=(5,10), loc=(3,0), rowspan=2, colspan=7)
    ax.plot(xcor*24, acf_emp-acf_mod, c='lightgrey')
    ax.set_xlim(xcor.min()*24,xcor.max()*24)
    ax.set_xticks(np.arange(0,xcor.max()*24,2))
    ax.set_xlabel('Lag time [hours]', fontsize=20)
    ax.set_ylabel('Residuals', fontsize=20)    
    
    return fig


# In[ ]:


# determine what data type each season has
season_dtype = []

if sc is not None:
    sc_seasons = np.unique(sc.season)
else:
    sc_seasons = np.array([])

if lc is not None:
    lc_seasons = np.unique(lc.season)
else:
    lc_seasons = np.array([])

for z in range(4):
    if np.isin(z, sc_seasons):
        season_dtype.append("short")
    elif np.isin(z, lc_seasons):
        season_dtype.append("long")
    else:
        season_dtype.append("none")


# In[ ]:


print('Generating autocorrelation function')
print("Season data types:", season_dtype, "\n")

# grab transit durations
durations = np.zeros(NPL)
for npl, p in enumerate(planets):
    durations[npl] = p.duration

    
# set cutoff between low and high frequency signals
fcut = 2/(LCIT/60/24)
fmin = 2/(5*durations.max())


# now estimate the ACF
acf_lag = []
acf_emp = []
acf_mod = []
acf_freqs = []


for z in range(4):
    if season_dtype[z] == "none":
        acf_lag.append(None)
        acf_emp.append(None)
        acf_mod.append(None)
        acf_freqs.append(None)
        
        
    else:
        if season_dtype[z] == "short":
            Npts = int(np.min([5*(1/24+durations.max()),2/3*periods.min()])*24*3600/SCIT)
            use = sc.season == z
            m_ = sc.mask.sum(axis=0) > 0
            m_ = m_[use]

            if np.sum(use) > 0:
                t_ = sc.time[use][~m_]
                f_ = sc.flux[use][~m_]
                c_ = sc.cadno[use][~m_]
                
        if season_dtype[z] == "long":
            Npts = int(np.min([5*(1/24+durations.max()),2/3*periods.min()])*24*60/LCIT)
            use = lc.season == z
            m_ = lc.mask.sum(axis=0) > 0
            m_ = m_[use]

            if np.sum(use) > 0:
                t_ = lc.time[use][~m_]
                f_ = lc.flux[use][~m_]
                c_ = lc.cadno[use][~m_]

                
        if np.sum(use) > 0:
            # generate the empirical acf (if generate_acf fails, use very low amplitde white noise)
            try:
                xcor, acor, wcor, acf_stats = noise.generate_acf(t_, f_, c_, Npts)
            except:
                xcor = np.arange(Npts, dtype="float")
                acor = np.random.normal(size=len(xcor))*np.std(f_)*np.finfo(float).eps
            
            if season_dtype[z] == "long":
                xcor = xcor*LCIT/60/24
                method = "smooth"
                window_length = 3
            
            if season_dtype[z] == "short":
                xcor = xcor*SCIT/3600/24
                method = "savgol"
                window_length = None

            # model the acf
            acor_emp, acor_mod, xf, yf, freqs = noise.model_acf(xcor, acor, fcut, fmin=fmin, 
                                                                method=method, window_length=window_length)

            # make some plots
            fig = plot_acf(xcor, acor_emp, acor_mod, xf, yf, freqs, TARGET, z)
            fig.savefig(FIGURE_DIR + TARGET + '_ACF_season_{0}.pdf'.format(z), bbox_inches='tight')
            
            
            # filter out high-frequency components in short cadence data
            if season_dtype[z] == "short":
                fring = freqs[freqs > fcut]
                bw = 1/(lcit-scit/np.sqrt(2))-1/(lcit+scit/np.sqrt(2))

                if len(fring) > 0:
                    flux_filtered = detrend.filter_ringing(sc, break_tolerance, fring, bw)

                    sc.flux[use] = flux_filtered[use]
                    f_ = sc.flux[use][~m_]

                # re-run the ACF modeling on the filtered lightcurve
                try:
                    xcor, acor, wcor, acf_stats = noise.generate_acf(t_, f_, c_, Npts)
                except:
                    xcor = np.arange(Npts, dtype="float")
                    acor = np.random.normal(size=len(xcor))*np.std(f_)*np.finfo(float).eps

                
                xcor = xcor*SCIT/3600/24
                
                acor_emp, acor_mod, xf, yf, freqs = noise.model_acf(xcor, acor, fcut, fmin=fmin, method='savgol')

                fig = plot_acf(xcor, acor_emp, acor_mod, xf, yf, freqs, TARGET, z)
                fig.savefig(FIGURE_DIR + TARGET + '_ACF_season_{0}_filtered.pdf'.format(z), bbox_inches='tight')
                
            
            # add to list
            acf_lag.append(xcor)
            acf_emp.append(acor_emp)
            acf_mod.append(acor_mod)
            acf_freqs.append(freqs)
            
        
        else:
            acf_lag.append(None)
            acf_emp.append(None)
            acf_mod.append(None)
            acf_freqs.append(None)   


# # Generate and model synthetic noise

# In[ ]:


print('Generating synthetic noise\n')

synth_time  = []
synth_red   = []
synth_white = []


for z in range(4):
    print("SEASON")
    print(season_dtype[z])
    
    if season_dtype[z] == "none":
        synth_time.append(None)
        synth_red.append(None)
        synth_white.append(None)
       
        
    else:
        if season_dtype[z] == "short":
            Npts = int(2*durations.max()*24*3600/SCIT)
            use = sc.season == z
            m_ = sc.mask.sum(axis=0) > 0
            m_ = m_[use]

            if np.sum(use) > 0:
                t_ = sc.time[use][~m_]
                f_ = sc.flux[use][~m_]
                
        
        if season_dtype[z] == "long":
            Npts = int(5*durations.max()*24*60/LCIT)
            use = lc.season == z
            m_ = lc.mask.sum(axis=0) > 0
            m_ = m_[use]

            if np.sum(use) > 0:
                t_ = lc.time[use][~m_]
                f_ = lc.flux[use][~m_]
                
                
        if np.sum(use) > 0:
            if season_dtype[z] == "long":
                vector_length = 15*len(acf_lag[z])
            if season_dtype[z] == "short":
                vector_length = len(acf_lag[z])
            
            
            # pull and split high/low frequencies
            freqs = np.copy(acf_freqs[z])

            low_freqs  = freqs[freqs <= fcut]
            high_freqs = freqs[freqs > fcut]


            # generate some synthetic correlated noise
            clipped_acf = (acf_mod[z][:Npts])*np.linspace(1,0,Npts)

            x, red_noise, white_noise = noise.generate_synthetic_noise(acf_lag[z][:Npts], clipped_acf, 
                                                                       vector_length, np.std(f_))

            # add to list
            synth_time.append(x)
            synth_red.append(red_noise)
            synth_white.append(white_noise)


            # plot the noise
            plt.figure(figsize=(20,5))
            plt.plot(x, white_noise + red_noise, '.', c='lightgrey')
            plt.plot(x, red_noise, c='red', lw=4)
            plt.axhline(4/RSRE**2, c='k', ls='--', lw=3)
            plt.axhline(-4/RSRE**2, c='k', ls='--', lw=3)
            plt.xlim(x.min(),x.max())
            plt.xlabel('Time [days]', fontsize=30)
            plt.ylabel('Flux', fontsize=30)
            plt.text(x.max()-0.02, white_noise.max()*0.95, '%s, SEASON %d' %(TARGET, z), va='center', ha='right', fontsize=24)
            plt.savefig(FIGURE_DIR + TARGET + '_synthetic_noise_season_{0}.pdf'.format(z), bbox_inches='tight')
            plt.close()

            
        else:
            synth_time.append(None)
            synth_red.append(None)
            synth_white.append(None)


# In[ ]:


print('Fitting a GP to synthetic noise\n')

gp_model  = []
gp_trace  = []
gp_priors = []


for z in range(4):
    if season_dtype[z] == "none":
        gp_model.append(None)
        gp_trace.append(None)
        gp_priors.append(None)
       
        
    else:
        # pull and split high/low frequencies
        freqs = np.copy(acf_freqs[z])
        
        if freqs is not None:

            low_freqs  = freqs[freqs <= fcut]
            high_freqs = freqs[freqs > fcut]

            if len(low_freqs) > 0:
                lf = [low_freqs[0]]
            else:
                lf = None
                
            if len(high_freqs) > 0:
                warnings.warn("there are remaining high-frequency noise components")
            
            
        else:
            lf = None


            
            
        # fit a GP model to the synthetic noise
        gp_model.append(noise.build_sho_model(synth_time[z], synth_red[z], var_method='local', test_freq=lf))
        
        
        with gp_model[z] as model:
            gp_map = exo.optimize(start=model.test_point, vars=[model.vars[0]])

            for mv in model.vars[1:]:
                gp_map = exo.optimize(start=gp_map, vars=[mv])

            gp_map = exo.optimize(start=gp_map)


        with gp_model[z] as model:
            trace = pm.sample(tune=4000, draws=1000, start=gp_map, chains=2, 
                              step=exo.get_dense_nuts_step(target_accept=0.9))
            
        
        gp_trace.append(trace)
        gp_priors.append(noise.make_gp_prior_dict(gp_trace[z]))

        plt.figure(figsize=(20,4))
        plt.plot(synth_time[z], synth_red[z], c='pink', lw=4)
        plt.plot(synth_time[z], np.nanmedian(gp_trace[z]['gp_pred'], axis=0), c='red')
        plt.xlim(synth_time[z].min(), synth_time[z].max())
        plt.savefig(FIGURE_DIR + TARGET + '_GP_noise_model_{0}.pdf'.format(z), bbox_inches='tight')
        plt.close()


# In[ ]:


# list of all quarters with data
quarters = []

if lc is not None:
    quarters = np.hstack([quarters, np.unique(lc.quarter)])
if sc is not None:
    quarters = np.hstack([quarters, np.unique(sc.quarter)])

quarters = np.array(np.sort(quarters), dtype="int")

# get variance of each quarter
if lc is not None:
    lcm = lc.mask.sum(axis=0) == 0
if sc is not None:
    scm = sc.mask.sum(axis=0) == 0

var_by_quarter = []

for i, q in enumerate(quarters):
    if (sc is not None) and (np.sum(sc.quarter==q) > 0):
        var_by_quarter.append(np.var(sc.flux[scm*(sc.quarter==q)]))
    elif (lc is not None) and (np.sum(lc.quarter==q) > 0):
        var_by_quarter.append(np.var(lc.flux[lcm*lc.quarter==q]))
    else:
        var_by_quarter.append(None)


# # Save detrended lightcurves and estimates of the noise properties

# In[ ]:


# Save detrended lightcurves as .fits files
try:
    lc.to_fits(TARGET, DLC_DIR + TARGET + '_lc_detrended.fits')
except:
    print("No long cadence data")

try:
    sc.to_fits(TARGET, DLC_DIR + TARGET + '_sc_detrended.fits')
except:
    print("No short cadence data")


# In[ ]:


# Save var_by_quarter
data_out  = np.vstack([quarters, var_by_quarter]).swapaxes(0,1)
fname_out = NOISE_DIR + TARGET + '_var_by_quarter.txt'
    
np.savetxt(fname_out, data_out, fmt=('%1d', '%.15f'), delimiter='\t')


# save gp_priors
for z in range(4):
    try:
        for k in gp_priors[z].keys():
            gp_priors[z][k] = list(gp_priors[z][k])

        fname_out = NOISE_DIR + TARGET + '_shoterm_gp_priors_{0}.txt'.format(z)

        with open(fname_out, 'w') as file:
            json.dump(gp_priors[z], file)
            
    except:
        pass


# In[ ]:


print('TOTAL RUNTIME = %.2f min' %((timer()-global_start_time)/60))


# In[ ]:





# In[ ]:





# In[ ]:




