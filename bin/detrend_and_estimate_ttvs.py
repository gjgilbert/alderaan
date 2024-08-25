#!/usr/bin/env python
# coding: utf-8

# # Detrend and Estimate TTVs


import os
import sys
import glob
import shutil
import warnings
from datetime import datetime
from timeit import default_timer as timer

print("")
print("+"*shutil.get_terminal_size().columns)
print("ALDERAAN Detrending and TTV Estimation")
print("Initialized {0}".format(datetime.now().strftime("%d-%b-%Y at %H:%M:%S")))
print("+"*shutil.get_terminal_size().columns)
print("")

# start program timer
global_start_time = timer()


# #### Parse inputs


import argparse
import matplotlib as mpl

try:
    parser = argparse.ArgumentParser(description="Inputs for ALDERAAN transit fiting pipeline")
    parser.add_argument("--mission", default=None, type=str, required=True, \
                        help="Mission name; can be 'Kepler' or 'Simulated'")
    parser.add_argument("--target", default=None, type=str, required=True, \
                        help="Target name; format should be K00000 or S00000")
    parser.add_argument("--project_dir", default=None, type=str, required=True, \
                        help="Project directory for saving outputs")
    parser.add_argument("--data_dir", default=None, type=str, required=True, \
                        help="Data directory for accessing MAST lightcurves")
    parser.add_argument("--catalog", default=None, type=str, required=True, \
                        help="CSV file containing input planetary parameters")
    parser.add_argument("--run_id", default=None, type=str, required=True, \
                        help="run identifier")
    parser.add_argument("--verbose", default=False, type=bool, required=False, \
                        help="'True' to enable verbose logging")
    parser.add_argument("--iplot", default=False, type=bool, required=False, \
                        help="'True' to enable interactive plotting; by default matplotlib backend will be set to 'Agg'")

    args = parser.parse_args()
    MISSION      = args.mission
    TARGET       = args.target
    PROJECT_DIR  = args.project_dir
    DATA_DIR     = args.data_dir
    CATALOG      = args.catalog
    RUN_ID       = args.run_id
    VERBOSE      = args.verbose
    IPLOT        = args.iplot
    
    # set plotting backend
    if not IPLOT:
        mpl.use('agg')
    
except:
    pass


USE_SC = False


print("")
if MISSION == 'Kepler':
    print("   MISSION : Kepler")
elif MISSION == 'Simulated':
    print("   MISSION : Simulated")
print("   TARGET  : {0}".format(TARGET))
print("   RUN ID  : {0}".format(RUN_ID))
print("")
print("   Project directory : {0}".format(PROJECT_DIR))
print("   Data directory    : {0}".format(DATA_DIR))
print("   Input catalog     : {0}".format(CATALOG))
print("")


# #### Build directory structure


# directories in which to place pipeline outputs for this run
RESULTS_DIR = os.path.join(PROJECT_DIR, 'Results', RUN_ID, TARGET)
FIGURE_DIR  = os.path.join(PROJECT_DIR, 'Figures', RUN_ID, TARGET)

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)


# #### Set environment variables


sys.path.append(PROJECT_DIR)


# #### Import packages


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import glob
from   copy import deepcopy

import astropy
from   astropy.io import fits
from   astropy.timeseries import LombScargle
import lightkurve as lk
import numpy.polynomial.polynomial as poly
from   scipy import ndimage
from   scipy import stats
from   scipy.interpolate import interp1d

import pymc3 as pm
import pymc3_ext as pmx
import exoplanet as exo
import aesara_theano_fallback.tensor as T
from   aesara_theano_fallback import aesara as theano
from   celerite2.theano import GaussianProcess
from   celerite2.theano import terms as GPterms
from   celerite2.backprop import LinAlgError

from   alderaan.constants import *
from   alderaan.utils import *
import alderaan.detrend as detrend
import alderaan.io as io
import alderaan.omc as omc
from   alderaan.LiteCurve import LiteCurve
from   alderaan.Planet import Planet


# flush buffer to avoid mixed outputs from progressbar
sys.stdout.flush()

# turn off FutureWarnings
warnings.filterwarnings('ignore', category=FutureWarning)

# supress UnitsWarnings (this code doesn't use astropy units)
warnings.filterwarnings(action='ignore', category=astropy.units.UnitsWarning, module='astropy')

# check for interactive matplotlib backends
if np.any(np.array(['agg', 'png', 'svg', 'pdf', 'ps']) == mpl.get_backend()):
    warnings.warn("Selected matplotlib backend does not support interactive plotting")
    IPLOT = False
    
# print theano compiledir cache
print("theano cache: {0}\n".format(theano.config.compiledir))


# MAIN SCRIPT BEGINS HERE
def main():    
    
    # # ################
    # # --- DATA I/O ---
    # # ################
    
    
    print("\nLoading data...\n")
    
    
    # ## Read in planet and stellar properties
    # 
    # ##### WARNING!!! Reference epochs are not always consistent between catalogs. If using DR25, you will need to correct from BJD to BJKD with an offset of 2454833 days - the cumulative exoplanet archive catalog has already converted epochs to BJKD
    
    
    # Read in the data from csv file
    target_dict = pd.read_csv(PROJECT_DIR + 'Catalogs/' + CATALOG)
    
    # set KOI_ID global variable
    if MISSION == "Kepler":
        KOI_ID = TARGET
    elif MISSION == "Simulated":
        KOI_ID = "K" + TARGET[1:]
    else:
        raise ValueError("MISSION must be 'Kepler' or 'Simulated'")
    
    # pull relevant quantities and establish GLOBAL variables
    use = np.array(target_dict['koi_id']) == KOI_ID
    
    KIC = np.array(target_dict['kic_id'], dtype='int')[use]
    NPL = np.array(target_dict['npl'], dtype='int')[use]
    
    U1 = np.array(target_dict['limbdark_1'], dtype='float')[use]
    U2 = np.array(target_dict['limbdark_2'], dtype='float')[use]
    
    PERIODS = np.array(target_dict['period'], dtype='float')[use]
    EPOCHS  = np.array(target_dict['epoch'],  dtype='float')[use]
    DEPTHS  = np.array(target_dict['depth'], dtype='float')[use]*1e-6          # [ppm] --> []
    DURS    = np.array(target_dict['duration'], dtype='float')[use]/24         # [hrs] --> [days]
    IMPACTS = np.array(target_dict['impact'], dtype='float')[use]
    
    # do some consistency checks
    if all(k == KIC[0] for k in KIC): KIC = KIC[0]
    else: raise ValueError("There are inconsistencies with KIC in the csv input file")
    
    if all(n == NPL[0] for n in NPL): NPL = NPL[0]
    else: raise ValueError("There are inconsistencies with NPL in the csv input file")
    
    if all(u == U1[0] for u in U1): U1 = U1[0]
    else: raise ValueError("There are inconsistencies with U1 in the csv input file")
    
    if all(u == U2[0] for u in U2): U2 = U2[0]
    else: raise ValueError("There are inconsistencies with U2 in the csv input file")
        
    if np.any(np.isnan(PERIODS)): raise ValueError("NaN values found in input catalog")
    if np.any(np.isnan(EPOCHS)):  raise ValueError("NaN values found in input catalog")
    if np.any(np.isnan(DEPTHS)):  raise ValueError("NaN values found in input catalog")
    if np.any(np.isnan(DURS)):    raise ValueError("NaN values found in input catalog")
    if np.any(np.isnan(IMPACTS)): raise ValueError("NaN values found in input catalog")
    
    
    # ## Read in pre-downloaded (or pre-simulated) lightcurve data
    
    
    mast_files = glob.glob(DATA_DIR + 'kplr{0:09d}*.fits'.format(KIC))
    mast_files.sort()
    
    
    # short cadence data
    sc_rawdata_list = []
    if USE_SC:
        for i, mf in enumerate(mast_files):
            with fits.open(mf) as hduL:
                if hduL[0].header['OBSMODE'] == 'short cadence':
                    sc_rawdata_list.append(lk.read(mf))
    
    sc_raw_collection = lk.LightCurveCollection(sc_rawdata_list)
    sc_data = io.cleanup_lkfc(sc_raw_collection, KIC)
    
    sc_quarters = []
    sc_lite = []
    for i, scd in enumerate(sc_data):
        sc_quarters.append(scd.quarter)
        sc_lite.append(io.LightKurve_to_LiteCurve(scd))
        
    sc_data = sc_lite
    
    
    # long cadence data (only use quarters w/o short cadence available)
    lc_rawdata_list = []
    for i, mf in enumerate(mast_files):
        with fits.open(mf) as hduL:
            if (hduL[0].header['OBSMODE'] == 'long cadence') \
            and ~np.isin(hduL[0].header['QUARTER'], sc_quarters):
                lc_rawdata_list.append(lk.read(mf))
                
    lc_raw_collection = lk.LightCurveCollection(lc_rawdata_list)
    lc_data = io.cleanup_lkfc(lc_raw_collection, KIC)
    
    lc_quarters = []
    lc_lite = []
    for i, lcd in enumerate(lc_data):
        lc_quarters.append(lcd.quarter)
        lc_lite.append(io.LightKurve_to_LiteCurve(lcd))
    
    lc_data = lc_lite
    
    
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
    
    
    # ## Initialize Planet objects
    
    
    print("Initializing {0} Planet objects".format(NPL))
    
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
    
    
    # ## Set oversampling factor
    
    
    long_cadence_oversample = [None]*NPL
    
    for npl in range(NPL):
        # rough ingress/egress timescale estimate following Winn 2010
        ror = np.sqrt(DEPTHS[npl])
        tau = 13*(PERIODS[npl]/365.25)**(1/3) * ror / 24
        
        # set sigma so binning error is < 0.1% of photometric uncertainty
        sigma = np.mean(lc_data[0].error/lc_data[0].flux) * 0.04
        
        N = int(np.ceil(np.sqrt((DEPTHS[npl]/tau) * (lcit/8/sigma))))
        N = N + (N % 2 + 1)
        
        long_cadence_oversample[npl] = np.min([np.max([N,7]),29])
        
    long_cadence_oversample = np.max(long_cadence_oversample)
    
    print("Oversampling factor = {0}".format(long_cadence_oversample))
    
    
    # # ############################
    # # ----- TRANSIT TIME SETUP -----
    # # ############################
    
    
    print("\nBuilding initial TTV model...\n")
    
    
    # use Holczer+ 2016 TTVs where they exist
    HOLCZER_FILE = PROJECT_DIR + 'Catalogs/holczer_2016_kepler_ttvs.txt'
    
    holczer_data = np.loadtxt(HOLCZER_FILE, usecols=[0,1,2,3])
    
    holczer_inds = []
    holczer_tts  = []
    holczer_pers = []
    
    for npl in range(NPL):
        koi = int(KOI_ID[1:]) + 0.01*(1+npl)
        use = np.isclose(holczer_data[:,0], koi, rtol=1e-10, atol=1e-10)
    
        # Holczer uses BJD -24548900; BJKD = BJD - 2454833
        if np.sum(use) > 0:
            holczer_inds.append(np.array(holczer_data[use,1], dtype='int'))
            holczer_tts.append(holczer_data[use,2] + holczer_data[use,3]/24/60 + 67)
            holczer_pers.append(np.median(holczer_tts[npl][1:] - holczer_tts[npl][:-1]))
    
        else:
            holczer_inds.append(None)
            holczer_tts.append(None)
            holczer_pers.append(np.nan)
    
    holczer_pers = np.asarray(holczer_pers)
    
    
    # smooth and interpolate Holczer+ 2016 TTVs where they exist
    for npl in range(NPL):
        if np.isfinite(holczer_pers[npl]):
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
    
            if len(yomc) > 16:
                ymed = boxcar_smooth(ndimage.median_filter(yomc, size=5, mode='mirror'), winsize=5)
            else:
                ymed = np.median(yomc)
                
            if len(yomc) > 4:
                out  = np.abs(yomc-ymed)/astropy.stats.mad_std(yomc-ymed) > 3.0
            else:
                out = np.zeros(len(yomc), dtype='bool')
            
            # estimate TTV signal with a regularized Matern-3/2 GP
            if np.sum(~out) > 4:
                holczer_model = omc.matern32_model(xtime[~out], yomc[~out], hephem)
            else:
                holczer_model = omc.poly_model(xtime[~out], yomc[~out], 1, hephem)
    
            with holczer_model:
                holczer_map = pmx.optimize(verbose=VERBOSE)
    
            htts = hephem + holczer_map['pred']
    
            holczer_inds[npl] = np.copy(hinds)
            holczer_tts[npl] = np.copy(htts)
    
            # plot the results
            plt.figure(figsize=(12,4))
            plt.plot(xtime[~out], yomc[~out]*24*60, 'o', c='grey', label="Holczer")
            plt.plot(xtime[out], yomc[out]*24*60, 'rx')
            plt.plot(hephem, (htts-hephem)*24*60, 'k+', label="Interpolation")
            plt.xlabel("Time [BJKD]", fontsize=20)
            plt.ylabel("O-C [min]", fontsize=20)
            plt.legend(fontsize=12)
            #plt.savefig(os.path.join(FIGURE_DIR, TARGET + '_ttvs_holczer_{0:02d}.png'.format(npl)), bbox_inches='tight')
            if IPLOT:
                plt.show()
            else:
                plt.close()
    
    
    # check if Holczer TTVs exist, and if so, replace the linear ephemeris
    for npl, p in enumerate(planets):
        match = np.isclose(holczer_pers, p.period, rtol=0.1, atol=DURS.max())
        
        if np.sum(match) > 1:
            raise ValueError("Something has gone wrong matching periods between DR25 and Holczer+2016")
            
        if np.sum(match) == 1:
            loc = np.squeeze(np.where(match))
        
            hinds = holczer_inds[loc]
            htts  = holczer_tts[loc]
            
            # first update to Holczer ephemeris
            epoch, period = poly.polyfit(hinds, htts, 1)
            
            p.epoch  = np.copy(epoch)
            p.period = np.copy(period)
            p.tts    = np.arange(p.epoch, TIME_END, p.period)
            p.index  = np.array(np.round((p.tts-p.epoch)/p.period),dtype='int')
            
            for i, t0 in enumerate(p.tts):
                for j, tH in enumerate(htts):
                    if np.abs(t0-tH)/p.period < 0.25:
                        p.tts[i] = tH
    
        else:
            pass
    
    
    # # ########################
    # # ----- SIMULATED DATA -----
    # # ########################
    
    
    if MISSION == 'Simulated':
        print("Simulating transits for injection-and-recovery test")
        
        # REMOVE KNOWN TRANSITS    
        tts  = []
        inds = []
        b    = np.zeros(NPL)
        ror  = np.zeros(NPL)
        dur  = np.zeros(NPL)
    
        for npl, p in enumerate(planets):
            tts.append(p.tts)
            inds.append(p.index)
            b[npl] = p.impact
            ror[npl] = np.sqrt(p.depth)
            dur[npl] = p.duration
    
        starrystar = exo.LimbDarkLightCurve([U1,U2])
        orbit = exo.orbits.TTVOrbit(transit_times=tts, transit_inds=inds, b=b, ror=ror, duration=dur)
    
        for i, lcd in enumerate(lc_data):
            light_curve = starrystar.get_light_curve(orbit=orbit, r=ror, t=lcd.time, oversample=long_cadence_oversample, texp=lcit)
            model_flux = pm.math.sum(light_curve, axis=-1) + T.ones(len(lcd.time))
    
            lcd.flux /= model_flux.eval()
    
        for i, scd in enumerate(sc_data):
            light_curve = starrystar.get_light_curve(orbit=orbit, r=ror, t=scd.time)
            model_flux = pm.math.sum(light_curve, axis=-1) + T.ones(len(scd.time))
    
            scd.flux /= model_flux.eval()
            
        
        # LOAD SIMULATED DATA
        target_dict = pd.read_csv(PROJECT_DIR + 'Simulations/{0}/{0}.csv'.format(RUN_ID))
    
        # pull relevant quantities and establish GLOBAL variables
        use = np.array(target_dict['koi_id']) == KOI_ID
    
        KIC = np.array(target_dict['kic_id'], dtype='int')[use]
        NPL = np.array(target_dict['npl'], dtype='int')[use]
    
        U1 = np.array(target_dict['limbdark_1'], dtype='float')[use]
        U2 = np.array(target_dict['limbdark_2'], dtype='float')[use]
    
        PERIODS = np.array(target_dict['period'], dtype='float')[use]
        EPOCHS  = np.array(target_dict['epoch'],  dtype='float')[use]
        DEPTHS  = np.array(target_dict['ror'], dtype='float')[use]**2
        DURS    = np.array(target_dict['duration'], dtype='float')[use]
        IMPACTS = np.array(target_dict['impact'], dtype='float')[use]
        
        # do some consistency checks
        if all(k == KIC[0] for k in KIC): KIC = KIC[0]
        else: raise ValueError("There are inconsistencies with KIC in the csv input file")
    
        if all(n == NPL[0] for n in NPL): NPL = NPL[0]
        else: raise ValueError("There are inconsistencies with NPL in the csv input file")
    
        if all(u == U1[0] for u in U1): U1 = U1[0]
        else: raise ValueError("There are inconsistencies with U1 in the csv input file")
    
        if all(u == U2[0] for u in U2): U2 = U2[0]
        else: raise ValueError("There are inconsistencies with U2 in the csv input file")
    
        if np.any(np.isnan(PERIODS)): raise ValueError("NaN values found in input catalog")
        if np.any(np.isnan(EPOCHS)):  raise ValueError("NaN values found in input catalog")
        if np.any(np.isnan(DEPTHS)):  raise ValueError("NaN values found in input catalog")
        if np.any(np.isnan(DURS)):    raise ValueError("NaN values found in input catalog")
        if np.any(np.isnan(IMPACTS)): raise ValueError("NaN values found in input catalog")
            
        # put epochs in range (TIME_START, TIME_START + PERIOD)
        for npl in range(NPL):
            if EPOCHS[npl] < TIME_START:
                adj = 1 + (TIME_START - EPOCHS[npl])//PERIODS[npl]
                EPOCHS[npl] += adj*PERIODS[npl]        
    
            if EPOCHS[npl] > (TIME_START + PERIODS[npl]):
                adj = (EPOCHS[npl] - TIME_START)//PERIODS[npl]
                EPOCHS[npl] -= adj*PERIODS[npl]
                
        
        # INITIALIZE NEW PLANET OBJECTS
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
    
            # load true transit times
            true_tts = np.loadtxt(PROJECT_DIR + 'Simulations/{0}/{1}_{2}.tts'.format(RUN_ID, TARGET, npl)).swapaxes(0,1)    
    
            p.tts = true_tts[1]
            p.index = np.array(true_tts[0], dtype='int')
    
            planets.append(p)
    
        order = np.argsort(PERIODS)
    
        sorted_planets = []
        for npl in range(NPL):
            sorted_planets.append(planets[order[npl]])
    
        planets = np.copy(sorted_planets)
        
        
        # INJECT SYNTHETIC TRANSITS
        tts  = []
        inds = []
        b    = np.zeros(NPL)
        ror  = np.zeros(NPL)
        dur  = np.zeros(NPL)
    
        for npl, p in enumerate(planets):
            tts.append(p.tts)
            inds.append(p.index)
            b[npl] = p.impact
            ror[npl] = np.sqrt(p.depth)
            dur[npl] = p.duration
    
        starrystar = exo.LimbDarkLightCurve([U1,U2])
        orbit = exo.orbits.TTVOrbit(transit_times=tts, transit_inds=inds, b=b, ror=ror, duration=dur)
    
        for i, lcd in enumerate(lc_data):
            light_curve = starrystar.get_light_curve(orbit=orbit, r=ror, t=lcd.time, oversample=long_cadence_oversample, texp=lcit)
            model_flux = pm.math.sum(light_curve, axis=-1) + T.ones(len(lcd.time))
    
            lcd.flux *= model_flux.eval()
    
        for i, scd in enumerate(sc_data):
            light_curve = starrystar.get_light_curve(orbit=orbit, r=ror, t=scd.time)
            model_flux = pm.math.sum(light_curve, axis=-1) + T.ones(len(scd.time))
    
            scd.flux *= model_flux.eval()
            
        lc_raw_sim_data = deepcopy(lc_data)
        sc_raw_sim_data = deepcopy(sc_data)
    
    
    # # #########################
    # # ----- 1ST DETRENDING -----
    # # #########################
    
    
    print("\nDetrending lightcurves (1st pass)...\n")
    
    
    # ## Detrend the lightcurves
    
    
    # array to hold dominant oscillation period for each quarter
    oscillation_period_by_quarter = np.ones(18)*np.nan
    
    # long cadence data
    min_period = 1.0
    
    for i, lcd in enumerate(lc_data):
        qmask = lk.KeplerQualityFlags.create_quality_mask(lcd.quality, bitmask='default')
        lcd.remove_flagged_cadences(qmask)
        
        # make transit mask
        lcd.mask = np.zeros(len(lcd.time), dtype='bool')
        for npl, p in enumerate(planets):
            lcd.mask += detrend.make_transitmask(lcd.time, p.tts, np.max([1/24,1.5*p.duration]))
        
        lcd.clip_outliers(kernel_size=13, sigma_upper=5, sigma_lower=5, mask=lcd.mask)
        lcd.clip_outliers(kernel_size=13, sigma_upper=5, sigma_lower=1000, mask=None)
        
        # identify primary oscillation period
        ls_estimate = LombScargle(lcd.time, lcd.flux)
        xf, yf = ls_estimate.autopower(minimum_frequency=1/(lcd.time.max()-lcd.time.min()), 
                                       maximum_frequency=1/min_period)
        
        peak_freq = xf[np.argmax(yf)]
        peak_per  = np.max([1./peak_freq, 1.001*min_period])
        
        oscillation_period_by_quarter[lcd.quarter[0]] = peak_per
        
        
    # short cadence data
    min_period = 1.0
    
    for i, scd in enumerate(sc_data):
        qmask = lk.KeplerQualityFlags.create_quality_mask(scd.quality, bitmask='default')
        scd.remove_flagged_cadences(qmask)
        
        # make transit mask
        scd.mask = np.zeros(len(scd.time), dtype='bool')
        for npl, p in enumerate(planets):
            scd.mask += detrend.make_transitmask(scd.time, p.tts, np.max([1/24,1.5*p.duration]))
        
        scd.clip_outliers(kernel_size=13, sigma_upper=5, sigma_lower=5, mask=scd.mask)
        scd.clip_outliers(kernel_size=13, sigma_upper=5, sigma_lower=1000, mask=None)
        
        # identify primary oscillation period
        ls_estimate = LombScargle(scd.time, scd.flux)
        xf, yf = ls_estimate.autopower(minimum_frequency=1/(scd.time.max()-scd.time.min()), 
                                       maximum_frequency=1/min_period)
        
        peak_freq = xf[np.argmax(yf)]
        peak_per  = np.max([1./peak_freq, 1.001*min_period])
        
        oscillation_period_by_quarter[scd.quarter[0]] = peak_per
        
        
    # seasonal approach assumes both stellar and instrumental effects are present
    oscillation_period_by_season = np.zeros((4,2))
    
    for i in range(4):
        oscillation_period_by_season[i,0] = np.nanmedian(oscillation_period_by_quarter[i::4])
        oscillation_period_by_season[i,1] = astropy.stats.mad_std(oscillation_period_by_quarter[i::4], ignore_nan=True)
    
    
    # detrend long cadence data
    break_tolerance = np.max([int(DURS.min()/lcit*5/2), 13])
    min_period = 1.0
    
    for i, lcd in enumerate(lc_data):
        print("QUARTER {}".format(lcd.quarter[0]))
        
        nom_per = oscillation_period_by_season[lcd.quarter[0] % 4][0]
        
        try:
            lcd = detrend.flatten_with_gp(lcd, break_tolerance, min_period, nominal_period=nom_per, verbose=VERBOSE)
        except LinAlgError:
            warnings.warn("Initial detrending model failed...attempting to refit without exponential ramp component")
            try:
                lcd = detrend.flatten_with_gp(lcd, break_tolerance, min_period, nominal_period=nom_per, 
                                              correct_ramp=False, verbose=VERBOSE)
            except LinAlgError:
                warnings.warn("Detrending with RotationTerm failed...attempting to detrend with SHOTerm")
                lcd = detrend.flatten_with_gp(lcd, break_tolerance, min_period, nominal_period=nom_per, 
                                              kterm="SHOTerm", correct_ramp=False, verbose=VERBOSE)
                         
    if len(lc_data) > 0:
        lc = detrend.stitch(lc_data)
    else:
        lc = None
    
    
    # detrend short cadence data
    break_tolerance = np.max([int(DURS.min()/(SCIT/3600/24)*5/2), 91])
    min_period = 1.0
    
    for i, scd in enumerate(sc_data):
        print("QUARTER {}".format(scd.quarter[0]))
        
        nom_per = oscillation_period_by_season[scd.quarter[0] % 4][0]
    
        try:
            scd = detrend.flatten_with_gp(scd, break_tolerance, min_period, nominal_period=nom_per, verbose=VERBOSE)
        except LinAlgError:
            warnings.warn("Initial detrending model failed...attempting to refit without exponential ramp component")
            try:
                scd = detrend.flatten_with_gp(scd, break_tolerance, min_period, nominal_period=nom_per, 
                                              correct_ramp=False, verbose=VERBOSE)
            except LinAlgError:
                warnings.warn("Detrending with RotationTerm failed...attempting to detrend with SHOTerm")
                scd = detrend.flatten_with_gp(scd, break_tolerance, min_period, nominal_period=nom_per, 
                                              kterm="SHOTerm", correct_ramp=False, verbose=VERBOSE)
                
    if len(sc_data) > 0:
        sc = detrend.stitch(sc_data)
    else:
        sc = None
    
    
    # ## Make wide masks that track each planet individually
    # #### These masks have width 2.5 transit durations, which is probably wider than the masks used for detrending
    
    
    if sc is not None:
        sc_mask = np.zeros((NPL,len(sc.time)),dtype='bool')
        for npl, p in enumerate(planets):
            sc_mask[npl] = detrend.make_transitmask(sc.time, p.tts, np.max([2/24,2.5*p.duration]))
            
        sc.mask = sc_mask.sum(axis=0) > 0
    
    else:
        sc_mask = None
    
        
    if lc is not None:
        lc_mask = np.zeros((NPL,len(lc.time)),dtype='bool')
        for npl, p in enumerate(planets):
            lc_mask[npl] = detrend.make_transitmask(lc.time, p.tts, np.max([2/24,2.5*p.duration]))
            
        lc.mask = lc_mask.sum(axis=0) > 0
    
    else:
        lc_mask = None
    
    
    # ## Flag high quality transits (quality = 1)
    # #### Good transits must have  at least 50% photometry coverage in/near transit
    
    
    for npl, p in enumerate(planets):
        count_expect_lc = np.max([1,int(np.floor(p.duration/lcit))])
        count_expect_sc = np.max([15,int(np.floor(p.duration/scit))])
            
        quality = np.zeros(len(p.tts), dtype='bool')
        
        for i, t0 in enumerate(p.tts):
            
            if sc is not None:
                in_sc = np.abs(sc.time - t0)/p.duration < 0.5
                near_sc = np.abs(sc.time - t0)/p.duration < 1.5
                
                qual_in = np.sum(in_sc) > 0.5*count_expect_sc
                qual_near = np.sum(near_sc) > 1.5*count_expect_sc
                
                quality[i] += qual_in*qual_near
            
            
            if lc is not None:
                in_lc = np.abs(lc.time - t0)/p.duration < 0.5
                near_lc = np.abs(lc.time - t0)/p.duration < 1.5
                
                qual_in = np.sum(in_lc) > 0.5*count_expect_lc
                qual_near = np.sum(near_lc) > 1.5*count_expect_lc
                
                quality[i] += qual_in*qual_near
                
        p.quality = np.copy(quality)
    
    
    # ## Flag transits that overlap
    
    
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
    
    
    # ## Count up transits and calculate initial fixed transit times
    
    
    num_transits = np.zeros(NPL)
    transit_inds = []
    fixed_tts = []
    
    for npl, p in enumerate(planets):
        transit_inds.append(np.array((p.index - p.index.min())[p.quality], dtype='int'))
        fixed_tts.append(np.copy(p.tts)[p.quality])
        
        num_transits[npl] = len(transit_inds[npl])
        transit_inds[npl] -= transit_inds[npl].min()
    
    
    # ## Grab data near transits
    
    
    # go quarter-by-quarter
    all_time = [None]*18
    all_flux = [None]*18
    all_error = [None]*18
    all_dtype = ['none']*18
    
    lc_flux = []
    sc_flux = []
    
    for q in range(18):
        if sc is not None:
            if np.isin(q, sc.quarter):
                use = (sc.mask)*(sc.quarter == q)
    
                if np.sum(use) > 45:
                    all_time[q] = sc.time[use]
                    all_flux[q] = sc.flux[use]
                    all_error[q] = sc.error[use]
                    all_dtype[q] = 'short'
    
                    sc_flux.append(sc.flux[use])
                    
                else:
                    all_dtype[q] = 'short_no_transits'
    
        
        if lc is not None:
            if np.isin(q, lc.quarter):
                use = (lc.mask)*(lc.quarter == q)
    
                if np.sum(use) > 5:
                    all_time[q] = lc.time[use]
                    all_flux[q] = lc.flux[use]
                    all_error[q] = lc.error[use]
                    all_dtype[q] = 'long'
    
                    lc_flux.append(lc.flux[use])
                    
                else:
                    all_dtype[q] = 'long_no_transits'
                    
    # check which quarters have coverage
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
            
    # set oversampling factors and expoure times
    oversample = np.zeros(18, dtype='int')
    texp = np.zeros(18)
    
    oversample[np.array(all_dtype)=='short'] = 1
    oversample[np.array(all_dtype)=='long'] = long_cadence_oversample
    
    texp[np.array(all_dtype)=='short'] = scit
    texp[np.array(all_dtype)=='long'] = lcit
    
    
    # ## Pull basic transit parameters
    
    
    periods = np.zeros(NPL)
    epochs  = np.zeros(NPL)
    depths  = np.zeros(NPL)
    durs    = np.zeros(NPL)
    impacts = np.zeros(NPL)
    
    for npl, p in enumerate(planets):
        periods[npl] = p.period
        epochs[npl]  = p.epoch
        depths[npl]  = p.depth
        durs[npl]    = p.duration
        impacts[npl] = p.impact
    
    
    # ## Define Legendre polynomials
    
    
    # use Legendre polynomials over transit times for better orthogonality; "x" is in the range (-1,1)
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
    
    
    print("")
    print("cumulative runtime = ", int(timer() - global_start_time), "s")
    print("")
    
    
    # # ############################
    # # ----- LIGHTCURVE FITTING -----
    # # ############################
    
    # ## Fit transit SHAPE model
    
    
    print('\nFitting transit SHAPE model...\n')
    
    
    with pm.Model() as shape_model:
        # planetary parameters
        log_r = pm.Uniform("log_r", lower=np.log(1e-5), upper=np.log(0.99), shape=NPL, testval=np.log(np.sqrt(depths)))
        r = pm.Deterministic("r", T.exp(log_r))    
        b = pm.Uniform("b", lower=0., upper=1., shape=NPL)
        
        log_dur = pm.Normal("log_dur", mu=np.log(durs), sd=5.0, shape=NPL)
        dur = pm.Deterministic("dur", T.exp(log_dur))
        
        # polynomial TTV parameters    
        C0 = pm.Normal("C0", mu=0.0, sd=durs/2, shape=NPL)
        C1 = pm.Normal("C1", mu=0.0, sd=durs/2, shape=NPL)
        
        transit_times = []
        for npl in range(NPL):
            transit_times.append(pm.Deterministic("tts_{0}".format(npl), 
                                                  fixed_tts[npl] + C0[npl]*Leg0[npl] + C1[npl]*Leg1[npl]))
        
        # set up stellar model and planetary orbit
        starrystar = exo.LimbDarkLightCurve([U1,U2])
        orbit = exo.orbits.TTVOrbit(transit_times=transit_times, transit_inds=transit_inds, 
                                    b=b, ror=r, duration=dur)
        
        # track period and epoch
        T0 = pm.Deterministic("T0", orbit.t0)
        P  = pm.Deterministic("P", orbit.period)
        
        # nuissance parameters
        flux0 = pm.Normal("flux0", mu=np.mean(good_flux), sd=np.std(good_flux), shape=len(quarters))
        log_jit = pm.Normal("log_jit", mu=np.log(np.var(good_flux)/10), sd=10, shape=len(quarters))
        
        # now evaluate the model for each quarter
        light_curves = [None]*nq
        model_flux = [None]*nq
        flux_err = [None]*nq
        obs = [None]*nq
        
        for j, q in enumerate(quarters):
            # calculate light curves
            light_curves[j] = starrystar.get_light_curve(orbit=orbit, r=r, t=all_time[q], 
                                                         oversample=oversample[q], texp=texp[q])
            
            model_flux[j] = pm.math.sum(light_curves[j], axis=-1) + flux0[j]*T.ones(len(all_time[q]))
            flux_err[j] = T.sqrt(np.mean(all_error[q])**2 + T.exp(log_jit[j]))/np.sqrt(2)
            
            obs[j] = pm.Normal("obs_{0}".format(j), 
                               mu=model_flux[j], 
                               sd=flux_err[j], 
                               observed=all_flux[q])
    
    
    with shape_model:
        shape_map = shape_model.test_point
        shape_map = pmx.optimize(start=shape_map, vars=[flux0, log_jit], progress=VERBOSE)
        shape_map = pmx.optimize(start=shape_map, vars=[b, r, dur], progress=VERBOSE)
        shape_map = pmx.optimize(start=shape_map, vars=[C0, C1], progress=VERBOSE)
        shape_map = pmx.optimize(start=shape_map, progress=VERBOSE)
    
    
    # grab transit times and ephemeris
    shape_transit_times = []
    shape_linear_ephemeris = []
    
    for npl, p in enumerate(planets):
        shape_transit_times.append(shape_map['tts_{0}'.format(npl)])
        shape_linear_ephemeris.append(shape_map['P'][npl]*transit_inds[npl] + shape_map['T0'][npl])
    
    
    # update parameter values
    periods = np.atleast_1d(shape_map['P'])
    epochs  = np.atleast_1d(shape_map['T0'])
    depths  = np.atleast_1d(get_transit_depth(shape_map['r'], shape_map['b']))
    durs    = np.atleast_1d(shape_map['dur'])
    impacts = np.atleast_1d(shape_map['b'])
    rors    = np.atleast_1d(shape_map['r'])
    
    for npl, p in enumerate(planets):
        p.period   = periods[npl]
        p.epoch    = epochs[npl]
        p.depth    = depths[npl]
        p.duration = durs[npl]
        p.impact   = impacts[npl]
    
    
    print("")
    print("cumulative runtime = ", int(timer() - global_start_time), "s")
    print("")
    
    
    # ## Fit SLIDE TTVs
    
    
    print('\nFitting TTVs..\n')
    
    
    # get list of quarters with observations
    if lc is not None:
        lc_quarters = np.unique(lc.quarter)
    else:
        lc_quarters = np.array([], dtype='int')
        
    if sc is not None:
        sc_quarters = np.unique(sc.quarter)
    else:
        sc_quarters = np.array([], dtype='int')
        
    quarters = np.sort(np.hstack([lc_quarters, sc_quarters]))
    seasons = np.sort(np.unique(quarters % 4))
    
    # get list of threshold times between quarters
    thresh = np.zeros(len(quarters)+1)
    thresh[0] = TIME_START
    
    for j, q in enumerate(quarters):
        if np.isin(q, sc_quarters):
            thresh[j+1] = sc.time[sc.quarter == q].max()
        if np.isin(q, lc_quarters):
            thresh[j+1] = lc.time[lc.quarter == q].max()
            
    thresh[0] -= 1.0
    thresh[-1] += 1.0
    
    # track individual transits
    transit_quarter = [None]*NPL
    
    for npl in range(NPL):
        tts = shape_transit_times[npl]
        transit_quarter[npl] = np.zeros(len(tts), dtype='int')
    
        for j, q in enumerate(quarters):
            transit_quarter[npl][(tts >= thresh[j])*(tts<thresh[j+1])] = q
    
    
    slide_transit_times = []
    slide_error = []
    
    for npl, p in enumerate(planets):
        print("\nPLANET", npl)
        
        slide_transit_times.append([])
        slide_error.append([])
        
        # create template transit
        starrystar = exo.LimbDarkLightCurve([U1,U2])
        orbit = exo.orbits.KeplerianOrbit(t0=0, period=p.period, b=p.impact, ror=rors[npl], duration=p.duration)
        
        slide_offset  = 1.0
        template_time = np.arange(0, (0.02+p.duration)*(slide_offset+1.6), scit)
        template_time = np.hstack([-template_time[:-1][::-1],template_time])    
        template_flux = 1.0 + starrystar.get_light_curve(orbit=orbit, r=rors[npl], t=template_time, oversample=1).sum(axis=-1).eval()
      
        # empty lists to hold new transit time and uncertainties
        tts = -99*np.ones_like(shape_transit_times[npl])
        err = -99*np.ones_like(shape_transit_times[npl])
    
        
        for i, t0 in enumerate(shape_transit_times[npl]):
            #print(i, np.round(t0,2))
            if ~p.overlap[p.quality][i]:
                # identify quarter
                q = transit_quarter[npl][i]
                
                # set exposure time and oversample factor
                if all_dtype[q] == 'long':
                    exptime = lcit
                    texp_offsets = np.linspace(-exptime/2., exptime/2., oversample[q])
                elif all_dtype[q] == 'short':
                    exptime = scit
                    texp_offsets = np.array([0.])
                else:
                    raise ValueError("data cadence expected to be 'long' or 'short'")
            
                # grab data near each non-overlapping transit
                use = np.abs(all_time[q] - t0)/p.duration < 2.5
                mask = np.abs(all_time[q] - t0)/p.duration < 1.0
                
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
                e_ = np.ones_like(f_)*np.std(f_[~m_])
                
                            
                # slide along transit time vector and calculate chisq
                gridstep  = scit/1.618/3
                tc_vector = np.arange(0, p.duration*slide_offset, gridstep)
                tc_vector = t0 + np.hstack([-tc_vector[:-1][::-1],tc_vector])    
                chisq_vector = np.zeros_like(tc_vector)
    
                for j, tc in enumerate(tc_vector):
                    y_ = np.interp(t_supersample-tc, template_time, template_flux)
                    y_ = bin_data(t_supersample, y_, exptime, bin_centers=t_)[1]
    
                    chisq_vector[j] = np.sum((f_ - y_)**2/e_**2)
                    
                    
                # grab points near minimum chisq
                delta_chisq = 1.0
                
                loop = True
                while loop:
                    # incrememnt delta_chisq and find minimum
                    delta_chisq *= 2
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
                    if len(x2fit) >= 7:
                        loop = False
                        
                    if delta_chisq >= 16:
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
                do_plots = False
                if do_plots:
                    if ~np.isnan(tts[i]):
    
                        fig, ax = plt.subplots(1,2, figsize=(10,3))
    
                        ax[0].plot(t_-tts[i], f_, 'ko')
                        ax[0].plot((t_-tts[i])[m_], f_[m_], "o", c='C{0}'.format(npl))
                        ax[0].plot(template_time, template_flux, c='C{0}'.format(npl), lw=2)
    
                        ax[1].plot(tcfit, x2fit, 'ko')
                        ax[1].plot(tcfit, quadfit, c='C{0}'.format(npl), lw=3)
                        ax[1].axvline(tts[i], color='k', ls="--", lw=2)
                        
                        if IPLOT:
                            plt.show()
                        else:
                            plt.close()
                        
            else:
                #print("OVERLAPPING TRANSITS")
                tts[i] = np.nan
                err[i] = np.nan
                
                
        slide_transit_times[npl] = np.copy(tts)
        slide_error[npl] = np.copy(err)
    
    
    # flag transits for which the slide method failed
    for npl, p in enumerate(planets):
        bad = np.isnan(slide_transit_times[npl]) + np.isnan(slide_error[npl])
        bad += slide_error[npl] > 8*np.nanmedian(slide_error[npl])
        
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
        sc_map_mask = np.zeros((NPL,len(sc.time)),dtype='bool')
        for npl, p in enumerate(planets):
            tts = slide_transit_times[npl][refit[npl]]
            sc_map_mask[npl] = detrend.make_transitmask(sc.time, tts, np.max([2/24,2.5*p.duration]))
            
        sc_map_mask = sc_map_mask.sum(axis=0) > 0
    
    else:
        sc_map_mask = None
    
    if lc is not None:
        lc_map_mask = np.zeros((NPL,len(lc.time)),dtype='bool')
        for npl, p in enumerate(planets):
            tts = slide_transit_times[npl][refit[npl]]
            lc_map_mask[npl] = detrend.make_transitmask(lc.time, tts, np.max([2/24,2.5*p.duration]))
            
        lc_map_mask = lc_map_mask.sum(axis=0) > 0
    
    else:
        lc_map_mask = None
    
    
    # grab data near transits for each quarter
    map_time = [None]*18
    map_flux = [None]*18
    map_error = [None]*18
    map_dtype = ['none']*18
    
    for q in range(18):
        if sc is not None:
            if np.isin(q, sc.quarter):
                use = (sc_map_mask)*(sc.quarter == q)
    
                if np.sum(use) > 45:
                    map_time[q] = sc.time[use]
                    map_flux[q] = sc.flux[use]
                    map_error[q] = sc.error[use]
                    map_dtype[q] = 'short'
                    
                else:
                    map_dtype[q] = 'short_no_transits'
    
        if lc is not None:
            if np.isin(q, lc.quarter):
                use = (lc_map_mask)*(lc.quarter == q)
    
                if np.sum(use) > 5:
                    map_time[q] = lc.time[use]
                    map_flux[q] = lc.flux[use]
                    map_error[q] = lc.error[use]
                    map_dtype[q] = 'long'
                    
                else:
                    map_dtype[q] = 'long_no_transits'
                    
    map_quarters = np.arange(18)[(np.array(map_dtype) == 'short') + (np.array(map_dtype) == 'long')]
    
    
    with pm.Model() as indep_model:
        # transit times
        tt_offset = []
        map_tts  = []
        map_inds = []
        
        for npl in range(NPL):
            use = np.copy(refit[npl])
            
            tt_offset.append(pm.Normal("tt_offset_{0}".format(npl), mu=0, sd=1, shape=np.sum(use)))
    
            map_tts.append(pm.Deterministic("tts_{0}".format(npl),
                                            shape_transit_times[npl][use] + tt_offset[npl]*durs[npl]/3))
            
            map_inds.append(transit_inds[npl][use])
            
        # set up stellar model and planetary orbit
        starrystar = exo.LimbDarkLightCurve([U1,U2])
        orbit  = exo.orbits.TTVOrbit(transit_times=map_tts, transit_inds=map_inds, 
                                     period=periods, b=impacts, ror=rors, duration=durs)
        
        # nuissance parameters
        flux0 = pm.Normal("flux0", mu=np.mean(good_flux), sd=np.std(good_flux), shape=len(map_quarters))
        log_jit = pm.Normal("log_jit", mu=np.log(np.var(good_flux)/10), sd=10, shape=len(map_quarters))
            
        # now evaluate the model for each quarter
        light_curves = [None]*len(map_quarters)
        model_flux = [None]*len(map_quarters)
        flux_err = [None]*len(map_quarters)
        obs = [None]*len(map_quarters)
        
        for j, q in enumerate(map_quarters):
            # calculate light curves
            light_curves[j] = starrystar.get_light_curve(orbit=orbit, r=rors, t=map_time[q], 
                                                         oversample=oversample[q], texp=texp[q])
            
            model_flux[j] = pm.math.sum(light_curves[j], axis=-1) + flux0[j]*T.ones(len(map_time[q]))
            flux_err[j] = T.sqrt(np.mean(map_error[q])**2 + T.exp(log_jit[j]))/np.sqrt(2)
            
            obs[j] = pm.Normal("obs_{0}".format(j), 
                               mu=model_flux[j], 
                               sd=flux_err[j], 
                               observed=map_flux[q])
    
    
    with indep_model:
        indep_map = indep_model.test_point
        indep_map = pmx.optimize(start=indep_map, vars=[flux0, log_jit], progress=VERBOSE)
        
        for npl in range(NPL):
            indep_map = pmx.optimize(start=indep_map, vars=[tt_offset[npl]], progress=VERBOSE)
            
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
            indep_transit_times[npl][replace] = indep_map["tts_{0}".format(npl)]
    
        pfit = poly.polyfit(transit_inds[npl], indep_transit_times[npl], 1)
    
        indep_linear_ephemeris.append(poly.polyval(transit_inds[npl], pfit))
        full_indep_linear_ephemeris.append(poly.polyval(p.index, pfit))
    
        if np.any(replace):
            indep_error[npl][replace] = astropy.stats.mad_std(indep_transit_times[npl][~replace] 
                                                              - indep_linear_ephemeris[npl][~replace])
    
    
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
        yomc  = indep_transit_times[npl] - indep_linear_ephemeris[npl]
    
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
                
                df_min = 1/(indep_linear_ephemeris[i].max() - indep_linear_ephemeris[i].min())
                
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
    
    
    omc_pers = []
    
    for npl in range(NPL):
        print("\nPLANET", npl)
        
        # roughly model OMC based on single frequency sinusoid (if found)
        if omc_freqs[npl] is not None:
            print("periodic signal found at P =", int(1/omc_freqs[npl]), "d")
            
            # store frequency
            omc_pers.append(1/omc_freqs[npl])
            
            # grab data and plot
            xtime = indep_linear_ephemeris[npl]
            yomc  = indep_transit_times[npl] - indep_linear_ephemeris[npl]
            LS = LombScargle(xtime, yomc)
            
            plt.figure(figsize=(12,3))
            plt.plot(xtime, yomc*24*60, "o", c="lightgrey")
            plt.plot(xtime, LS.model(xtime, omc_freqs[npl])*24*60, c="C{0}".format(npl), lw=3)
            if IPLOT:
                plt.show()
            else:
                plt.close()
        
        else:
            print("no sigificant periodic component found")
            omc_pers.append(2*(indep_linear_ephemeris[npl].max()-indep_linear_ephemeris[npl].min()))
    
    
    # ## Determine best OMC model
    
    
    print("...running model selection routine")
    
    quick_transit_times = []
    full_quick_transit_times = []
    
    outlier_prob = []
    outlier_class = []
    
    for npl, p in enumerate(planets):
        print("\nPLANET", npl)
        
        # grab data
        xtime = indep_linear_ephemeris[npl]
        yomc  = indep_transit_times[npl] - indep_linear_ephemeris[npl]
        
        if len(yomc) > 16:
            ymed = boxcar_smooth(ndimage.median_filter(yomc, size=5, mode='mirror'), winsize=5)
        else:
            ymed = np.median(yomc)
        
        if len(yomc) > 4:
            out = np.abs(yomc-ymed)/astropy.stats.mad_std(yomc-ymed) > 5.0
        else:
            out = np.zeros(len(yomc), dtype='bool')
            
        # compare various models
        aiclist = []
        biclist = []
        fgplist = []
        outlist = []
        plylist = []
        
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
        
        for polyorder in range(min_polyorder, max_polyorder+1):
            if polyorder == -1:
                omc_model = omc.matern32_model(xtime[~out], yomc[~out], xtime)
            elif polyorder == 0:
                omc_model = omc.sin_model(xtime[~out], yomc[~out], omc_pers[npl], xtime)
            elif polyorder >= 1:
                omc_model = omc.poly_model(xtime[~out], yomc[~out], polyorder, xtime)
    
            with omc_model:
                omc_map = omc_model.test_point
                omc_map = pmx.optimize(start=omc_map, progress=VERBOSE)
                omc_trace = pmx.sample(tune=8000, draws=2000, start=omc_map, chains=2, target_accept=0.95, progressbar=VERBOSE)
    
            omc_trend = np.nanmedian(omc_trace['pred'], 0)
            residuals = yomc - omc_trend
            
            # flag outliers via mixture model of the residuals
            mix_model = omc.mix_model(residuals)
    
            with mix_model:
                mix_trace = pmx.sample(tune=8000, draws=2000, chains=1, target_accept=0.95, progressbar=VERBOSE)
    
            loc = np.nanmedian(mix_trace['mu'], axis=0)
            scales = np.nanmedian(1/np.sqrt(mix_trace['tau']), axis=0)
    
            fg_prob, bad = omc.flag_outliers(residuals, loc, scales)
            
            while np.sum(bad)/len(bad) > 0.3:
                thresh = np.max(fg_prob[bad])
                bad = fg_prob < thresh
                
            fgplist.append(fg_prob)
            outlist.append(bad)
            plylist.append(polyorder)
            
            print("{0} outliers found out of {1} transit times ({2}%)".format(np.sum(bad), len(bad), 
                                                                              np.round(100.*np.sum(bad)/len(bad),1)))
            
            plt.figure(figsize=(12,3))
            plt.plot(xtime, yomc*24*60, 'o', c='lightgrey')
            plt.plot(xtime[bad], yomc[bad]*24*60, 'rx')
            plt.plot(xtime, omc_trend*24*60, c='C{0}'.format(npl), lw=2)
            plt.xlabel("Time [BJKD]", fontsize=16)
            plt.ylabel("O-C [min]", fontsize=16)
            if IPLOT: 
                plt.show()
            else:
                plt.close()
                
                
            # calculate AIC & BIC
            n = len(yomc)
            
            if polyorder <= 0:
                k = 3
            else:
                k = polyorder + 1
            
            aic = n*np.log(np.sum(residuals[~bad]**2)/np.sum(~bad)) + 2*k
            bic = n*np.log(np.sum(residuals[~bad]**2)/np.sum(~bad)) + k*np.log(n)
            
            aiclist.append(aic)
            biclist.append(bic)
            
            print("AIC:", np.round(aic,1))
            print("BIC:", np.round(bic,1))
            
            
        # choose the best model and recompute
        out = outlist[np.argmin(aiclist)]
        fg_prob = fgplist[np.argmin(aiclist)]
        polyorder = plylist[np.argmin(aiclist)]
        xt_predict = full_indep_linear_ephemeris[npl]
    
        if polyorder == -1:
            omc_model = omc.matern32_model(xtime[~out], yomc[~out], xt_predict)
        elif polyorder == 0:
            omc_model = omc.sin_model(xtime[~out], yomc[~out], omc_pers[npl], xt_predict)
        elif polyorder >= 1:
            omc_model = omc.poly_model(xtime[~out], yomc[~out], polyorder, xt_predict)
    
        with omc_model:
            omc_map = omc_model.test_point
            omc_map = pmx.optimize(start=omc_map, progress=VERBOSE)
            omc_trace = pmx.sample(tune=8000, draws=2000, start=omc_map, chains=2, target_accept=0.95, progressbar=VERBOSE)
        
        omc_trend = np.nanmedian(omc_trace['pred'], 0)
        residuals = yomc - omc_trend[np.isin(xt_predict, xtime)]
        mix_model = omc.mix_model(residuals)
    
        with mix_model:
            mix_trace = pmx.sample(tune=8000, draws=2000, chains=1, target_accept=0.95, progressbar=VERBOSE)
    
        loc = np.nanmedian(mix_trace['mu'], axis=0)
        scales = np.nanmedian(1/np.sqrt(mix_trace['tau']), axis=0)
    
        fg_prob, bad = omc.flag_outliers(residuals, loc, scales)
    
        while np.sum(bad)/len(bad) > 0.3:
            thresh = np.max(fg_prob[bad])
            bad = fg_prob < thresh
        
        
        # save the final results
        full_omc_trend = np.nanmedian(omc_trace['pred'], 0)
        
        full_quick_transit_times.append(full_indep_linear_ephemeris[npl] + full_omc_trend)
        quick_transit_times.append(full_quick_transit_times[npl][transit_inds[npl]])
        
        outlier_prob.append(1-fg_prob)
        outlier_class.append(bad)
        
        plt.figure(figsize=(12,4))
        plt.scatter(xtime, yomc*24*60, c=1-fg_prob, cmap='viridis', label="MAP TTVs")
        plt.plot(xtime[bad], yomc[bad]*24*60, 'rx')
        plt.plot(full_indep_linear_ephemeris[npl], full_omc_trend*24*60, 'k', label="Quick model")
        plt.xlabel("Time [BJKD]", fontsize=20)
        plt.ylabel("O-C [min]", fontsize=20)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.legend(fontsize=14, loc='upper right')
        plt.title(TARGET, fontsize=20)
        plt.savefig(os.path.join(FIGURE_DIR, TARGET + '_ttvs_quick_{0:02d}.png'.format(npl)), bbox_inches='tight')
        if IPLOT:
            plt.show()
        else:
            plt.close()
    
    
    # ## Estimate TTV scatter w/ uncertainty buffer
    
    
    ttv_scatter = np.zeros(NPL)
    ttv_buffer  = np.zeros(NPL)
    
    for npl in range(NPL):
        # estimate TTV scatter
        ttv_scatter[npl] = astropy.stats.mad_std(indep_transit_times[npl]-quick_transit_times[npl])
    
        # based on scatter in independent times, set threshold so not even one outlier is expected
        N = len(transit_inds[npl])
        eta = np.max([3., stats.norm.interval((N-1)/N)[1]])
    
        ttv_buffer[npl] = eta*ttv_scatter[npl] + lcit
    
    
    # ## Update TTVs
    
    
    for npl, p in enumerate(planets):
        # update transit time info in Planet objects
        epoch, period = poly.polyfit(p.index, full_quick_transit_times[npl], 1)
    
        p.epoch = np.copy(epoch)
        p.period = np.copy(period)
        p.tts = np.copy(full_quick_transit_times[npl])
    
    
    print("")
    print("cumulative runtime = ", int(timer() - global_start_time), 's')
    print("")
    
    
    # ## Flag outliers based on transit model
    # #### Cadences must be flagged as outliers from BOTH the quick ttv model and the independent ttv model to be rejected
    
    
    print("\nFlagging outliers based on transit model...\n")
    
    
    res_i = []
    res_q = []
    
    for j, q in enumerate(quarters):
        # grab time and flux data
        if (all_dtype[q] != 'long') and (all_dtype[q] != 'short'):
            res_i.append(None)
            res_q.append(None)
            
        else:
            if all_dtype[q] == 'long':
                use = lc.quarter == q
                t_ = lc.time[use]
                f_ = lc.flux[use]
            elif all_dtype[q] == 'short':
                use = sc.quarter == q
                t_ = sc.time[use]
                f_ = sc.flux[use]
            else:
                raise ValueError("cadence data type must be 'short' or 'long'")
    
            # grab transit times for each planet
            wp_i = []
            tts_i = []
            inds_i = []
    
            wp_q = []
            tts_q = []
            inds_q = []
    
            for npl in range(NPL):
                itt = indep_transit_times[npl]
                qtt = quick_transit_times[npl]
    
                use_i = (itt > t_.min())*(itt < t_.max())
                use_q = (qtt > t_.min())*(qtt < t_.max())
    
                if np.sum(use_i) > 0:
                    wp_i.append(npl)
                    tts_i.append(itt[use_i])
                    inds_i.append(transit_inds[npl][use_i] - transit_inds[npl][use_i][0])
    
                if np.sum(use_q) > 0:
                    wp_q.append(npl)
                    tts_q.append(itt[use_q])
                    inds_q.append(transit_inds[npl][use_q] - transit_inds[npl][use_q][0])
    
            # first check independent transit times
            if len(tts_i) > 0:
                # set up model
                starrystar = exo.LimbDarkLightCurve([U1,U2])
                orbit  = exo.orbits.TTVOrbit(transit_times=tts_i, transit_inds=inds_i, period=list(periods[wp_i]), 
                                             b=impacts[wp_i], ror=rors[wp_i], duration=durs[wp_i])
    
                # calculate light curves
                light_curves = starrystar.get_light_curve(orbit=orbit, r=rors[wp_i], t=t_, oversample=oversample[q], texp=texp[q])
                model_flux = 1.0 + pm.math.sum(light_curves, axis=-1).eval()
    
            else:
                model_flux = np.ones_like(f_)*np.mean(f_)
    
            # calculate residuals
            res_i.append(f_ - model_flux)
    
            # then check matern transit times
            if len(tts_q) > 0:
                # set up model
                starrystar = exo.LimbDarkLightCurve([U1,U2])
                orbit  = exo.orbits.TTVOrbit(transit_times=tts_q, transit_inds=inds_q, period=list(periods[wp_q]), 
                                             b=impacts[wp_q], ror=rors[wp_q], duration=durs[wp_q])
    
                # calculate light curves
                light_curves = starrystar.get_light_curve(orbit=orbit, r=rors[wp_q], t=t_, oversample=oversample[q], texp=texp[q])
                model_flux = 1.0 + pm.math.sum(light_curves, axis=-1).eval()
    
            else:
                model_flux = np.ones_like(f_)*np.mean(f_)
    
            # calculate residuals
            res_q.append(f_ - model_flux)
    
    
    for j, q in enumerate(quarters):
        if (all_dtype[q] == 'long') or (all_dtype[q] == 'short'):
            print("\nQUARTER", q)
            res = 0.5*(res_i[j] + res_q[j])
            x_ = np.arange(len(res))
    
            bad_i = np.abs(res_i[j] - np.mean(res_i[j]))/astropy.stats.mad_std(res_i[j]) > 5.0
            bad_q = np.abs(res_q[j] - np.mean(res_q[j]))/astropy.stats.mad_std(res_q[j]) > 5.0
    
            bad = bad_i * bad_q
    
            print(" outliers rejected:", np.sum(bad))
            print(" marginal outliers:", np.sum(bad_i*~bad_q)+np.sum(~bad_i*bad_q))
    
            plt.figure(figsize=(20,3))
            plt.plot(x_, res, 'k', lw=0.5)
            plt.plot(x_[bad], res[bad], 'rx')
            plt.xlim(x_.min(), x_.max())
            if IPLOT:
                plt.show()
            else:
                plt.close()
    
    
    bad_lc = []
    bad_sc = []
    
    for q in range(18):
        if all_dtype[q] == 'long_no_transits':
            bad = np.ones(np.sum(lc.quarter == q), dtype='bool')
            bad_lc = np.hstack([bad_lc, bad])
            
        if all_dtype[q] == 'short_no_transits':
            bad = np.ones(np.sum(sc.quarter == q), dtype='bool')
            bad_sc = np.hstack([bad_sc, bad])    
        
        if (all_dtype[q] == 'short') + (all_dtype[q] == 'long'):
            j = np.where(quarters == q)[0][0]
    
            res = 0.5*(res_i[j] + res_q[j])
            x_ = np.arange(len(res))
    
            bad_i = np.abs(res_i[j] - np.mean(res_i[j]))/astropy.stats.mad_std(res_i[j]) > 5.0
            bad_q = np.abs(res_q[j] - np.mean(res_q[j]))/astropy.stats.mad_std(res_q[j]) > 5.0
    
            bad = bad_i * bad_q
    
            if all_dtype[q] == 'short':
                bad_sc = np.hstack([bad_sc, bad])
    
            if all_dtype[q] == 'long':
                bad_lc = np.hstack([bad_lc, bad])
            
    bad_lc = np.array(bad_lc, dtype='bool')
    bad_sc = np.array(bad_sc, dtype='bool')
    
    if sc is not None:
        good_cadno_sc = sc.cadno[~bad_sc]
        
    if lc is not None:
        good_cadno_lc = lc.cadno[~bad_lc]
    
    
    # # #########################
    # # ----- 2ND DETRENDING -----
    # # #########################
    
    
    print("\nResetting to raw MAST data and performing 2nd DETRENDING...\n")
    
    
    # ## Reset to raw input data
    
    
    if MISSION == 'Kepler':
        # reset LONG CADENCE data
        if lc is not None:
            lc_data = io.cleanup_lkfc(lc_raw_collection, KIC)
    
        # make sure there is at least one transit in the long cadence data
        # this shouldn't be an issue for real KOIs, but can happen for simulated data
        if np.sum(np.array(all_dtype) == 'long') == 0:
            lc_data = []
    
        lc_quarters = []
        for i, lcd in enumerate(lc_data):
            lc_quarters.append(lcd.quarter)
    
        # reset SHORT CADENCE data
        if sc is not None:
            sc_data = io.cleanup_lkfc(sc_raw_collection, KIC)
    
        # make sure there is at least one transit in the short cadence data
        # this shouldn't be an issue for real KOIs, but can happen for simulated data
        if np.sum(np.array(all_dtype) == 'short') == 0:
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
            
        
    elif MISSION == 'Simulated':
        sc_lite = deepcopy(sc_raw_sim_data)
        lc_lite = deepcopy(lc_raw_sim_data)
    
    
    # ## Remove flagged cadences
    
    
    sc_data = []
    for i, scl in enumerate(sc_lite):
        qmask = np.isin(scl.cadno, good_cadno_sc)
        
        if np.sum(qmask)/len(qmask) > 0.1:
            sc_data.append(scl.remove_flagged_cadences(qmask))
    
    lc_data = []
    for i, lcl in enumerate(lc_lite):
        qmask = np.isin(lcl.cadno, good_cadno_lc)
        
        if np.sum(qmask)/len(qmask) > 0.1:
            lc_data.append(lcl.remove_flagged_cadences(qmask))
    
    
    # ## Detrend the lightcurves
    
    
    # array to hold dominant oscillation period for each quarter
    oscillation_period_by_quarter = np.ones(18)*np.nan
    
    # long cadence data
    min_period = 1.0
    
    for i, lcd in enumerate(lc_data):
        qmask = lk.KeplerQualityFlags.create_quality_mask(lcd.quality, bitmask='default')
        lcd.remove_flagged_cadences(qmask)
        
        # make transit mask
        lcd.mask = np.zeros(len(lcd.time), dtype='bool')
        for npl, p in enumerate(planets):
            lcd.mask += detrend.make_transitmask(lcd.time, p.tts, np.max([1/24,0.5*p.duration+ttv_buffer[npl]]))
        
        lcd.clip_outliers(kernel_size=13, sigma_upper=5, sigma_lower=5, mask=lcd.mask)
        lcd.clip_outliers(kernel_size=13, sigma_upper=5, sigma_lower=1000, mask=None)
        
        # identify primary oscillation period
        ls_estimate = LombScargle(lcd.time, lcd.flux)
        xf, yf = ls_estimate.autopower(minimum_frequency=1/(lcd.time.max()-lcd.time.min()), 
                                       maximum_frequency=1/min_period)
        
        peak_freq = xf[np.argmax(yf)]
        peak_per  = np.max([1./peak_freq, 1.001*min_period])
        
        oscillation_period_by_quarter[lcd.quarter[0]] = peak_per
        
        
    # short cadence data
    min_period = 1.0
    
    for i, scd in enumerate(sc_data):
        qmask = lk.KeplerQualityFlags.create_quality_mask(scd.quality, bitmask='default')
        scd.remove_flagged_cadences(qmask)
        
        # make transit mask
        scd.mask = np.zeros(len(scd.time), dtype='bool')
        for npl, p in enumerate(planets):
            scd.mask += detrend.make_transitmask(scd.time, p.tts, np.max([1/24,0.5*p.duration+ttv_buffer[npl]]))
        
        scd.clip_outliers(kernel_size=13, sigma_upper=5, sigma_lower=5, mask=scd.mask)
        scd.clip_outliers(kernel_size=13, sigma_upper=5, sigma_lower=1000, mask=None)
        
        # identify primary oscillation period
        ls_estimate = LombScargle(scd.time, scd.flux)
        xf, yf = ls_estimate.autopower(minimum_frequency=1/(scd.time.max()-scd.time.min()), 
                                       maximum_frequency=1/min_period)
        
        peak_freq = xf[np.argmax(yf)]
        peak_per  = np.max([1./peak_freq, 1.001*min_period])
        
        oscillation_period_by_quarter[scd.quarter[0]] = peak_per
        
        
    # seasonal approach assumes both stellar and instrumental effects are present
    oscillation_period_by_season = np.zeros((4,2))
    
    for i in range(4):
        oscillation_period_by_season[i,0] = np.nanmedian(oscillation_period_by_quarter[i::4])
        oscillation_period_by_season[i,1] = astropy.stats.mad_std(oscillation_period_by_quarter[i::4], ignore_nan=True)
    
    
    # detrend long cadence data
    break_tolerance = np.max([int(DURS.min()/lcit*5/2), 13])
    min_period = 1.0
    
    for i, lcd in enumerate(lc_data):
        print("QUARTER {}".format(lcd.quarter[0]))
        
        nom_per = oscillation_period_by_season[lcd.quarter[0] % 4][0]
        
        try:
            lcd = detrend.flatten_with_gp(lcd, break_tolerance, min_period, nominal_period=nom_per, verbose=VERBOSE)
        except LinAlgError:
            warnings.warn("Initial detrending model failed...attempting to refit without exponential ramp component")
            try:
                lcd = detrend.flatten_with_gp(lcd, break_tolerance, min_period, nominal_period=nom_per, 
                                              correct_ramp=False, verbose=VERBOSE)
            except LinAlgError:
                warnings.warn("Detrending with RotationTerm failed...attempting to detrend with SHOTerm")
                lcd = detrend.flatten_with_gp(lcd, break_tolerance, min_period, nominal_period=nom_per, 
                                              kterm="SHOTerm", correct_ramp=False, verbose=VERBOSE)
    
    if len(lc_data) > 0:
        lc = detrend.stitch(lc_data)
    else:
        lc = None
    
    
    # detrend short cadence data
    break_tolerance = np.max([int(DURS.min()/(SCIT/3600/24)*5/2), 91])
    min_period = 1.0
    
    for i, scd in enumerate(sc_data):
        print("QUARTER {}".format(scd.quarter[0]))
        
        nom_per = oscillation_period_by_season[scd.quarter[0] % 4][0]
    
        try:
            scd = detrend.flatten_with_gp(scd, break_tolerance, min_period, nominal_period=nom_per, verbose=VERBOSE)
        except LinAlgError:
            warnings.warn("Initial detrending model failed...attempting to refit without exponential ramp component")
            try:
                scd = detrend.flatten_with_gp(scd, break_tolerance, min_period, nominal_period=nom_per, 
                                              correct_ramp=False, verbose=VERBOSE)
            except LinAlgError:
                warnings.warn("Detrending with RotationTerm failed...attempting to detrend with SHOTerm")
                scd = detrend.flatten_with_gp(scd, break_tolerance, min_period, nominal_period=nom_per, 
                                              kterm="SHOTerm", correct_ramp=False, verbose=VERBOSE)
                
    if len(sc_data) > 0:
        sc = detrend.stitch(sc_data)
    else:
        sc = None
    
    
    # # ##############################################
    # # ----- MAKE PLOTS, OUTPUT DATA, & CLEAN UP -----
    # # ##############################################
    
    # ## Make individual mask for where each planet transits
    # #### These masks have width 1.5 transit durations, which may be wider than the masks used for detrending
    
    
    if sc is not None:
        sc_mask = np.zeros((NPL,len(sc.time)),dtype='bool')
        for npl, p in enumerate(planets):
            sc_mask[npl] = detrend.make_transitmask(sc.time, p.tts, np.max([3/24,1.5*p.duration]))
            
        sc.mask = sc_mask.sum(axis=0) > 0
    
    else:
        sc_mask = None
    
    if lc is not None:
        lc_mask = np.zeros((NPL,len(lc.time)),dtype='bool')
        for npl, p in enumerate(planets):
            lc_mask[npl] = detrend.make_transitmask(lc.time, p.tts, np.max([3/24,1.5*p.duration]))
            
        lc.mask = lc_mask.sum(axis=0) > 0
    
    else:
        lc_mask = None
    
    
    # ## Flag high quality transits (quality = 1)
    # #### Good transits must have  at least 50% photometry coverage in/near transit
    
    
    for npl, p in enumerate(planets):
        count_expect_lc = np.max([1,int(np.floor(p.duration/lcit))])
        count_expect_sc = np.max([15,int(np.floor(p.duration/scit))])
            
        quality = np.zeros(len(p.tts), dtype='bool')
        
        for i, t0 in enumerate(p.tts):
            
            if sc is not None:
                in_sc = np.abs(sc.time - t0)/p.duration < 0.5
                near_sc = np.abs(sc.time - t0)/p.duration < 1.5
                
                qual_in = np.sum(in_sc) > 0.5*count_expect_sc
                qual_near = np.sum(near_sc) > 1.5*count_expect_sc
                
                quality[i] += qual_in*qual_near
            
            if lc is not None:
                in_lc = np.abs(lc.time - t0)/p.duration < 0.5
                near_lc = np.abs(lc.time - t0)/p.duration < 1.5
                
                qual_in = np.sum(in_lc) > 0.5*count_expect_lc
                qual_near = np.sum(near_lc) > 1.5*count_expect_lc
                
                quality[i] += qual_in*qual_near
                
        p.quality = np.copy(quality)
    
    
    # ## Flag which transits overlap (overlap = 1)
    
    
    # identify overlapping transits
    overlap = []
    
    for i in range(NPL):
        overlap.append(np.zeros(len(planets[i].tts), dtype='bool'))
        
        for j in range(NPL):
            if i != j:
                for ttj in planets[j].tts:
                    overlap[i] += np.abs(planets[i].tts - ttj)/durs.max() < 1.5
                    
        planets[i].overlap = np.copy(overlap[i])
    
    
    # ## Make phase-folded transit plots
    
    
    print("\nMaking phase-folded transit plots...\n")
    
    for npl, p in enumerate(planets):
        tts = p.tts[p.quality*~p.overlap]
        
        if len(tts) == 0:
            print("No non-overlapping high quality transits found for planet {0} (P = {1} d)".format(npl, p.period))
        
        else:
            t_folded = []
            f_folded = []
    
            # grab the data
            for t0 in tts:
                if sc is not None:
                    use = np.abs(sc.time-t0)/p.duration < 1.5
                    
                    if np.sum(use) > 0:
                        t_folded.append(sc.time[use]-t0)
                        f_folded.append(sc.flux[use])
                        
                if lc is not None:
                    use = np.abs(lc.time-t0)/p.duration < 1.5
                    
                    if np.sum(use) > 0:
                        t_folded.append(lc.time[use]-t0)
                        f_folded.append(lc.flux[use])
            
            # sort the data
            t_folded = np.hstack(t_folded)
            f_folded = np.hstack(f_folded)
    
            order = np.argsort(t_folded)
            t_folded = t_folded[order]
            f_folded = f_folded[order]
            
            # bin the data
            t_binned, f_binned = bin_data(t_folded, f_folded, p.duration/11)
            
            # set undersampling factor and plotting limits
            inds = np.arange(len(t_folded), dtype="int")
            inds = np.random.choice(inds, size=np.min([3000,len(inds)]), replace=False)
            
            ymin = 1 - 3*np.std(f_folded) - p.depth
            ymax = 1 + 3*np.std(f_folded)
            
            # plot the data
            plt.figure(figsize=(12,4))
            plt.plot(t_folded[inds]*24, f_folded[inds], '.', c='lightgrey')
            plt.plot(t_binned*24, f_binned, 'o', ms=8, color='C{0}'.format(npl), label="{0}-{1}".format(TARGET, npl))
            plt.xlim(t_folded.min()*24, t_folded.max()*24)
            plt.ylim(ymin, ymax)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.xlabel("Time from mid-transit [hrs]", fontsize=20)
            plt.ylabel("Flux", fontsize=20)
            plt.legend(fontsize=20, loc='upper right', framealpha=1)
            plt.savefig(os.path.join(FIGURE_DIR, TARGET + '_folded_transit_{0:02d}.png'.format(npl)), bbox_inches='tight')
            if IPLOT:
                plt.show()
            else:
                plt.close()
    
    
    # ## Save transit times
    
    
    for npl, p in enumerate(planets):
        keep = np.isin(quick_transit_times[npl], p.tts[p.quality])
        
        data_out  = np.vstack([transit_inds[npl][keep],
                               indep_transit_times[npl][keep],
                               quick_transit_times[npl][keep],
                               outlier_prob[npl][keep], 
                               outlier_class[npl][keep]]).swapaxes(0,1)
        
        fname_out = os.path.join(RESULTS_DIR, '{0}_{1:02d}_quick.ttvs'.format(TARGET, npl))
        np.savetxt(fname_out, data_out, fmt=('%1d', '%.8f', '%.8f', '%.8f', '%1d'), delimiter='\t')
    
    
    # ## Save detrended lightcurves
    
    
    print("\nSaving detrended lightcurves...\n")
    
    if lc is not None:
        filename = os.path.join(RESULTS_DIR, '{0}_lc_detrended.fits'.format(TARGET))
        lc.to_fits(TARGET, filename, cadence='LONG')
    else:
        print("No long cadence data")
    
    if sc is not None:
        filename = os.path.join(RESULTS_DIR, '{0}_sc_detrended.fits'.format(TARGET))
        sc.to_fits(TARGET, filename, cadence='SHORT')
    else:
        print("No short cadence data")
    
    
    # ## Exit program
    
    
    print("")
    print("+"*shutil.get_terminal_size().columns)
    print("Automated lightcurve detrending complete {0}".format(datetime.now().strftime("%d-%b-%Y at %H:%M:%S")))
    print("Total runtime = %.1f min" %((timer()-global_start_time)/60))
    print("+"*shutil.get_terminal_size().columns)
    
    
if __name__ == '__main__':
    main()