#!/usr/bin/env python
# coding: utf-8

# # Fit transit shape with nested sampling


import os
import sys
import glob
import shutil
import warnings
from datetime import datetime
from timeit import default_timer as timer

print("")
print("+"*shutil.get_terminal_size().columns)
print("ALDERAAN Transit Fitting")
print("Initialized {0}".format(datetime.now().strftime("%d-%b-%Y at %H:%M:%S")))
print("+"*shutil.get_terminal_size().columns)
print("")

# start program timer
global_start_time = timer()


# #### Parse inputs


# Automatically set inputs (when running batch scripts)
import argparse
import matplotlib as mpl

try:
    parser = argparse.ArgumentParser(description="Inputs for ALDERAAN transit fiting pipeline")
    parser.add_argument("--mission", default=None, type=str, required=True, \
                        help="Mission name; can be 'Kepler' or 'Simulated'")
    parser.add_argument("--target", default=None, type=str, required=True, \
                        help="Target name; format should be K00000 or S00000")
    parser.add_argument("--project_dir", default=None, type=str, required=True, \
                        help="Project directory for accessing lightcurve data and saving outputs")
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


# #### Set environment variables


sys.path.append(PROJECT_DIR)


# #### Build directory structure


# directories in which to place pipeline outputs for this run
RESULTS_DIR = os.path.join(PROJECT_DIR, 'Results', RUN_ID, TARGET)
FIGURE_DIR  = os.path.join(PROJECT_DIR, 'Figures', RUN_ID, TARGET)

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)


# #### Import packages


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import multiprocessing as multipro
import pickle

import astropy.stats
from   astropy.io import fits
import batman
import dynesty
from   dynesty import plotting as dyplot
import numpy.polynomial.polynomial as poly
from   scipy import stats

import aesara_theano_fallback.tensor as T
from   aesara_theano_fallback import aesara as theano
from   celerite2 import GaussianProcess
from   celerite2 import terms as GPterms

from   alderaan.constants import *
from   alderaan.detrend import make_transitmask
import alderaan.dynesty_helpers as dynhelp
from   alderaan.Ephemeris import Ephemeris
import alderaan.io as io
from   alderaan.LiteCurve import LiteCurve
from   alderaan.Planet import Planet
from   alderaan.utils import *


# flush buffer to avoid mixed outputs from progressbar
sys.stdout.flush()

# turn off FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)

# check for interactive matplotlib backends
if np.any(np.array(['agg', 'png', 'svg', 'pdf', 'ps']) == mpl.get_backend()):
    warnings.warn("Selected matplotlib backend does not support interactive plotting")
    IPLOT = False
    
# print theano compiledir cache
print("theano cache: {0}\n".format(theano.config.compiledir))


# MAIN SCRIPT BEGINS HERE
def main():    
    
    # # ################
    # # ----- DATA I/O -----
    # # ################
    
    
    print("\nLoading data...\n")
    
    
    # ## Read in planet and stellar properties
    
    
    # Read in the data from csv file
    if MISSION == 'Kepler':
        target_dict = pd.read_csv(PROJECT_DIR + 'Catalogs/' + CATALOG)
    elif MISSION == 'Simulated':
        target_dict = pd.read_csv(PROJECT_DIR + 'Simulations/{0}/{0}.csv'.format(RUN_ID))
    
    
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
    
    PERIODS = np.array(target_dict['period'], dtype='float')[use]
    DEPTHS  = np.array(target_dict['depth'], dtype='float')[use]/1e6
    DURS    = np.array(target_dict['duration'], dtype='float')[use]
    
    if MISSION == 'Kepler':
        DURS /= 24.  # [hrs] --> [days]
    
    U1 = np.array(target_dict['limbdark_1'], dtype='float')[use]
    U2 = np.array(target_dict['limbdark_2'], dtype='float')[use]
    
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
    if np.any(np.isnan(DEPTHS)):  raise ValueError("NaN values found in input catalog")
    if np.any(np.isnan(DURS)):    raise ValueError("NaN values found in input catalog")
        
    # sort planet parameters by period
    order = np.argsort(PERIODS)
    
    PERIODS = PERIODS[order]
    DEPTHS  = DEPTHS[order]
    DURS    = DURS[order]
    
    
    # ## Read in filtered lightcurves
    # #### These can be generated by running the script "analyze_autocorrelated_noise.py"
    
    
    if os.path.exists(os.path.join(RESULTS_DIR , '{0}_lc_filtered.fits'.format(TARGET))):
        lc = io.load_detrended_lightcurve(os.path.join(RESULTS_DIR, '{0}_lc_filtered.fits'.format(TARGET)))
        lc.season = lc.quarter % 4
    else:
        lc = None
        
    if os.path.exists(os.path.join(RESULTS_DIR , '{0}_sc_filtered.fits'.format(TARGET))):
        sc = io.load_detrended_lightcurve(os.path.join(RESULTS_DIR, '{0}_sc_filtered.fits'.format(TARGET)))
        sc.season = sc.quarter % 4
    else:
        sc = None
    
    
    # ## Read in quick transit times
    # #### These can be generated by running the script "detrend_and_estimate_ttvs.py"
    
    
    # transit times
    epochs = np.zeros(NPL)
    periods = np.zeros(NPL)
    ephemeris = [None]*NPL
    
    transit_inds = []
    indep_transit_times = []
    quick_transit_times = []
    
    for npl in range(NPL):
        # read in transit time data
        fname_in = os.path.join(RESULTS_DIR, '{0}_{1:02d}_quick.ttvs'.format(TARGET, npl))
        data_in  = np.genfromtxt(fname_in)
        
        transit_inds.append(np.array(data_in[:,0], dtype='int'))
        indep_transit_times.append(np.array(data_in[:,1], dtype='float'))
        quick_transit_times.append(np.array(data_in[:,2], dtype='float'))
        
        # make sure transits are zero-indexed
        transit_inds[npl] -= transit_inds[npl][0]
        
        # do a quick fit to get a linear ephemeris
        pfit = poly.polyfit(transit_inds[npl], quick_transit_times[npl], 1)
        
        epochs[npl] = pfit[0]
        periods[npl] = pfit[1]
        ephemeris[npl] = poly.polyval(transit_inds[npl], pfit)
    
        
    # calculate centered transit indexes
    centered_transit_inds = [None]*NPL
    for npl in range(NPL):
        centered_transit_inds[npl] = (transit_inds[npl] - transit_inds[npl][-1]//2)
        
        
    if IPLOT:
        fig, axes = plt.subplots(NPL, figsize=(12,3*NPL))
        if NPL == 1: axes = [axes]
    
        for npl in range(NPL):
            xtime = ephemeris[npl]
            yomc_i = (indep_transit_times[npl] - ephemeris[npl])*24*60
            yomc_q = (quick_transit_times[npl] - ephemeris[npl])*24*60
    
            axes[npl].plot(xtime, yomc_i, 'o', c='lightgrey')
            axes[npl].plot(xtime, yomc_q, lw=2, c='C{0}'.format(npl))
            axes[npl].set_ylabel('O-C [min]', fontsize=20)
        
        axes[NPL-1].set_xlabel('Time [BJKD]', fontsize=20)
    
        plt.show()
    
    
    # # ####################
    # # --- PRELIMINARIES ---
    # # ####################
    
    
    print("\nRunning preliminaries...\n")
    
    
    # ## Establish time baseline
    
    
    time_min = []
    time_max = []
    
    if sc is not None:
        time_min.append(sc.time.min())
        time_max.append(sc.time.max()) 
    
    if lc is not None:
        time_min.append(lc.time.min())
        time_max.append(lc.time.max())     
    
    TIME_START = np.min(time_min)
    TIME_END   = np.max(time_max)
    
    # put epochs in range (TIME_START, TIME_START + PERIOD)
    for npl in range(NPL):
        if epochs[npl] < TIME_START:
            adj = 1 + (TIME_START - epochs[npl])//periods[npl]
            epochs[npl] += adj*periods[npl]        
            
        if epochs[npl] > (TIME_START + periods[npl]):
            adj = (epochs[npl] - TIME_START)//periods[npl]
            epochs[npl] -= adj*periods[npl]
    
    
    # ## Identify overlapping transits
    
    
    overlap = []
    
    for i in range(NPL):
        overlap.append(np.zeros(len(ephemeris[i]), dtype='bool'))
        
        for j in range(NPL):
            if i != j:
                for tt in ephemeris[j]:
                    overlap[i] += np.abs(ephemeris[i] - tt) < (DURS[i] + DURS[j] + lcit)
    
    
    # ## Track which quarter each transit falls in
    
    
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
        tts = ephemeris[npl]
        transit_quarter[npl] = np.zeros(len(tts), dtype='int')
    
        for j, q in enumerate(quarters):
            transit_quarter[npl][(tts >= thresh[j])*(tts<thresh[j+1])] = q
    
    
    # ## Make transit masks
    
    
    if sc is not None:
        sc_mask = np.zeros((NPL,len(sc.time)), dtype='bool')
        for npl in range(NPL):
            sc_mask[npl] = make_transitmask(sc.time, quick_transit_times[npl], masksize=np.max([1/24,1.5*DURS[npl]]))
            
    if lc is not None:
        lc_mask = np.zeros((NPL,len(lc.time)), dtype='bool')
        for npl in range(NPL):
            lc_mask[npl] = make_transitmask(lc.time, quick_transit_times[npl], masksize=np.max([1/24,1.5*DURS[npl]]))
    
    
    # ## Grab data near transits
    
    
    # grab data near transits for each quarter
    all_time = [None]*18
    all_flux = [None]*18
    all_error = [None]*18
    all_mask  = [None]*18
    all_dtype = ['none']*18
    
    for q in range(18):
        if sc is not None:
            if np.isin(q, sc.quarter):
                use = (sc_mask.sum(0) != 0)*(sc.quarter == q)
    
                if np.sum(use) > 45:
                    all_time[q] = sc.time[use]
                    all_flux[q] = sc.flux[use]
                    all_error[q] = sc.error[use]
                    all_mask[q]  = sc_mask[:,use]
                    all_dtype[q] = 'short'
                    
                else:
                    all_dtype[q] = 'short_no_transits'
    
    
        if lc is not None:
            if np.isin(q, lc.quarter):
                use = (lc_mask.sum(0) != 0)*(lc.quarter == q)
    
                if np.sum(use) > 3:
                    all_time[q] = lc.time[use]
                    all_flux[q] = lc.flux[use]
                    all_error[q] = lc.error[use]
                    all_mask[q]  = lc_mask[:,use]
                    all_dtype[q] = 'long'
                    
                else:
                    all_dtype[q] = 'long_no_transits'
    
    
    # ## Track mean, variance, oversampling factors, and exposure times
    
    
    # track mean and variance of each quarter
    mean_by_quarter = np.ones(18)*np.nan
    var_by_quarter = np.ones(18)*np.nan
    
    for q in range(18):
        if sc is not None:
            if np.isin(q, sc.quarter):
                mean_by_quarter[q] = np.mean(sc.flux[sc.quarter == q])
                var_by_quarter[q] = np.var(sc.flux[sc.quarter == q])
                
        if lc is not None:
            if np.isin(q, lc.quarter):
    
                mean_by_quarter[q] = np.mean(lc.flux[lc.quarter == q])
                var_by_quarter[q] = np.var(lc.flux[lc.quarter == q])
    
    
    # determine best oversampling factor following Kipping 2010
    long_cadence_oversample = [None]*NPL
    
    for npl in range(NPL):
        # rough ingress/egress timescale estimate following Winn 2010
        ror = np.sqrt(DEPTHS[npl])
        tau = 13*(PERIODS[npl]/365.25)**(1/3) * ror / 24
        
        # set sigma so binning error is < 0.1% of photometric uncertainty
        sigma = np.mean(lc.error/lc.flux) * 0.04
        
        N = int(np.ceil(np.sqrt((DEPTHS[npl]/tau) * (lcit/8/sigma))))
        N = N + (N % 2 + 1)
        
        long_cadence_oversample[npl] = np.min([np.max([N,7]),29])
        
    long_cadence_oversample = np.max(long_cadence_oversample)
    
    print("Oversampling factor = {0}".format(long_cadence_oversample))
    
    
    # set oversampling factors and expoure times
    oversample = np.zeros(18, dtype='int')
    exptime = np.zeros(18)
    
    oversample[np.array(all_dtype)=='short'] = 1
    oversample[np.array(all_dtype)=='long'] = long_cadence_oversample
    
    exptime[np.array(all_dtype)=='short'] = scit
    exptime[np.array(all_dtype)=='long'] = lcit
    
    # precompute exposure integration time offsets
    texp_offsets = [None]*18
    for j, q in enumerate(quarters):
        
        if all_dtype[q] == 'short':
            texp_offsets[q] = np.array([0.])
        elif all_dtype[q] == 'long':
            texp_offsets[q] = np.linspace(-exptime[q]/2., exptime[q]/2., oversample[q])
    
    
    # ## Set up GP noise priors
    
    
    # Read in noise model GP priors from analyze_autocorrelated_noise.py
    gp_percs = []
    
    for z in range(4):
        try:
            fname_in = os.path.join(RESULTS_DIR, '{0}_shoterm_gp_priors_{1}.txt'.format(TARGET,z))
    
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
                if k != 'percentiles':
                    perc = np.array(gp_percs[z]['percentiles'])
    
                    med = np.array(gp_percs[z][k])[perc == 50.0][0]
                    err1 = np.array(gp_percs[z][k])[perc == 84.135][0]
                    err2 = np.array(gp_percs[z][k])[perc == 15.865][0]
    
                    dev = np.sqrt((err1-med)**2/2 + (err2-med)**2/2)
    
                    gpz[k] = (med, dev)
    
            gp_priors.append(gpz)
            
        else:
            # these are dummy values that effectively create a zero-amplitude kernel
            gpz = {}
            gpz["logw0"] = [np.log(2*pi/(7*DURS.max()))]
            gpz["logSw4"] = [-100.]
            gpz["logQ"] = [np.log(1/np.sqrt(2))]
            
            gp_priors.append(gpz)
            
    # calculate a few convenience quantities
    for z in range(4):
        gpz = gp_priors[z]
        
        logS = gpz["logSw4"][0] - 4*gpz["logw0"][0]
        
        if len(gpz["logSw4"]) == 1:
            print(1)
            gp_priors[z]["logS"] = [logS]
            
        if len(gpz["logSw4"]) == 2:
            logS_var = gpz["logSw4"][1]**2 + 16*gpz["logw0"][1]**2
            gp_priors[z]["logS"] = np.array([logS, np.sqrt(logS_var)])
    
    
    # # ############################
    # # ----- LIGHTCURVE FITTING -----
    # # ############################
    
    
    print("\nModeling Lightcurve...\n")
    
    
    # identify which quarters and seasons have data
    no_transits = (np.array(all_dtype) == 'long_no_transits') + (np.array(all_dtype) == 'short_no_transits')
    which_quarters = np.sort(np.unique(np.hstack(transit_quarter)))
    which_quarters = which_quarters[~np.isin(which_quarters, np.squeeze(np.argwhere(no_transits)))]
    which_seasons  = np.sort(np.unique(which_quarters % 4))
    
    # mean flux and jitter for each quarter
    nq = len(which_quarters)
    mbq = mean_by_quarter[which_quarters]
    vbq = var_by_quarter[which_quarters]
    
    # initialize alderaan.Ephemeris objects
    ephem = [None]*NPL
    for npl in range(NPL):
        ephem[npl] = Ephemeris(transit_inds[npl], quick_transit_times[npl])
    
    # initialize TransitParams objects
    theta = [None]*NPL
    for npl in range(NPL):
        theta[npl] = batman.TransitParams()
        theta[npl].per = ephem[npl].period
        theta[npl].t0  = 0.                     # t0 must be set to zero b/c we are warping TTVs
        theta[npl].rp  = np.sqrt(DEPTHS[npl])
        theta[npl].b   = 0.5
        theta[npl].T14 = DURS[npl]
        theta[npl].u   = [U1, U2]
        theta[npl].limb_dark = 'quadratic'
        
    # grab data
    t_ = [None]*nq
    f_ = [None]*nq
    e_ = [None]*nq
    m_ = [None]*nq     # m_ is "mask" not "model"
    
    for j, q in enumerate(which_quarters):
        m_[j] = all_mask[q].sum(axis=0) > 0
        t_[j] = all_time[q][m_[j]]
        f_[j] = all_flux[q][m_[j]]
        e_[j] = all_error[q][m_[j]]
        
    '''# initialize TransitModel objects
    transit_model = []
    for npl in range(NPL):
        transit_model.append([])
        for j, q in enumerate(which_quarters):
            transit_model[npl].append(batman.TransitModel(theta[npl], 
                                                          ephem[npl]._warp_times(t_[j]),
                                                          supersample_factor=oversample[q],
                                                          exp_time=exptime[q]
                                                         )
                                     )
    '''    
    # build the GP kernel using a different noise model for each season
    kernel = [None]*4
    for z in which_seasons:
        kernel[z] = GPterms.SHOTerm(S0=np.exp(gp_priors[z]['logS'][0]),
                                    w0=np.exp(gp_priors[z]['logw0'][0]),
                                    Q =np.exp(gp_priors[z]['logQ'][0])
                                   )
    
    
    # #### TODO:
    # 1. Rename x --> theta (this is the parameter vector that is fed into dynesty)
    # 2. Infer num_planets as (len(theta)-2)//5
    # 3. Eliminate redundant batman theta
    # 4. Generalize ld_priors to allow for non-static Gaussian uncertainty
    
    
    warped_t = []
    warped_x = []
    
    for npl in range(NPL):
        warped_t.append([])
        warped_x.append([])
        
        for j, q in enumerate(which_quarters):
            _warp_time, _warp_index = ephem[npl]._warp_times(t_[j], return_inds=True)
            _warp_legx = (_warp_index - transit_inds[npl][-1]//2)/(transit_inds[npl][-1]/2)
            
            warped_t[npl].append(_warp_time)
            warped_x[npl].append(_warp_legx)
    
    
    transit_legx = []
    
    for npl in range(NPL):
        transit_legx.append(centered_transit_inds[npl]/(transit_inds[npl][-1]/2))
    
    
    phot_args = {}
    
    phot_args['time'] = t_
    phot_args['flux'] = f_
    phot_args['error'] = e_
    phot_args['quarters'] = which_quarters
    phot_args['warped_t'] = warped_t
    phot_args['warped_x'] = warped_x
    phot_args['exptime']  = exptime
    phot_args['oversample'] = oversample
    phot_args['texp_offsets'] = texp_offsets
    
    ephem_args = {}
    ephem_args['transit_inds']  = centered_transit_inds
    ephem_args['transit_times'] = quick_transit_times
    ephem_args['transit_legx']  = transit_legx
    
    
    ncores      = multipro.cpu_count() - 2
    ndim        = 5*NPL+2
    logl        = dynhelp.lnlike
    ptform      = dynhelp.prior_transform
    logl_args   = (NPL, theta, ephem_args, phot_args, [U1,U2], kernel)
    ptform_args = (NPL, DURS)
    chk_file    = os.path.join(RESULTS_DIR, '{0}-dynesty.checkpoint'.format(TARGET))
    
    
    #%prun [logl(ptform([0.5,0.5,0.5,0.5,0.5,0.5,0.5],1,[1]), *logl_args) for i in range(1000)]
    
    
    USE_MULTIPRO = False
    
    if USE_MULTIPRO:
        with dynesty.pool.Pool(ncores, logl, ptform, logl_args=logl_args, ptform_args=ptform_args) as pool:
            sampler = dynesty.DynamicNestedSampler(pool.loglike, pool.prior_transform, ndim, bound='multi', sample='rwalk', pool=pool)
            sampler.run_nested(checkpoint_file=chk_file, checkpoint_every=60, print_progress=VERBOSE)
            results = sampler.results
            
    if not USE_MULTIPRO:
        sampler = dynesty.DynamicNestedSampler(logl, ptform, ndim, bound='multi', sample='rwalk', logl_args=logl_args, ptform_args=ptform_args)
        sampler.run_nested(checkpoint_file=chk_file, checkpoint_every=60, print_progress=VERBOSE)
        results = sampler.results
    
    
    labels = []
    for npl in range(NPL):
        labels = labels + 'C0_{0} C1_{0} r_{0} b_{0} T14_{0}'.format(npl).split()
    
    labels = labels + 'q1 q2'.split()
    
    
    rfig, raxes = dyplot.runplot(results, logplot=True)
    
    
    tfig, taxes = dyplot.traceplot(results, labels=labels)
    
    
    cfig, caxes = dyplot.cornerplot(results, labels=labels)
    
    
    # # Save results as fits file
    
    
    hduL = io.to_fits(results, PROJECT_DIR, RUN_ID, TARGET, NPL)
    
    path = os.path.join(PROJECT_DIR, 'Results', RUN_ID, TARGET)
    os.makedirs(path, exist_ok=True)
    hduL.writeto(os.path.join(path, '{0}-results.fits'.format(TARGET)), overwrite=True)
    
    
    # ## Exit program
    
    
    print("")
    print("+"*shutil.get_terminal_size().columns)
    print("Exoplanet recovery complete {0}".format(datetime.now().strftime("%d-%b-%Y at %H:%M:%S")))
    print("Total runtime = %.1f min" %((timer()-global_start_time)/60))
    print("+"*shutil.get_terminal_size().columns)
    
    
if __name__ == '__main__':
    main()