#!/usr/bin/env python
# coding: utf-8

# # Fit transit shape with nested sampling

# In[ ]:


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

# In[ ]:


# Automatically set inputs (when running batch scripts)
import argparse
import matplotlib as mpl

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
parser.add_argument("--interactive", default=False, type=bool, required=False, \
                    help="'True' to enable interactive plotting; by default matplotlib backend will be set to 'Agg'")

args = parser.parse_args()
MISSION      = args.mission
TARGET       = args.target
PROJECT_DIR  = args.project_dir
DATA_DIR     = args.data_dir
CATALOG      = args.catalog
RUN_ID       = args.run_id

# set plotting backend
if args.interactive == False:
    mpl.use('agg')
    

# In[ ]:


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

# In[ ]:


sys.path.append(PROJECT_DIR)


# #### Build directory structure

# In[ ]:


# directories in which to place pipeline outputs for this run
RESULTS_DIR = os.path.join(PROJECT_DIR, 'Results', RUN_ID, TARGET)
FIGURE_DIR  = os.path.join(PROJECT_DIR, 'Figures', RUN_ID, TARGET)

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)


# #### Import packages

# In[ ]:


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


# In[ ]:


# flush buffer to avoid mixed outputs from progressbar
sys.stdout.flush()

# turn off FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)


# check for interactive matplotlib backends
if np.any(np.array(['agg', 'png', 'svg', 'pdf', 'ps']) == mpl.get_backend()):
    iplot = False
else:
    iplot = True
    
# print theano compiledir cache
print("theano cache: {0}\n".format(theano.config.compiledir))


# # ################
# # ----- DATA I/O -----
# # ################

# In[ ]:


print("\nLoading data...\n")


# ## Read in planet and stellar properties

# In[ ]:


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
DEPTHS  = np.array(target_dict['depth'], dtype='float')[use]
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

# In[ ]:


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

# In[ ]:


# transit times
epochs = np.zeros(NPL)
periods = np.zeros(NPL)
ephemeris = [None]*NPL

transit_inds = []
indep_transit_times = []
quick_transit_times = []

for npl in range(NPL):
    fname_in = os.path.join(RESULTS_DIR, '{0}_{1:02d}_quick.ttvs'.format(TARGET, npl))
    data_in  = np.genfromtxt(fname_in)
    
    transit_inds.append(np.array(data_in[:,0], dtype='int'))
    indep_transit_times.append(np.array(data_in[:,1], dtype='float'))
    quick_transit_times.append(np.array(data_in[:,2], dtype='float'))
    
    # do a quick fit to get a linear ephemeris
    pfit = poly.polyfit(transit_inds[npl], quick_transit_times[npl], 1)
    
    epochs[npl] = pfit[0]
    periods[npl] = pfit[1]
    ephemeris[npl] = poly.polyval(transit_inds[npl], pfit)
    
# make sure transit_inds are zero-indexed
for npl in range(NPL):
    transit_inds[npl] = np.array(transit_inds[npl] - transit_inds[npl][0], dtype='int')

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

if iplot: plt.show()
else: plt.close()


# # ####################
# # --- PRELIMINARIES ---
# # ####################

# In[ ]:


print("\nRunning preliminaries...\n")


# ## Establish time baseline

# In[ ]:


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

# In[ ]:


overlap = []

for i in range(NPL):
    overlap.append(np.zeros(len(ephemeris[i]), dtype='bool'))
    
    for j in range(NPL):
        if i != j:
            for tt in ephemeris[j]:
                overlap[i] += np.abs(ephemeris[i] - tt) < (DURS[i] + DURS[j] + lcit)


# ## Track which quarter each transit falls in

# In[ ]:


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

# In[ ]:


if sc is not None:
    sc_mask = np.zeros((NPL,len(sc.time)), dtype='bool')
    for npl in range(NPL):
        sc_mask[npl] = make_transitmask(sc.time, quick_transit_times[npl], masksize=np.max([1/24,1.5*DURS[npl]]))
        
if lc is not None:
    lc_mask = np.zeros((NPL,len(lc.time)), dtype='bool')
    for npl in range(NPL):
        lc_mask[npl] = make_transitmask(lc.time, quick_transit_times[npl], masksize=np.max([1/24,1.5*DURS[npl]]))


# ## Grab data near transits

# In[ ]:


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

# In[ ]:


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
            
            
# set oversampling factors and expoure times
oversample = np.zeros(18, dtype='int')
texp = np.zeros(18)

oversample[np.array(all_dtype)=='short'] = 1
oversample[np.array(all_dtype)=='long'] = 15

texp[np.array(all_dtype)=='short'] = scit
texp[np.array(all_dtype)=='long'] = lcit


# ## Define Legendre polynomials

# In[ ]:


# Legendre polynomials for better orthogonality when fitting period and epoch; "x" is in the range (-1,1)
# In practice only linear perturbations are used; higher orders are vestiges of a previous pipeline version
Leg0 = []
Leg1 = []
Leg2 = []
Leg3 = []
t = []

# this assumes a baseline in the range (TIME_START,TIME_END)
for npl in range(NPL):    
    t.append(ephemeris[npl])
    x = 2*(t[npl]-TIME_START)/(TIME_END-TIME_START) - 1

    Leg0.append(np.ones_like(x))
    Leg1.append(x.copy())
    Leg2.append(0.5*(3*x**2 - 1))
    Leg3.append(0.5*(5*x**3 - 3*x))


# ## Set up GP noise priors

# In[ ]:


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

# In[ ]:


print("\nModeling Lightcurve...\n")


# In[ ]:


# identify which quarters and seasons have data
which_quarters = np.sort(np.unique(np.hstack(transit_quarter)))
which_seasons = np.sort(np.unique(which_quarters % 4))

# mean flux and jitter for each quarter
nq = len(which_quarters)
mbq = mean_by_quarter[which_quarters]
vbq = var_by_quarter[which_quarters]

# initialize alderaan.Ephemeris object
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
    theta[npl].u   = [0.25, 0.4]
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
    
# initialize TransitModel objects
transit_model = []
for npl in range(NPL):
    transit_model.append([])
    for j, q in enumerate(which_quarters):
        transit_model[npl].append(batman.TransitModel(theta[npl], 
                                                      ephem[npl]._warp_times(t_[j]),
                                                      supersample_factor=oversample[q],
                                                      exp_time=texp[q]
                                                     )
                                 )
    
# build the GP kernel using a different noise model for each season
kernel = [None]*4
for z in which_seasons:
    kernel[z] = GPterms.SHOTerm(S0=np.exp(gp_priors[z]['logS'][0]),
                                w0=np.exp(gp_priors[z]['logw0'][0]),
                                Q =np.exp(gp_priors[z]['logQ'][0])
                               )


# In[ ]:


# package everything in dictionaries to simplify passing to dynesty
# this slightly inefficient method maintains easy backward compatibility
ephem_args = {}
ephem_args['ephem'] = ephem
ephem_args['transit_inds'] = transit_inds
ephem_args['transit_times'] = quick_transit_times
ephem_args['Leg0'] = Leg0
ephem_args['Leg1'] = Leg1

phot_args = {}
phot_args['time'] = t_
phot_args['flux'] = f_
phot_args['error'] = e_
phot_args['mask']  = m_


# In[ ]:


ncores      = multipro.cpu_count()
ndim        = 5*NPL+2
logl        = dynhelp.lnlike
ptform      = dynhelp.prior_transform
logl_args   = (NPL, theta, transit_model, which_quarters, ephem_args, phot_args, kernel, [U1,U2])
ptform_args = (NPL, DURS)
chk_file    = os.path.join(RESULTS_DIR, '{0}-dynesty.checkpoint'.format(TARGET))


USE_MULTIPRO = False

if USE_MULTIPRO:
    with dynesty.pool.Pool(ncores, logl, ptform, logl_args=logl_args, ptform_args=ptform_args) as pool:
        sampler = dynesty.DynamicNestedSampler(pool.loglike, pool.prior_transform, ndim, pool=pool)
        sampler.run_nested(n_effective=1000, checkpoint_file=chk_file, checkpoint_every=600)
        results = sampler.results
        
if ~USE_MULTIPRO:
    sampler = dynesty.DynamicNestedSampler(logl, ptform, ndim, logl_args=logl_args, ptform_args=ptform_args)
    sampler.run_nested(n_effective=1000, checkpoint_file=chk_file, checkpoint_every=600)
    results = sampler.results


# In[ ]:


labels = []
for npl in range(NPL):
    labels = labels + 'C0_{0} C1_{0} r_{0} b_{0} T14_{0}'.format(npl).split()

labels = labels + 'q1 q2'.split()


# In[ ]:


rfig, raxes = dyplot.runplot(results, logplot=True)


# In[ ]:


tfig, taxes = dyplot.traceplot(results, labels=labels)


# In[ ]:


cfig, caxes = dyplot.cornerplot(results, labels=labels)


# # Save results as fits file

# In[ ]:


hduL = io.to_fits(results, PROJECT_DIR, RUN_ID, TARGET, NPL)

path = os.path.join(PROJECT_DIR, 'Results', RUN_ID, TARGET)
os.makedirs(path, exist_ok=True)
hduL.writeto(os.path.join(path, '{0}-results.fits'.format(TARGET)), overwrite=True)


# ## Exit program

# In[ ]:


print("")
print("+"*shutil.get_terminal_size().columns)
print("Exoplanet recovery complete {0}".format(datetime.now().strftime("%d-%b-%Y at %H:%M:%S")))
print("Total runtime = %.1f min" %((timer()-global_start_time)/60))
print("+"*shutil.get_terminal_size().columns)


# In[ ]:




