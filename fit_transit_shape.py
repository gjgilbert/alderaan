#!/usr/bin/env python
# coding: utf-8

# # Fit transit shape

# In[ ]:


import numpy as np
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt

import scipy.optimize as op
import scipy.signal as sig
from   scipy import stats
from   scipy import fftpack
from   scipy import ndimage
import astropy
from   astropy.io import fits as pyfits
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

import lightkurve as lk
import exoplanet as exo
import theano.tensor as T
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


# # Manually set I/O parameters
# #### User should manually set MISSION, TARGET, PRIMARY_DIR,  and CSV_FILE

# In[ ]:


# here's where we parse the inputs
try:
    parser = argparse.ArgumentParser(description="Inputs for ALDERAAN transit fiting pipeline")
    parser.add_argument("--mission", default=None, type=str, required=True,                         help="Mission name")
    parser.add_argument("--target", default=None, type=str, required=True,                         help="Target name; see ALDERAAN documentation for acceptable formats")
    parser.add_argument("--primary_dir", default=None, type=str, required=True,                         help="Primary directory path for accessing lightcurve data and saving outputs")
    parser.add_argument("--csv_file", default=None, type=str, required=True,                         help="Path to .csv file containing input planetary parameters")


    args = parser.parse_args()
    MISSION     = args.mission
    TARGET      = args.target
    PRIMARY_DIR = args.primary_dir
    CSV_FILE    = args.csv_file
    
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


# # Read in planet and stellar parameters from Kepler DR25 & Gaia DR2

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

#RHO = np.array(target_dict['rhostar'],  dtype='float')[use]
#RHO_ERR1 = np.array(target_dict['rhostar_err1'],  dtype='float')[use]
#RHO_ERR2 = np.array(target_dict['rhostar_err2'],  dtype='float')[use]

U1 = np.array(target_dict['limbdark_1'], dtype='float')[use]
U2 = np.array(target_dict['limbdark_2'], dtype='float')[use]

PERIODS = np.array(target_dict['period'], dtype='float')[use]
EPOCHS  = np.array(target_dict['epoch'],  dtype='float')[use]
DEPTHS  = np.array(target_dict['depth'], dtype='float')[use]*1e-6          # [ppm] --> []
DURS    = np.array(target_dict['duration'], dtype='float')[use]/24         # [hrs] --> [days]


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
    
if all(u == U1[0] for u in U1): U1 = U1[0]
else: raise ValueError('There are inconsistencies with U1 in the csv input file')

if all(u == U2[0] for u in U2): U2 = U2[0]
else: raise ValueError('There are inconsistencies with U2 in the csv input file')


# In[ ]:


# sort planet parameters by period
order = np.argsort(PERIODS)

PERIODS = PERIODS[order]
EPOCHS  = EPOCHS[order]
DEPTHS  = DEPTHS[order]
DURS    = DURS[order]

# initialize and radius arrays
RADII = np.sqrt(DEPTHS)*RSTAR
IMPACTS = 0.5*np.ones(NPL)


# In[ ]:


# combline stellar radius uncertainties
RSTAR_ERR = np.sqrt(RSTAR_ERR1**2 + RSTAR_ERR2**2)/np.sqrt(2)

# Kipping 2013 limb darkening parameterization
Q1 = (U1 + U2)**2
Q2 = 0.5*U1/(U1+U2)


# # Read in detrended lightcurves and initial transit time estimates
# #### The data can be generated by running the script "detrend_and_estimate_noise.py"

# In[ ]:


# Load detrended lightcurves
try:
    lc = io.load_detrended_lightcurve(DLC_DIR + TARGET + '_lc_detrended.fits')
except:
    lc = None
    
try:
    sc = io.load_detrended_lightcurve(DLC_DIR + TARGET + '_sc_detrended.fits')
except:
    sc = None


# In[ ]:


# Read in QuickTTV estimates and calculate linear ephemeris for each planet
EPOCHS  = np.zeros(NPL)
PERIODS = np.zeros(NPL)

transit_inds = []
indep_transit_times = []
indep_ephemeris = []


for npl in range(NPL):
    # read in predetermined transit times
    fname_in = QUICK_TTV_DIR + TARGET + '_{:02d}'.format(npl) + '_quick_ttvs.txt'
    data_in  = np.genfromtxt(fname_in)
    
    transit_inds.append(data_in[:,0])
    indep_transit_times.append(data_in[:,1])
    
    # do a quick fit to get a linear ephemeris
    pfit = np.polyfit(transit_inds[npl], indep_transit_times[npl], 1)
    
    indep_ephemeris.append(np.polyval(pfit, transit_inds[npl]))
    
    EPOCHS[npl] = pfit[1]
    PERIODS[npl] = pfit[0]
    
    
# make sure transit_inds are zero-indexed
for npl in range(NPL):
    transit_inds[npl] = np.array(transit_inds[npl] - transit_inds[npl][0], dtype="int")


# In[ ]:


# determine scatter relative to linear ephemeris
# this is a deliberate overestimate of the true scatter
omc_scatter = np.zeros(NPL)

for npl in range(NPL):
    xtime = indep_ephemeris[npl]
    yomc  = indep_transit_times[npl] - indep_ephemeris[npl]
    
    omc_scatter[npl] = np.std(yomc)


# In[ ]:


fixed_ephemeris = np.copy(indep_ephemeris)
fixed_transit_times = np.copy(indep_transit_times)


# # Set time baseline

# In[ ]:


# determine the time baseline
time_min = []
time_max = []

try:
    time_min.append(sc.time.min())
    time_max.append(sc.time.max()) 
except:
    pass


try:
    time_min.append(lc.time.min())
    time_max.append(lc.time.max())     
except:
    pass
    
    
TIME_START = np.min(time_min)
TIME_END   = np.max(time_max)

# put epochs in range (TIME_START, TIME_START + PERIOD)
for npl in range(NPL):
    if EPOCHS[npl] < TIME_START:
        adj = 1 + (TIME_START - EPOCHS[npl])//PERIODS[npl]
        EPOCHS[npl] += adj*PERIODS[npl]        
        
    if EPOCHS[npl] > (TIME_START + PERIODS[npl]):
        adj = (EPOCHS[npl] - TIME_START)//PERIODS[npl]
        EPOCHS[npl] -= adj*PERIODS[npl]


# # Set up noise GP

# In[ ]:


# Read in noise model GP priors
gp_percs = []

for z in range(4):
    try:
        fname_in = NOISE_DIR + TARGET + '_shoterm_gp_priors_{0}.txt'.format(z)

        with open(fname_in) as infile:
            gp_percs.append(json.load(infile))

    except:
        gp_percs.append(None)

# Read in quarter-by-quarter variances
var_by_quarter = np.genfromtxt(NOISE_DIR + TARGET + '_var_by_quarter.txt')[:,1]


# In[ ]:


gp_priors = []

for z in range(4):
    if gp_percs[z] is not None:
        # set GP priors baed on outputs of alderaan.detrend_and_estimate_noise
        # expected for any season with short cadence data
        gpz = {}

        for k in gp_percs[z].keys():
            if k != "percentiles":
                perc = np.array(gp_percs[z]['percentiles'])

                med = np.array(gp_percs[z][k])[perc == 50.0][0]
                err1 = np.array(gp_percs[z][k])[perc == 84.1][0]
                err2 = np.array(gp_percs[z][k])[perc == 15.9][0]

                dev = np.sqrt((err1-med)**2/2 + (err2-med)**2/2)

                gpz[k] = (med, dev)

        gp_priors.append(gpz)
        
    else:
        # these are dummy values that effectively create a zero-amplitude kernel
        gpz = {}
        gpz['logw0'] = [np.log(2*pi/(7*DURS.max()))]
        gpz['logSw4'] = [-100.]
        gpz['logQ'] = [np.log(1/np.sqrt(2))]
        
        gp_priors.append(gpz)


# In[ ]:


for z in range(4):
    gpz = gp_priors[z]
    
    logS = gpz["logSw4"][0] - 4*gpz["logw0"][0]
    
    if len(gpz["logSw4"]) == 1:
        gp_priors[z]["logS"] = np.copy(logS)
        
    if len(gpz["logSw4"]) == 2:
        logS_var = gpz["logSw4"][1]**2 + 16*gpz["logw0"][1]**2
        gp_priors[z]["logS"] = np.array([logS, np.sqrt(logS_var)])


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
                all_dtype.append('none')


        elif np.isin(q, lc.quarter):
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
                all_dtype.append('none')


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
                all_dtype.append('none')


        else:
            all_time.append(None)
            all_flux.append(None)
            all_error.append(None)
            all_dtype.append('none')



# check which quarters have data
good = (np.array(all_dtype) == 'short') + (np.array(all_dtype) == 'long')
quarters = np.arange(18)[good]
nq = len(quarters)

seasons = np.sort(np.unique(quarters % 4))

lc_quarters = np.arange(18)[np.array(all_dtype) == 'long']
sc_quarters = np.arange(18)[np.array(all_dtype) == 'short']


# expand var_by_quarter to have None for missing quarters
vbq_all = [None]*18

for j, q in enumerate(quarters):
    vbq_all[q] = var_by_quarter[j]
    
vbq_all = np.asarray(vbq_all, dtype="float")


# In[ ]:


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


print('')
print('cumulative runtime = ', int(timer() - global_start_time), 's')
print('')


# # Define Legendre polynomials

# In[ ]:


# use Legendre polynomials for better orthogonality; "x" is in the range (-1,1)
Leg0 = []
Leg1 = []
Leg2 = []
Leg3 = []
t = []

# this assumes a baseline in the range (TIME_START,TIME_END)
for npl in range(NPL):    
    t.append(fixed_ephemeris[npl])
    x = 2*(t[npl]-TIME_START)/(TIME_END-TIME_START) - 1

    Leg0.append(np.ones_like(x))
    Leg1.append(x.copy())
    Leg2.append(0.5*(3*x**2 - 1))
    Leg3.append(0.5*(5*x**3 - 3*x))


# # Fit a transit model with fixed transit times

# In[ ]:


with pm.Model() as shape_model:
    # stellar parameters (limb darkening using Kipping 2013)
    #u = exo.distributions.QuadLimbDark('u', testval=np.array([U1,U2]))
    q_limbdark = pm.Uniform("q_limbdark", lower=0, upper=1, testval=np.array([Q1,Q2]), shape=2)
    
    u1 = 2*T.sqrt(q_limbdark[0])*q_limbdark[1]
    u2 = T.sqrt(q_limbdark[0])*(1-2*q_limbdark[1])
    u  = pm.Deterministic("u", T.stack([u1,u2]))

    Rstar = pm.Bound(pm.Normal, lower=RSTAR-3*RSTAR_ERR, upper=RSTAR+3*RSTAR_ERR)('Rstar', mu=RSTAR, sd=RSTAR_ERR)
    #Mstar = pm.Bound(pm.Normal, lower=MSTAR-3*MSTAR_ERR, upper=MSTAR+3*MSTAR_ERR)('Mstar', mu=MSTAR, sd=MSTAR_ERR)
    logrho = pm.Normal("logrho", mu=np.log(1.408), sd=10, shape=NPL)
    rho    = pm.Deterministic("rho", T.exp(logrho))
    
    # planetary parameters
    logr = pm.Uniform('logr', lower=np.log(0.0003), upper=np.log(0.3), testval=np.log(RADII), shape=NPL)
    rp   = pm.Deterministic('rp', T.exp(logr))
    
    beta = pm.Exponential('beta', lam=1, testval=-np.log(IMPACTS), shape=NPL)
    b    = pm.Deterministic('b', T.exp(-beta))
    
    # polynomial TTV parameters    
    C0 = pm.Normal('C0', mu=np.zeros(NPL), sd=3*omc_scatter, shape=NPL)
    C1 = pm.Normal('C1', mu=np.zeros(NPL), sd=3*omc_scatter, shape=NPL)
    
    
    # transit times
    transit_times = []
    
    for npl in range(NPL):
        transit_times.append(pm.Deterministic("tts_{0}".format(npl),
                                              fixed_transit_times[npl]
                                              + C0[npl]*Leg0[npl] + C1[npl]*Leg1[npl]))
   
    # set up stellar model and planetary orbit
    exoSLC = exo.StarryLightCurve(u)
    orbit  = exo.orbits.TTVOrbit(transit_times=transit_times, transit_inds=transit_inds,                                  b=b, r_star=Rstar, rho_star=rho)
    
    # track period and epoch
    T0 = pm.Deterministic('T0', orbit.t0)
    P  = pm.Deterministic('P', orbit.period)
    
    
    # build the GP kernel using a different noise model for each season
    logSw4   = [None]*4
    logw0    = [None]*4
    logQ_off = [None]*4
    logQ     = [None]*4
    logS     = [None]*4
    penalty  = [None]*4
    
    kernel  = [None]*4
    
    for z in range(4):
        gpz = gp_priors[z]
        
        try:
            logSw4[z] = pm.Normal('logSw4_{0}'.format(z), mu=gpz['logSw4'][0], sd=gpz['logSw4'][1])
        except:
            logSw4[z] = gpz['logSw4'][0]
        
        try:
            logw0[z] = pm.Normal('logw0_{0}'.format(z), mu=gpz['logw0'][0], sd=gpz['logw0'][1])
        except:
            logw0[z] = gpz['logw0'][0]

        try:
            logQ_off[z] = pm.Normal('logQ_off_{0}'.format(z), 
                                    mu=np.log(np.exp(gpz['logQ'][0])-1/np.sqrt(2)), 
                                    sd=gpz['logQ'][1])
            logQ[z] = pm.Deterministic('logQ_{0}'.format(z), T.log(1/T.sqrt(2) + T.exp(logQ_off[z]))) 
        except:
            logQ[z] = gpz['logQ'][0]
            
            
        try:
            logS[z] = pm.Deterministic('logS_{0}'.format(z), logSw4[z]-4*logw0[z])
            penalty[z] = pm.Potential('penalty_{0}'.format(z), -T.exp((logS[z]-gpz['logS'][0])/gpz['logS'][1]))
            
        except:
            pass
            
            
            

        if np.isin(z, seasons):
            kernel[z] = exo.gp.terms.SHOTerm(log_Sw4=logSw4[z], log_w0=logw0[z], log_Q=logQ[z])
        else:
            kernel[z] = None
        
        
    # nuissance parameters
    flux0 = pm.Normal('flux0', mu=np.mean(good_flux), sd=np.std(good_flux), shape=len(quarters))
    logjit = pm.Normal('logjit', mu=np.var(good_flux), sd=10, shape=len(quarters))
    
    
    # now evaluate the model for each quarter
    light_curves       = [None]*nq
    summed_light_curve = [None]*nq
    model_flux         = [None]*nq
    
    gp      = [None]*nq
    gp_pred = [None]*nq
    
    
    for j, q in enumerate(quarters):
        # set oversampling factor
        if all_dtype[q] == 'short':
            oversample = 1
        elif all_dtype[q] == 'long':
            oversample = 15
            
        # calculate light curves
        light_curves[j] = exoSLC.get_light_curve(orbit=orbit, r=rp, t=all_time[q], oversample=oversample)
        summed_light_curve[j] = pm.math.sum(light_curves[j], axis=-1) + flux0[j]*T.ones(len(all_time[q]))
        model_flux[j] = pm.Deterministic('model_flux_{0}'.format(j), summed_light_curve[j])
        
        # here's the GP (w/ kernel by season)
        if all_dtype[q] == 'short':
            gp[j] = exo.gp.GP(kernel[q%4], all_time[q], T.exp(logjit[j])*T.ones(len(all_time[q])))
            
        elif all_dtype[q] == 'long':
            gp[j] = exo.gp.GP(kernel[q%4], all_time[q], T.exp(logjit[j])*T.ones(len(all_time[q])))
            
        else:
            raise ValueError("Cadence data type must be 'short' or 'long'")


        # add custom potential (log-prob fxn) with the GP likelihood
        pm.Potential('obs_{0}'.format(j), gp[j].log_likelihood(all_flux[q] - model_flux[j]))


        # track GP prediction
        gp_pred[j] = pm.Deterministic('gp_pred_{0}'.format(j), gp[j].predict())


# In[ ]:


with shape_model:
    shape_map = exo.optimize(start=shape_model.test_point, vars=[flux0, logjit])
    shape_map = exo.optimize(start=shape_map, vars=[C0, C1])
    shape_map = exo.optimize(start=shape_map, vars=[b, rp, Rstar, rho])
    shape_map = exo.optimize(start=shape_map, vars=[u])


# In[ ]:


with shape_model:
    shape_trace = pm.sample(tune=9000, draws=1000, start=shape_map, chains=2,                             step=exo.get_dense_nuts_step(target_accept=0.9))


# In[ ]:


# select which variables to save (don't save full GP or model traces or "under the hood" variables)
shape_map_keys = list(shape_map.keys())
shape_varnames = []

for i, smk in enumerate(shape_map_keys):
    skip = ("gp_pred" in smk) + ("model_flux" in smk) + ("__" in smk)

    if skip == False:
        shape_varnames.append(smk)


# In[ ]:


shape_hdulist = io.trace_to_hdulist(shape_trace, shape_varnames, TARGET)
shape_hdulist.writeto(TRACE_DIR + TARGET + '_transit_shape.fits', overwrite=True)


# In[ ]:


print('TOTAL RUNTIME = %.2f min' %((timer()-global_start_time)/60))


# In[ ]:




