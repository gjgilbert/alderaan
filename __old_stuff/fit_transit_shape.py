#!/usr/bin/env python
# coding: utf-8

# # Fit transit shape

# In[ ]:


import numpy as np
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt

import numpy.polynomial.polynomial as poly
import scipy.stats as stats
from   astropy.io import fits as pyfits

import sys
import os
import importlib as imp
import warnings
import argparse
import json
from   timeit import default_timer as timer

import pymc3 as pm
import pymc3_ext as pmx
import exoplanet as exo
import aesara_theano_fallback.tensor as T
from   aesara_theano_fallback import aesara as theano
from   celerite2.theano import GaussianProcess
from   celerite2.theano import terms as GPterms

from alderaan.constants import *
from alderaan.utils import *
from alderaan.Planet import *
from alderaan.LiteCurve import *
import alderaan.io as io

# these lines make progressbar work with SLURM 
from fastprogress.fastprogress import force_console_behavior
master_bar, progress_bar = force_console_behavior()
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
LOGRHO = np.array(target_dict['logrho'], dtype='float')[use]

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

if all(r == LOGRHO[0] for r in LOGRHO): LOGRHO = LOGRHO[0]
else: raise ValueError('There are inconsistencies with LOGRHO in the csv input file')

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

# initialize impact parameter and radius arrays
RADII   = np.sqrt(DEPTHS)*RSTAR
IMPACTS = 0.5*np.ones(NPL)


# In[ ]:


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
    pfit = poly.polyfit(transit_inds[npl], indep_transit_times[npl], 1)
    
    indep_ephemeris.append(poly.polyval(transit_inds[npl], pfit))
    
    EPOCHS[npl] = pfit[1]
    PERIODS[npl] = pfit[0]
    
    
# make sure transit_inds are zero-indexed
for npl in range(NPL):
    transit_inds[npl] = np.array(transit_inds[npl] - transit_inds[npl][0], dtype="int")


# In[ ]:


fig, axes = plt.subplots(NPL, figsize=(12,3*NPL))

for npl in range(NPL):
    xtime = indep_ephemeris[npl]
    yomc = (indep_transit_times[npl] - indep_ephemeris[npl])*24*60
    
    axes[npl].plot(xtime, yomc, '-', c='C{0}'.format(npl))
    axes[npl].set_ylabel('O-C [min]', fontsize=20)
axes[NPL-1].set_xlabel('Time [BJKD]', fontsize=20)
plt.close()


# In[ ]:


# determine scatter relative to linear ephemeris
# this is a deliberate overestimate of the true scatter
omc_scatter = np.zeros(NPL)

for npl in range(NPL):
    xtime = indep_ephemeris[npl]
    yomc  = indep_transit_times[npl] - indep_ephemeris[npl]
    
    omc_scatter[npl] = np.std(yomc)


# In[ ]:


fixed_ephemeris = [x for x in indep_ephemeris]
fixed_transit_times = [x for x in indep_transit_times]


# # Set time baseline

# In[ ]:


# determine the time baseline
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
#var_by_quarter = np.genfromtxt(NOISE_DIR + TARGET + '_var_by_quarter.txt')[:,1]


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
                err1 = np.array(gp_percs[z][k])[perc == 84.135][0]
                err2 = np.array(gp_percs[z][k])[perc == 15.865][0]

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


# # Numerically determine priors

# In[ ]:


Ndraw = int(1e6)
edraw = np.random.uniform(0, 1, size=Ndraw)
wdraw = np.random.uniform(0, 2*pi, size=Ndraw)
gdraw = (1 + edraw*np.sin(wdraw))/np.sqrt(1-edraw**2)

xbins = np.linspace(np.log(gdraw).min(),np.log(gdraw).max(), int(np.sqrt(Ndraw)))

ypdf_logg, xpdf_logg = np.histogram(np.log(gdraw), bins=xbins, density=True)
xpdf_logg = 0.5*(xpdf_logg[1:]+xpdf_logg[:-1])


plt.figure(figsize=(8,5))
plt.hist(np.log(gdraw), bins=xbins, density=True)
plt.plot(xpdf_logg, ypdf_logg, lw=3)
plt.xlabel("$\log(g)$", fontsize=16)
plt.ylabel("PDF", fontsize=16)
plt.close()


# In[ ]:


n = 1000000
r = 0.2

# pm.Interpolated expects strictly increasing x-vectors
# pm.Interpolated also gets angry with flat pdfs
# stats.halfnorm will be undefined at x=0
# all these issues are solved with a small eps parameter
eps = 1e-6

x1 = np.linspace(0,1,n)
y1 = np.linspace(1,1-eps,n)

x2 = np.linspace(1+eps,2,n)
y2 = stats.halfnorm.pdf(x2, loc=1, scale=r)
y2 /= y2.max()


xpdf_b = np.hstack([x1,x2])
ypdf_b = np.hstack([y1,y2])

plt.figure()
plt.plot(xpdf_b, ypdf_b, lw=2)
plt.xlabel("$b$", fontsize=16)
plt.ylabel("PDF", fontsize=16)
plt.close()


# # Grab the relevant data

# In[ ]:


# grab data near transits for each quarter
all_time = [None]*18
all_flux = [None]*18
all_error = [None]*18
all_dtype = ["none"]*18

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
                all_dtype[q] = "short"

                sc_flux.append(sc.flux[use])
                
            else:
                all_dtype[q] = "short_no_transits"

    
    if lc is not None:
        if np.isin(q, lc.quarter):
            use = (lc.mask)*(lc.quarter == q)

            if np.sum(use) > 5:
                all_time[q] = lc.time[use]
                all_flux[q] = lc.flux[use]
                all_error[q] = lc.error[use]
                all_dtype[q] = "long"

                lc_flux.append(lc.flux[use])
                
            else:
                all_dtype[q] = "long_no_transits"


# In[ ]:


# check which quarters have data and transits
good = (np.array(all_dtype) == "short") + (np.array(all_dtype) == "long")
quarters = np.arange(18)[good]
nq = len(quarters)

seasons = np.sort(np.unique(quarters % 4))

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


# set oversampling factors and exposure times
oversample = np.zeros(18, dtype="int")
texp = np.zeros(18)

oversample[np.array(all_dtype)=="short"] = 1
oversample[np.array(all_dtype)=="long"] = 15

texp[np.array(all_dtype)=="short"] = scit
texp[np.array(all_dtype)=="long"] = lcit


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
    q_limbdark = pm.Uniform("q_limbdark", lower=0, upper=1, testval=np.array([Q1,Q2]), shape=2)
    u1 = 2*T.sqrt(q_limbdark[0])*q_limbdark[1]
    u2 = T.sqrt(q_limbdark[0])*(1-2*q_limbdark[1])
    u  = pm.Deterministic("u", T.stack([u1,u2]))
    
    # planetary parameters (rho is circular density, see Ford, Quinn, & Veras 2008)
    log_ror = pm.Normal("log_ror", mu=np.log(RADII/RSTAR), sd=10, shape=NPL)
    ror = pm.Deterministic("ror", T.exp(log_ror))
    rp = pm.Deterministic("rp", ror*RSTAR)
        
    #beta = pm.Exponential("beta", lam=1, testval=-np.log(IMPACTS), shape=NPL)
    #b = pm.Deterministic("b", T.exp(-beta))
    #b = pm.Interpolated("b", xpdf_b, ypdf_b, shape=NPL)
    b = exo.distributions.ImpactParameter("b", ror=ror, shape=NPL)
    
    #pm.Potential("grazing_penalty", pm.Interpolated.dist(xpdf_b, ypdf_b).logp(b))
    
    log_dur = pm.Normal("log_dur", mu=np.log(DURS), sd=5.0, shape=NPL)
    dur = pm.Deterministic("dur", T.exp(log_dur))
    
    #loggamma = pm.Interpolated("loggamma", xpdf_logg, ypdf_logg, shape=NPL)
    #gamma    = pm.Deterministic("gamma", T.exp(loggamma))
    #rho      = pm.Deterministic("rho", T.exp(LOGRHO)*gamma**3)
    
    
    # polynomial TTV parameters (coefficients for Legendre polynomicals)
    C0 = pm.Normal("C0", mu=np.zeros(NPL), sd=3*omc_scatter, shape=NPL)
    C1 = pm.Normal("C1", mu=np.zeros(NPL), sd=3*omc_scatter, shape=NPL)
    
    
    # transit times
    transit_times = []
    
    for npl in range(NPL):
        transit_times.append(pm.Deterministic("tts_{0}".format(npl),
                                              fixed_transit_times[npl]
                                              + C0[npl]*Leg0[npl] + C1[npl]*Leg1[npl]))
   
    # set up stellar model and planetary orbit
    starrystar = exo.LimbDarkLightCurve(u)
    orbit  = exo.orbits.TTVOrbit(transit_times=transit_times, transit_inds=transit_inds, 
                                 b=b, duration=dur, ror=ror, r_star=RSTAR)
    
    # track period, epoch, and stellar density
    P = pm.Deterministic("P", orbit.period)
    T0 = pm.Deterministic("T0", orbit.t0)
    rho = pm.Deterministic("rho", orbit.rho_star)
    
    
    # build the GP kernel using a different noise model for each season
    logSw4   = [None]*4
    logw0    = [None]*4
    #logQ_off = [None]*4
    logQ     = [None]*4
    logS     = [None]*4
    
    kernel  = [None]*4
    
    for z in range(4):
        gpz = gp_priors[z]
        
        logSw4[z] = gpz["logSw4"][0]
        logw0[z] = gpz["logw0"][0]
        logQ[z] = gpz["logQ"][0]
        logS[z] = logSw4[z]-4*logw0[z]
        
        
        #try:
        #    logSw4[z] = pm.Normal('logSw4_{0}'.format(z), mu=gpz['logSw4'][0], sd=gpz['logSw4'][1])
        #except:
        #    logSw4[z] = gpz['logSw4'][0]
        
        #try:
        #    logw0[z] = pm.Normal('logw0_{0}'.format(z), mu=gpz['logw0'][0], sd=gpz['logw0'][1])
        #except:
        #    logw0[z] = gpz['logw0'][0]

        #try:
        #    logQ_off[z] = pm.Normal('logQ_off_{0}'.format(z), 
        #                            mu=np.log(np.exp(gpz['logQ'][0])-1/np.sqrt(2)), 
        #                            sd=gpz['logQ'][1])
        #    logQ[z] = pm.Deterministic('logQ_{0}'.format(z), T.log(1/T.sqrt(2) + T.exp(logQ_off[z]))) 
        
        #except:
        #    logQ[z] = gpz['logQ'][0]
            
        #try:
        #    logS[z] = pm.Deterministic('logS_{0}'.format(z), logSw4[z]-4*logw0[z])        
        #except:
        #    logS[z] = logSw4[z]-4*logw0[z]
        
            
        if np.isin(z, seasons):
            kernel[z] = GPterms.SHOTerm(S0=T.exp(logS[z]), w0=T.exp(logw0[z]), Q=T.exp(logQ[z]))
        else:
            kernel[z] = None
        
        
    # mean flux and jitter
    flux0 = pm.Normal("flux0", mu=np.median(good_flux), sd=np.std(good_flux), shape=len(quarters))
    logjit = pm.Normal("logjit", mu=np.var(good_flux), sd=5.0, shape=len(quarters))
    
    
    # now evaluate the model for each quarter
    light_curves = [None]*nq
    model_flux   = [None]*nq
    
    gp = [None]*nq
    pred = [None]*nq
    
    
    for j, q in enumerate(quarters):
        # calculate light curves
        light_curves[j] = starrystar.get_light_curve(orbit=orbit, r=rp, t=all_time[q],
                                                     oversample=oversample[j], texp=texp[j])
        
        model_flux[j] = pm.math.sum(light_curves[j], axis=-1) + flux0[j]*T.ones(len(all_time[q]))
        pm.Deterministic("model_flux_{0}".format(j), model_flux[j])
        
        # here's the GP (w/ kernel by season)
        gp[j] = GaussianProcess(kernel[q%4], 
                                t=all_time[q], 
                                diag=T.exp(logjit[j])*T.ones(len(all_time[q])), 
                                mean=model_flux[j])

        gp[j].marginal("gp_{0}".format(j), observed=all_flux[q])


# In[ ]:


with shape_model:
    shape_map = shape_model.test_point
    shape_map = pmx.optimize(start=shape_map, vars=[flux0, logjit])
    shape_map = pmx.optimize(start=shape_map, vars=[C0, C1])
    shape_map = pmx.optimize(start=shape_map, vars=[b, dur])


# In[ ]:


with shape_model:
    shape_trace = pm.sample(tune=10000, 
                            draws=5000, 
                            start=shape_map, 
                            chains=2, 
                            target_accept=0.9,
                            init="adapt_full")


# In[ ]:


# select which variables to save (don't save full GP or model traces or "under the hood" variables)
shape_map_keys = list(shape_map.keys())
shape_varnames = []

for i, smk in enumerate(shape_map_keys):
    skip = ("pred" in smk) + ("model_flux" in smk) + ("__" in smk)

    if skip == False:
        shape_varnames.append(smk)


# In[ ]:


shape_hdulist = io.trace_to_hdulist(shape_trace, shape_varnames, TARGET)
shape_hdulist.writeto(TRACE_DIR + TARGET + '_transit_shape.fits', overwrite=True)


# In[ ]:


print('TOTAL RUNTIME = %.2f min' %((timer()-global_start_time)/60))


# In[ ]:




