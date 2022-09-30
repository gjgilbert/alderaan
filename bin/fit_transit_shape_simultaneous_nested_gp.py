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
print("ALDERAAN Transit Fitting (single planet)")
print("Initialized {0}".format(datetime.now().strftime("%d-%b-%Y at %H:%M:%S")))
print("+"*shutil.get_terminal_size().columns)
print("")

# start program timer
global_start_time = timer()


# #### Parse inputs

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
parser.add_argument("--catalog", default=None, type=str, required=True, \
                    help="CSV file containing input planetary parameters")
parser.add_argument("--interactive", default=False, type=bool, required=False, \
                    help="'True' to enable interactive plotting; by default matplotlib backend will be set to 'Agg'")

args = parser.parse_args()
MISSION      = args.mission
TARGET       = args.target
PROJECT_DIR  = args.project_dir
CATALOG      = args.catalog  

# set plotting backend
if args.interactive == False:
    mpl.use('agg')

# #### Set environment variables
sys.path.append(PROJECT_DIR)


# #### Build directory structure


# directories in which to place pipeline outputs
FIGURE_DIR    = PROJECT_DIR + 'Figures/' + TARGET + '/'
TRACE_DIR     = PROJECT_DIR + 'Traces/' + TARGET + '/'
QUICK_TTV_DIR = PROJECT_DIR + 'QuickTTVs/' + TARGET + '/'
DLC_DIR       = PROJECT_DIR + 'Detrended_lightcurves/' + TARGET + '/'
NOISE_DIR     = PROJECT_DIR + 'Noise_models/' + TARGET + '/'

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


# #### Import packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
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
from   alderaan.utils import *
import alderaan.io as io
from   alderaan.detrend import make_transitmask
from   alderaan.LiteCurve import LiteCurve
from   alderaan.Planet import Planet



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



print("\nLoading data...\n")


# ## Read in planet and stellar properties



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

PERIODS = np.array(target_dict['period'], dtype='float')[use]
DEPTHS  = np.array(target_dict['depth'], dtype='float')[use]*1e-6          # [ppm] --> []
DURS    = np.array(target_dict['duration'], dtype='float')[use]/24         # [hrs] --> [days]


# do some consistency checks
if all(k == KIC[0] for k in KIC): KIC = KIC[0]
else: raise ValueError("There are inconsistencies with KIC in the csv input file")

if all(n == NPL[0] for n in NPL): NPL = NPL[0]
else: raise ValueError("There are inconsistencies with NPL in the csv input file")
    
    
# sort planet parameters by period
order = np.argsort(PERIODS)

PERIODS = PERIODS[order]
DEPTHS  = DEPTHS[order]
DURS    = DURS[order]


# ## Read in filtered lightcurves
# #### These can be generated by running the script "analyze_autocorrelated_noise.py"



# detrended lightcurves
try:
    lc = io.load_detrended_lightcurve(DLC_DIR + TARGET + '_lc_filtered.fits')
    lc.season = lc.quarter % 4
except:
    lc = None
    
try:
    sc = io.load_detrended_lightcurve(DLC_DIR + TARGET + '_sc_filtered.fits')
    sc.season = sc.quarter % 4
except:
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
    fname_in = QUICK_TTV_DIR + TARGET + '_{:02d}'.format(npl) + '_quick.ttvs'
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


# ## Identify and remove overlapping transits



overlap = []

for i in range(NPL):
    overlap.append(np.zeros(len(ephemeris[i]), dtype='bool'))
    
    for j in range(NPL):
        if i != j:
            for tt in ephemeris[j]:
                overlap[i] += np.abs(ephemeris[i] - tt) < (DURS[i] + DURS[j] + lcit)
                
ephemeris = [ephemeris[npl][~overlap[npl]] for npl in range(NPL)]
transit_inds = [transit_inds[npl][~overlap[npl]] for npl in range(NPL)]
quick_transit_times = [quick_transit_times[npl][~overlap[npl]] for npl in range(NPL)]


# ## Track which quarter each transit falls in



# get list of quarters with observations
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

# In[16]:


if sc is not None:
    sc_mask = np.zeros((NPL,len(sc.time)), dtype='bool')
    for npl in range(NPL):
        sc_mask[npl] = make_transitmask(sc.time, quick_transit_times[npl], masksize=1.5)
        
        
if lc is not None:
    lc_mask = np.zeros((NPL,len(lc.time)), dtype='bool')
    for npl in range(NPL):
        lc_mask[npl] = make_transitmask(lc.time, quick_transit_times[npl], masksize=1.5)


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

            if np.sum(use) > 5:
                all_time[q] = lc.time[use]
                all_flux[q] = lc.flux[use]
                all_error[q] = lc.error[use]
                all_mask[q]  = lc_mask[:,use]
                all_dtype[q] = 'long'
                
            else:
                all_dtype[q] = 'long_no_transits'
                
                
# set oversampling factors and expoure times
oversample = np.zeros(18, dtype='int')
texp = np.zeros(18)

oversample[np.array(all_dtype)=='short'] = 1
oversample[np.array(all_dtype)=='long'] = 15

texp[np.array(all_dtype)=='short'] = scit
texp[np.array(all_dtype)=='long'] = lcit



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


# ## Define Legendre polynomials



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



# Read in noise model GP priors from analyze_autocorrelated_noise.py
gp_percs = []

for z in range(4):
    try:
        fname_in = NOISE_DIR + TARGET + '_shoterm_gp_priors_{0}.txt'.format(z)

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


class Ephemeris:
    def __init__(self, inds, tts):
        self.inds = inds
        self.tts  = tts
    
        # calculate least squares period and epoch
        self.t0, self.period = poly.polyfit(inds, tts, 1)
        
        # calculate ttvs
        self.ttvs = tts - poly.polyval(inds, [self.t0, self.period])
    
        # calculate full set of transit times
        self.full_transit_times = self.t0 + self.period*np.arange(self.inds.max()+1)
        self.full_transit_times[self.inds] = self.tts

        # set up histogram for identifying transit offsets
        ftts = self.full_transit_times

        self._bin_edges = np.concatenate([[ftts[0] - 0.5*self.period],
                                         0.5*(ftts[1:] + ftts[:-1]),
                                         [ftts[-1] + 0.5*self.period]
                                        ])

        self._bin_values = np.concatenate([[ftts[0]], ftts, [ftts[-1]]])
    
    
    def _get_model_dt(self, t):
        _inds = np.searchsorted(self._bin_edges, t)
        _vals = self._bin_values[_inds]
        return _vals
    
    def _warp_times(self, t):
        return t - self._get_model_dt(t)



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
        transit_model[npl].append(batman.TransitModel(theta[npl], ephem[npl]._warp_times(t_[j])))
    
    
# build the GP kernel using a different noise model for each season
kernel = [None]*4
for z in which_seasons:
    kernel[z] = GPterms.SHOTerm(S0=np.exp(gp_priors[z]['logS'][0]),
                                w0=np.exp(gp_priors[z]['logw0'][0]),
                                Q =np.exp(gp_priors[z]['logQ'][0])
                               )

    
# functions for dynesty (hyperparameters are currently hard-coded)
def prior_transform(uniform_hypercube):
    x_ = np.array(uniform_hypercube)

    # 5*NPL parameters: {C0, C1, r, b, T14}
    dists = []
    for npl in range(NPL):
        C0  = stats.norm(loc=0., scale=0.1).ppf(x_[0+npl*5])
        C1  = stats.norm(loc=0., scale=0.1).ppf(x_[1+npl*5])
        r   = stats.loguniform(1e-5, 0.99).ppf(x_[2+npl*5])
        b   = stats.uniform(0., 1+r).ppf(x_[3+npl*5])
        T14 = stats.loguniform(scit, 3*DURS[npl]).ppf(x_[4+npl*5])
        
        dists = np.hstack([dists, [C0, C1, r, b, T14]])
    
    return np.array(dists)


def lnlike(x):
    for npl in range(NPL):
        C0, C1, rp, b, T14 = np.array(x[5*npl:5*(npl+1)])

        # update ephemeris
        ephem[npl] = Ephemeris(transit_inds[npl], quick_transit_times[npl] + C0*Leg0[npl] + C1*Leg1[npl])

        # update transit parameters
        theta[npl].per = ephem[npl].period
        theta[npl].t0  = 0.                     # t0 must be set to zero b/c we are warping TTVs
        theta[npl].rp  = rp
        theta[npl].b   = b
        theta[npl].T14 = T14

    # calculate likelihood
    loglike = 0.

    for j, q in enumerate(which_quarters):
        light_curve = np.ones(len(t_[j]), dtype='float')
        
        for npl in range(NPL):
            transit_model[npl][j] = batman.TransitModel(theta[npl], ephem[npl]._warp_times(t_[j]))
            light_curve += transit_model[npl][j].light_curve(theta[npl]) - 1.0

        gp = GaussianProcess(kernel[q%4], mean=light_curve)
        gp.compute(t_[j], yerr=e_[j])

        loglike += gp.log_likelihood(f_[j])

    if not np.isfinite(loglike):
        return -1e300

    return loglike



# now run the sampler
sampler = dynesty.DynamicNestedSampler(lnlike, prior_transform, 5*NPL)
sampler.run_nested()
results = sampler.results

labels = []
for npl in range(NPL):
    labels = labels + 'C0_{0} C1_{0} r_{0} b_{0} T14_{0}'.format(npl).split()

#rfig, raxes = dyplot.runplot(results, logplot=True)
#tfig, taxes = dyplot.traceplot(results, labels=labels)
#cfig, caxes = dyplot.cornerplot(results, labels=labels)


# # Save results as dictionary
f_name = PROJECT_DIR + 'Traces/{0}/{0}-nested.pkl'.format(TARGET)

with open(f_name, 'wb') as f:
    pickle.dump(results.asdict(), f)


print("")
print("+"*shutil.get_terminal_size().columns)
print("Exoplanet recovery complete {0}".format(datetime.now().strftime("%d-%b-%Y at %H:%M:%S")))
print("Total runtime = %.1f min" %((timer()-global_start_time)/60))
print("+"*shutil.get_terminal_size().columns)

