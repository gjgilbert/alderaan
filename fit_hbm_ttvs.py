#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt

import numpy.polynomial.polynomial as poly
import scipy.optimize as op
import scipy.signal as sig
from   scipy import stats
from   scipy import fftpack
from   scipy.interpolate import UnivariateSpline
import astropy
from   astropy.io import fits as pyfits
from   astropy.timeseries import LombScargle
import random

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

# In[2]:


# here's where you can hard code in mission, target, and paths
#MISSION = "Simulated"
#TARGET  = "K00137"
#PRIMARY_DIR = '/Users/research/projects/alderaan/'

USE_HBM = False


# In[3]:


# here's where we parse the inputs from an sbatch script
try:
    parser = argparse.ArgumentParser(description="Inputs for ALDERAAN transit fiting pipeline")
    parser.add_argument("--mission", default=None, type=str, required=True,                         help="Mission name")
    parser.add_argument("--target", default=None, type=str, required=True,                         help="Target name; see ALDERAAN documentation for acceptable formats")
    parser.add_argument("--primary_dir", default=None, type=str, required=True,                         help="Primary directory path for accessing lightcurve data and saving outputs")
    
    args = parser.parse_args()
    MISSION     = args.mission
    TARGET      = args.target
    PRIMARY_DIR = args.primary_dir
    
except:
    pass


# # Make sure the necessary paths exist

# In[4]:


# directory in which to find lightcurve data
#if MISSION == 'Kepler': DOWNLOAD_DIR = PRIMARY_DIR + 'MAST_downloads/'
#if MISSION == 'Simulated': DOWNLOAD_DIR = PRIMARY_DIR + 'Simulations/'

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


# # Get shape model posteriors

# In[5]:


TRACE_FILE = TRACE_DIR + TARGET + '_transit_shape.fits'


# In[6]:


# Read in the fits file with saved traces
with pyfits.open(TRACE_FILE) as trace:
    header  = trace[0].header
    hdulist = pyfits.HDUList(trace)
    
    NDRAWS, NPL = trace['RP'].shape
    
    # stellar parameters
    RSTAR  = trace['RSTAR'].data
    MSTAR  = trace['MSTAR'].data
    U      = trace['U'].data
    U1, U2 = U[:,0], U[:,1]
    
    # planetary parameters
    T0   = trace['T0'].data
    P    = trace['P'].data
    LOGR = trace['LOGR'].data
    RP   = trace['RP'].data
    B    = trace['B'].data
    
    # TTV parameters
    TTS = [None]*NPL

    for npl in range(NPL):    
        TTS[npl] = trace['TTS_{0}'.format(npl)].data
            

    C0 = trace['C0'].data
    C1 = trace['C1'].data
    
    try: C2 = trace['C2'].data
    except: pass
    
    try: C3 = trace['C3'].data
    except: pass


# In[7]:


with pyfits.open(TRACE_FILE) as trace:
    header  = trace[0].header
    hdulist = pyfits.HDUList(trace)
    
    NDRAWS, NPL = trace['RP'].shape
    
    
    # GP parameters
    LOGSW4 = np.zeros((NDRAWS,4))
    LOGW0  = np.zeros((NDRAWS,4))
    LOGQ   = np.zeros((NDRAWS,4))
    
    for z in range(4):
        try: LOGSW4[:,z] = trace['LOGSW4_{0}'.format(z)].data
        except: LOGSW4[:,z] = np.nan
        
        try: LOGW0[:,z] = trace['LOGW0_{0}'.format(z)].data
        except: LOGW0[:,z] = np.nan
        
        try: LOGQ[:,z] = trace['LOGQ_{0}'.format(z)].data
        except: LOGQ[:,z] = np.nan


# # Get transit times, linear ephemeris and O-C

# In[8]:


epochs = np.zeros(NPL)
periods = np.zeros(NPL)

shape_transit_times = []
shape_ephemeris = []
transit_inds = []

for npl in range(NPL):   
    shape_transit_times.append(np.median(TTS[npl], axis=0))
    
    t0  = np.median(T0[:,npl])
    per = np.median(P[:,npl])
    tts = shape_transit_times[npl]
    
    tinds = np.array(np.floor((tts - t0 + per/2) / per), dtype='int')
    
    epochs[npl] = t0
    periods[npl] = per
    shape_ephemeris.append(t0 + per*tinds)
    transit_inds.append(tinds)


# In[9]:


fig, axes = plt.subplots(NPL, figsize=(12,8))

for npl in range(NPL):
    xtime = shape_transit_times[npl]
    yomc  = (shape_transit_times[npl]-shape_ephemeris[npl])*24*60
    
    axes[npl].plot(xtime, yomc, color='C{0}'.format(npl), lw=3)
    axes[npl].set_ylabel('O-C [min]', fontsize=20)
    
axes[NPL-1].set_xlabel('Time [BJKD]', fontsize=20)    
plt.close()


# # Read in detrended lightcurves and independent TTV estimates

# In[10]:


# Load detrended lightcurves
try:
    lc = io.load_detrended_lightcurve(DLC_DIR + TARGET + '_lc_detrended.fits')
except:
    lc = None
    
try:
    sc = io.load_detrended_lightcurve(DLC_DIR + TARGET + '_sc_detrended.fits')
except:
    sc = None


# In[11]:


# Read in QuickTTV estimates and calculate linear ephemeris for each planet
transit_inds = []
indep_transit_times = []
indep_ephemeris = []


for npl in range(NPL):
    fname_in = QUICK_TTV_DIR + TARGET + '_{:02d}'.format(npl) + '_quick_ttvs.txt'
    data_in  = np.genfromtxt(fname_in)
    
    transit_inds.append(data_in[:,0])
    indep_transit_times.append(data_in[:,1])
    
    pfit = poly.polyfit(transit_inds[npl], indep_transit_times[npl], 1)
    
    indep_ephemeris.append(poly.polyval(transit_inds[npl], pfit))
    
    
fig, axes = plt.subplots(NPL, figsize=(12,8))

for npl in range(NPL):
    xtime = indep_ephemeris[npl]
    yomc  = (indep_transit_times[npl] - indep_ephemeris[npl])*24*60
    
    axes[npl].plot(xtime, yomc, '.', c='C{0}'.format(npl))
    axes[npl].set_ylabel('O-C [min]', fontsize=20)
axes[NPL-1].set_xlabel('Time [BJKD]', fontsize=20)
plt.close()


# In[12]:


# get estimate of ttv uncertainty and a reasonable buffer
ttv_scatter = np.zeros(NPL)
ttv_rms_amp = np.zeros(NPL)

for npl in range(NPL):
    ttv_scatter[npl] = np.std(indep_transit_times[npl] - shape_transit_times[npl])
    ttv_rms_amp[npl] = np.std(indep_transit_times[npl] - shape_ephemeris[npl])


# # Break the transits into subgroups
# #### Groups are indexed starting at 1, with 0 reserved for overlapping transits

# In[13]:


MAX_GROUP_SIZE = 32
MIN_GROUP_SIZE = 16

tts_group = []
local_trend_type = []

sin_priors = []

fig, axes = plt.subplots(NPL, figsize=(12,8))


for npl in range(NPL):
    print("\nPLANET", npl)
    
    xtime = indep_ephemeris[npl]
    yomc  = (indep_transit_times[npl] - indep_ephemeris[npl])*24*60
    
    ntrans = len(xtime)
    peakfreq = None

    # search for a period component
    try:
        xf, yf, freqs, faps, levels = LS_estimator(xtime, yomc, return_levels=True)

        if len(freqs) > 0:
            if freqs[0] > xf.min():
                peakfreq = freqs[0]
                
    except:
        pass
        
        
    # roughly model OMC based on single frequency sinusoid (if found)
    if peakfreq is not None:
        trend_type = "sinusoid"

        print("periodic component found at P = ", int(1/peakfreq), "d")
        
        LS = LombScargle(xtime, yomc)
        
        trend = LS.model(xtime, peakfreq)
        _, Asin, Bcos = LS.model_parameters(peakfreq)
        
        sin_priors.append((peakfreq, Asin, Bcos))
        
        edges = [xtime.min()-0.5, xtime.max()+0.5]
        ngroups = int(np.ceil((xtime.max()-xtime.min())*peakfreq))

        for i in range(1,ngroups):
            edges.append(xtime.min()+i/ngroups*(xtime.max()-xtime.min()))
            
        edges = list(np.sort(edges))
        
    
    # otherwise, chose best polynomial model based on the BIC
    else:
        trend_type = "polynomial"
        sin_priors.append(None)
        
        biclist = []
        for i in range(1, int(np.ceil(np.sqrt(ntrans)))):
            poly_coeffs = poly.polyfit(xtime, yomc, i)
            poly_trend = poly.polyval(xtime, poly_coeffs)

            RSS = np.sum((yomc-poly_trend)**2)
            BIC = ntrans*np.log(RSS/ntrans) + (i+1)*np.log(ntrans)

            biclist.append(BIC)

        polyorder = np.argmin(biclist)+1
        trend = np.polyval(np.polyfit(xtime, yomc, polyorder), xtime)

        print("lowest BIC at polyorder = {0}".format(polyorder))


        # fit a spline and find inflecton points
        if polyorder > 2:
            spline = UnivariateSpline(xtime, trend, k=5)
            inflects = spline.derivative(2).roots()
            curvature = np.array(np.sign(spline.derivative(2)(xtime)), dtype="int")

            # define edges of groups
            edges = np.hstack([xtime.min()-0.5, inflects, xtime.max()+0.5])
            edges = list(np.sort(edges))
            
        else:
            curvature = np.zeros(len(trend), dtype="int")
            
            edges = [xtime.min()-0.5, xtime.max()+0.5]

        
    # makes sure the groups are not too small (i.e. edges near endpoints)
    if len(edges) > 2:
        if int((edges[-1]-edges[-2])/periods[npl]) < MIN_GROUP_SIZE:
            edges.remove(edges[-2])
        if int((edges[1]-edges[0])/periods[npl]) < MIN_GROUP_SIZE:
            edges.remove(edges[1])

    if len(edges) < 2:
        raise ValueError("There should be at least two edges to indicate endpoints")


    # make sure the groups are not too big
    for i in range(len(edges)-1):
        count = int(np.ceil((edges[i+1]-edges[i])/periods[npl]/MAX_GROUP_SIZE))

        for j in range(1,count):
            new = edges[i] + (j/count)*(edges[i+1]-edges[i])
            edges.append(new)

    edges = np.sort(edges)


    # assign each transit to a group
    tg = np.zeros(ntrans, dtype="int")
    
    for edj in edges:
        tg = tg + np.array(xtime >= edj, dtype="int")
        
    tts_group.append(tg)
    
    
    # determine the best local model for each group
    if trend_type == "sinusoid":
        local_trend_type.append(["sinusoid"]*tts_group[npl].max())
    
    
    elif trend_type == "polynomial":
        if polyorder <= 2:
            local_trend_type.append(["linear"]*tts_group[npl].max())
        else:
            local_trend_type.append(["quadratic"]*tts_group[npl].max())
        

    # display the results
    axes[npl].plot(xtime, yomc, 'o', c='lightgrey')
    axes[npl].plot(xtime, trend, c='C{0}'.format(npl), lw=2)
    for edj in edges:
        axes[npl].axvline(edj, c='k', lw=1, ls='--')
    axes[npl].set_ylabel('O-C [min]', fontsize=20)
    
axes[NPL-1].set_xlabel('Time [BJKD]', fontsize=20)
plt.close()    


# In[14]:


# interlace groups
for npl in range(NPL):
    for g in range(1, tts_group[npl].max()):
        
        count = np.min([np.sum(tts_group[npl] == g), np.sum(tts_group[npl] == g+1)])
        loc = np.where(tts_group[npl] == g)[0].max()
        
        for c in range(np.min([count//6,3])):
            tts_group[npl][loc-2*c] = g+1
            tts_group[npl][loc+2*c+1] = g


# # Set fixed values for star and planet parameters

# In[15]:


# identify which sample is closest to the median for all parameters

dist_sq = ((U1 - np.median(U1))/np.std(U1))**2
dist_sq = ((U2 - np.median(U2))/np.std(U2))**2
dist_sq = ((RSTAR - np.median(RSTAR))/np.std(RSTAR))**2
dist_sq = ((MSTAR - np.median(MSTAR))/np.std(MSTAR))**2

for npl in range(NPL):
    dist_sq += ((RP[:,npl] - np.median(RP[:,npl]))/np.std(RP[:,npl]))**2
    dist_sq += ((B[:,npl] - np.median(B[:,npl]))/np.std(B[:,npl]))**2
    dist_sq += ((P[:,npl] - np.median(P[:,npl]))/np.std(P[:,npl]))**2
    dist_sq += ((T0[:,npl] - np.median(T0[:,npl]))/np.std(T0[:,npl]))**2


loc = np.argmin(dist_sq)


# In[16]:


# grab star and planet parameters for that sample
u = U1[loc], U2[loc]

Rstar = RSTAR[loc]
Mstar = MSTAR[loc]

rp = np.array(RP[loc], dtype="float")
b  = np.array(B[loc], dtype="float")

periods = np.array(P[loc], dtype="float")
epochs = np.array(T0[loc], dtype="float")

sma = get_sma(periods, Mstar)
durs = get_dur_tot(periods, rp, Rstar, b, sma)


# # Identify overlapping transits

# In[17]:


if lc is not None and sc is not None:
    quarters = np.unique(np.hstack([lc.quarter, sc.quarter]))
    
elif lc is not None and sc is None:
    quarters = np.unique(lc.quarter)
    
elif lc is None and sc is not None:
    quarters = np.unique(sc.quarter)
    
else:
    raise ValueError("Expected either short cadence or long cadence data...how did you get this far?")


# In[18]:


dur_max = np.max(durs)
overlap = []

for i in range(NPL):
    overlap.append(np.zeros(len(shape_transit_times[i]), dtype='bool'))
    
    for j in range(NPL):
        if i != j:
            for tt in shape_transit_times[j]:
                overlap[i] += np.abs(shape_transit_times[i] - tt) < dur_max


# In[19]:


thresh = np.zeros(len(quarters)+1)

try:
    thresh[0] = np.min(np.hstack([lc.time, sc.time]))
except:
    thresh[0] = np.min(lc.time)

for j, q in enumerate(quarters):
    try:
        thresh[j+1] = np.max(np.hstack([lc.time[lc.quarter==q],sc.time[sc.quarter==q]]))
    except:
        thresh[j+1] = np.max([lc.time[lc.quarter==q]])
    
thresh[0] -= 1.0
thresh[-1] += 1.0

transit_quarter = [None]*NPL

for npl in range(NPL):
    stt = shape_transit_times[npl]
    transit_quarter[npl] = np.zeros(len(stt), dtype='int')

    for j, q in enumerate(quarters):
        transit_quarter[npl][(stt >= thresh[j])*(stt<thresh[j+1])] = q


# In[20]:


# assign overlapping transits to group -99
for npl in range(NPL):
    tts_group[npl][overlap[npl]] = -99


# # Renormalize individual transits

# In[21]:


if lc is not None:
    t_ = lc.time
    f_ = lc.flux

    for npl in range(NPL):
        print("\nPLANET", npl)
        for i, t0 in enumerate(shape_transit_times[npl]):
            if tts_group[npl][i] != -99:
                wide_mask = np.abs(t_-t0)/durs[npl] < 1.5
                narrow_mask = np.abs(t_-t0)/durs[npl] < 0.67
                
                m_ = wide_mask*~narrow_mask
                
                if (np.sum(wide_mask)==0) + (np.sum(narrow_mask)==0) + (np.sum(m_)==0):
                    warnings.warn("Found a transit with no photometric data...this is unexpected")
                    
                else:
                    trend = poly.polyval(t_[wide_mask], poly.polyfit(t_[m_], f_[m_], 1))

                    f_[wide_mask] /= trend
                    f_[wide_mask] /= np.mean(f_[m_])
                

    lc.flux = np.copy(f_)


# # Set up GP

# In[22]:


# Read in quarter-by-quarter variances
var_by_quarter = np.genfromtxt(NOISE_DIR + TARGET + '_var_by_quarter.txt')[:,1]

# this grabs a single sample based on the transit parameters
gp_priors = []

for z in range(4):
    gpz = {}
    
    # check if any noise GP variables are missing
    bad = np.isnan(LOGW0[:,z][loc]) + np.isnan(LOGSW4[:,z][loc]) + np.isnan(LOGQ[:,z][loc])
    
    # where noise GP values exist from the transit shape fits, use these
    if ~bad:
        gpz['logw0'] = (LOGW0[:,z][loc], 0.0)
        gpz['logSw4'] = (LOGSW4[:,z][loc], 0.0)
        gpz['logQ'] = (LOGQ[:,z][loc], 0.0)
        
    # otherwise, these dummy values establish an essentially zero-amplitude kernel
    else:
        gpz['logw0'] = (np.log(2*pi/(5*durs.max())), 0.0)
        gpz['logSw4'] = (-40., 0.0)
        gpz['logQ'] = (np.log(1/np.sqrt(2)), 0.0)
        
    gp_priors.append(gpz)


# # Grab the relevant data

# In[23]:


# grab data near transits for each quarter
all_time = []
all_flux = []
all_mask = []
all_dtype = []


if sc is not None:
    for q in range(18):
        if np.isin(q, sc.quarter)*np.isin(q, lc.quarter):
            raise ValueError("Double counting data in both short and long cadence")


        elif np.isin(q, sc.quarter):
            use = (sc.mask.sum(axis=0) > 0)*(sc.quarter == q)

            if np.sum(use) > 45:
                all_time.append(sc.time[use])
                all_flux.append(sc.flux[use])
                all_mask.append(sc.mask[:,use])
                all_dtype.append('short')
                
            else:
                all_time.append(None)
                all_flux.append(None)
                all_mask.append(None)
                all_dtype.append('none')


        elif np.isin(q, lc.quarter):
            use = (lc.mask.sum(axis=0) > 0)*(lc.quarter == q)
            
            if np.sum(use) > 3:
                all_time.append(lc.time[use])
                all_flux.append(lc.flux[use])
                all_mask.append(lc.mask[:,use])
                all_dtype.append('long')
                
            else:
                all_time.append(None)
                all_flux.append(None)
                all_mask.append(None)
                all_dtype.append('none')


        else:
            all_time.append(None)
            all_flux.append(None)
            all_mask.append(None)
            all_dtype.append('none')
            
else:
    for q in range(18):
        if np.isin(q, lc.quarter):
            use = (lc.mask.sum(axis=0) > 0)*(lc.quarter == q)

            if np.sum(use) > 3:
                all_time.append(lc.time[use])
                all_flux.append(lc.flux[use])
                all_mask.append(lc.mask[:,use])
                all_dtype.append('long')
                
            else:
                all_time.append(None)
                all_flux.append(None)
                all_mask.append(None)
                all_dtype.append('none')


        else:
            all_time.append(None)
            all_flux.append(None)
            all_mask.append(None)
            all_dtype.append('none')



# check which quarters have data
good = (np.array(all_dtype) == 'short') + (np.array(all_dtype) == 'long')
quarters = np.arange(18)[good]
nq = len(quarters)

lc_quarters = np.arange(18)[np.array(all_dtype) == 'long']
sc_quarters = np.arange(18)[np.array(all_dtype) == 'short']


# expand var_by_quarter to have None for missing quarters
vbq_all = [None]*18

for j, q in enumerate(quarters):
    vbq_all[q] = var_by_quarter[j]
    
vbq_all = np.asarray(vbq_all, dtype="float")


# In[24]:


print('')
print('cumulative runtime = ', int(timer() - global_start_time), 's')
print('')


# # Make masks for the photometry to separate transit groups

# In[25]:


# group mask for non-overlapping transits
group_mask = [None]*18

for q in range(18):
    if all_mask[q] is not None:        
        t_ = all_time[q]
        m_ = np.array(all_mask[q].copy(), dtype="int")
    
        for npl in range(NPL):
            m_[npl] *= 0
            
            for i, t0 in enumerate(shape_transit_times[npl]):
                m_[npl][np.abs(t_-t0)/durs[npl] < 3.0] = tts_group[npl][i]
                
        group_mask[q] = np.copy(m_)


# In[27]:


# plot non-overlapping tranits each planet, color-coded by group 
for npl in range(NPL):
    plt.figure(figsize=(20,2))
    
    for j, q in enumerate(quarters):        
        t_ = all_time[q]
        f_ = all_flux[q]
        m_ = group_mask[q][npl]
        
        for ng in range(1,tts_group[npl].max()+1):
            plt.plot(t_[m_==ng], f_[m_==ng], ".", c="C{0}".format((ng-1)%10))
        
    plt.close()
    


plt.figure(figsize=(20,3))


# plot all overlapping transits for all planets
for j, q in enumerate(quarters):        
    t_ = all_time[q]
    f_ = all_flux[q]
    m_ = np.sum(group_mask[q] == -99, axis=0) > 0

    for ng in range(1,tts_group[npl].max()+1):
        plt.plot(t_[m_], f_[m_], "k.")


plt.close()


# # Fit transit times

# In[28]:


tts_chains = []
offset_chains = []
pop_sd_chains = []
omc_trend_chains = []

for npl in range(NPL):
    for ng in range(1,tts_group[npl].max()+1):
        print("\nPLANET {0}, GROUP {1}".format(npl,ng))
        
        # identify which quarters and which_seasons have data
        use = tts_group[npl] == ng
        
        wq = np.unique(transit_quarter[npl][use])
        ws = np.unique(wq % 4)
                        
        print(np.sum(use), "transits")
        print("quarters:", wq)
        print("local trend:", local_trend_type[npl][ng-1])
        
        # grab transit times
        fixed_tts = np.copy(shape_transit_times[npl][use])
        fixed_inds = np.array(transit_inds[npl][use], dtype="int")
        fixed_ephem = np.copy(shape_ephemeris[npl][use])
        
        fixed_inds -= fixed_inds[0]
                
        # define Legendre polynomials
        x = 2*(fixed_ephem-np.min(fixed_ephem))/(np.max(fixed_ephem)-np.min(fixed_ephem)) - 1
                
        Leg0 = np.ones_like(x)
        Leg1 = x.copy()
        Leg2 = 0.5*(3*x**2 - 1)
        
        
        # now build the model
        with pm.Model() as hbm_model:
            # local trend parameters
            C0 = pm.Normal("C0", mu=0.0, sd=2*ttv_rms_amp[npl])
            C1 = pm.Normal("C1", mu=0.0, sd=2*ttv_rms_amp[npl])
            
            if local_trend_type[npl][ng-1] == "linear":
                C2 = 0.0
                sin_omc = T.zeros(len(fixed_ephem)) 
                
            elif local_trend_type[npl][ng-1] == "quadratic":
                C2 = pm.Normal("C2", mu=0.0, sd=2*ttv_rms_amp[npl])
                sin_omc = T.zeros(len(fixed_ephem)) 

            elif local_trend_type[npl][ng-1] == "sinusoid":
                C2 = 0.0
                
                freq = pm.Normal("freq", mu=sin_priors[npl][0], sd=0.1*sin_priors[npl][0])
                A = pm.Normal("A", mu=0, sd=ttv_rms_amp[npl], testval=sin_priors[npl][1])
                B = pm.Normal("B", mu=0, sd=ttv_rms_amp[npl], testval=sin_priors[npl][2])
                
                sin_omc = pm.Deterministic("sin_omc", 
                                           A*T.sin(2*pi*freq*fixed_ephem) + 
                                           B*T.cos(2*pi*freq*fixed_ephem))
                
                
            else:
                raise ValueError("local_trend_type must be 'linear', 'quadratic', or 'sinusoid'")
                
                
            # hierarchical (hyper)parameters
            if USE_HBM:
                log_pop_var = pm.Normal('log_pop_var', mu=2*np.log(ttv_rms_amp[npl]), 
                                        sd=np.log(4))
            
                pop_sd = pm.Deterministic('pop_sd', T.sqrt(T.exp(log_pop_var)))
                
            else:
                pop_sd = np.sqrt(2)*ttv_rms_amp[npl]
            
            # transit times
            tt_offset = pm.StudentT('tt_offset', nu=2, shape=len(fixed_ephem))
            
            poly_omc  = pm.Deterministic('poly_omc', C0*Leg0 + C1*Leg1 + C2*Leg2)
            omc_trend = pm.Deterministic('omc_trend', poly_omc + sin_omc)
            
            transit_times = pm.Deterministic('tts', fixed_ephem + omc_trend + tt_offset*pop_sd)
            
                        
            # set up stellar model and planetary orbit
            exoSLC = exo.StarryLightCurve(u)
            orbit  = exo.orbits.TTVOrbit(transit_times=[transit_times], transit_inds=[fixed_inds], 
                                         b=b[npl], r_star=Rstar, m_star=Mstar)

            # track period and epoch
            T0 = pm.Deterministic('T0', orbit.t0)
            P  = pm.Deterministic('P', orbit.period)
            
            # nuissance parameters (one mean flux; variance by quarter)
            flux0 = pm.Normal('flux0', mu=np.ones(len(wq)), sd=np.sqrt(vbq_all[wq])/4, shape=len(wq))
            logvar = pm.Normal('logvar', mu=np.log(vbq_all[wq]), sd=np.log(4)*np.ones(len(wq)), shape=len(wq))

            
            # build the GP kernel using a different noise model for each season
            logSw4 = [None]*4
            logw0  = [None]*4
            logQ   = [None]*4
            kernel = [None]*4

            for i in range(4):
                if np.isin(i, ws):
                    gpi = gp_priors[i]              

                    logSw4[i] = gpi['logSw4'][0]
                    logw0[i] = gpi['logw0'][0]
                    logQ[i] = gpi['logQ'][0]

                    kernel[i] = exo.gp.terms.SHOTerm(log_Sw4=logSw4[i], log_w0=logw0[i], log_Q=logQ[i])
                
                else:
                    kernel[i] = None
            
            
            # now evaluate the model for each quarter
            light_curves       = [None]*len(wq)
            summed_light_curve = [None]*len(wq)
            model_flux         = [None]*len(wq)
            
            gp = [None]*len(wq)
            

            for j, q in enumerate(wq):
                # here's the data
                t_ = all_time[q]
                f_ = all_flux[q]
                m_ = group_mask[q][npl] == ng
                
                # sometimes there is a missing transit (?) in the last quarter
                # I think this is an edge/boundary issue but I haven't been able to locate it
                if np.sum(m_) == 0:
                    warnings.warn("Got unexpected empty data array for PLANET {0}, QUARTER {1}".format(npl,q))
                    
                else:
                    # set oversampling factor
                    if all_dtype[q] == 'short':
                        oversample = 1
                    elif all_dtype[q] == 'long':
                        oversample = 15

                    # calculate light curves
                    light_curves[j] = exoSLC.get_light_curve(orbit=orbit, r=rp[npl], t=t_[m_], oversample=oversample)
                    summed_light_curve[j] = pm.math.sum(light_curves[j], axis=-1) + flux0[j]*T.ones(len(t_[m_]))
                    model_flux[j] = pm.Deterministic('model_flux_{0}'.format(j), summed_light_curve[j])

                    # here's the GP (w/ kernel by season)
                    gp[j] = exo.gp.GP(kernel[q%4], t_[m_], T.exp(logvar[j])*T.ones(len(t_[m_])))
                
                
                    # add custom potential (log-prob fxn) with the GP likelihood
                    pm.Potential('obs_{0}'.format(j), gp[j].log_likelihood(f_[m_] - model_flux[j]))
                
                
        with hbm_model:
            hbm_map = exo.optimize(start=hbm_model.test_point, vars=[flux0, logvar])

            if local_trend_type[npl][ng-1] == "linear":
                hbm_map = exo.optimize(start=hbm_map, vars=[C0, C1]) 
            if local_trend_type[npl][ng-1] == "quadratic":
                hbm_map = exo.optimize(start=hbm_map, vars=[C0, C1, C2]) 
            if local_trend_type[npl][ng-1] == "sinusoid":
                hbm_map = exo.optimize(start=hbm_map, vars=[C0, C1, A, B])
            
                            
        # sample from the posterior
        with hbm_model:
            hbm_trace = pm.sample(tune=3000, draws=1000, start=hbm_map, chains=2,
                                  step=exo.get_dense_nuts_step(target_accept=0.9))

        # save the results
        omc_trend_chains.append(np.copy(np.array(hbm_trace['omc_trend'])))
        tts_chains.append(np.copy(np.array(hbm_trace['tts'])))
        offset_chains.append(np.copy(np.array(hbm_trace['tt_offset'])))
        
        if USE_HBM:
            pop_sd_chains.append(np.copy(np.array(hbm_trace['pop_sd'])))
        else:
            pop_sd_chains.append(pop_sd*np.ones(hbm_trace['tts'].shape[0]))


# # Fit overlapping transits

# In[ ]:


overlap_count = 0

for npl in range(NPL):
    overlap_count += np.sum(tts_group[npl]==-99)
    
    
print(overlap_count, "overlapping transits")


# In[ ]:


if overlap_count > 0:
    with pm.Model() as overlap_model:
        fixed_tts = []
        fixed_inds = []
        fixed_ephem = []

        which_planets = []
        which_quarters = []


        # grab the relevant data
        for npl in range(NPL):
            use = tts_group[npl] == -99

            if np.sum(use) > 0:
                fixed_tts.append(np.copy(shape_transit_times[npl][use]))
                fixed_inds.append(np.array(transit_inds[npl][use], dtype="int"))
                fixed_ephem.append(np.copy(shape_ephemeris[npl][use]))

                fixed_inds[npl] -= fixed_inds[npl][0]

                which_planets.append(npl)
                which_quarters.append(transit_quarter[npl][use])

        wq = np.unique(np.hstack(which_quarters))
        ws = np.unique(wq % 4)


        # use a fixed scatter parameter for now
        pop_sd = np.sqrt(2)*ttv_rms_amp

        # transit times
        tt_offset = []
        transit_times = []

        for i, npl in enumerate(which_planets):
            tt_offset.append(pm.StudentT('tt_offset_{0}'.format(npl), nu=2, shape=len(fixed_ephem[i])))
            transit_times.append(pm.Deterministic('tts_{0}'.format(npl), fixed_ephem[i] + tt_offset[i]*pop_sd[npl]))


        # set up stellar model and planetary orbit
        exoSLC = exo.StarryLightCurve(u)
        orbit  = exo.orbits.TTVOrbit(transit_times=transit_times, transit_inds=fixed_inds, 
                                     b=b[npl], r_star=Rstar, m_star=Mstar)

        # track period and epoch
        T0 = pm.Deterministic('T0', orbit.t0)
        P  = pm.Deterministic('P', orbit.period)


        # nuissance parameters (one mean flux; variance by quarter)
        flux0 = pm.Normal('flux0', mu=np.ones(len(wq)), sd=np.sqrt(vbq_all[wq])/4, shape=len(wq))
        logvar = pm.Normal('logvar', mu=np.log(vbq_all[wq]), sd=np.log(4)*np.ones(len(wq)), shape=len(wq))



        # build the GP kernel using a different noise model for each season
        logSw4 = [None]*4
        logw0  = [None]*4
        logQ   = [None]*4
        kernel = [None]*4

        for i in range(4):
            if np.isin(i, ws):
                gpi = gp_priors[i]              

                logSw4[i] = gpi['logSw4'][0]
                logw0[i] = gpi['logw0'][0]
                logQ[i] = gpi['logQ'][0]

                kernel[i] = exo.gp.terms.SHOTerm(log_Sw4=logSw4[i], log_w0=logw0[i], log_Q=logQ[i])

            else:
                kernel[i] = None


        # now evaluate the model for each quarter
        light_curves       = [None]*len(wq)
        summed_light_curve = [None]*len(wq)
        model_flux         = [None]*len(wq)

        gp = [None]*len(wq)


        for j, q in enumerate(wq):
            # here's the data
            t_ = all_time[q]
            f_ = all_flux[q]
            m_ = np.sum(group_mask[q] == -99, axis=0) > 0

            # sometimes there is a missing transit (?) in the last quarter
            # I think this is an edge/boundary issue but I haven't been able to locate it
            if np.sum(m_) == 0:
                warnings.warn("Got unexpected empty data array for PLANET {0}, QUARTER {1}".format(npl,q))

            else:
                # set oversampling factor
                if all_dtype[q] == 'short':
                    oversample = 1
                elif all_dtype[q] == 'long':
                    oversample = 15

                # calculate light curves
                light_curves[j] = exoSLC.get_light_curve(orbit=orbit, r=rp, t=t_[m_], oversample=oversample)
                summed_light_curve[j] = pm.math.sum(light_curves[j], axis=-1) + flux0[j]*T.ones(len(t_[m_]))
                model_flux[j] = pm.Deterministic('model_flux_{0}'.format(j), summed_light_curve[j])

                # here's the GP (w/ kernel by season)
                gp[j] = exo.gp.GP(kernel[q%4], t_[m_], T.exp(logvar[j])*T.ones(len(t_[m_])))


                # add custom potential (log-prob fxn) with the GP likelihood
                pm.Potential('obs_{0}'.format(j), gp[j].log_likelihood(f_[m_] - model_flux[j]))



    # optimize nuissance variables
    with overlap_model:
        overlap_map = exo.optimize(start=overlap_model.test_point, vars=[flux0, logvar])


    # sample from the posterior
    with overlap_model:
        overlap_trace = pm.sample(tune=3000, draws=1000, start=overlap_map, chains=2,
                                  step=exo.get_dense_nuts_step(target_accept=0.9))


    for i, npl in enumerate(which_planets):
        tts_chains.append(np.copy(np.array(overlap_trace['tts_{0}'.format(npl)])))
        offset_chains.append(np.copy(np.array(overlap_trace['tt_offset_{0}'.format(npl)])))
        pop_sd_chains.append(pop_sd[npl]*np.ones(overlap_trace['tts_{0}'.format(npl)].shape[0]))
        omc_trend_chains.append(np.zeros_like(overlap_trace['tts_{0}'.format(npl)]))


# In[ ]:


# make a list of ordered pairs (npl,group) to help organize the chains
chain_organizer = []

for npl in range(NPL):
    for ng in range(1, 1+tts_group[npl].max()):
        chain_organizer.append((npl,ng))

if overlap_count > 0:
    for i, npl in enumerate(which_planets):
        chain_organizer.append((npl,-99))


# In[ ]:


# make primary HDU
primary_hdu = pyfits.PrimaryHDU()
header = primary_hdu.header
header['TARGET'] = TARGET
primary_hdu.header = header
    
# add it to HDU list
hbm_hdulist = []
hbm_hdulist.append(primary_hdu)


# grab all samples from trace
for npl in range(NPL):
    combo_tts = []
    combo_omctrend = []
    combo_offset = []
    combo_pop_sd = []
    combo_groupno = []
    
    for i, chorg in enumerate(chain_organizer):
        if chorg[0] == npl:
            combo_tts.append(tts_chains[i])
            combo_offset.append(offset_chains[i])
            combo_groupno.append(chorg[1]*np.ones(tts_chains[i].shape[1], dtype="int"))
            combo_pop_sd.append(pop_sd_chains[i])
            combo_omctrend.append(omc_trend_chains[i])

        
    combo_tts = np.hstack(combo_tts)
    combo_omctrend = np.hstack(combo_omctrend)
    combo_offset = np.hstack(combo_offset)
    combo_pop_sd = np.vstack(combo_pop_sd).swapaxes(0,1)
    combo_groupno = np.hstack(combo_groupno)

    order = np.argsort(np.nanmedian(combo_tts,0))

    combo_tts = combo_tts[:,order]
    combo_omctrend = combo_omctrend[:,order]
    combo_offset = combo_offset[:,order]
    combo_groupno = combo_groupno[order]
    
    
    # add to HDUList
    hbm_hdulist.append(pyfits.ImageHDU(combo_tts, name='TTS_{0}'.format(npl)))
    hbm_hdulist.append(pyfits.ImageHDU(combo_omctrend, name='TREND_{0}'.format(npl)))
    hbm_hdulist.append(pyfits.ImageHDU(combo_offset, name='OFFSET_{0}'.format(npl)))
    hbm_hdulist.append(pyfits.ImageHDU(combo_pop_sd, name='POP_SD_{0}'.format(npl)))
    hbm_hdulist.append(pyfits.ImageHDU(combo_groupno, name='GROUP_{0}'.format(npl)))

    
hbm_hdulist = pyfits.HDUList(hbm_hdulist)
hbm_hdulist.writeto(TRACE_DIR + TARGET + '_hbm_ttvs.fits', overwrite=True)

print('TOTAL RUNTIME = %.2f min' %((timer()-global_start_time)/60))


# In[ ]:





# In[ ]:




