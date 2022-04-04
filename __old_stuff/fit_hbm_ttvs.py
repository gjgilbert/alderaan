#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt

import numpy.polynomial.polynomial as poly
import astropy.stats
from   astropy.timeseries import LombScargle
from   astropy.io import fits as pyfits
from   scipy.interpolate import UnivariateSpline
from   scipy import stats

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

# In[ ]:


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

# In[ ]:


TRACE_FILE = TRACE_DIR + TARGET + '_transit_shape.fits'


# In[ ]:


with pyfits.open(TRACE_FILE) as trace:
    print(trace.info())


# In[ ]:


# Read in the fits file with saved traces

with pyfits.open(TRACE_FILE) as trace:
    header  = trace[0].header
    hdulist = pyfits.HDUList(trace)
    
    NDRAWS, NPL = trace['P'].shape
    
    # limb darkening parameters
    U_LIMBDARK = np.array(trace['U'].data, dtype="float")
    Q_LIMBDARK = np.array(trace['Q_LIMBDARK'].data, dtype="float")
    
    # basis parameters
    C0 = np.array(trace['C0'].data, dtype="float")
    C1 = np.array(trace['C1'].data, dtype="float")
    BETA = np.array(trace['BETA'].data, dtype="float")
    LOG_ROR = np.array(trace['LOG_ROR'].data, dtype="float")/np.log(10)
    LOG_DUR = np.array(trace['LOG_DUR'].data, dtype="float")/np.log(10)
    
    # physical parameters
    P   = np.array(trace['P'].data, dtype="float")
    T0  = np.array(trace['T0'].data, dtype="float")
    RP  = np.array(trace['RP'].data, dtype="float")*RSRE
    B   = np.array(trace['B'].data, dtype="float")
    RHO = np.array(trace['RHO'].data, dtype="float")
    DUR = np.array(trace['DUR'].data, dtype="float")
    
    # TTV parameters
    TTS = [None]*NPL

    for npl in range(NPL):    
        TTS[npl] = trace['TTS_{0}'.format(npl)].data


# # Set fixed values for star and planet parameters

# #### TODO: Fit a "cannonical" model rather than pulling a single trace

# In[ ]:


basis = [C0, C1, BETA, LOG_ROR, LOG_DUR]

# identify which sample is closest to the median for all parameters
dist_sq = np.zeros(NDRAWS)


for i, var in enumerate(basis):
    for npl in range(NPL):
        dist_sq += ((var[:,npl] - np.median(var[:,npl]))/np.std(var[:,npl]))**2
        
loc = np.argmin(dist_sq)


# In[ ]:


# grab star and planet parameters for that sample
u = U_LIMBDARK[loc]

ror = np.exp(np.array(LOG_ROR[loc], dtype="float"))
b   = np.array(B[loc], dtype="float")
dur = np.array(DUR[loc], dtype="float")

periods = np.array(P[loc], dtype="float")
epochs  = np.array(T0[loc], dtype="float")


# # Read in detrended lightcurves and QuickTTV estimates

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


transit_inds = []
ephemeris = []

quick_transit_times = []
map_transit_times = []


# load Quick TTVs
for npl in range(NPL):
    fname_in = QUICK_TTV_DIR + TARGET + '_{:02d}'.format(npl) + '_quick_ttvs.txt'
    data_in  = np.genfromtxt(fname_in)
    
    quick_transit_times.append(np.array(data_in[:,1], dtype="float"))
    
    transit_inds.append(np.array(data_in[:,0], dtype="int"))
    ephemeris.append(poly.polyval(transit_inds[npl], [epochs[npl], periods[npl]]))    
    

# load MAP TTVs
for npl in range(NPL):
    fname_in = QUICK_TTV_DIR + TARGET + '_{:02d}'.format(npl) + '_map_ttvs.txt'
    data_in  = np.genfromtxt(fname_in)
    
    map_transit_times.append(np.array(data_in[:,1], dtype="float"))    
    
    
fig, axes = plt.subplots(NPL, figsize=(12,3*NPL))

for npl in range(NPL):
    xtime = ephemeris[npl]
    yomc_q = (quick_transit_times[npl] - ephemeris[npl])*24*60
    yomc_m = (map_transit_times[npl] - ephemeris[npl])*24*60
    
    axes[npl].plot(xtime, yomc_q, '-', c='C{0}'.format(npl))
    axes[npl].plot(xtime, yomc_m, '.', c='C{0}'.format(npl))
    axes[npl].set_ylabel('O-C [min]', fontsize=20)
axes[NPL-1].set_xlabel('Time [BJKD]', fontsize=20)
plt.close()


# In[ ]:


# get estimate of ttv amplitude and a reasonable buffer
ttv_rms_amp = np.zeros(NPL)
ttv_buffer  = np.zeros(NPL)

for npl in range(NPL):
    # estimate TTV amplitude
    ttv_rms_amp[npl] = astropy.stats.mad_std(map_transit_times[npl] - ephemeris[npl])

    # based on scatter in independent times, set threshold so not even one outlier is expected
    N   = len(transit_inds[npl])
    eta = np.max([3., stats.norm.interval((N-1)/N)[1]])

    ttv_buffer[npl] = eta*ttv_rms_amp[npl] + lcit


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


# # Determine time baseline

# In[ ]:


time_min = []
time_max = []

if lc is not None:
    time_min.append(lc.time.min())
    time_max.append(lc.time.max())

if sc is not None:
    time_min.append(sc.time.min())
    time_max.append(sc.time.max())

TIME_START = np.min(time_min)
TIME_END   = np.max(time_max)

if TIME_START < 0:
    raise ValueError("START TIME [BKJD] is negative...this will cause problems")


# # Break the transits into subgroups
# #### Groups are indexed starting at 1, with 0 reserved for overlapping transits

# In[ ]:


# fit at most MAX_GROUP_SIZE transits at a time
tts_group = []

for npl in range(NPL):
    ntrans = len(ephemeris[npl])
    ngroups = int(np.ceil(ntrans)/MAX_GROUP_SIZE)
    groupsize = int(np.ceil(ntrans)/ngroups) + 1
    
    tts_group.append(np.repeat(np.arange(1,ngroups+1), groupsize)[:ntrans])


# In[ ]:


if INTERLACE:
    # interlace groups
    for npl in range(NPL):
        for g in range(1, tts_group[npl].max()):

            count = np.min([np.sum(tts_group[npl] == g), np.sum(tts_group[npl] == g+1)])
            loc = np.where(tts_group[npl] == g)[0].max()

            for c in range(np.min([count//6,3])):
                tts_group[npl][loc-2*c] = g+1
                tts_group[npl][loc+2*c+1] = g


# # Identify overlapping transits

# In[ ]:


if lc is not None:
    lc_quarters = np.unique(lc.quarter)
else:
    lc_quarters = np.array([])
    
if sc is not None:
    sc_quarters = np.unique(sc.quarter)
else:
    sc_quarters = np.array([])
    
    
quarters = np.sort(np.hstack([lc_quarters, sc_quarters]))


# In[ ]:


overlap = []

for i in range(NPL):
    overlap.append(np.zeros(len(ephemeris[i]), dtype='bool'))
    
    for j in range(NPL):
        if i != j:
            for tt in ephemeris[j]:
                overlap[i] += np.abs(ephemeris[i] - tt) < (dur[i] + dur[j] + lcit)
                
                
# assign overlapping transits to group -99
for npl in range(NPL):
    tts_group[npl][overlap[npl]] = -99


# # Track which quarter each transit falls in

# In[ ]:


thresh = np.zeros(len(quarters)+1)

thresh[0] = TIME_START

for j, q in enumerate(quarters):
    if np.isin(q, sc_quarters):
        thresh[j+1] = sc.time[sc.quarter == q].max()
    if np.isin(q, lc_quarters):
        thresh[j+1] = lc.time[lc.quarter == q].max()
        
thresh[0] -= 1.0
thresh[-1] += 1.0


# In[ ]:


transit_quarter = [None]*NPL

for npl in range(NPL):
    tts = ephemeris[npl]
    transit_quarter[npl] = np.zeros(len(tts), dtype='int')

    for j, q in enumerate(quarters):
        transit_quarter[npl][(tts >= thresh[j])*(tts<thresh[j+1])] = q


# # Renormalize individual transits

# In[ ]:


if lc is not None:
    t_ = lc.time
    f_ = lc.flux

    for npl in range(NPL):
        print("\nPLANET", npl)
        for i, t0 in enumerate(ephemeris[npl]):
            if tts_group[npl][i] != -99:
                wide_mask = np.abs(t_-t0) < 1.5*dur[npl] + ttv_buffer[npl]
                narrow_mask = np.abs(t_-t0) < 0.5*dur[npl] + ttv_buffer[npl]
                
                m_ = wide_mask*~narrow_mask
                
                if (np.sum(wide_mask)==0) + (np.sum(narrow_mask)==0) + (np.sum(m_)==0):
                    warnings.warn("Found a transit with no photometric data...this is unexpected")
                    
                else:
                    f_[wide_mask] /= np.mean(f_[m_])
                
    lc.flux = np.copy(f_)


# # Make photometry masks

# In[ ]:


if sc is not None:
    sc_mask = np.zeros((NPL,len(sc.time)),dtype='int')
    
    for npl in range(NPL):
        masksize = np.max([2/24,1.5*dur[npl]])        
        
        for i, t0 in enumerate(quick_transit_times[npl]):
            use = np.abs(sc.time - t0) <= masksize
            
            sc_mask[npl][use] = tts_group[npl][i]

else:
    sc_mask = None

    
if lc is not None:
    lc_mask = np.zeros((NPL,len(lc.time)),dtype='int')
    
    for npl in range(NPL):
        masksize = np.max([2/24,1.5*dur[npl]])        
        
        for i, t0 in enumerate(quick_transit_times[npl]):
            use = np.abs(lc.time - t0) <= masksize
            
            lc_mask[npl][use] = tts_group[npl][i]
        
else:
    lc_mask = None


# In[ ]:


for npl in range(NPL):
    plt.figure(figsize=(20,4))
    
    for g in np.unique(tts_group[npl]):
        m_ = lc_mask[npl] == g
        
        if g != -99:
            plt.plot(lc.time[m_], lc.flux[m_], c="C{0}".format(g))
    
    plt.close()


# # Grab the relevant data

# In[ ]:


# grab data near transits for each quarter
all_time = [None]*18
all_flux = [None]*18
all_mask = [None]*18
all_dtype = ["none"]*18

lc_flux = []
sc_flux = []


for q in range(18):
    if sc is not None:
        if np.isin(q, sc.quarter):
            use = (sc_mask.sum(0) != 0)*(sc.quarter == q)

            if np.sum(use) > 45:
                all_time[q] = sc.time[use]
                all_flux[q] = sc.flux[use]
                all_mask[q] = sc_mask[:,use]
                all_dtype[q] = "short"

                sc_flux.append(sc.flux[use])
                
            else:
                all_dtype[q] = "short_no_transits"

    
    if lc is not None:
        if np.isin(q, lc.quarter):
            use = (lc_mask.sum(0) != 0)*(lc.quarter == q)

            if np.sum(use) > 5:
                all_time[q] = lc.time[use]
                all_flux[q] = lc.flux[use]
                all_mask[q] = lc_mask[:,use]
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


print('')
print('cumulative runtime = ', int(timer() - global_start_time), 's')
print('')


# # Fit transit times

# In[ ]:


print("Fitting transit times\n\n")


# In[ ]:


tts_chains = []
offset_chains = []
pop_sd_chains = []

for npl in range(NPL):
    for ng in range(1,tts_group[npl].max()+1):
        print("\nPLANET {0}, GROUP {1}".format(npl,ng))
        
        # identify which quarters and which_seasons have data
        use = tts_group[npl] == ng
        
        group_quarters = np.unique(transit_quarter[npl][use])
        group_seasons = np.unique(group_quarters % 4)
                        
        print(np.sum(use), "transits")
        print("quarters:", group_quarters)
        
        # grab transit times
        fixed_inds = np.copy(transit_inds[npl][use])
        fixed_inds -= fixed_inds[0]
        
        fixed_ephem = ephemeris[npl][use]
        quick_tts = quick_transit_times[npl][use]
          
        
        # define Legendre polynomials
        #x = 2*(fixed_ephem-np.min(fixed_ephem))/(np.max(fixed_ephem)-np.min(fixed_ephem)) - 1
                
        
        # now build the model
        with pm.Model() as hbm_model:
            # hierarchical (hyper)parameters
            if USE_HBM:
                raise ValueError("Not set up for HBM yet")
            else:
                pop_sd = ttv_rms_amp[npl]
            
            
            # transit times
            tt_offset = pm.StudentT("tt_offset", nu=2, shape=len(fixed_ephem))
            transit_times = pm.Deterministic("tts", fixed_ephem + tt_offset*pop_sd)
            
                        
            # set up stellar model and planetary orbit
            starrystar = exo.LimbDarkLightCurve(u)
            orbit = exo.orbits.TTVOrbit(transit_times=[transit_times], transit_inds=[fixed_inds], 
                                        b=b[npl], ror=ror[npl], duration=dur[npl])

            # track period and epoch
            T0 = pm.Deterministic("T0", orbit.t0)
            P  = pm.Deterministic("P", orbit.period)
            
            # nuissance parameters (one mean flux; variance by quarter)
            flux0 = pm.Normal("flux0", mu=np.median(good_flux), sd=np.std(good_flux), shape=len(group_quarters))
            logjit = pm.Normal("logjit", mu=np.var(good_flux), sd=5.0, shape=len(group_quarters))
            
            # build the GP kernel using a different noise model for each season
            kernel = [None]*4

            for z in range(4):
                gpz = gp_priors[z]
                if np.isin(z, group_seasons):
                    kernel[z] = GPterms.SHOTerm(S0=T.exp(gpz["logSw4"][0]-4*gpz["logw0"][0]), 
                                                w0=T.exp(gpz["logw0"][0]), 
                                                Q=T.exp(gpz["logQ"][0]))
            
            
            # now evaluate the model for each quarter
            light_curves = [None]*len(group_quarters)
            model_flux = [None]*len(group_quarters)
            gp = [None]*len(group_quarters)

            
            for j, q in enumerate(group_quarters):
                # here's the data
                t_ = all_time[q][all_mask[q][npl] == ng]
                f_ = all_flux[q][all_mask[q][npl] == ng]
                
                if all_dtype[q] == "short":
                    oversample = 1
                    texp = 1.0*scit
                if all_dtype[q] == "long":
                    oversample = 15
                    texp = 1.0*lcit
                

                # calculate light curves
                light_curves[j] = starrystar.get_light_curve(orbit=orbit, r=ror[npl], t=t_, 
                                                             oversample=oversample, texp=texp)

                model_flux[j] = pm.math.sum(light_curves[j], axis=-1) + flux0[j]*T.ones(len(t_))
                pm.Deterministic("model_flux_{0}".format(j), model_flux[j])

                # here's the GP (w/ kernel by season)
                gp[j] = GaussianProcess(kernel[q%4], 
                                        t=t_, 
                                        diag=T.exp(logjit[j])*T.ones(len(t_)), 
                                        mean=0.0)

                gp[j].marginal("gp_{0}".format(j), observed=f_-model_flux[j])
            

        with hbm_model:
            hbm_map = hbm_model.test_point
            hbm_map = pmx.optimize(start=hbm_map, vars=[flux0, logjit])
            hbm_map = pmx.optimize(start=hbm_map)
  
            
        # sample from the posterior
        with hbm_model:
            hbm_trace = pmx.sample(tune=5000, draws=1000, start=hbm_map, chains=2, target_accept=0.9)
        
        
        # save the results
        tts_chains.append(np.copy(np.array(hbm_trace["tts"])))
        offset_chains.append(np.copy(np.array(hbm_trace["tt_offset"])))


# # Fit overlapping transits

# ### TODO: fit overlapping transits in batches of < MAX_GROUP_SIZE

# In[ ]:


overlap_count = 0

for npl in range(NPL):
    overlap_count += np.sum(tts_group[npl]==-99)
    
print(overlap_count, "overlapping transits")


# In[ ]:


if overlap_count > 0:
    # grab the relevant data
    fixed_inds  = []
    fixed_ephem = []
    quick_tts   = []

    overlap_planets = []
    overlap_quarters = []

    for npl in range(NPL):
        use = tts_group[npl] == -99

        if np.sum(use) > 0:
            fixed_inds.append(transit_inds[npl][use])
            fixed_inds[npl] -= fixed_inds[npl][0]

            fixed_ephem.append(ephemeris[npl][use])
            quick_tts.append(quick_transit_times[npl][use])

            overlap_planets.append(npl)
            overlap_quarters.append(transit_quarter[npl][use])
    
    overlap_quarters = np.unique(np.hstack(overlap_quarters))
    overlap_seasons = np.unique(overlap_quarters % 4)
    
    
    # now build the model
    with pm.Model() as overlap_model:
        # hierarchical (hyper)parameters
        if USE_HBM:
            raise ValueError("Not set up for HBM yet")
        else:
            pop_sd = ttv_rms_amp
            
            
        # transit times
        tt_offset = []
        transit_times = []

        for i, npl in enumerate(overlap_planets):
            tt_offset.append(pm.StudentT('tt_offset_{0}'.format(npl), nu=2, shape=len(fixed_ephem[i])))
            transit_times.append(pm.Deterministic('tts_{0}'.format(npl), 
                                                  fixed_ephem[i] + tt_offset[i]*pop_sd[npl]))


        # set up stellar model and planetary orbit
        starrystar = exo.LimbDarkLightCurve(u)
        orbit = exo.orbits.TTVOrbit(transit_times=transit_times, transit_inds=fixed_inds, 
                                    b=b[overlap_planets], ror=ror[overlap_planets], duration=dur[overlap_planets])


        # track period and epoch
        T0 = pm.Deterministic("T0", orbit.t0)
        P  = pm.Deterministic("P", orbit.period)

        # nuissance parameters (one mean flux; variance by quarter)
        flux0 = pm.Normal("flux0", mu=np.median(good_flux), sd=np.std(good_flux), shape=len(overlap_quarters))
        logjit = pm.Normal("logjit", mu=np.var(good_flux), sd=5.0, shape=len(overlap_quarters))


        # build the GP kernel using a different noise model for each season
        kernel = [None]*4

        for z in range(4):
            gpz = gp_priors[z]
            if np.isin(z, overlap_seasons):
                kernel[z] = GPterms.SHOTerm(S0=T.exp(gpz["logSw4"][0]-4*gpz["logw0"][0]), 
                                            w0=T.exp(gpz["logw0"][0]), 
                                            Q=T.exp(gpz["logQ"][0]))



        # now evaluate the model for each quarter
        light_curves = [None]*len(overlap_quarters)
        model_flux = [None]*len(overlap_quarters)
        gp = [None]*len(overlap_quarters)


        for j, q in enumerate(overlap_quarters):
            # here's the data
            t_ = all_time[q][(all_mask[q] == -99).sum(0) > 0]
            f_ = all_flux[q][(all_mask[q] == -99).sum(0) > 0]


            if all_dtype[q] == "short":
                oversample = 1
                texp = 1.0*scit
            if all_dtype[q] == "long":
                oversample = 15
                texp = 1.0*lcit


            # calculate light curves
            light_curves[j] = starrystar.get_light_curve(orbit=orbit, r=ror[overlap_planets], t=t_, 
                                                         oversample=oversample, texp=texp)

            model_flux[j] = pm.math.sum(light_curves[j], axis=-1) + flux0[j]*T.ones(len(t_))
            pm.Deterministic("model_flux_{0}".format(j), model_flux[j])

            # here's the GP (w/ kernel by season)
            gp[j] = GaussianProcess(kernel[q%4], 
                                    t=t_, 
                                    diag=T.exp(logjit[j])*T.ones(len(t_)), 
                                    mean=0.0)

            gp[j].marginal("gp_{0}".format(j), observed=f_-model_flux[j])
        
        
    # optimize the MAP solution
    with overlap_model:
        overlap_map = pmx.optimize()


    # sample from the posterior
    with overlap_model:
        overlap_trace = pmx.sample(tune=5000, draws=1000, start=overlap_map, chains=2, target_accept=0.9)


    # save the results
    for i, npl in enumerate(overlap_planets):
        tts_chains.append(np.copy(np.array(overlap_trace['tts_{0}'.format(npl)])))
        offset_chains.append(np.copy(np.array(overlap_trace['tt_offset_{0}'.format(npl)])))


# # Save traces to fits file

# In[ ]:


# make a list of ordered pairs (npl,group) to help organize the chains
chain_organizer = []

for npl in range(NPL):
    for ng in range(1, 1+tts_group[npl].max()):
        chain_organizer.append((npl,ng))

if overlap_count > 0:
    for i, npl in enumerate(overlap_planets):
        chain_organizer.append((npl,-99))


# In[ ]:


# make primary HDU
primary_hdu = pyfits.PrimaryHDU()
header = primary_hdu.header
header["TARGET"] = TARGET
header["U1"] = u[0]
header["U2"] = u[1]

for npl in range(NPL):
    header["ROR_{0}".format(npl)] = ror[npl]
    header["B_{0}".format(npl)]   = b[npl]
    header["DUR_{0}".format(npl)] = dur[npl]   
    
primary_hdu.header = header
    
# add it to HDU list
hbm_hdulist = []
hbm_hdulist.append(primary_hdu)


# grab all samples from trace
for npl in range(NPL):
    combo_tts = []
    combo_offset = []
    combo_groupno = []
    
    for i, chorg in enumerate(chain_organizer):
        if chorg[0] == npl:
            combo_tts.append(tts_chains[i])
            combo_offset.append(offset_chains[i])
            combo_groupno.append(chorg[1]*np.ones(tts_chains[i].shape[1], dtype="int"))
        
    combo_tts = np.hstack(combo_tts)
    combo_offset = np.hstack(combo_offset)
    combo_groupno = np.hstack(combo_groupno)

    order = np.argsort(np.nanmedian(combo_tts,0))

    combo_tts = combo_tts[:,order]
    combo_offset = combo_offset[:,order]
    combo_groupno = combo_groupno[order]
    
    
    # add to HDUList
    hbm_hdulist.append(pyfits.ImageHDU(combo_tts, name='TTS_{0}'.format(npl)))
    hbm_hdulist.append(pyfits.ImageHDU(combo_offset, name='OFFSET_{0}'.format(npl)))
    hbm_hdulist.append(pyfits.ImageHDU(combo_groupno, name='GROUP_{0}'.format(npl)))

    
hbm_hdulist = pyfits.HDUList(hbm_hdulist)
hbm_hdulist.writeto(TRACE_DIR + TARGET + '_hbm_ttvs.fits', overwrite=True)


# In[ ]:


print('TOTAL RUNTIME = %.2f min' %((timer()-global_start_time)/60))


# In[ ]:




