# import relevant modules
import numpy as np
import scipy.optimize as op
import scipy.signal as sig
from   scipy import stats
from   scipy.interpolate import interp1d
import astropy
from   astropy.io import fits as pyfits

import lightkurve as lk
import exoplanet as exo
import pymc3 as pm
import theano.tensor as T

import csv
import sys
import os
import warnings
from   copy import deepcopy

from .constants import *
from .LiteCurve import *
from .utils import *

import matplotlib.pyplot as plt


__all__ = ["cleanup_lkfc",
           "remove_flagged_cadences",
           "clip_outliers",
           "make_transitmask",
           "identify_gaps",
           "stitch_lkc",
           "filter_ringing",
           "flatten_with_gp"]



def cleanup_lkfc(lkf_collection, kic):
    """
    Join each quarter in a lk.LightCurveFileCollection into a single lk.LightCurveFile
    Performs only the minimal detrending step remove_nans()
    
    Parameters
    ----------
    lkf_collection : lk.LightCurveFileCollection
        
    kic : int
        Kepler Input Catalogue (KIC) number for target
    
    Returns
    -------
    lkc : lk.LightCurveCollection
    
    """
    lkf_col = deepcopy(lkf_collection)
    
    quarters = []
    for i, lkf in enumerate(lkf_col):
        quarters.append(lkf.quarter)

    data_out = []
    for q in np.unique(quarters):
        lkf_list = []
        cadno   = []

        for i, lkf in enumerate(lkf_col):
            if (lkf.quarter == q)*(lkf.targetid==kic):
                lkf_list.append(lkf)
                cadno.append(lkf.cadenceno.min())
        
        order = np.argsort(cadno)
        
        lkfc_list = []
        
        for j in order:
            lkfc_list.append(lkf_list[j])

        # the operation "stitch" converts a LightCurveFileCollection to a single LightCurve
        lkfc_list = lk.LightCurveFileCollection(lkfc_list)
        lklc = lkfc_list.PDCSAP_FLUX.stitch().remove_nans()

        data_out.append(lklc)

    return lk.LightCurveCollection(data_out)



def remove_flagged_cadences(lklc, bitmask='default', return_mask=False):
    """
    Remove cadences flagged by Kepler pipeline data quality flags
    See lightkurve documentation for description of input parameters

    Parameters
    ----------
    lklc : lightkurve.KeplerLightCurve() object
    
    bitmask : string
        'default', 'hard', or 'hardest' -- how aggressive to be on cutting flagged cadences
    return_mask : bool
        'True' to return a mask of flagged cadences (1=good); default=False

    Returns
    -------
    lklc : LightCurve() object
        with flagged cadences removed
    qmask : array-like, bool
        boolean mask indicating with cadences were removed (True=data is good quality)
    """
    qmask = lk.KeplerQualityFlags.create_quality_mask(lklc.quality, bitmask)
    
    lklc.time         = lklc.time[qmask]
    lklc.flux         = lklc.flux[qmask]
    lklc.flux_err     = lklc.flux_err[qmask]
    lklc.centroid_col = lklc.centroid_col[qmask]
    lklc.centroid_row = lklc.centroid_row[qmask]
    lklc.cadenceno    = lklc.cadenceno[qmask]
    lklc.quality      = lklc.quality[qmask]
    
    if return_mask:
        return lklc, qmask
    else:
        return lklc



def clip_outliers(lklc, kernel_size, sigma_upper, sigma_lower):
    """
    Docstring
    
    Parameters
    ----------
    lklc : lightkurve.KeplerLightCurve
    
    """
    loop = True
    count = 0
    
    while loop:
        smoothed = sig.medfilt(lklc.flux, kernel_size=kernel_size)

        bad = astropy.stats.sigma_clip(lklc.flux-smoothed, sigma_upper=sigma_upper, sigma_lower=sigma_lower, \
                                       stdfunc=astropy.stats.mad_std).mask

        lklc.time = lklc.time[~bad]
        lklc.flux = lklc.flux[~bad]
        lklc.quality = lklc.quality[~bad]
        lklc.flux_err = lklc.flux_err[~bad]
        lklc.cadenceno = lklc.cadenceno[~bad]
        lklc.centroid_row = lklc.centroid_row[~bad]
        lklc.centroid_col = lklc.centroid_col[~bad]
        
        if np.sum(bad) == 0:
            loop = False
        else:
            count += 1
            
        if count >= 3:
            loop = False

    return lklc    


    
def make_transitmask(time, tts, duration, masksize=1.5):
    """
    Make a transit mask for a Planet
    
    Parameters
    ----------
    time : array-like
        time values at each cadence
    tts : array-like
        transit times for a single planet
    duration : float
        transit duration
    masksize : float
        size of mask window in number of transit durations from transit center (default=1.5 --> 3 transit durations masked)
    
    Returns
    -------
        transitmask : array-like, bool
            boolean array (1=near transit; 0=not)
    """  
    transitmask = np.zeros_like(time, dtype='bool')
    
    tts_here = tts[(tts >= time.min())*(tts <= time.max())]
    
    for t0 in tts_here:
        neartransit = np.abs(time-t0)/duration < masksize
        transitmask += neartransit
    
    return transitmask



def identify_gaps(lc, break_tolerance, jump_tolerance=5.0):
    """
    Search a LiteCurve for large time breaks and flux jumps
    
    Parameters
    ----------
        lc : alderaan.LiteCurve() object
            must have time, flux, and cadno attributes
        break_tolerance : int
            number of cadences considered a large gap in time
        jump_tolerance : float
            number of sigma from median flux[i+1]-flux[i] to be considered a large jump in flux (default=5.0)
            
    Returns
    -------
        gap_locs : array
            indexes of identified gaps, including endpoints
    """
    # 1D mask
    mask = np.sum(np.atleast_2d(lc.mask, 0) == 0)
    
    # identify time gaps
    breaks = lc.cadno[1:]-lc.cadno[:-1]
    breaks = np.pad(breaks, (1,0), 'constant', constant_values=(1,0))
    break_locs = np.where(breaks > break_tolerance)[0]
    break_locs = np.pad(break_locs, (1,1), 'constant', constant_values=(0,len(breaks)+1))
    
    # identify flux jumps
    jumps = lc.flux[1:]-lc.flux[:-1]
    jumps = np.pad(jumps, (1,0), 'constant', constant_values=(0,0))
    big_jump = np.abs(jumps - np.median(jumps))/astropy.stats.mad_std(jumps) > 5.0
    jump_locs = np.where(mask*big_jump)[0]
    
    return np.sort(np.unique(np.hstack([break_locs, jump_locs])))


def stitch_lkc(lkc):
    """
    Stitch together multiple quarters of a lk.LightCurveCollection into a custom LiteCurve
    
    Parameters
    ----------
    lkc : lk.LightCurveCollection()
    
    Returns
    -------
    litecurve : LiteCurve() object
        This is a custom class, similar but not idential to a lk.LightCurve object
    """
    litecurve = LiteCurve()
    
    time = []
    flux = []
    error = []
    cadno = []
    quarter = []
    channel = []
    centroid_col = []
    centroid_row = []
    mask = []
    
    # combine
    for i, lc in enumerate(lkc):
        time.append(lc.time)
        flux.append(lc.flux)
        error.append(lc.flux_err)
        cadno.append(lc.cadenceno)
        quarter.append(lc.quarter)
        channel.append(lc.channel)
        centroid_col.append(lc.centroid_col)
        centroid_row.append(lc.centroid_row)
    
    # linearize
    litecurve.time = np.asarray(np.hstack(time), dtype='float')
    litecurve.flux = np.asarray(np.hstack(flux), dtype='float')
    litecurve.error = np.asarray(np.hstack(error), dtype='float')
    litecurve.cadno = np.asarray(np.hstack(cadno), dtype='int')
    litecurve.quarter = np.asarray(np.hstack(quarter), dtype='int')
    litecurve.channel = np.asarray(np.hstack(channel), dtype='int')
    litecurve.centroid_col = np.asarray(np.hstack(centroid_col), dtype='int')
    litecurve.centroid_row = np.asarray(np.hstack(centroid_row), dtype='int')
    
    return litecurve



def filter_ringing(lc, break_tolerance, fring, bw):
    """
    Filter out known long cadence instrumental ringing modes (see Gilliland+ 2010)
    Applies a notch filter (narrow bandstop filter) at a set of user specified frequencies
    
    Parameters
    ----------
        lc : LiteCurve() object
            must have time, flux, and cadno attributes
        break_tolerance : int
            number of cadences considered a large gap in time
        fring : array-like
            ringing frequencies in same units as lklc.time (i.e. if time is in days, fring is in days^-1)
        bw : float
            bandwidth of stopband (same units as fring)
             
    Returns
    -------
        flux_filtered : ndarray
            flux with ringing modes filtered out
    """
    # make lists to hold outputs
    flux_filtered = []

    # identify gaps
    gap_locs = identify_gaps(lc, break_tolerance, jump_tolerance=5.0)
    

    # break the data into contiguous segments and detrend
    for i, gloc in enumerate(gap_locs[:-1]):
        
        # grab segments of time, flux, cadno, masks
        t = lc.time[gap_locs[i]:gap_locs[i+1]]
        f = lc.flux[gap_locs[i]:gap_locs[i+1]]
        c = lc.cadno[gap_locs[i]:gap_locs[i+1]]

        # fill small gaps with white noise
        npts = c[-1]-c[0] + 1
        dt = np.min(t[1:]-t[:-1])

        t_interp = np.linspace(t.min(),t.max()+dt*3/2, npts)
        f_interp = np.ones_like(t_interp)
        c_interp = np.arange(c.min(), c.max()+1)

        data_exists = np.isin(c_interp, c)

        f_interp[data_exists] = f
        f_interp[~data_exists] = np.random.normal(loc=np.median(f), scale=np.std(f), size=np.sum(~data_exists))
        
        
        # now apply the filter
        f_fwd_back = np.copy(f_interp)
        f_back_fwd = np.copy(f_interp)
        f_ramp = np.linspace(0,1,len(f_interp))
        
        for j, f0 in enumerate(fring):
            b, a = sig.iirnotch(f0, Q=2*f0/bw, fs=1/dt)
            f_fwd_back = sig.filtfilt(b, a, f_fwd_back, padlen=np.min([120, len(f_fwd_back)-2]))
            f_back_fwd = sig.filtfilt(b, a, f_back_fwd[::-1], padlen=np.min([120, len(f_back_fwd/2)-1]))[::-1]
            
            f_filt = f_fwd_back*f_ramp + f_fwd_back*f_ramp[::-1]
            
        flux_filtered.append(f_filt[data_exists])
          
    return np.hstack(flux_filtered)



def flatten_with_gp(lc, break_tolerance, min_period, bin_factor=None, return_trend=False):
    """
    Detrend the flux from an alderaan LiteCurve using a celerite RotationTerm GP kernel
    The mean function of each uninterrupted segment of flux is modeled as an exponential
    
        Fmean = F0*(1+A*exp(-t/tau))
    
    Parameters
    ----------
    lc : alderaan.LiteCurve
        must have .time, .flux and .mask attributes
    break_tolerance : int
        number of cadences considered a large gap in time
    min_period : float
        lower bound on primary period for RotationTerm kernel 
    return_trend : bool (default=False)
        if True, return the trend inferred from the GP fit
        
    Returns
    -------
    lc : alderaan.LiteCurve
        LiteCurve with trend removed from lc.flux
    gp_trend : ndarray
        trend inferred from GP fit (only returned if return_trend == True)
    """
    # find gaps/breaks/jumps in the data
    gaps = identify_gaps(lc, break_tolerance=break_tolerance)
    gaps[-1] -= 1

    
    # initialize data arrays and lists of segments
    gp_time = np.array(lc.time, dtype="float64")
    gp_flux = np.array(lc.flux, dtype="float64")
    gp_mask = np.sum(lc.mask, 0) == 0

    time_segs = []
    flux_segs = []
    mask_segs = []

    for i in range(len(gaps)-1):
        time_segs.append(gp_time[gaps[i]:gaps[i+1]])
        flux_segs.append(gp_flux[gaps[i]:gaps[i+1]])
        mask_segs.append(gp_mask[gaps[i]:gaps[i+1]])

    mean_flux = []
    approx_var = []

    for i in range(len(gaps)-1):
        m = mask_segs[i]
        mean_flux.append(np.mean(flux_segs[i][m]))
        approx_var.append(np.var(flux_segs[i] - sig.medfilt(flux_segs[i], 13)))

    
    # put segments into groups of ten
    nseg = len(time_segs)
    ngroup = int(np.ceil(nseg/10))
    seg_groups = np.array(np.arange(ngroup+1)*np.ceil(nseg/ngroup), dtype="int")
    seg_groups[-1] = len(gaps) -1
    

    # identify rotation period to initialize GP
    ls_estimate = exo.estimators.lomb_scargle_estimator(gp_time, gp_flux, max_peaks=1, \
                                                        min_period=min_period, max_period=91.0, \
                                                        samples_per_peak=50)

    peak_per = ls_estimate["peaks"][0]["period"]
    
    
    # set up lists to hold trend info
    trend_maps = [None]*ngroup
    
    
    # optimize the GP for each group of segments
    for j in range(ngroup):
        sg0 = seg_groups[j]
        sg1 = seg_groups[j+1]
        nuse = sg1-sg0

        with pm.Model() as trend_model:

            log_amp = pm.Normal("log_amp", mu=np.log(np.std(gp_flux)), sd=5)
            log_per_off = pm.Normal("log_per_off", mu=0, sd=5, testval=np.log(peak_per-min_period))
            log_Q0_off = pm.Normal("log_Q0_off", mu=0, sd=10)
            log_deltaQ = pm.Normal("log_deltaQ", mu=2, sd=10)
            mix = pm.Uniform("mix", lower=0, upper=1)

            P = pm.Deterministic("P", min_period + T.exp(log_per_off))
            Q0 = pm.Deterministic("Q0", 1/T.sqrt(2) + T.exp(log_Q0_off))

            kernel = exo.gp.terms.RotationTerm(log_amp=log_amp, period=P, Q0=Q0, log_deltaQ=log_deltaQ, mix=mix)


            # exponential trend
            logtau = pm.Normal("logtau", mu=np.log(3)*np.ones(nuse), sd=5*np.ones(nuse), shape=nuse)
            exp_amp = pm.Normal('exp_amp', mu=np.zeros(nuse), sd=np.std(gp_flux)*np.ones(nuse), shape=nuse)


            # nuissance parameters per segment
            flux0 = pm.Normal('flux0', mu=np.array(mean_flux[sg0:sg1]), sd=np.std(gp_flux)*np.ones(nuse), shape=nuse)
            logvar = pm.Normal('logvar', mu=np.log(approx_var[sg0:sg1]), sd=10*np.ones(nuse), shape=nuse)


            # now set up the GP
            gp      = [None]*nuse
            gp_pred = [None]*nuse

            for i in range(nuse):
                m = mask_segs[sg0+i]
                t = time_segs[sg0+i][m] - time_segs[sg0+i][m][0]

                ramp = 1 + exp_amp[i]*np.exp(-t/np.exp(logtau[i]))

                gp[i] = exo.gp.GP(kernel, time_segs[sg0+i][m], T.exp(logvar[i])*T.ones(len(time_segs[sg0+i][m])))
                pm.Potential('obs_{0}'.format(i), gp[i].log_likelihood(flux_segs[sg0+i][m] - flux0[i]*ramp))


        with trend_model:
            trend_maps[j] = exo.optimize(start=trend_model.test_point)



    # set up mean and variance vectors
    gp_mean  = np.ones_like(gp_flux)
    gp_var   = np.ones_like(gp_flux)


    for i in range(nseg):
        j = np.argmin(seg_groups <= i) - 1

        g0 = gaps[i]
        g1 = gaps[i+1]

        F0_  = trend_maps[j]["flux0"][i-seg_groups[j]]
        A_   = trend_maps[j]["exp_amp"][i-seg_groups[j]]
        tau_ = np.exp(trend_maps[j]["logtau"][i-seg_groups[j]])

        t_ = gp_time[g0:g1] - gp_time[g0]

        gp_mean[g0:g1] = F0_*(1 + A_*np.exp(-t_/tau_))
        gp_var[g0:g1] = np.ones(g1-g0)*np.exp(trend_maps[j]["logvar"][i-seg_groups[j]])


    # increase variance for cadences in transit (yes, this is hacky, but it works)
    # using gp.predict() inside the initial model was crashing jupyter, making debugging slow
    gp_var[~gp_mask] *= 1e12


    # now evaluate the GP to get the final trend
    gp_trend = np.zeros_like(gp_flux)

    for j in range(ngroup):
        start = gaps[seg_groups[j]]
        end = gaps[seg_groups[j+1]]

        m = gp_mask[start:end]

        with pm.Model() as trend_model:

            log_amp = trend_maps[j]["log_amp"]
            P = trend_maps[j]["P"]
            Q0 = trend_maps[j]["Q0"]
            log_deltaQ = trend_maps[j]["log_deltaQ"]
            mix = trend_maps[j]["mix"]

            kernel = exo.gp.terms.RotationTerm(log_amp=log_amp, period=P, Q0=Q0, log_deltaQ=log_deltaQ, mix=mix)


            gp = exo.gp.GP(kernel, gp_time[start:end], gp_var[start:end])

            gp.log_likelihood(gp_flux[start:end] - gp_mean[start:end])

            gp_trend[start:end] = gp.predict().eval() + gp_mean[start:end]
            
            
    # now remove the trend
    lc.flux  = lc.flux - gp_trend + 1.0
    
    
    # return results
    if return_trend:
        return lc, gp_trend
    else:
        return lc