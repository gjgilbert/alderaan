__all__ = [
    "remove_bad_cadences",
    "estimate_oscillation_period",
    "identify_gaps",
    "flatten_litecurve",
    "flatten_with_gp",
    "stitch",
    "filter_ringing",
]


from copy import deepcopy
import warnings

import aesara_theano_fallback.tensor as T
import astropy
from astropy.timeseries import LombScargle
from celerite2.backprop import LinAlgError
from celerite2.theano import GaussianProcess
from celerite2.theano import terms as GPterms
import lightkurve as lk
import numpy as np
import pymc3 as pm
import pymc3_ext as pmx
import scipy.signal as sig

from .astro import make_transit_mask
from .constants import *


def remove_bad_cadences(litecurve, planets, rel_masksize, min_masksize):
    """
    Docstring
    """
    # remove flagged cadences
    qmask = lk.KeplerQualityFlags.create_quality_mask(
        litecurve.quality, bitmask="default"
    )
    litecurve.remove_flagged_cadences(qmask)

    # make transit mask
    litecurve.mask = np.zeros(len(litecurve.time), dtype="bool")
    for n, p in enumerate(planets):
        masksize = np.max([min_masksize, rel_masksize * p.duration])
        litecurve.mask += make_transit_mask(litecurve.time, p.tts, masksize)

    litecurve.clip_outliers(
        kernel_size=13, sigma_upper=5, sigma_lower=5, mask=litecurve.mask
    )
    litecurve.clip_outliers(kernel_size=13, sigma_upper=5, sigma_lower=1000, mask=None)

    return litecurve


def estimate_oscillation_period(litecurve, min_period):
    """
    Docstring
    """
    ls_estimate = LombScargle(litecurve.time, litecurve.flux)

    min_freq = 1 / (litecurve.time.max() - litecurve.time.min())
    max_freq = 1 / min_period

    xf, yf = ls_estimate.autopower(
        minimum_frequency=min_freq, maximum_frequency=max_freq
    )

    peak_freq = xf[np.argmax(yf)]
    peak_per = np.max([1.0 / peak_freq, 1.001 * min_period])

    return peak_per


def identify_gaps(lc, break_tolerance, jump_tolerance=5.0):
    """
    Find gaps (breaks in time) and jumps (sudden flux changes) in a LiteCurve

    Parameters
    ----------
        lc : alderaan.LiteCurve
            Litecurve to be analyzed
        break_tolerance : int
            number of cadences to be considered a (time) gap
        jump_tolerance : float
            sigma threshold for identifying (flux) jumps (i.e. cadence-to-cadence flux variation)

    Returns
    -------
        gaps : ndarray, dtype=int
            array of indexes corresponding to the locations of gaps in lc.flux
    """
    # 1D mask
    mask = np.sum(np.atleast_2d(lc.mask, 0) == 0)

    # identify time gaps
    breaks = lc.cadno[1:] - lc.cadno[:-1]
    breaks = np.pad(breaks, (1, 0), "constant", constant_values=(1, 0))
    break_locs = np.where(breaks > break_tolerance)[0]
    break_locs = np.pad(
        break_locs, (1, 1), "constant", constant_values=(0, len(breaks) + 1)
    )

    # identify flux jumps
    jumps = lc.flux[1:] - lc.flux[:-1]
    jumps = np.pad(jumps, (1, 0), "constant", constant_values=(0, 0))
    big_jump = (
        np.abs(jumps - np.median(jumps)) / astropy.stats.mad_std(jumps) > jump_tolerance
    )
    jump_locs = np.where(mask * big_jump)[0]

    gaps = np.sort(np.unique(np.hstack([break_locs, jump_locs])))

    # flag nearly-consecutive cadences identified as gaps
    bad = np.hstack([False, (gaps[1:] - gaps[:-1]) < break_tolerance])

    if bad[-1]:
        bad[-1] = False
        bad[-2] = True

    gaps = gaps[~bad]

    return gaps


def flatten_litecurve(
    litecurve, break_tolerance, min_period, nominal_period, verbose=False
):
    """
    Remove trends from a LiteCurve using celerite Gaussian Process
    Calls flatten_with_gp() using successively simpler models as fallbacks
    """
    try:
        litecurve = flatten_with_gp(
            litecurve,
            break_tolerance,
            min_period,
            nominal_period=nominal_period,
            kterm="RotationTerm",
            correct_ramp=True,
            verbose=verbose,
        )
    except LinAlgError:
        warnings.warn(
            "Initial detrending failed...attempting to refit without exponential ramp component"
        )

        try:
            litecurve = flatten_with_gp(
                litecurve,
                break_tolerance,
                min_period,
                nominal_period=nominal_period,
                kterm="RotationTerm",
                correct_ramp=False,
                verbose=verbose,
            )
        except LinAlgError:
            warnings.warn(
                "Detrending with RotationTerm failed...attempting to detrend with SHOTerm"
            )

            litecurve = detrend.flatten_with_gp(
                litecurve,
                break_tolerance,
                min_period,
                nominal_period=nominal_period,
                kterm="SHOTerm",
                correct_ramp=False,
                verbose=verbose,
            )

    return litecurve


def flatten_with_gp(
    lc,
    break_tolerance,
    min_period,
    nominal_period=None,
    kterm="RotationTerm",
    correct_ramp=True,
    return_trend=False,
    verbose=True,
):
    """
    Remove trends from a LiteCurve using celerite Gaussian processes

    Parameters
    ----------
        lc : alderaan.LiteCurve
            LiteCurve to be flattened
        break_tolerance : int
            number of cadences to be considered a gap in data
        min_period : float
            minimum allowed period of GP kernel
        nominal_period : float
            mean estimate of period to pass to GP kernel
        kterm : string
            must be either 'RotationTerm' (default) or 'SHOTerm'
        correct_ramp : bool
            True to include an exponential ramp in the model for each disjoint section of photometry
        return_trend : bool
            True to return the predicted GP trend

    Returns
    -------
        lc : alderaan.LiteCurve
            LiteCurve with lc.flux and lc.error flattened and normalized
    """
    # identify primary oscillation period
    if nominal_period is None:
        ls_estimate = LombScargle(lc.time, lc.flux)
        xf, yf = ls_estimate.autopower(
            minimum_frequency=1 / (lc.time.max() - lc.time.min()),
            maximum_frequency=1 / min_period,
        )

        peak_freq = xf[np.argmax(yf)]
        peak_per = np.max([1.0 / peak_freq, 1.001 * min_period])

    else:
        peak_per = np.copy(nominal_period)

    # find gaps/jumps in the data
    gaps = identify_gaps(lc, break_tolerance=break_tolerance)
    gaps[-1] -= 1
    nseg = len(gaps) - 1

    # make adjustments for masked transits
    inds_ = np.arange(len(lc.time), dtype="int")[~lc.mask]
    gaps_ = [np.sum(inds_ < g) for g in gaps]
    time_ = lc.time[~lc.mask]
    flux_ = lc.flux[~lc.mask]

    # break up data into segments bases on gaps/jumps
    seg = np.zeros(len(lc.time), dtype="int")
    seg_ = np.zeros(len(time_), dtype="int")

    for i in range(nseg):
        seg[gaps[i] : gaps[i + 1]] = i
        seg_[gaps_[i] : gaps_[i + 1]] = i

    # define the mean function (exponential ramp)
    if correct_ramp:

        def mean_fxn(_t, _s, flux0, ramp_amp, log_tau):
            mean = T.zeros(len(_t))

            for i in range(len(np.unique(_s))):
                t0 = _t[_s == i].min()
                mean += (
                    flux0[i]
                    * (1 + ramp_amp[i] * T.exp(-(_t - t0) / T.exp(log_tau[i])))
                    * (_s == i)
                )

            return mean

    else:

        def mean_fxn(_t, _s, flux0, ramp_amp=None, log_tau=None):
            mean = T.zeros(len(_t))

            for i in range(len(np.unique(_s))):
                mean += flux0[i] * (_s == i)

            return mean

    # here's the stellar rotation model
    with pm.Model() as trend_model:
        # set up the kernal
        log_sigma = pm.Normal("log_sigma", mu=np.log(np.std(flux_)), sd=5.0)
        logP_off = pm.Normal("logP", mu=np.log(peak_per - min_period), sd=2.0)
        log_Q0 = pm.Normal("log_Q0", mu=0.0, sd=5.0, testval=np.log(0.5))

        sigma = pm.Deterministic("sigma", T.exp(log_sigma))
        P = pm.Deterministic("P", min_period + T.exp(logP_off))

        if kterm == "RotationTerm":
            log_dQ = pm.Normal("log_dQ", mu=0.0, sd=5.0, testval=np.log(1e-3))
            mix = pm.Uniform("mix", lower=0, upper=1, testval=0.1)
            kernel = GPterms.RotationTerm(
                sigma=sigma, period=P, Q0=T.exp(log_Q0), dQ=T.exp(log_dQ), f=mix
            )

        elif kterm == "SHOTerm":
            kernel = GPterms.SHOTerm(sigma=sigma, w0=2 * pi / P, Q=0.5 + T.exp(log_Q0))

        else:
            raise ValueError("kterm must be 'RotationTerm' or 'SHOTerm'")

        # mean function is an exponential trend (per segment)
        approx_mean_flux = [np.mean(flux_[seg_ == i]) for i in range(nseg)]

        if correct_ramp:
            flux0 = pm.Normal(
                "flux0", mu=approx_mean_flux, sd=np.ones(nseg), shape=nseg
            )
            ramp_amp = pm.Normal("ramp_amp", mu=0, sd=np.std(lc.flux), shape=nseg)
            log_tau = pm.Normal("log_tau", mu=0, sd=5, shape=nseg)
            mean_ = pm.Deterministic(
                "mean_", mean_fxn(time_, seg_, flux0, ramp_amp, log_tau)
            )

        else:
            flux0 = pm.Normal(
                "flux0", mu=approx_mean_flux, sd=np.ones(nseg), shape=nseg
            )
            ramp_amp = None
            log_tau = None
            mean_ = pm.Deterministic("mean_", mean_fxn(time_, seg_, flux0))

        # variance
        log_yvar = pm.Normal(
            "log_yvar", mu=np.var(flux_ - sig.medfilt(flux_, 13)), sd=5.0
        )

        # now set up the GP
        gp = GaussianProcess(
            kernel, t=time_, diag=T.exp(log_yvar) * T.ones(len(time_)), mean=mean_
        )
        gp.marginal("gp", observed=flux_)

        # track mean predictions
        full_mean_pred = pm.Deterministic(
            "full_mean_pred", mean_fxn(lc.time, seg, flux0, ramp_amp, log_tau)
        )

    # optimize the GP hyperparameters
    with trend_model:
        trend_map = trend_model.test_point
        trend_map = pmx.optimize(start=trend_map, vars=[flux0], progress=verbose)
        trend_map = pmx.optimize(
            start=trend_map, vars=[flux0, log_yvar], progress=verbose
        )

        for i in range(1 + correct_ramp):
            if kterm == "RotationTerm":
                trend_map = pmx.optimize(
                    start=trend_map,
                    vars=[log_yvar, flux0, sigma, P, log_Q0, log_dQ, mix],
                    progress=verbose,
                )
            if kterm == "SHOTerm":
                trend_map = pmx.optimize(
                    start=trend_map,
                    vars=[log_yvar, flux0, sigma, P, log_Q0],
                    progress=verbose,
                )
            if correct_ramp:
                trend_map = pmx.optimize(
                    start=trend_map,
                    vars=[log_yvar, flux0, ramp_amp, log_tau],
                    progress=verbose,
                )

        trend_map = pmx.optimize(start=trend_map, progress=verbose)

    # reconstruct the GP to interpolate over masked transits
    if kterm == "RotationTerm":
        kernel = GPterms.RotationTerm(
            sigma=trend_map["sigma"],
            period=trend_map["P"],
            Q0=T.exp(trend_map["log_Q0"]),
            dQ=T.exp(trend_map["log_dQ"]),
            f=trend_map["mix"],
        )

    elif kterm == "SHOTerm":
        kernel = GPterms.SHOTerm(
            sigma=trend_map["sigma"],
            w0=2 * pi / trend_map["P"],
            Q=0.5 + T.exp(trend_map["log_Q0"]),
        )

    gp = GaussianProcess(kernel, mean=0.0)
    gp.compute(time_, diag=T.exp(trend_map["log_yvar"]) * T.ones(len(time_)))

    full_trend = (
        gp.predict(flux_ - trend_map["mean_"], lc.time).eval()
        + trend_map["full_mean_pred"]
    )

    lc.flux /= full_trend
    lc.error /= full_trend

    if return_trend:
        return lc, full_trend

    return lc


def stitch(litecurves):
    """
    Combine a list of LiteCurves in a single LiteCurve

    Parameters
    ----------
        litecurves : list of alderaan.LiteCurve() objects

    Returns
    -------
        lc : a single combined alderaan.LiteCurve() object
    """
    combo = deepcopy(litecurves[0])

    for i in range(1, len(litecurves)):
        for k in combo.__dict__.keys():
            if type(combo.__dict__[k]) is np.ndarray:
                combo.__dict__[k] = np.hstack(
                    [combo.__dict__[k], litecurves[i].__dict__[k]]
                )

    return combo


def filter_ringing(lc, break_tolerance, fring, bw):
    """
    Filter out known long cadence instrumental ringing modes (see Gilliland+ 2010)
    Applies a notch filter (narrow bandstop filter) at a set of user specified frequencies

    The function does NOT change the lc.flux attribute directly, but rather returns a new flux array

    Parameters
    ----------
        lc : alderaan.LiteCurve
            Litecurve must have time, flux, and cadno attributes
        break_tolerance : int
            number of cadences considered a large gap in time
        fring : array-like
            ringing frequencies in same units as lc.time (i.e. if time is in days, fring is in days^-1)
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
        t = lc.time[gap_locs[i] : gap_locs[i + 1]]
        f = lc.flux[gap_locs[i] : gap_locs[i + 1]]
        c = lc.cadno[gap_locs[i] : gap_locs[i + 1]]

        # fill small gaps with white noise
        npts = c[-1] - c[0] + 1
        dt = np.min(t[1:] - t[:-1])

        t_interp = np.linspace(t.min(), t.max() + dt * 3 / 2, npts)
        f_interp = np.ones_like(t_interp)
        c_interp = np.arange(c.min(), c.max() + 1)

        data_exists = np.isin(c_interp, c)

        f_interp[data_exists] = f
        f_interp[~data_exists] = np.random.normal(
            loc=np.median(f), scale=np.std(f), size=np.sum(~data_exists)
        )

        # now apply the filter
        f_fwd_back = np.copy(f_interp)
        f_back_fwd = np.copy(f_interp)
        f_ramp = np.linspace(0, 1, len(f_interp))

        for j, f0 in enumerate(fring):
            b, a = sig.iirnotch(f0, Q=2 * f0 / bw, fs=1 / dt)
            f_fwd_back = sig.filtfilt(
                b, a, f_fwd_back, padlen=np.min([120, len(f_fwd_back) - 2])
            )
            f_back_fwd = sig.filtfilt(
                b, a, f_back_fwd[::-1], padlen=np.min([120, len(f_back_fwd / 2) - 1])
            )[::-1]

            f_filt = f_fwd_back * f_ramp + f_fwd_back * f_ramp[::-1]

        flux_filtered.append(f_filt[data_exists])

    return np.hstack(flux_filtered)
