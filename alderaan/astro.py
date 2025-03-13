__all__ = [
    "get_transit_depth",
    "get_sma",
    "get_dur_14",
    "get_dur_23",
    "get_dur_cc",
    "predict_tc_error",
    "make_transit_mask",
    "set_oversample_factor",
]


import numpy as np
from .constants import *


def get_transit_depth(p, b):
    """
    Calculate approximate transit depth
    See Mandel & Agol 2002

    Parameters
    ----------
    p : planet-to-star radius ratio Rp/Rstar
    b : impact parameter

    Returns
    -------
    d : transit depth
    """
    # broadcasting
    p = p * np.ones(np.atleast_1d(b).shape)
    b = b * np.ones(np.atleast_1d(p).shape)

    # non-grazing transit (b <= 1-p)
    d = p**2

    # grazing transit (1-p < b < 1+p)
    grazing = (b > 1 - p) * (b < 1 + p)

    pg = p[grazing]
    bg = b[grazing]

    k0 = np.arccos((pg**2 + bg**2 - 1) / (2 * pg * bg))
    k1 = np.arccos((1 - pg**2 + bg**2) / (2 * bg))
    s0 = np.sqrt((4 * bg**2 - (1 + bg**2 - pg**2) ** 2) / 4)

    d[grazing] = (1 / pi) * (pg**2 * k0 + k1 - s0)

    # non-transiting (b >= 1+p)
    d[b >= 1 + p] = 0.0

    return np.squeeze(d)


def get_sma(P, Ms):
    """
    Calculate semi-major axis in units of Solar radii from Kepler's law

    Parameters
    ----------
    P : period [days]
    Ms : stellar mass [Solar masses]

    Returns
    -------
    sma : semimajor axis [Solar radii]
    """
    return Ms ** (1 / 3) * (P / 365.24) ** (2 / 3) / RSAU


def get_dur_14(P, aRs, b, ror, ecc=None, w=None):
    """
    Calculate total transit duration (I-IV contacts)
    See Winn 2010 Eq. 14 & 16

    P : period
    aRs : semimajor axis over stellar radius
    b : impact parameter
    ror : planet-to-star radius ratio
    ecc : eccentricity
    w : argument of periastron [radians]
    """
    if ecc is not None:
        Xe = np.sqrt(1 - ecc**2) / (1 + ecc * np.sin(w))
        We = (1 - ecc**2) / (1 + ecc * np.sin(w))
    else:
        Xe = 1.0
        We = 1.0

    sini = np.sin(np.arccos((b / We) / (aRs)))
    argument = (1 / aRs) * np.sqrt((1 + ror) ** 2 - b**2) / sini

    Ttot = (P / pi) * np.arcsin(argument) * Xe

    return Ttot


def get_dur_23(P, aRs, b, ror, ecc=None, w=None):
    """
    Calculate full duration (II-III contacts)
    See Winn 2010 Eq. 15 & 16

    P : period
    aRs : semimajor axis over stellar radius
    b : impact parameter
    ror : planet-to-star radius ratio
    ecc : eccentricity
    w : argument of periastron [radians]
    """
    if ecc is not None:
        Xe = np.sqrt(1 - ecc**2) / (1 + ecc * np.sin(w))
        We = (1 - ecc**2) / (1 + ecc * np.sin(w))
    else:
        Xe = 1.0
        We = 1.0

    sini = np.sin(np.arccos((b / We) / (aRs)))
    argument = (1 / aRs) * np.sqrt((1 - ror) ** 2 - b**2) / sini

    Tfull = np.asarray((P / pi) * np.arcsin(argument)) * Xe

    # correct for grazing transits
    grazing = np.asarray(b) > np.asarray(1 - ror)
    Tfull[grazing] = np.nan

    return Tfull


def get_dur_cc(P, aRs, b, ecc=None, w=None):
    """
    Calculate ingress/egrees midpoint transit duration (1.5-3.5 contacts)
    See Winn 2010

    P : period
    aRs : semimajor axis over stellar radius
    b : impact parameter
    ecc : eccentricity
    w : argument of periastron [radians]
    """
    if ecc is not None:
        Xe = np.sqrt(1 - ecc**2) / (1 + ecc * np.sin(w))
        We = (1 - ecc**2) / (1 + ecc * np.sin(w))
    else:
        Xe = 1.0
        We = 1.0

    sini = np.sin(np.arccos((b / We) / (aRs)))
    argument = (1 / aRs) * np.sqrt(1 - b**2) / sini

    Tmid = np.asarray((P / pi) * np.arcsin(argument)) * Xe

    # correct for grazing transits
    grazing = np.asarray(b) > 1
    Tmid[grazing] = np.nan

    return Tmid


def predict_tc_error(ror, b, T14, texp, sigma_f):
    """
    Predict uncertainty on mid-transit time
    See Carter+2008

    Parameters
    ----------
    ror : planet-to-star radius ratio
    b : impact parameter
    T14 : first-to-fourth contact transit duration
    texp : exposure time (in same units as T14)
    sigma_f : photometric error on flux
    """
    Q = np.sqrt(T14 / texp) * (ror**2 / sigma_f)

    if b <= 1 - ror:
        tau_over_T14 = ror / (1 - b**2)
    else:
        tau_over_T14 = 0.5

    sigma_tc = (T14 / Q) * np.sqrt(0.5 * tau_over_T14)

    return sigma_tc


def make_transit_mask(time, tts, masksize):
    """
    Make a transit mask for an alderaan.Planet() object

    Parameters
    ----------
        time : array-like
            time values at each cadence
        tts : array-like
            transit times for a single planet
        masksize : float
            size of mask window in units of time

    Returns
    -------
        transitmask : ndarray, bool
            boolean array (1=near transit; 0=not)
    """
    transitmask = np.zeros(len(time), dtype="bool")

    tts_here = tts[(tts >= time.min()) * (tts <= time.max())]

    for t0 in tts_here:
        neartransit = np.abs(time - t0) < masksize
        transitmask += neartransit

    return transitmask


def set_oversample_factor(periods, depths, durs, flux, error, texp):
    """
    Docstring
    """
    if (len(periods) != len(depths)) or (len(periods) != len(durs)):
        raise ValueError("Input array shape mismatch")

    npl = len(periods)

    # ingress/egress timescale estimate following Winn 2010
    ror = np.sqrt(depths)
    tau = 13 * (periods / 365.25) ** (1 / 3) * ror / 24

    # set sigma so binning error is < 0.1% of photometric uncertainty
    sigma = np.mean(error / flux) * 0.04

    npts = np.array(np.ceil(np.sqrt((depths / tau) * (texp / 8 / sigma))), dtype="int")
    npts = npts + (npts % 2 + 1)

    npts = np.max(np.hstack([npts, 7]))

    return npts
