"""Astrophysical conversions: limb darkening bases, depths, timing errors"""

__all__ = [
    "approx_tc_error",
    "approx_ror",
    "approx_transit_depth",
    "ld_q_to_u",
    "ld_u_to_q",
]


import numpy as np
from scipy.optimize import brentq


def ld_q_to_u(q1, q2):
    """Quadratic limb darkening (q1, q2) --> (u1, u2), see Kipping 2013"""
    u1 = 2 * np.sqrt(q1) * q2
    u2 = np.sqrt(q1) * (1 - 2 * q2)
    return u1, u2


def ld_u_to_q(u1, u2):
    """Quadratic limb darkening (u1, u2) --> (q1, q2), see Kipping 2013"""
    q1 = (u1 + u2) ** 2
    q2 = u1 / (2 * (u1 + u2))
    return q1, q2


def approx_transit_depth(p, b):
    """Approximate transit depth following Mandel & Agol (2002)

    Parameters
    ----------
    p : array-like
        planet-to-star radius ratio Rp/Rs
    b : array-like
        impact parameter

    Returns
    -------
    depth : array-like
        transit depth
    """
    p, b = np.asarray(p, dtype=float), np.asarray(b, dtype=float)

    # grazing transit (1-p < b < 1+p); terms are unused outside this regime, and
    # clipped against roundoff at the contact points
    with np.errstate(divide="ignore", invalid="ignore"):
        k0 = np.arccos(np.clip((p**2 + b**2 - 1) / (2 * p * b), -1, 1))
        k1 = np.arccos(np.clip((1 - p**2 + b**2) / (2 * b), -1, 1))
        s0 = np.sqrt(np.maximum(4 * b**2 - (1 + b**2 - p**2) ** 2, 0) / 4)
    grazing = (b > 1 - p) & (b < 1 + p)

    # non-grazing transit (b <= 1-p), non-transiting (b >= 1+p)
    d = np.where(grazing, (p**2 * k0 + k1 - s0) / np.pi, p**2)
    return np.where(b >= 1 + p, 0.0, d)


def approx_ror(depth, b):
    """Invert approx_transit_depth: radius ratio from transit depth

    Parameters
    ----------
    depth : array-like
        transit depth, 0 < depth < 1
    b : array-like
        impact parameter, b >= 0

    Returns
    -------
    ror : array-like
        planet-to-star radius ratio Rp/Rs
    """
    if np.any(~((np.asarray(depth) > 0) & (np.asarray(depth) < 1))):
        raise ValueError(f"depth = {depth} must be in (0, 1)")

    def _ror(d, b):
        p = np.sqrt(d)
        if b <= 1 - p:
            return p
        # grazing: depth < p**2 at p = sqrt(d), depth = 1 at p = 1 + b
        return brentq(lambda x: approx_transit_depth(x, b) - d, p, 1 + b)

    return np.vectorize(_ror, otypes=[float])(depth, b)


def approx_tc_error(ror, b, T14, exptime, sigma_f):
    """
    Predicted mid-transit time uncertainty, following Price & Rogers 2014
    (finite integration time; Carter+2008 as exptime -> 0)

    Parameters
    ----------
    ror : planet-to-star radius ratio
    b : impact parameter
    T14 : first-to-fourth contact transit duration
    exptime : exposure time (in same units as T14)
    sigma_f : photometric error on flux
    """
    Q = np.sqrt(T14 / exptime) * (ror**2 / sigma_f)

    if b <= 1 - ror:
        tau_over_T14 = ror / (1 - b**2)
    else:
        tau_over_T14 = 0.5

    tau = tau_over_T14 * T14

    if tau >= exptime:
        sigma_tc = np.sqrt(tau * T14 / 2) / Q / np.sqrt(1 - exptime / (3 * tau))
    else:
        sigma_tc = np.sqrt(exptime * T14 / 2) / Q / np.sqrt(1 - tau / (3 * exptime))

    return sigma_tc
