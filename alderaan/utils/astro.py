"""Astrophysical conversions: limb darkening bases, depths, timing errors"""

__all__ = [
    "approx_tc_error",
    "approx_transit_depth",
    "ld_q_to_u",
    "ld_u_to_q",
]


import numpy as np


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
    """Calculate approximate transit depth following Mandel & Agol (2002)

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

    d[grazing] = (1 / np.pi) * (pg**2 * k0 + k1 - s0)

    # non-transiting (b >= 1+p)
    d[b >= 1 + p] = 0.0

    return np.squeeze(d)


def approx_tc_error(ror, b, T14, exptime, sigma_f):
    """
    Predict uncertainty on mid-transit time
    See Price & Rogers 2014 (finite integration time; Carter+2008 as exptime -> 0)

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
