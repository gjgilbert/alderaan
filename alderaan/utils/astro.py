__all__ = [
    "bin_data",
    "estimate_transit_depth",
    "ld_q_to_u",
    "ld_u_to_q",
    "predict_tc_error",
    "transit_obsmode",
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


def transit_obsmode(planets, litecurve):
    """
    Observing mode at the cadence nearest each transit

    Returns:
        transit_obsmode (list) : one array of obsmodes per planet
        unique_obsmodes (ndarray) : unique obsmodes across all transits
    """
    time = litecurve.time
    if np.any(np.diff(time) < 0):
        raise ValueError("litecurve.time must be sorted")

    obsmode = [None] * len(planets)
    for n, p in enumerate(planets):
        # nearest cadence; ties go to the earlier cadence
        i = np.clip(np.searchsorted(time, p.transit_times), 1, len(time) - 1)
        earlier = (p.transit_times - time[i - 1]) <= (time[i] - p.transit_times)
        obsmode[n] = litecurve.obsmode[i - earlier]

    return obsmode, np.unique(np.hstack(obsmode))


def bin_data(time, data, binsize, bin_centers=None):
    """Bin a 1D timeseries data array

    If bin_centers is not provided, bins will be linearly spaced
    between (time.min(),time.max())

    Args:
        time (ndarray) : time values
        data (ndarray) : corresponding data to be binned
        binsize (float): bin size for output data, in same units as time
        bin_centers (optional) : pre-defined bin centers

    Returns:
        tuple : a tuple containing:
          - time_binned (array-like)
          - data_binned (array-like)
    """
    if bin_centers is None:
        bin_centers = np.hstack(
            [
                np.arange(time.mean(), time.min() - binsize / 2, -binsize)[::-1],
                np.arange(time.mean(), time.max() + binsize / 2, binsize)[1:],
            ]
        )

    binned_data = np.zeros(len(bin_centers))
    for i, t0 in enumerate(bin_centers):
        binned_data[i] = np.mean(data[np.abs(time - t0) < binsize / 2])

    return bin_centers, binned_data


def estimate_transit_depth(p, b):
    """Calculate approximate transit depth following Mandel & Agol (2002)

    Args:
        p (array-like) : planet-to-star radius ratio Rp/Rstar
        b (array-like) : impact parameter

    Returns:
        array-like : transit depth
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
