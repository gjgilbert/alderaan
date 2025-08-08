__all__ = ['bin_data',
           'estimate_transit_depth',
           ]

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from alderaan.constants import pi


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

    d[grazing] = (1 / pi) * (pg**2 * k0 + k1 - s0)

    # non-transiting (b >= 1+p)
    d[b >= 1 + p] = 0.0

    return np.squeeze(d)