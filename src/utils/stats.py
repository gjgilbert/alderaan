__all__ = ['LS_estimator']

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from astropy.stats import mad_std
from astropy.timeseries import LombScargle
import numpy as np
from scipy import signal


def LS_estimator(x, y, fsamp=None, fap=0.1, return_levels=False, max_peaks=2):
    """
    Generate a Lomb-Scargle periodogram and identify significant frequencies
      * assumes that data are nearly evenly sampled
      * optimized for finding marginal periodic TTV signals in OMC data
      * may not perform well for other applications

    Arguments
    ---------
    x : array-like
        1D array of x data values; should be monotonically increasing
    y : array-like
        1D array of corresponding y data values, len(x)
    fsamp: float
        nominal sampling frequency; if not provided it will be calculated from the data
    fap : float
        false alarm probability threshold to consider a frequency significant (default=0.1)

    Returns
    -------
    xf : ndarray
        1D array of frequencies
    yf : ndarray
        1D array of corresponding response
    freqs : list
        signficant frequencies
    faps : list
        corresponding false alarm probabilities
    """
    # get sampling frequency
    if fsamp is None:
        fsamp = 1 / np.min(x[1:] - x[:-1])

    # Hann window to reduce ringing
    hann = signal.windows.hann(len(x))
    hann /= np.sum(hann)

    # identify any egregious outliers
    out = np.abs(y - np.median(y)) / mad_std(y - np.median(y)) > 5.0
    xt = x[~out]
    yt = y[~out]

    freqs = []
    faps = []

    loop = True
    while loop:
        lombscargle = LombScargle(xt, yt * hann[~out])
        xf, yf = lombscargle.autopower(
            minimum_frequency=1.5 / (xt.max() - xt.min()),
            maximum_frequency=0.25 * fsamp,
            samples_per_peak=10,
        )

        peak_freq = xf[np.argmax(yf)]
        peak_fap = lombscargle.false_alarm_probability(yf.max(), method="bootstrap")

        # output first iteration of LS periodogram
        if len(freqs) == 0:
            xf_out = xf.copy()
            yf_out = yf.copy()
            levels = lombscargle.false_alarm_level([0.1, 0.01, 0.001])

        if peak_fap < fap:
            yt -= lombscargle.model(xt, peak_freq) * len(xt)
            freqs.append(peak_freq)
            faps.append(peak_fap)

        else:
            loop = False

        if len(freqs) >= max_peaks:
            loop = False

    if return_levels:
        return xf_out, yf_out, freqs, faps, levels

    else:
        return xf_out, yf_out, freqs, faps
