__all__ = [
    "boxcar_smooth",
    "FFT_estimator",
    "LS_estimator",
    "bin_data",
    "autocorr_length",
    "weighted_percentile",
    "lorentzian",
    "heavyside",
    "draw_random_samples",
    "get_bw",
    "generate_synthetic_samples",
]


import warnings

import aesara_theano_fallback.tensor as T
import astropy.stats
from astropy.timeseries import LombScargle
import numpy as np
from scipy import fftpack
import scipy.optimize as op
import scipy.signal as sig
from scipy import stats
from scipy.interpolate import interp1d
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from sklearn.covariance import EmpiricalCovariance

from .constants import *


def boxcar_smooth(x, winsize, passes=1):
    """
    Smooth a data array with a sliding boxcar filter

    Parameters
    ----------
    x : ndarray
        data to be smoothed
    winsize : int
        size of boxcar window
    passes : int
        number of passes (default=1)

    Returns
    -------
    xsmooth : ndarray
        smoothed data array,same size as input 'x'
    """
    win = sig.boxcar(winsize) / winsize
    xsmooth = np.pad(x, (winsize, winsize), mode="reflect")

    for i in range(passes):
        xsmooth = sig.convolve(xsmooth, win, mode="same")

    xsmooth = xsmooth[winsize:-winsize]

    return xsmooth


def FFT_estimator(
    x,
    y,
    fmin=None,
    fmax=None,
    crit_fap=0.003,
    nboot=1000,
    return_levels=False,
    max_peaks=2,
):
    """
    Identify significant frequencies in a (uniformly sampled) data series

    Parameters
    ----------
    x : array-like
        1D array of x time domain data values; should be monotonically increasing
    y : array-like
        1D array of corresponding y time domain data values, len(x)
    fmin : float (optional)
        minimum frequency to check; if not provided this will be set to 1/baseline
    fmax : float (optional)
        maximum frequency to check; if not provided this will be set to the Nyquist frequency
    crit_fap : float
        critical false alarm probability for significant signals (default=0.003)
    nboot : int
        number of bootstrap samples for calculating false alarm probabilities (default=1000)
    return_levels : bool
        if True, return the FAP for each frequency in the grid

    Returns
    -------
    xf : ndarray
        1D array of frequency values
    yf : ndarray
        1D array of response values, len(xf); |yf|^2 = power
    freqs : ndarray
        array of significant frequencies
    faps : ndarray
        array of corresponding false alarm probabilities
    levels : ndarray
        array of FAPs for each grid frequency
    """
    # set baseline and Nyquist frequencies
    fbas = 1 / (x.max() - x.min())
    fnyq = 1 / (2 * (x[1] - x[0]))

    if fmin is None:
        fmin = 1.0 * fbas
    if fmax is None:
        fmax = 1.0 * fnyq

    # generate FFT (convolve w/ hann window to reduce spectral leakage)
    w = sig.hann(len(x))

    xf = np.linspace(0, fnyq, len(x) // 2)
    yf = np.abs(fftpack.fft(y * w)[: len(x) // 2])

    keep = (xf >= fmin) * (xf <= fmax)

    xf = xf[keep]
    yf = yf[keep]

    # calculate false alarm probabilities w/ bootstrap test
    yf_max = np.zeros(nboot)

    for i in range(nboot):
        y_shuff = np.random.choice(y, size=len(y), replace=False)
        yf_shuff = np.abs(fftpack.fft(y_shuff * w)[: len(x) // 2])
        yf_max[i] = yf_shuff[keep].max()

    yf_fap = np.zeros(len(xf))

    for i in range(len(xf)):
        yf_fap[i] = 1 - np.sum(yf[i] > yf_max) / nboot

    levels = np.array(
        [
            np.percentile(yf_max, 90),
            np.percentile(yf_max, 99),
            np.percentile(yf_max, 99.9),
        ]
    )

    # now search for significant frequencies
    m = np.zeros(len(xf), dtype="bool")
    freqs = []
    faps = []

    if len(xf) > 3:
        loop = True
        while loop:
            peakfreq = xf[~m][np.argmax(yf[~m])]
            peakfap = yf_fap[~m][np.argmax(yf[~m])]

            if peakfreq == xf[~m].min():
                m[xf <= xf[~m].min()] = 1

            elif peakfap < crit_fap:
                fxn_ = lambda theta, x, y: y - lorentzian(theta, x)

                theta_in = np.array([peakfreq, fbas, yf.max(), np.median(yf)])
                theta_out, success = op.leastsq(fxn_, theta_in, args=(xf, yf))

                width = np.max(5 * [theta_out[1], 3 * (xf[1] - xf[0])])
                m += np.abs(xf - theta_out[0]) / width < 1

                freqs.append(theta_out[0])
                faps.append(peakfap)

            else:
                loop = False

            if len(freqs) >= max_peaks:
                loop = False

            if np.sum(m) / len(m) > 0.5:
                loop = False

    freqs = np.asarray(freqs)
    faps = np.asarray(faps)

    if return_levels:
        return xf, yf, freqs, faps, levels

    return xf, yf, freqs, faps


def LS_estimator(x, y, fsamp=None, fap=0.1, return_levels=False, max_peaks=2):
    """
    Generate a Lomb-Scargle periodogram and identify significant frequencies
      * assumes that data are nearly evenly sampled
      * optimized for finding marginal periodic TTV signals in OMC data
      * may not perform well for other applications

    Parameters
    ----------
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
    hann = sig.windows.hann(len(x))
    hann /= np.sum(hann)

    # identify any egregious outliers
    out = np.abs(y - np.median(y)) / astropy.stats.mad_std(y) > 5.0

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


def bin_data(time, data, binsize, bin_centers=None):
    """
    Parameters
    ----------
    time : ndarray
        vector of time values
    data : ndarray
        corresponding vector of data values to be binned
    binsize : float
        bin size for output data, in same units as time

    Returns
    -------
    bin_centers : ndarray
        center of each data (i.e. binned time)
    binned_data : ndarray
        data binned to selcted binsize
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


def autocorr_length(x):
    """
    Determine the autocorrelation length of a 1D data vector

    Parameters
    ----------
    x : array-like
        1D data vector

    Returns
    -------
    tau : float
        estimated autocorrelation length
    """
    # subtract off mean
    y = x - np.mean(x)

    # generate empirical ACF
    acf = np.correlate(y, y, mode="full")
    acf = acf[len(acf) // 2 :]
    acf /= acf[0]

    # automatic windowing following Sokal 1989
    taus = 2 * np.cumsum(acf) - 1

    m = np.arange(len(taus)) < 5.0 * taus
    if np.any(m):
        win = np.argmin(m)
    else:
        win = len(taus) - 1

    return np.max([taus[win], 1.0])


def weighted_percentile(a, q, w=None):
    """
    Compute the q-th percentile of data array a
    Similar to np.percentile, but allows for weights (but not axis-wise computation)
    """
    a = np.array(a)
    q = np.array(q)

    if w is None:
        return np.percentile(a, q)

    else:
        w = np.array(w)

        assert np.all(q >= 0) and np.all(q <= 100), "quantiles should be in [0, 100]"

        order = np.argsort(a)
        a = a[order]
        w = w[order]

        weighted_quantiles = np.cumsum(w) - 0.5 * w
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]

        return np.interp(q / 100.0, weighted_quantiles, a)


def lorentzian(theta, x):
    """
    Model a Lorentzian (Cauchy) function

    Parameters
    ----------
    theta : array-like
        parameters for Lorentzian = [loc, scale, height, baseline]
    x : array-like
        1D array of x data values

    Returns
    -------
    pdf : array-like
        Lorentzian probability density function
    """
    loc, scale, height, baseline = theta

    return height * stats.cauchy(loc=loc, scale=scale).pdf(x) + baseline


def heavyside(x, x0=0.0, k=1000.0):
    """
    Approximates the Heavyside step function with a smooth distribution
    Implemented using theano tensors for easily setting bounds on PyMC3 deterministic variables

    H0 = 1/(1+T.exp(-2*k*(x-x0)))

    Parameters
    ----------
    x : array-like
        input values at which to compute function
    x0 : float
        location of step
    k : float (optional)
        exponent factor in approximation (default=1000.)
    """
    return 1 / (1 + T.exp(-2 * k * (x - x0)))


def draw_random_samples(pdf, domain, N, *args, **kwargs):
    """
    Draw random samples from a given pdf using inverse transform sampling

    Paramters
    --------
    pdf : univarite function
        outputs pdf as a function of input value
    domain : tuple
        domain over which pdf is defined
    N : int
        number of random samples to draw

    Returns
    -------
    samples : ndarray
        vector of random samples drawn
    """
    x = np.linspace(domain[0], domain[1], int(1e5))
    y = pdf(x, *args, **kwargs)
    cdf_y = np.cumsum(y)
    cdf_y = cdf_y / cdf_y.max()
    inverse_cdf = interp1d(cdf_y, x)

    return inverse_cdf(np.random.uniform(cdf_y.min(), cdf_y.max(), N))


def get_bw(samples, weights=None, max_draws=1000):
    """
    Use cross-validation to determine KDE bandwidth for a set of 1D data samples
    Iteratively performs first coarse then fine estimation

    Parameters
    ----------
    samples : array-like
        1D data samples
    weights : array-like (optional)
        weights corresponding to samples
    max_draws : int (default=1000)
        maximum number of samples to use during estimation

    Returns
    -------
    bw : float
        estimated bandwidth
    """
    N = len(samples)
    x = np.random.choice(samples, p=weights, size=np.min([max_draws, N]), replace=True)

    coarse_mesh = np.linspace((x.max() - x.min()) / N, np.std(x), int(np.sqrt(N)))
    grid = GridSearchCV(KernelDensity(), {"bandwidth": coarse_mesh}, cv=5)
    grid.fit(x[:, None])

    fine_mesh = grid.best_params_["bandwidth"] + np.linspace(-1, 1, int(np.sqrt(N))) * (
        coarse_mesh[1] - coarse_mesh[0]
    )
    grid = GridSearchCV(KernelDensity(), {"bandwidth": fine_mesh}, cv=5)
    grid.fit(x[:, None])

    return grid.best_params_["bandwidth"]


def generate_synthetic_samples(samples, bandwidths, n_up, weights=None):
    """
    Use PDF Over-Sampling (PDFOS - Gao+ 2014) to generate synthetic samples

    Parameters
    ----------
    samples : ndarray, (N x M)
        array of data samples arranged N_samples, M_parameters
    bandwitdhs : array-like, (M)
        pre-estimated KDE bandwidths for each of M parameters
    n_up : int
        number of upsampled synthetic data points to generate
    weights : ndarray, (N x M)
        array of weights corresponding to samples

    Returns
    -------
    new_samples : ndarray
        array containing synthetic samples
    """
    n_samp, n_dim = samples.shape
    index = np.arange(0, n_samp, dtype="int")

    # we'll generate a few more samples than needed in anticipation of rejecting a few
    n_up101 = int(1.01 * n_up)

    # naive resampling (used only to estimate covariance matrix)
    naive_resamples = samples[np.random.choice(index, p=weights, size=3 * n_samp)]

    # compute empirical covariance matrix and lower Cholesky decomposition
    emp_cov = EmpiricalCovariance().fit(naive_resamples).covariance_
    L = np.linalg.cholesky(emp_cov)

    # scale each parameter by precomputed bandwidths so they have similar variance
    samples_scaled = (samples - np.mean(samples, axis=0)) / bandwidths

    # calculate synthetic samples following PDFOS (Gao+ 2014)
    random_index = np.random.choice(index, p=weights, size=n_up101)
    random_samples = samples_scaled[random_index]
    random_jitter = np.random.normal(0, 1, n_up101 * n_dim).reshape(n_up101, n_dim)
    new_samples = random_samples + np.dot(L.T, random_jitter.T).T

    # rescale each parameter to invert previous scaling
    new_samples = new_samples * bandwidths + np.mean(samples, axis=0)

    # reject any synthetic samples pushed out of bounds of original samples
    bad = np.zeros(n_up101, dtype="bool")

    for i in range(n_dim):
        bad += new_samples[:, i] < samples[:, i].min()
        bad += new_samples[:, i] > samples[:, i].max()

    if np.sum(bad) / len(bad) > 0.01:
        warnings.warn(
            "More than 1% of PDFOS generated samples were beyond min/max values of original samples"
        )

    new_samples = new_samples[~bad]

    # only return n_up samples
    if new_samples.shape[0] >= n_up:
        return new_samples[:n_up]

    else:
        # use naive resampling to replace rejected samples
        replacement_samples = samples[
            np.random.choice(index, p=weights, size=n_up - new_samples.shape[0])
        ]

        return np.vstack([new_samples, replacement_samples])
