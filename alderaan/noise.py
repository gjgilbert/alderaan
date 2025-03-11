__all__ = [
    "make_chunklist",
    "build_sho_model",
    "generate_acf",
    "model_acf",
    "make_covariance_matrix",
    "generate_synthetic_noise",
    "make_gp_prior_dict",
]


import warnings

import aesara_theano_fallback.tensor as T
import astropy
from celerite2.theano import GaussianProcess
from celerite2.theano import terms as GPterms
import numpy as np
from numpy.linalg import LinAlgError
import pymc3 as pm
import scipy.signal as sig

from .constants import *
from .utils import boxcar_smooth, FFT_estimator


def make_chunklist(
    time,
    flux,
    cadno,
    Npts,
    sigma_reject=5.0,
    gap_tolerance=13,
    interpolate=True,
    cover=0.95,
):
    """
    Make an array of 'chunks' of data uninterupted by transits or data gaps

    Parameters
    ----------
    time : array-like
        out-of-transit time values (i.e. transits masked BEFORE passing array into make_chucklist)
    flux : array-like
        out-of-transit flux values
    cadno : array-like
        out-of-transit cadence numbers
    Npts : int
        number of points to use in each chunk; should be ~3x max transit duration
    sigma_reject : float
        sigma threshold for rejection of noisy chunks (default=5.0)
    gap_tolerance : int
        maximum number of consecutive missing cadences allowed (default=2)
    interpolate : bool
        True to perform linear interpolation with additive white noise over small gaps (default=True)
    cover : float between (0,1)
        fractional coverage required to consider a chunk 'good'

    Returns
    -------
    chunklist : list
        M x N list of data 'chunks' uninterrupted by transits or data gaps
    """
    chunklist = []

    i = 0
    loop = True
    while loop:
        # mark start and end cadence for this chunk
        cad_low = cadno[i]
        cad_high = cad_low + Npts + 1

        use = (cadno >= cad_low) * (cadno < cad_high)

        # check if there are gaps
        index = cadno[use] - cad_low
        gaps = np.hstack([1, index[1:] - index[:-1]])

        # make sure there are no long stretches of consecutive missing cadences
        if gaps.max() <= gap_tolerance:
            # pull time and flux chunks
            t_chunk = np.ones(Npts + 1) * 99  # use '99' to mark missing cadences
            f_chunk = np.ones(Npts + 1) * 99

            t_chunk[index] = time[use]
            f_chunk[index] = flux[use]

            # interpolate over missing cadences
            empty = t_chunk == 99

            t_chunk[empty] = np.interp(
                np.arange(Npts + 1)[empty], index, t_chunk[~empty]
            )
            f_chunk[empty] = np.interp(
                np.arange(Npts + 1)[empty], index, f_chunk[~empty]
            )
            f_chunk[empty] += np.random.normal(size=np.sum(empty)) * np.std(
                f_chunk[~empty]
            )

            # require at least X% coverage (default = 95%)
            if np.sum(~empty) / len(empty) > cover:
                chunklist.append(f_chunk)

            i += Npts

        # if there are missing cadences, move forward past any gaps
        else:
            i += np.argmax(gaps > gap_tolerance)

        # finish the loop if there is no more unusued flux
        if i > (len(time) - Npts - 1):
            loop = False

    # convert list to array
    chunklist = np.array(chunklist)

    # reject any chunks with unusually high medians or variability
    loop = True
    while loop:
        mad_chunk = astropy.stats.mad_std(chunklist, axis=1)
        med_chunk = np.median(chunklist, axis=1)

        bad = (
            np.abs(mad_chunk - np.median(mad_chunk)) / astropy.stats.mad_std(mad_chunk)
            > sigma_reject
        )
        bad += (
            np.abs(med_chunk - np.median(med_chunk)) / astropy.stats.mad_std(med_chunk)
            > sigma_reject
        )

        chunklist = chunklist[~bad]

        if np.sum(bad) == 0:
            loop = False

    return chunklist


def build_sho_model(t, y, var_method, fmin=None, fmax=None, f0=None, Q0=None):
    """
    Build PyMC3/celerite model for correlated noise using a single SHOTerm

    Parameters
    ----------
    t : array-like
        independent variable data (e.g. time)
    y : array-like
        corresponding dependent variable data (e.g. empirical ACF or flux)
    var_method : string
        automatic method for selecting y data variance
        'global' --> yvar = np.var(y)
        'local' --> yvar = np.var(y - local_trend)
        'fit' --> log_yvar is a free hyperparameter in the GP model
    fmin : float (optional)
        lower bound on (ordinary, not angular) frequency
    fmax : float (optional)
        upper bound on (ordinary, not angular) frequency
    f0 : float (optional)
        if provided, tight priors will be placed on this frequency
    Q0 : float (optional)
        if provided, Q will be fixed to this value

    Returns
    -------
    model : a PyMC3 model with a celerite SHOTERM
    """
    with pm.Model() as model:
        # frequency
        if fmin is None:
            fmin = 1 / (t.max() - t.min())
        if fmax is None:
            fmax = 2 / np.mean(np.diff(t))

        df = 1 / (t.max() - t.min())

        if f0 is None:
            logw0 = pm.Uniform(
                "logw0", lower=np.log(2 * pi * fmin), upper=np.log(2 * pi * fmax)
            )
        else:
            logw0 = pm.Bound(
                pm.Normal, lower=np.log(2 * pi * fmin), upper=np.log(2 * pi * fmax)
            )("logw0", mu=np.log(2 * pi * f0), sd=np.log((f0 + df) / f0))

        w0 = pm.Deterministic("w0", T.exp(logw0))

        # quality factor (downstream transit models accept logQ - easier to keep using it here)
        if Q0 is None:
            logQ_off = pm.Normal("logQ_off", mu=0, sd=5, testval=np.log(1e-3))
            logQ = pm.Deterministic("logQ", T.log(1 / np.sqrt(2) + T.exp(logQ_off)))
        else:
            logQ = pm.Deterministic("logQ", T.log(Q0))

        # amplitude parameter
        logSw4 = pm.Normal("logSw4", mu=np.log(np.var(y) * fmax**3), sd=10.0)
        S0 = pm.Deterministic("S0", T.exp(logSw4) / w0**4)

        # mean
        mean = pm.Normal("mean", mu=np.mean(y), sd=np.std(y))

        # diagonal variance
        if var_method == "global":
            yvar = np.var(y)
        elif var_method == "local":
            yvar = np.var(y - boxcar_smooth(y, 7))
        elif var_method == "fit":
            log_yvar = pm.Normal("log_yvar", mu=np.log(np.var(y)), sd=10.0)
            yvar = T.exp(log_yvar)
        else:
            raise ValueError("Must specify var_method as 'global', 'local', or 'fit'")

        # set up the GP
        kernel = GPterms.SHOTerm(S0=S0, w0=w0, Q=T.exp(logQ))
        gp = GaussianProcess(kernel, t=t, diag=yvar * T.ones(len(t)), mean=mean)
        gp.marginal("gp", observed=y)

        # track GP prediction
        pm.Deterministic("pred", gp.predict(y))

    return model


def generate_acf(time, flux, cadno, Npts, sigma_reject=5.0, keep_zero_lag=False):
    """
    Generate an autocorrelation function from a collection of out-of-transit data 'chunks'

    Parameters
    ----------
    time : array-like
        out-of-transit time values (i.e. transits masked BEFORE passing array into make_chucklist)
    flux : array-like
        out-of-transit flux values
    cadno : array-like
        out-of-transit cadence numbers
    Npts : int
        number of points to use in each chunk; should be ~3x max transit duration
    sigma_reject : float
        sigma threshold for rejection of noisy chunks (default=5.0)
    keep_zero_lag : bool
        default behavior (False) is to remove lag-zero term from ACF before returning

    Returns
    -------
    xcor : ndarray
        time-lag used to generate autocorrelation function with lag-zero value removed
    acor : ndarray
        autocorrelation function with lag-zero value removed
    """
    chunklist = make_chunklist(time, flux, cadno, Npts, sigma_reject=sigma_reject)

    Nsamples = chunklist.shape[0]

    # generate the autocorrelation function
    acor = np.zeros((Nsamples, 2 * Npts + 1))

    for i in range(Nsamples):
        chunk = chunklist[i] - np.median(chunklist[i])
        acor[i] = np.correlate(chunk, chunk, mode="full")

    acor = np.mean(acor, axis=0)
    acor = acor[Npts:] / acor[Npts]
    xcor = np.arange(Npts + 1)

    if keep_zero_lag:
        return xcor, acor
    else:
        return xcor[1:], acor[1:]


def model_acf(
    xcor,
    acor,
    fcut,
    fmin=None,
    fmax=None,
    crit_fap=0.003,
    method="smooth",
    window_length=None,
):
    """
    Model an empirical autocorrelation function (ACF) using one of several methods

    Parameters
    ----------
    xcor : array-like
        lag time values
    acor : array-like
        empirical autocorrelation function power at each time lag
    fcut : float
        cutoff value for seperating high vs. low frequencies
    fmin : float (optional)
        minimum frequency to check; if not provided this will be set to 1/baseline
    fmax : float (optional)
        maximum frequency to check; if not provided this will be set to the Nyquist frequency
    crit_fap : float
        critical false alarm probability for significant signals (default=0.003)
    method : string
        method to model low frequency component; either 'smooth' or 'savgol'  (default='smooth')
    window_length : int
        size of smoothing window  (required if method is 'smooth')

    Returns
    -------
    acor_emp, acor_mod, xf, yf, freqs
    """
    # arrays to hold empirical and model ACF
    acor_mod = np.zeros_like(acor)
    acor_emp = acor.copy()

    # 1st model component (low frequency)
    xf_L, yf_L, freqs_L, faps_L = FFT_estimator(
        xcor, acor_emp, fmin=fmin, fmax=fmax, crit_fap=crit_fap
    )

    low_freqs = freqs_L[freqs_L < fcut]

    # model the ACF with chosen method
    if method == "smooth":
        if window_length is None:
            raise ValueError("Must specify window_length if method == 'smooth'")

        acor_mirror = np.hstack([acor_emp[::-1], acor_emp])
        acor_mod = boxcar_smooth(acor_mirror, window_length)[len(acor_emp) :]

    elif method == "savgol":
        if window_length is None:
            if len(low_freqs) == 0:
                window_length = 45
            else:
                window_length = int((24 * 60) / low_freqs[0]) // 2
                window_length = window_length + (window_length % 2) + 1

        acor_mirror = np.hstack([acor_emp[::-1], acor_emp])
        acor_mod = sig.savgol_filter(
            acor_mirror, polyorder=2, window_length=window_length
        )[len(acor_emp) :]
        acor_mod = boxcar_smooth(acor_mod, 5)

    else:
        raise ValueError("method must be either 'smooth', or 'savgol'")

    # check for any high-frequency components
    xf_H, yf_H, freqs_H, faps_H = FFT_estimator(
        xcor, acor_emp - acor_mod, fmin=fmin, fmax=fmax, crit_fap=crit_fap, max_peaks=5
    )

    high_freqs = freqs_H[freqs_H > fcut]

    # combine freqs, return ACF and power spectrum
    freqs = np.hstack([low_freqs, high_freqs])

    return acor_emp, acor_mod, xf_L, yf_L, freqs


def make_covariance_matrix(acf, size=None):
    """
    Generate a square 2D covariance matrix from a 1D autocorrelation function

    Parameters
    ----------
    acf : array-like
        1D autocorrelation function not including lag-zero term (length N)
    size : int
        size of output covariance matrix (optional; if not given size --> N+1 x N+1)

    Returns
    -------
    covmatrix : ndarray
        n x n array; diagonal terms all equal 1.0
    """
    N = len(acf) + 1

    if size is None:
        n = N
    else:
        n = size

    if n > N:
        acf = np.hstack([acf, np.zeros(n - N)])

    covmatrix = np.zeros((n, n))

    for i in range(n):
        covmatrix[i, i + 1 :] = acf[: n - i - 1]

    covmatrix += covmatrix.swapaxes(0, 1)
    covmatrix += np.eye(n)

    return covmatrix


def generate_synthetic_noise(xcor, acor, n, sigma):
    """
    Generate synthetic correlated noise given a specified autorrelation function


    Parameters
    ----------
    xcor : array-like
        lag time values
    acor : array-like
        autocorrelation function power at each time lag
    n : int
        size of n x n covariance matrix
    sigma : float
        scale of white noise

    Returns
    -------
    x : ndarray
        1D array of time values (or some general independent coordinate)
    red_noise : ndarray
        synthetic correlated noise
    white_noise: ndarray
        gaussian noise vector used to generate red noise
    """
    # first make the covariance matrix
    covmatrix = make_covariance_matrix(acor, n)

    # decompose it
    try:
        L = np.linalg.cholesky(covmatrix)

    except LinAlgError:
        try:
            warnings.warn(
                "Covariance matrix not positive definite...adjusting automatically"
            )

            # decompose for eigenvalues and eigenvectors
            # matrix was constructed to be symmetric - if eigh doesn't work there is a serious problem
            eigenvalues, eigenvectors = np.linalg.eigh(covmatrix)

            # diagonalize
            D = np.diag(eigenvalues)

            # elementwise comparison with zero
            Z = np.zeros_like(D)
            Dz = np.maximum(D, Z)

            # generate a positive semi-definite matrix
            psdm = np.dot(np.dot(eigenvectors, Dz), eigenvectors.T)

            # now make it positive definite
            eps = np.min(np.abs(acor[acor != 0])) * 1e-6
            covmatrix = psdm + np.eye(n) * eps

            # renormalize
            covmatrix = covmatrix / covmatrix.max()

            # do Cholesky decomposition
            L = np.linalg.cholesky(covmatrix)

        except LinAlgError:
            warnings.warn(
                "Covariance matrix fatally broken...returning identity matrix"
            )
            covmatrix = np.eye(n)
            L = np.linalg.cholesky(covmatrix)

    # generate a vector of gaussian noise and remove any random covariance
    z = np.random.normal(size=n) * sigma

    # make correlated noise
    x = np.arange(n) * (xcor[1] - xcor[0])
    y = np.dot(L, z)

    return x, y - z, z


def make_gp_prior_dict(
    sho_trace, percs=[0.135, 2.275, 15.865, 50.0, 84.135, 97.725, 99.865]
):
    """
    Generates a list of percentiles from posteriors for each hyperparameter of a GP noise model
    The expected sho_trace should be the output of a PyMC3/Exoplanet model built with noise.build_sho_model()

    Assumes a specific set of input variable names from sho_trace:
      - ['logw0', 'logSw4', 'logQ']

    Parameters
    ----------
    sho_trace : PyMC3 multitrace
        trace output of a PyMC3/Exoplanet model built with noise.build_sho_model()
        will also work if sho_trace a dictionary with the necessary keys
    percs : list
        list of percentiles to return, by default 1- 2- 3- sigma and median

    Returns
    -------
    priors : dict
        Dictionary keys can be any combination of ['logw0', 'logSw4', 'logQ']
        Each key gives a list of values corresponding to specified percentiles from sho_trace
    """
    priors = {}
    priors["percentiles"] = percs

    try:
        varnames = sho_trace.varnames
    except Exception:
        varnames = list(sho_trace.keys())

    # get posterior percentiles of each hyperparameter and assign to a dictionary
    if np.isin("logw0", varnames):
        priors["logw0"] = np.percentile(sho_trace["logw0"], percs)

    if np.isin("logSw4", varnames):
        priors["logSw4"] = np.percentile(sho_trace["logSw4"], percs)

    if np.isin("logQ", varnames):
        priors["logQ"] = np.percentile(sho_trace["logQ"], percs)

    return priors
