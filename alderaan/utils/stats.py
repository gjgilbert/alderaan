"""Statistical helpers: ppf transforms, robust fits, sigma cutoffs, binning, R-hat"""

__all__ = [
    "uniform_ppf",
    "loguniform_ppf",
    "norm_ppf",
    "sigma_cutoff",
    "robust_polyfit",
    "weighted_percentile",
    "bin_data",
    "legendre_p1",
    "split_rhat",
]


import numpy as np
from numpy.polynomial import Legendre, Polynomial
from scipy.special import erfinv
from scipy.stats import norm as scipy_norm


def uniform_ppf(u, a, b):
    """
    Transform U(0,1) samples to Uniform(a, b)

    Parameters
    ----------
    u : array-like
        samples from U(0,1)
    a, b : float
        lower and upper bounds

    Returns
    -------
    x : array-like
    """
    return u * (b - a) + a


def loguniform_ppf(u, a, b):
    """
    Transform U(0,1) samples to LogUniform(a, b)

    Parameters
    ----------
    u : array-like
        samples from U(0,1)
    a, b : float
        lower and upper bounds

    Returns
    -------
    x : array-like
    """
    return np.exp(u * np.log(b) + (1 - u) * np.log(a))


def norm_ppf(u, mu, sig, eps=1e-12):
    """
    Transform U(0,1) samples to Normal(mu, sig)

    Parameters
    ----------
    u : array-like
        samples from U(0,1)
    mu, sig : float
        mean and standard deviation
    eps : float
        keeps erfinv finite at u = 0 and u = 1

    Returns
    -------
    x : array-like
    """
    return mu + sig * np.sqrt(2) * erfinv((2 * u - 1) / (1 + eps))


def sigma_cutoff(npts, sigma_min=3.0):
    """
    Two-sided sigma cutoff beyond which one of npts normal draws is expected,
    floored at sigma_min

    Parameters
    ----------
    npts : int
        number of points
    sigma_min : float
        minimum cutoff
    """
    return np.max([sigma_min, scipy_norm.interval((npts - 1) / npts)[1]])


def robust_polyfit(x, y, deg):
    """
    Drop-in replacement for np.polyfit(x, y, deg) that solves the least squares fit
    in a Legendre basis over the span of x, which stays well conditioned where the
    power-basis Vandermonde matrix does not

    Parameters
    ----------
    x : ndarray
        independent variable
    y : ndarray
        dependent variable, shape (M,) or (M, K)
    deg : int
        degree of the fitted polynomial

    Returns
    -------
    coeffs : ndarray
        power-basis coefficients, highest order first (same as np.polyfit);
        shape (deg + 1,) or (deg + 1, K)
    """
    y = np.asarray(y)
    if y.ndim == 2:
        return np.column_stack([robust_polyfit(x, yk, deg) for yk in y.T])

    series = Legendre.fit(x, y, deg)

    # convert() trims trailing zero coefficients; np.polyfit always returns deg + 1
    coeffs = np.zeros(deg + 1)
    converted = series.convert(kind=Polynomial).coef
    coeffs[: converted.size] = converted

    return coeffs[::-1]


def weighted_percentile(a, q, w=None):
    """
    q-th percentile of a, with optional weights w (like np.percentile, but
    not axis-wise)

    Parameters
    ----------
    a : array-like
        data
    q : float or array-like
        percentile(s) in [0, 100]
    w : array-like, optional
        weights, same length as a

    Returns
    -------
    percentile : float or ndarray
        percentile value(s)
    """
    a = np.asarray(a)
    q = np.asarray(q)

    if w is None:
        return np.percentile(a, q)

    if not (np.all(q >= 0) and np.all(q <= 100)):
        raise ValueError("quantiles should be in [0, 100]")

    order = np.argsort(a)
    a = a[order]
    w = np.asarray(w)[order]

    weighted_quantiles = np.cumsum(w) - 0.5 * w
    weighted_quantiles -= weighted_quantiles[0]
    weighted_quantiles /= weighted_quantiles[-1]

    return np.interp(q / 100.0, weighted_quantiles, a)


def bin_data(time, data, binsize, bin_centers=None):
    """
    Bin a 1D timeseries

    Default bin_centers: spaced by binsize from time.mean(), spanning the data

    Parameters
    ----------
    time : ndarray
        time values
    data : ndarray
        corresponding data to be binned
    binsize : float
        bin width, in units of time
    bin_centers : ndarray, optional

    Returns
    -------
    bin_centers : ndarray
    binned_data : ndarray
        mean of data in each bin
    binned_error : ndarray
        standard error of the mean in each bin
    """
    if bin_centers is None:
        bin_centers = np.hstack(
            [
                np.arange(time.mean(), time.min() - binsize / 2, -binsize)[::-1],
                np.arange(time.mean(), time.max() + binsize / 2, binsize)[1:],
            ]
        )

    binned_data = np.zeros(len(bin_centers))
    binned_error = np.zeros(len(bin_centers))
    for i, t0 in enumerate(bin_centers):
        d = data[np.abs(time - t0) < binsize / 2]
        binned_data[i] = np.mean(d)
        binned_error[i] = np.std(d) / np.sqrt(d.size)

    return bin_centers, binned_data, binned_error


def legendre_p1(x, domain):
    """
    First Legendre polynomial P1 of x, with domain (x_min, x_max) mapped onto
    the Legendre window [-1, 1] (numpy convention): x_min -> -1, x_max -> +1

    Parameters
    ----------
    x : array-like
    domain : (float, float)

    Returns
    -------
    ndarray
    """
    return Legendre.basis(1, domain=domain)(x)


def split_rhat(draws):
    """
    Split R-hat of a single chain: first vs second half of the draws

    Parameters
    ----------
    draws : ndarray, shape (ndraw, nparam)

    Returns
    -------
    rhat : ndarray, shape (nparam,)
    """
    m = len(draws) // 2
    halves = np.stack([draws[:m], draws[m : 2 * m]])
    within = halves.var(axis=1, ddof=1).mean(axis=0)
    between = halves.mean(axis=1).var(axis=0, ddof=1)
    return np.sqrt(((m - 1) / m * within + between) / within)
