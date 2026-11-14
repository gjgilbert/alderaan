__all__ = [
    "uniform_ppf",
    "loguniform_ppf",
    "norm_ppf",
    "set_sigma_cutoff",
    "robust_polyfit",
    "weighted_percentile",
]


import numpy as np
from numpy.polynomial import Legendre, Polynomial
from scipy.special import erfinv
from scipy.stats import norm as scipy_norm


def uniform_ppf(u, a, b):
    """
    Transform from U(0,1) --> Uniform(a,b)

    Args:
        u (array-like) : samples from standard uniform distribution U(0,1)
        a (float) : lower bound on transformed distribution U(a,b)
        b (float) : upper bound on transformed distribution U(a,b)

    Returns:
        array-like : transformed samples from uniform distribution
    """
    return u * (b - a) + a


def loguniform_ppf(u, a, b):
    """
    Transform from U(0,1) --> LogUniform(a,b)

    Args:
        u (array-like) : samples from standard uniform distribution U(0,1)
        a (float) : lower bound on transformed distribution lnU(a,b)
        b (float) : upper bound on transformed distribution lnU(a,b)

    Returns:
        x (array-like) : transformed samples from log-uniform distribution
    """
    return np.exp(u * np.log(b) + (1 - u) * np.log(a))


def norm_ppf(u, mu, sig, eps=1e-12):
    """
    Transform from U(0,1) --> Normal(mu,sig)

    Args:
        u (array-like) : samples from standard uniform distribution U(0,1)
        mu (float) : mean of transformed distribution N(mu,sig)
        sig (float) : standard deviation of transformed distribution N(mu,sig)
        eps (float) : provides numerical stability in the inverse error function
                      at edges

    Returns
    -------
        array-like : transformed samples from Normal distribution
    """
    return mu + sig * np.sqrt(2) * erfinv((2 * u - 1) / (1 + eps))


def set_sigma_cutoff(npts, sigma_min=3.0):
    """
    Set sigma cutoff such that not even one outlier is expected
    if distribution is normal

    Args:
        npts (int) : number of points
        sigma_min (float) : minimum sigma cutoff, default=3.0
    """
    return np.max([sigma_min, scipy_norm.interval((npts - 1) / npts)[1]])


def robust_polyfit(x, y, deg):
    """
    Drop-in replacement for np.polyfit(x, y, deg) that solves the least squares fit
    in a Legendre basis over the span of x, which stays well conditioned where the
    power-basis Vandermonde matrix does not

    Args:
        x (ndarray) : independent variable
        y (ndarray) : dependent variable, shape (M,) or (M, K)
        deg (int) : degree of the fitted polynomial

    Returns:
        ndarray : power-basis coefficients, highest order first (same as np.polyfit);
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
    Compute the q-th percentile of data array a, with optional weights w

    Similar to np.percentile, but allows for weights (not axis-wise)

    Args:
        a (array-like) : data
        q (float or array-like) : percentile(s) in [0, 100]
        w (array-like) : optional weights, same length as a

    Returns:
        float or ndarray : percentile value(s)
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
