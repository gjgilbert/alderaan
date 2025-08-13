__all__ = ['uniform_ppf',
           'loguniform_ppf',
           'norm_ppf',
           ]


import numpy as np
from scipy.special import erfinv


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
        eps (float) : provides numerical stability in the inverse error function at edges

    Returns
    -------
        array-like : transformed samples from Normal distribution
    """
    return mu + sig * np.sqrt(2) * erfinv((2 * u - 1) / (1 + eps))
