__all__ = ['prior_transform', '_uniform_ppf', '_loguniform_ppf',  '_norm_ppf']

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from dynesty.utils import print_fn
import io
import numpy as np
from scipy.special import erfinv
import time


def prior_transform(uniform_hypercube, fixed_durations):
    """
    Prior transform over physical ALDERAAN basis {C0, C1, r, b, T14}...{q1,q2}

    Distributions are hard-coded to be:
        Normal on ephemeris perturbations {C0,C1}
        Log-uniform on radius ratio r and transit duration T14
        Uniform on impact parameter b
        Uniform on quadratic limb darkening coefficients {q1,q2}

    For motivation behind this prior-parameter choice see:
        Carter+ 2008 (2008ApJ...689..499C)
        Kipping 2013 (2013MNRAS.435.2152K)
        Gilbert, MacDougall, & Petigura 2022 (2022AJ....164...92G)
        MacDougal, Gilbert, & Petigura 2023 (2023AJ....166...61M)

    Assumes that transit durations are known with reasonable confidence
      * sampled T14 will be restricted between (0.01*fixed_duration, 3*fixed_duration)

    Parameters
    ----------
        uniform_hypercube : array-like, length 5 * N + 2
            list of parameters N * [C0, C1, r, b, T14] + [q1, q2]
        fixed_durations : array-like
            list of transit durations for N planets (used for setting prior limits)

    Returns
    -------
        transformed_hypercube : array_like, length 5 * N + 2
            transformed samples
    """
    npl = len(fixed_durations)

    u = np.array(uniform_hypercube)  # U(0,1) priors
    x = np.zeros_like(u)             # physical priors

    # 5 * npl + 2 parameters: [C0, C1, r, b, T14]...[q1, q2]
    for n in range(npl):
        x[5 * n + 0] = _norm_ppf(u[0 + n * 5], 0.0, 0.1)
        x[5 * n + 1] = _norm_ppf(u[1 + n * 5], 0.0, 0.1)
        x[5 * n + 2] = _loguniform_ppf(u[2 + n * 5], 1e-5, 0.99)
        x[5 * n + 3] = _uniform_ppf(u[3 + n * 5], 0.0, 1 + x[5 * n + 2])
        x[5 * n + 4] = _loguniform_ppf(u[4 + n * 5], 0.01*fixed_durations[n], 3 * fixed_durations[n])

    # limb darkening coefficients (see Kipping 2013)
    x[-2] = _uniform_ppf(u[-2], 0, 1)
    x[-1] = _uniform_ppf(u[-1], 0, 1)

    return x


def _uniform_ppf(u, a, b):
    """
    Transform from U(0,1) --> Uniform(a,b)

    Parameters
    ----------
        u : array-like
            samples from standard uniform distribution U(0,1)
        a : float
            lower bound on transformed distribution U(a,b)
        b : float
            upper bound on transformed distribution U(a,b)

    Returns
    -------
        x : array-like
            transformed samples from uniform distribution
    """
    return u * (b - a) + a


def _loguniform_ppf(u, a, b):
    """
    Transform from U(0,1) --> LogUniform(a,b)

    Parameters
    ----------
        u : array-like
            samples from standard uniform distribution U(0,1)
        a : float
            lower bound on transformed distribution lnU(a,b)
        b : float
            upper bound on transformed distribution lnU(a,b)

    Returns
    -------
        x : array-like
            transformed samples from log-uniform distribution
    """
    return np.exp(u * np.log(b) + (1 - u) * np.log(a))


def _norm_ppf(u, mu, sig, eps=1e-12):
    """
    Transform from U(0,1) --> Normal(mu,sig)

    Parameters
    ----------
        u : array-like
            samples from standard uniform distribution U(0,1)
        mu : float
            mean of transformed distribution N(mu,sig)
        sig : float
            standard deviation of transformed distribution N(mu,sig)
        eps : float (optional)
            provides numerical stability in the inverse error function at edges

    Returns
    -------
        x : array-like
            transformed samples from Normal distribution
    """
    return mu + sig * np.sqrt(2) * erfinv((2 * u - 1) / (1 + eps))


def throttled_print_fn(interval=10):
    last = [0]
    start = time.time()

    def callback(results, niter, ncall, *args, **kwargs):
        now = time.time()
        if now - last[0] >= interval:
            runtime = (now - start)/60

            buf = io.StringIO()
            stdout = sys.stdout
            sys.stdout = buf
            print_fn(results, niter, ncall, *args, **kwargs)
            sys.stdout = stdout

            line = buf.getvalue().rstrip()
            print(f"{line} | runtime: {runtime:.1f} min")
            last[0] = now

    return callback