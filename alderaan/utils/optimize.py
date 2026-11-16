"""Staged MAP optimization of a log-probability function"""

import numpy as np
from scipy.optimize import minimize


def optimize(logp, x0, stages):
    """
    Maximize logp with scipy BFGS, in stages: each stage optimizes the entries
    of x listed by its index array, holding all others at the current point

    Gradients are finite differences. A stage's result is kept only if the log
    probability is finite and improved

    Parameters
    ----------
    logp : callable
        log probability of a parameter vector (unconstrained parameters,
        including any transform Jacobians)
    x0 : ndarray
        starting point
    stages : list of ndarray
        integer indices into x to optimize, one array per stage, in order

    Returns
    -------
    x : ndarray
        optimized parameter vector
    info : scipy.optimize.OptimizeResult
        result of the last stage
    """
    x = np.array(x0, dtype=float)

    for idx in stages:

        def objective(xs, idx=idx):
            trial = x.copy()
            trial[idx] = xs
            return -logp(trial)

        start = x[idx].copy()
        info = minimize(objective, start, method="BFGS")
        if np.isfinite(info.fun) and info.fun < objective(start):
            x[idx] = info.x
        else:
            info.x = start

    return x, info
