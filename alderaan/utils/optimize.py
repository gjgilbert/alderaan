"""Staged MAP optimization of a log-probability function"""

import logging

import numpy as np
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


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
    logp_x0 = logp(x)

    for i, idx in enumerate(stages):

        def objective(xs, idx=idx):
            trial = x.copy()
            trial[idx] = xs
            return -logp(trial)

        start = x[idx].copy()
        logp_start = -objective(start)
        info = minimize(objective, start, method="BFGS")
        improved = np.isfinite(info.fun) and -info.fun > logp_start
        if improved:
            x[idx] = info.x
        else:
            info.x = start
        logger.debug(
            "stage %d/%d (%d parameters): logp %.6g -> %.6g, %s (%s)",
            i + 1,
            len(stages),
            len(idx),
            logp_start,
            -info.fun,
            "kept" if improved else "rejected",
            info.message,
        )
        # BFGS status 1 = iteration limit, 3 = NaN; 2 (precision loss) is routine
        if info.status in (1, 3):
            logger.warning(
                "stage %d/%d did not converge (%s)",
                i + 1,
                len(stages),
                info.message,
            )

    if not logp(x) > logp_x0:
        logger.warning("logp did not improve (%.6g)", logp_x0)

    return x, info
