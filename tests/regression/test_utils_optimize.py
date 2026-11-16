"""Tests for alderaan/utils/optimize.py"""

import numpy as np

from alderaan.utils.optimize import optimize

MEAN = np.array([1.0, -2.0])
PRECISION = np.array([[2.0, 1.5], [1.5, 2.0]])  # strongly correlated


def _logp(x):
    d = x - MEAN
    return -0.5 * d @ PRECISION @ d


def test_final_joint_stage_reaches_optimum():
    stages = [np.array([0]), np.array([1]), np.array([0, 1])]
    x, _ = optimize(_logp, np.zeros(2), stages)
    np.testing.assert_allclose(x, MEAN, atol=1e-4)


def test_stage_kept_only_if_improved():
    # logp does not depend on x[1], so its stage cannot improve and is rejected
    def logp(x):
        return -((x[0] - 1.0) ** 2)

    x0 = np.array([0.0, 0.3])
    x, info = optimize(logp, x0, [np.array([0]), np.array([1])])
    assert np.isclose(x[0], 1.0, atol=1e-5)
    assert x[1] == x0[1]
    assert info.x[0] == x0[1]
