"""Tests for alderaan/utils/stats.py: prior transforms, fits, and summaries"""

import numpy as np
import numpy.testing as npt
from scipy import stats as sps

from alderaan.utils.stats import (
    bin_data,
    legendre_p1,
    loguniform_ppf,
    norm_ppf,
    robust_polyfit,
    sigma_cutoff,
    split_rhat,
    uniform_ppf,
    weighted_percentile,
)

U = np.linspace(0.001, 0.999, 101)


class TestPriorTransforms:
    """U(0,1) --> target distribution, as used by nested sampling"""

    def test_uniform_ppf(self):
        npt.assert_allclose(uniform_ppf(U, -2.0, 3.0), sps.uniform(-2, 5).ppf(U))

    def test_loguniform_ppf(self):
        npt.assert_allclose(
            loguniform_ppf(U, 1e-3, 10.0), sps.loguniform(1e-3, 10).ppf(U)
        )

    def test_norm_ppf(self):
        npt.assert_allclose(norm_ppf(U, 1.5, 0.3), sps.norm(1.5, 0.3).ppf(U), rtol=1e-9)

    def test_norm_ppf_finite_at_edges(self):
        assert np.all(np.isfinite(norm_ppf(np.array([0.0, 1.0]), 0.0, 1.0)))


class TestSigmaCutoff:
    def test_floor_for_small_samples(self):
        assert sigma_cutoff(10) == 3.0

    def test_expect_one_outlier_for_large_samples(self):
        npts = 100000
        c = sigma_cutoff(npts)
        npt.assert_allclose(npts * 2 * sps.norm.sf(c), 1.0, rtol=1e-6)


class TestRobustPolyfit:
    def test_matches_polyfit_when_well_conditioned(self):
        rng = np.random.default_rng(1)
        x = np.linspace(-1, 1, 50)
        y = 1 - 2 * x + 0.5 * x**2 + rng.normal(0, 0.01, x.size)
        npt.assert_allclose(robust_polyfit(x, y, 2), np.polyfit(x, y, 2), atol=1e-10)

    def test_two_dimensional_y(self):
        x = np.linspace(0, 10, 30)
        y = np.column_stack([2 * x + 1, -x + 4])
        coeffs = robust_polyfit(x, y, 1)
        assert coeffs.shape == (2, 2)
        npt.assert_allclose(coeffs, [[2, -1], [1, 4]], atol=1e-10)


class TestWeightedPercentile:
    def test_unweighted_is_np_percentile(self):
        a = np.random.default_rng(2).normal(size=101)
        npt.assert_allclose(
            weighted_percentile(a, [16, 50, 84]), np.percentile(a, [16, 50, 84])
        )

    def test_heavy_weight_dominates(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        w = np.array([1.0, 1.0, 1.0, 1.0, 1000.0])
        assert weighted_percentile(a, 50, w) > 4.0 > np.median(a)


class TestBinData:
    def test_bins_linear_trend(self):
        time = np.arange(0, 10, 0.01)
        centers, binned, error = bin_data(time, 2 * time + 1, binsize=1.0)
        npt.assert_allclose(np.diff(centers), 1.0)
        # end bins are only partly filled
        npt.assert_allclose(binned[1:-1], 2 * centers[1:-1] + 1, atol=0.02)
        # 100 points uniform over a span of 2: std 2/sqrt(12), error std/sqrt(100)
        npt.assert_allclose(error[1:-1], 2 / np.sqrt(12) / 10, rtol=0.02)


class TestLegendreP1:
    def test_maps_domain_to_unit_window(self):
        domain = (131.5, 625.0)
        x = np.array([domain[0], np.mean(domain), domain[1]])
        npt.assert_allclose(legendre_p1(x, domain), [-1.0, 0.0, 1.0], atol=1e-12)


class TestSplitRhat:
    def test_stationary_near_one_drift_large(self):
        rng = np.random.default_rng(0)
        iid = rng.normal(size=(2000, 1))
        drift = iid + np.linspace(0, 3, 2000)[:, None]
        rhat = split_rhat(np.hstack([iid, drift]))
        assert rhat[0] < 1.01
        assert rhat[1] > 1.1
