"""Tests for alderaan/utils/astro.py: limb darkening bases and transit approximations"""

import numpy as np
import numpy.testing as npt
import pytest

from alderaan.utils.astro import (
    approx_ror,
    approx_tc_error,
    approx_transit_depth,
    ld_q_to_u,
    ld_u_to_q,
)


class TestLimbDarkeningBasis:
    """Kipping 2013 (q1, q2) <--> (u1, u2)"""

    def test_round_trip(self):
        rng = np.random.default_rng(42)
        q1, q2 = rng.uniform(0.01, 0.99, (2, 1000))
        u1, u2 = ld_q_to_u(q1, q2)
        npt.assert_allclose(ld_u_to_q(u1, u2), (q1, q2), rtol=1e-12)

    def test_unit_square_maps_to_physical_region(self):
        # physical: positive, monotonically decreasing intensity profile
        rng = np.random.default_rng(0)
        q1, q2 = rng.uniform(0, 1, (2, 10000))
        u1, u2 = ld_q_to_u(q1, q2)
        assert np.all(u1 + u2 <= 1)
        assert np.all(u1 >= 0)
        assert np.all(u1 + 2 * u2 >= 0)

    def test_physical_region_maps_to_unit_square(self):
        u1, u2 = np.meshgrid(np.linspace(0.01, 2, 80), np.linspace(-1, 1, 80))
        physical = (u1 + u2 < 1) & (u1 > 0) & (u1 + 2 * u2 > 0)
        q1, q2 = ld_u_to_q(u1[physical], u2[physical])
        assert np.all((q1 > 0) & (q1 < 1))
        assert np.all((q2 > 0) & (q2 < 1))

    def test_solar_like_values(self):
        # Kepler-band Sun-like star
        q1, q2 = ld_u_to_q(0.4, 0.25)
        npt.assert_allclose(q1, 0.65**2)
        npt.assert_allclose(q2, 0.4 / 1.3)


class TestApproxTransitDepth:
    def test_regimes(self):
        p = 0.1
        b = np.array([0.0, 0.5, 0.95, 1.05, 1.2])
        d = approx_transit_depth(p, b)
        npt.assert_allclose(d[:2], p**2)
        assert 0 < d[3] < d[2] < p**2
        assert d[4] == 0.0

    def test_continuous_at_contact_points(self):
        p = 0.1
        eps = 1e-7
        b = np.array([1 - p - eps, 1 - p + eps, 1 + p - eps, 1 + p + eps])
        d = approx_transit_depth(p, b)
        npt.assert_allclose(d[0], d[1], atol=1e-5)
        npt.assert_allclose(d[2], d[3], atol=1e-5)


class TestApproxRor:
    def test_inverts_approx_transit_depth(self):
        # non-grazing and grazing
        p = np.array([0.01, 0.1, 0.1, 0.1, 0.3])
        b = np.array([0.0, 0.0, 0.85, 1.05, 0.9])
        npt.assert_allclose(approx_ror(approx_transit_depth(p, b), b), p, rtol=1e-8)

    def test_unphysical_depth_raises(self):
        with pytest.raises(ValueError, match="must be in"):
            approx_ror(0.0, 0.0)


class TestApproxTcError:
    def test_scaling(self):
        base = approx_tc_error(0.05, 0.3, 0.15, 0.02, 1e-3)
        # linear in photometric noise
        npt.assert_allclose(approx_tc_error(0.05, 0.3, 0.15, 0.02, 2e-3), 2 * base)
        # deeper transits are timed more precisely
        assert approx_tc_error(0.1, 0.3, 0.15, 0.02, 1e-3) < base
        # grazing geometry degrades timing precision
        assert approx_tc_error(0.05, 0.99, 0.15, 0.02, 1e-3) > base

    def test_finite_integration_time(self):
        # Price & Rogers 2014; ingress tau = T14 * ror / (1 - b^2) = 0.01
        ror, b, T14, sigma_f = 0.05, 0.0, 0.2, 1e-3
        tau = T14 * ror
        # instantaneous limit (exptime << tau) recovers Carter+2008, with the
        # sampling rate 1/exptime factored out
        exptime = 1e-6
        carter = np.sqrt(tau * T14 / 2) / (np.sqrt(T14 / exptime) * ror**2 / sigma_f)
        npt.assert_allclose(
            approx_tc_error(ror, b, T14, exptime, sigma_f), carter, rtol=1e-4
        )
        # continuous at exptime = tau
        lo = approx_tc_error(ror, b, T14, tau * (1 - 1e-9), sigma_f)
        hi = approx_tc_error(ror, b, T14, tau * (1 + 1e-9), sigma_f)
        npt.assert_allclose(lo, hi, rtol=1e-6)
