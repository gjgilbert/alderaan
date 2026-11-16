"""Tests for alderaan/utils/astro.py: limb darkening bases and transit approximations"""

import numpy as np
import numpy.testing as npt

from alderaan.utils.astro import (
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


class TestApproxTcError:
    def test_scaling(self):
        base = approx_tc_error(0.05, 0.3, 0.15, 0.02, 1e-3)
        # linear in photometric noise
        npt.assert_allclose(approx_tc_error(0.05, 0.3, 0.15, 0.02, 2e-3), 2 * base)
        # deeper transits are timed more precisely
        assert approx_tc_error(0.1, 0.3, 0.15, 0.02, 1e-3) < base
        # grazing geometry degrades timing precision
        assert approx_tc_error(0.05, 0.99, 0.15, 0.02, 1e-3) > base
