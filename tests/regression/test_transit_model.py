"""
Tests for alderaan/transit_model.py: transit_flux, exposure_offsets, TransitModel

Fast tests inject transits into a synthetic two-planet long cadence litecurve
with TTVs; slow tests build the model on K00137.
"""

import numpy as np
import numpy.testing as npt
import pytest

from alderaan.constants import KEPLER_LCIT, KEPLER_SCIT
from alderaan.data_models import LiteCurve, Planet, Star, TransitTimes
from alderaan.transit_model import TransitModel, exposure_offsets, transit_flux
from alderaan.utils.astro import ld_q_to_u, ld_u_to_q
from alderaan.utils.stats import legendre_p1

U1, U2 = 0.4, 0.25

# (period, epoch, ror, b, T14, ttv amplitude)
PLANETS = [
    (5.3, 102.1, 0.08, 0.3, 0.15, 0.008),
    (11.7, 105.4, 0.05, 0.6, 0.20, 0.015),
]

# linear perturbation (C0, C1) applied to each planet's transit times
PERTURBATION = [(0.002, -0.001), (-0.003, 0.002)]


def _transit_signal(time, ttimes, exptime):
    """Flux of all PLANETS transiting at ttimes, finely integrated over exptime"""
    flux = np.ones_like(time)
    for (period, _, ror, b, T14, _), ttime in zip(PLANETS, ttimes):
        # nearest transit to each cadence
        k = np.argmin(np.abs(time[:, None] - ttime[None, :]), axis=1)
        flux += (
            transit_flux(
                time - ttime[k],
                0.0,
                period,
                ror,
                b,
                T14,
                U1,
                U2,
                exptime_offsets=exposure_offsets(exptime, 101),
            )
            - 1.0
        )
    return flux


def _synthetic_system(inject_perturbation=False):
    """Star, planets, and a noiseless long cadence litecurve with TTVs"""
    time = np.arange(100.0, 160.0, KEPLER_LCIT)
    t_start, t_end = time.min(), time.max()

    star = Star("Kepler", "SYN", "K99999", 1, U1, U2)
    planets, ttimes = [], []
    for (period, epoch, ror, b, T14, ttv), (C0, C1) in zip(PLANETS, PERTURBATION):
        lin = TransitTimes.from_linear_ephemeris(
            "Kepler", "SYN", period, epoch, t_start, t_end
        )
        ttime = lin.ttime_mod + ttv * np.sin(2 * np.pi * lin.index / 4.3)
        p = Planet("Kepler", "SYN", period, epoch, ror**2, T14, b)
        p.set_transit_times(TransitTimes("Kepler", "SYN", lin.index, ttime))
        p.set_static_period_and_epoch()
        planets.append(p)

        if inject_perturbation:
            T = p.epoch + p.period * lin.index
            ttime = ttime + C0 + C1 * legendre_p1(T, (t_start, t_end))
        ttimes.append(ttime)

    flux = _transit_signal(time, ttimes, KEPLER_LCIT)

    n = time.size
    litecurve = LiteCurve(
        "Kepler",
        "SYN",
        time,
        flux,
        np.full(n, 1e-4),
        np.arange(n),
        np.full(n, 5),
        np.full(n, "long cadence"),
    )
    return star, planets, litecurve


def _theta_true(perturbed=False):
    theta = []
    for (_, _, ror, b, T14, _), (C0, C1) in zip(PLANETS, PERTURBATION):
        theta += [C0, C1] if perturbed else [0.0, 0.0]
        theta += [ror, b, T14]
    theta += list(ld_u_to_q(U1, U2))
    return np.array(theta)


class TestExposureOffsets:
    def test_midpoints_span_exposure_symmetrically(self):
        exptime, n = KEPLER_LCIT, 9
        dt = exposure_offsets(exptime, n)
        assert dt.size == n
        npt.assert_allclose(dt, -dt[::-1], atol=1e-15)
        npt.assert_allclose(np.diff(dt), exptime / n)
        npt.assert_allclose(dt[0] - exptime / n / 2, -exptime / 2)
        npt.assert_allclose(dt[-1] + exptime / n / 2, exptime / 2)


class TestTransitFlux:
    T14 = 0.2
    PERIOD = 10.0

    def test_depth_is_ror_squared_without_limb_darkening(self):
        for ror in (0.01, 0.05, 0.1):
            f = transit_flux(
                np.array([0.0]), 0.0, self.PERIOD, ror, 0.0, self.T14, 0, 0
            )
            npt.assert_allclose(1 - f, ror**2, rtol=1e-6)

    def test_unity_outside_first_to_fourth_contact(self):
        t = np.array([-self.T14 / 2 - 1e-4, self.T14 / 2 + 1e-4, 3.0, -4.9])
        f = transit_flux(t, 0.0, self.PERIOD, 0.1, 0.5, self.T14, U1, U2)
        npt.assert_array_equal(f, 1.0)
        f_in = transit_flux(
            np.array([self.T14 / 2 - 1e-3]),
            0.0,
            self.PERIOD,
            0.1,
            0.5,
            self.T14,
            U1,
            U2,
        )
        assert f_in[0] < 1.0

    def test_periodic(self):
        t = np.linspace(-0.15, 0.15, 31)
        args = (0.0, self.PERIOD, 0.1, 0.3, self.T14, U1, U2)
        npt.assert_allclose(
            transit_flux(t + 3 * self.PERIOD, *args), transit_flux(t, *args), atol=1e-12
        )

    def test_limb_darkening_deepens_center(self):
        t = np.array([0.0])
        f_ld = transit_flux(t, 0.0, self.PERIOD, 0.1, 0.0, self.T14, U1, U2)
        f_uniform = transit_flux(t, 0.0, self.PERIOD, 0.1, 0.0, self.T14, 0, 0)
        assert f_ld[0] < f_uniform[0]

    def test_supersampling_conserves_deficit_and_smooths(self):
        # long exposure relative to ingress
        exptime = 0.05
        t = np.arange(-0.3, 0.3, 1e-4)
        args = (0.0, self.PERIOD, 0.1, 0.3, self.T14, U1, U2)
        f = transit_flux(t, *args)
        f_ss = transit_flux(t, *args, exptime_offsets=exposure_offsets(exptime, 51))

        npt.assert_allclose(np.sum(1 - f_ss), np.sum(1 - f), rtol=1e-3)
        assert np.max(np.abs(np.diff(f_ss))) < np.max(np.abs(np.diff(f)))
        assert f_ss[t.size // 2] > f[t.size // 2]
        # flux leaks outside T14 by half an exposure, no further
        assert np.all(f_ss[np.abs(t) > self.T14 / 2 + exptime / 2] == 1.0)
        assert np.all(f_ss[np.abs(t) < self.T14 / 2 + exptime / 2 - 1e-3] < 1.0)


class TestTransitModel:
    def test_requires_static_ephemeris(self):
        star, planets, litecurve = _synthetic_system()
        planets[1].static_period = None
        with pytest.raises(ValueError, match="planet 1: static period/epoch"):
            TransitModel(star, planets, litecurve)

    def test_theta_initial_layout(self):
        star, planets, litecurve = _synthetic_system()
        theta = TransitModel(star, planets, litecurve).theta_initial()

        assert theta.shape == (5 * len(planets) + 2,)
        for n, p in enumerate(planets):
            C0, C1, ror, b, T14 = theta[5 * n : 5 * (n + 1)]
            assert (C0, C1) == (0.0, 0.0)
            npt.assert_allclose(ror, np.sqrt(p.depth))
            assert 0 < b < 1
            assert T14 == p.duration
        npt.assert_allclose(theta[-2:], [star.q1, star.q2])

    def test_supersampling_at_least_seven_and_odd(self):
        star, planets, litecurve = _synthetic_system()
        model = TransitModel(star, planets, litecurve)
        ss = model.supersample["long cadence"]
        assert ss >= 7 and ss % 2 == 1
        assert model.exptime_offsets["long cadence"].size == ss

    def test_reproduces_injected_ttv_transits(self):
        star, planets, litecurve = _synthetic_system()
        model = TransitModel(star, planets, litecurve)
        theta = _theta_true()

        flux_mod = model.model_flux(theta)
        assert np.min(flux_mod) < 1 - 0.9 * PLANETS[0][2] ** 2
        npt.assert_allclose(model.residuals(theta), 0.0, atol=0.05)

    def test_reproduces_linear_perturbation(self):
        star, planets, litecurve = _synthetic_system(inject_perturbation=True)
        model = TransitModel(star, planets, litecurve)

        npt.assert_allclose(
            model.residuals(_theta_true(perturbed=True)), 0.0, atol=0.05
        )
        # the unperturbed ephemeris no longer fits
        assert np.max(np.abs(model.residuals(_theta_true()))) > 5

    def test_near_transit_evaluation_is_exact(self):
        # model_flux skips cadences far from each planet's transits; the result
        # must equal evaluating every planet on every cadence, bit for bit
        star, planets, litecurve = _synthetic_system()
        model = TransitModel(star, planets, litecurve)
        theta = _theta_true(perturbed=True)
        u1, u2 = ld_q_to_u(*theta[-2:])

        expected = np.ones(litecurve.time.size)
        for n in range(model.npl):
            C0, C1, ror, b, T14 = theta[5 * n : 5 * (n + 1)]
            t = model.warped_time[n] - (C0 + C1 * model.warped_legx[n])
            expected += (
                transit_flux(
                    t,
                    0.0,
                    model.static_period[n],
                    ror,
                    b,
                    T14,
                    u1,
                    u2,
                    exptime_offsets=model.exptime_offsets["long cadence"],
                )
                - 1.0
            )

        assert np.array_equal(model.model_flux(theta), expected)

    def test_reproduces_transits_in_mixed_long_and_short_cadence(self):
        star, planets, _ = _synthetic_system()
        ttimes = [p.transit_times.ttime_mod for p in planets]
        segments = [
            (np.arange(100.0, 130.0, KEPLER_LCIT), KEPLER_LCIT, "long cadence"),
            (np.arange(130.0, 160.0, KEPLER_SCIT), KEPLER_SCIT, "short cadence"),
        ]
        time = np.hstack([t for t, _, _ in segments])
        flux = np.hstack([_transit_signal(t, ttimes, exp) for t, exp, _ in segments])
        obsmode = np.hstack([np.full(t.size, om) for t, _, om in segments])
        n = time.size
        litecurve = LiteCurve(
            "Kepler",
            "SYN",
            time,
            flux,
            np.full(n, 1e-4),
            np.arange(n),
            np.full(n, 5),
            obsmode,
        )
        model = TransitModel(star, planets, litecurve)

        assert model.supersample["short cadence"] == 1
        assert model.supersample["long cadence"] >= 7
        residuals = model.residuals(_theta_true())
        is_lc = obsmode == "long cadence"
        npt.assert_allclose(residuals[is_lc], 0.0, atol=0.05)
        # short cadence is not integrated over its 1 min exposure (by design)
        npt.assert_allclose(residuals[~is_lc], 0.0, atol=0.1)

    def test_models_short_cadence_inside_a_long_cadence_transit(self):
        # short cadence covers only the second half of one transit, so no transit
        # midpoint's nearest cadence is short cadence
        star, planets, _ = _synthetic_system()
        ttimes = [p.transit_times.ttime_mod for p in planets]
        tc = ttimes[0][3]
        lc_time = np.arange(100.0, 160.0, KEPLER_LCIT)
        lc_time = lc_time[(lc_time < tc + 0.01) | (lc_time > tc + 0.06)]
        sc_time = np.arange(tc + 0.01, tc + 0.06, KEPLER_SCIT)
        time = np.hstack([lc_time, sc_time])
        order = np.argsort(time)
        obsmode = np.hstack(
            [
                np.full(lc_time.size, "long cadence"),
                np.full(sc_time.size, "short cadence"),
            ]
        )[order]
        flux = np.hstack(
            [
                _transit_signal(lc_time, ttimes, KEPLER_LCIT),
                _transit_signal(sc_time, ttimes, KEPLER_SCIT),
            ]
        )[order]
        n = time.size
        litecurve = LiteCurve(
            "Kepler",
            "SYN",
            time[order],
            flux,
            np.full(n, 1e-4),
            np.arange(n),
            np.full(n, 5),
            obsmode,
        )
        model = TransitModel(star, planets, litecurve)

        is_sc = obsmode == "short cadence"
        assert np.all(model.model_flux(_theta_true())[is_sc] < 1)
        npt.assert_allclose(model.residuals(_theta_true())[is_sc], 0.0, atol=0.1)

    def test_ln_likelihood_peaks_at_truth(self):
        star, planets, litecurve = _synthetic_system()
        model = TransitModel(star, planets, litecurve)
        theta = _theta_true()
        lnl_true = model.ln_likelihood(theta)

        # C0, ror, b, T14 of each planet, then q1, q2
        for i in [0, 2, 3, 4, 5, 7, 8, 9, 10, 11]:
            for step in (-1e-3, 1e-3):
                perturbed = theta.copy()
                perturbed[i] += step
                assert model.ln_likelihood(perturbed) < lnl_true

    def test_trim_keeps_selected_cadences(self):
        star, planets, litecurve = _synthetic_system()
        n_cadences = litecurve.time.size
        model = TransitModel(star, planets, litecurve)
        theta = _theta_true()
        flux_full = model.model_flux(theta)

        mask = np.ones(n_cadences, dtype=bool)
        mask[::3] = False
        model.trim(mask)

        npt.assert_allclose(model.model_flux(theta), flux_full[mask], atol=1e-15)
        assert model.residuals(theta).size == mask.sum()
        # caller's litecurve is untouched
        assert litecurve.time.size == n_cadences


@pytest.mark.slow
class TestTransitModelK00137:
    def test_initial_transits_improve_fit_over_flat(self, k00137):
        star, planets, litecurve = k00137
        model = TransitModel(star, planets, litecurve)
        theta = model.theta_initial()

        flat = theta.copy()
        flat[2 : 5 * model.npl : 5] = 0.0

        chisq = np.sum(model.residuals(theta) ** 2)
        chisq_flat = np.sum(model.residuals(flat) ** 2)
        assert np.isfinite(model.ln_likelihood(theta))
        assert chisq < chisq_flat
