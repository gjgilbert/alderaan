"""
Tests for alderaan/algorithms/detrend_visits.py

Fast tier: segment identification, GP trend model (mean function ramp origin,
finite log posterior, no RotationTerm timescale collapse), single-visit ACF
periodicity test, star-level kernel choice and nominal period, method check,
GP fallback chains, and GP detrending of a synthetic two-visit litecurve with a
slow stellar trend, a data gap, and injected transits. No slow tier.
"""

import copy
from types import SimpleNamespace

import celerite2
import numpy as np
import pytest
from celerite2 import terms as GPterms
from celerite2.driver import LinAlgError

from alderaan.algorithms import DetrendVisits
from alderaan.algorithms.detrend_visits import _GPTrendModel
from alderaan.constants import EXPTIME
from alderaan.data_models import LiteCurve, Planet, Star, TransitTimes
from alderaan.transit_model import exposure_offsets, transit_flux

LC = "long cadence"
U1, U2 = 0.4, 0.25
PERIOD, EPOCH, ROR, IMPACT, T14 = 5.3, 102.1, 0.08, 0.3, 0.15
NOISE = 1e-4


def _synthetic_system(seed=0):
    """
    Star, one planet, and a 60-day long cadence litecurve over two visits with a
    1.5-day gap; returns (star, planets, litecurve, clean) where litecurve
    carries a slow multiplicative trend and clean is the same data without it
    """
    rng = np.random.default_rng(seed)
    cadno = np.arange(int(60 / EXPTIME[LC]))
    time = 100.0 + cadno * EXPTIME[LC]
    index = np.arange(int((time.max() - EPOCH) / PERIOD) + 1)

    flux = np.ones_like(time)
    offsets = exposure_offsets(EXPTIME[LC], 15)
    for tc in EPOCH + PERIOD * index:
        near = np.abs(time - tc) < PERIOD / 2
        flux[near] = transit_flux(
            time[near], tc, PERIOD, ROR, IMPACT, T14, U1, U2, exptime_offsets=offsets
        )
    flux += NOISE * rng.standard_normal(len(time))

    visit = np.where(time < 130, 1, 2)
    clean = LiteCurve(
        "Kepler",
        "SYN",
        time,
        flux,
        np.full_like(time, NOISE),
        cadno,
        visit,
        np.full(len(time), LC),
    )
    clean.select(~((time > 114) & (time < 115.5)))

    trend = (
        1 + 2e-3 * np.sin(2 * np.pi * clean.time / 9.0) + 1e-3 * (clean.time - 100) / 60
    )
    litecurve = copy.deepcopy(clean)
    litecurve.flux = clean.flux * trend
    litecurve.error = clean.error * trend

    star = Star("Kepler", "SYN", 1.01, 1, U1, U2)
    planet = Planet(
        "Kepler",
        "SYN",
        PERIOD,
        EPOCH,
        ROR,
        IMPACT,
        T14,
    )
    planet.set_transit_times(
        TransitTimes("Kepler", "SYN", index, EPOCH + PERIOD * index)
    )
    planet.set_static_period_and_epoch()
    planet.transit_times.ttv_scatter = 0.0  # no TTVs (RegularizeOMC sets this)

    return star, [planet], litecurve, clean


class TestIdentifyBreaks:
    @staticmethod
    def _series(n=1000, seed=0):
        rng = np.random.default_rng(seed)
        return np.arange(n), 1 + 1e-4 * rng.standard_normal(n), np.zeros(n, bool)

    def test_time_gap_splits_segments(self):
        cadno, flux, mask = self._series()
        cadno[500:] += 50
        seg_id = DetrendVisits._identify_breaks(cadno, flux, mask, 13, 5.0)
        np.testing.assert_array_equal(seg_id, np.arange(len(cadno)) >= 500)

    def test_flux_jump_splits_segments_outside_transit_mask_only(self):
        cadno, flux, mask = self._series()
        flux[300:] += 0.01

        seg_id = DetrendVisits._identify_breaks(cadno, flux, mask, 13, 5.0)
        np.testing.assert_array_equal(seg_id, np.arange(len(cadno)) >= 300)

        mask[295:305] = True
        seg_id = DetrendVisits._identify_breaks(cadno, flux, mask, 13, 5.0)
        assert not seg_id.any()

    def test_fully_masked_segment_merges_into_previous(self):
        cadno, flux, mask = self._series()
        cadno[500:] += 50
        cadno[530:] += 50
        mask[500:530] = True
        seg_id = DetrendVisits._identify_breaks(cadno, flux, mask, 13, 5.0)
        np.testing.assert_array_equal(seg_id, np.arange(len(cadno)) >= 530)


class TestGPTrendModel:
    def test_ramp_origin_independent_of_evaluated_cadences(self):
        # segment 1 starts at t=10; drop its first cadences, as a transit mask would
        time = np.arange(0.0, 20.0, 0.1)
        flux = 1 + 1e-3 * np.random.default_rng(1).normal(size=time.size)
        seg_id = (time >= 10).astype(int)
        oot = (time < 10) | (time > 12)
        model = _GPTrendModel(
            time, flux, oot, seg_id, np.array([0.0, 10.0]), "SHOTerm", 2.0, 1.0
        )
        p = {"flux0": np.ones(2), "ramp_amp": np.full(2, 0.1), "log_tau": np.zeros(2)}
        np.testing.assert_allclose(model._mean(p, oot), model._mean(p)[oot])

    @pytest.mark.parametrize("gp_term", ["RotationTerm", "SHOTerm"])
    def test_finite_logp_and_full_length_prediction(self, gp_term):
        rng = np.random.default_rng(2)
        time = np.arange(0.0, 20.0, EXPTIME[LC])
        flux = 1 + 1e-3 * np.sin(time / 3) + NOISE * rng.normal(size=time.size)
        seg_id = (time >= 10).astype(int)
        oot = np.abs(time - 5) > 0.2
        model = _GPTrendModel(
            time,
            flux,
            oot,
            seg_id,
            np.array([0.0, 10.0]),
            gp_term,
            2.0,
            1.0,
        )

        assert np.isfinite(model.logp(model.x0))
        pred = model.predict(model.x0)
        assert pred.shape == time.shape and np.all(np.isfinite(pred))

    @pytest.mark.parametrize(
        "name, value",
        [
            ("log_Q0", 600.0),  # Q0 overflows
            ("log_dQ", 600.0),  # Q1**2 overflows: celerite2 raises OverflowError
            ("log_sigma", 400.0),  # sigma**2 overflows
        ],
    )
    def test_logp_is_minus_inf_beyond_truncated_prior(self, name, value):
        time = np.arange(0.0, 20.0, EXPTIME[LC])
        flux = 1 + NOISE * np.random.default_rng(3).normal(size=time.size)
        model = _GPTrendModel(
            time,
            flux,
            np.ones(time.size, dtype=bool),
            np.zeros(time.size, dtype=int),
            np.array([0.0]),
            "RotationTerm",
            2.0,
            1.0,
        )
        x = model.x0.copy()
        x[model.ix[name]] = value
        assert model.logp(x) == -np.inf

    def test_logp_is_minus_inf_where_covariance_is_singular(self):
        # within the truncated prior, but sigma, P and Q so large that celerite2
        # cannot factorize (as K00001 visit 9's optimizer probe)
        time = np.arange(0.0, 20.0, EXPTIME[LC])
        flux = 1 + NOISE * np.random.default_rng(3).normal(size=time.size)
        model = _GPTrendModel(
            time,
            flux,
            np.ones(time.size, dtype=bool),
            np.zeros(time.size, dtype=int),
            np.array([0.0]),
            "SHOTerm",
            2.0,
            1.0,
        )
        x = model.x0.copy()
        for name in ("log_sigma", "log_P", "log_Q0"):
            mu, sd = model.priors[name]
            x[model.ix[name]] = mu + 3.9 * sd
        with pytest.raises(LinAlgError):
            model._gp(model.unpack(x))
        assert model.logp(x) == -np.inf

    def test_rotation_term_decay_time_does_not_collapse(self):
        # at the lowest allowed Q0, the P/2 term's decay time 2Q/w0 stays at
        # its Q = 1 value, sqrt(3) P / (4 pi), instead of going to zero
        time = np.arange(0.0, 20.0, EXPTIME[LC])
        flux = 1 + NOISE * np.random.default_rng(4).normal(size=time.size)
        model = _GPTrendModel(
            time,
            flux,
            np.ones(time.size, dtype=bool),
            np.zeros(time.size, dtype=int),
            np.array([0.0]),
            "RotationTerm",
            2.0,
            1.0,
        )
        x = model.x0.copy()
        mu, sd = model.priors["log_Q0"]
        x[model.ix["log_Q0"]] = mu - 4 * sd
        p = model.unpack(x)
        period = model.hyperparameters(p)[0]
        damping_rate = model._gp(p).kernel.get_coefficients()[4]
        assert 1 / damping_rate[1] >= np.sqrt(3) * period / (4 * np.pi) * (1 - 1e-9)


def _acf_visit(kernel, exptime=EXPTIME[LC], seed=0):
    """
    One 90-day visit: a celerite2 GP draw plus unit white noise, with a 2-day
    gap and one masked transit; returns (cadno, flux, mask)
    """
    rng = np.random.default_rng(seed)
    cadno = np.arange(int(90 / exptime))
    time = cadno * exptime
    keep = (time < 40) | (time > 42)
    cadno, time = cadno[keep], time[keep]
    flux = rng.standard_normal(time.size)
    if kernel is not None:
        gp = celerite2.GaussianProcess(kernel)
        gp.compute(time, diag=np.full(time.size, 1e-12))
        flux += gp.dot_tril(rng.standard_normal(time.size))
    return cadno, flux, np.abs(time - 60) < 0.25


ROTATION = GPterms.RotationTerm(sigma=10.0, period=4.0, Q0=5.0, dQ=1.0, f=0.5)


class TestACFPeriod:
    @pytest.mark.parametrize("obsmode", [LC, "short cadence"])
    def test_detects_rotation_period(self, obsmode):
        exptime = EXPTIME[obsmode]
        cadno, flux, mask = _acf_visit(ROTATION, exptime)
        periodic, period_acf, _ = DetrendVisits._acf_period(
            cadno, flux, mask, exptime, 0.5
        )
        assert periodic
        assert period_acf == pytest.approx(4.0, rel=0.1)

    def test_aperiodic_red_noise_is_not_periodic_and_gives_timescale(self):
        red = GPterms.SHOTerm(sigma=10.0, rho=5.0, Q=1 / np.sqrt(2))
        cadno, flux, mask = _acf_visit(red)
        periodic, _, period_half = DetrendVisits._acf_period(
            cadno, flux, mask, EXPTIME[LC], 0.5
        )
        assert not periodic
        assert 5.0 / 1.5 < period_half < 5.0 * 1.5

    def test_white_noise_is_not_periodic(self):
        cadno, flux, mask = _acf_visit(None)
        periodic, _, _ = DetrendVisits._acf_period(cadno, flux, mask, EXPTIME[LC], 0.5)
        assert not periodic


class TestNominalPeriod:
    @pytest.mark.parametrize(
        "visit_results, periodic, period",
        [
            # (periodic, period_acf, period_half) per visit; the P/2 visit is clipped
            (
                [
                    (True, 4.0, 6.0),
                    (True, 4.1, 6.0),
                    (True, 3.9, 6.0),
                    (True, 2.0, 6.0),
                ],
                True,
                4.0,
            ),
            # 1 of 4 visits periodic: an SHOTerm star, period from period_half
            (
                [
                    (True, 4.0, 6.0),
                    (False, np.nan, 6.0),
                    (False, np.nan, 6.0),
                    (False, np.nan, 6.0),
                ],
                False,
                6.0,
            ),
        ],
    )
    def test_star_kernel_and_clipped_geometric_mean(
        self, monkeypatch, visit_results, periodic, period
    ):
        results = iter(visit_results)
        monkeypatch.setattr(
            DetrendVisits,
            "_acf_period",
            staticmethod(lambda *args: next(results)),
        )
        star, planets, litecurve, _ = _synthetic_system()
        alg = DetrendVisits(star, planets, litecurve)
        lc = SimpleNamespace(
            visit=np.arange(4),
            obsmode=np.full(4, LC),
            cadno=np.arange(4),
            flux=np.ones(4),
        )
        masks = {v: np.zeros(1, dtype=bool) for v in range(4)}

        got_periodic, got_period = alg._nominal_period(lc, masks)
        assert got_periodic == periodic
        assert got_period == pytest.approx(period, rel=0.02)


class TestTransitMask:
    @staticmethod
    def _width(period, ror, duration, tight_mask, ttv_scatter=0.01):
        """Full width (days) of the mask around one transit of a lone planet"""
        planet = Planet("Kepler", "SYN", period, 0.0, ror, 0.0, duration)
        index = np.arange(3)
        planet.set_transit_times(TransitTimes("Kepler", "SYN", index, period * index))
        planet.transit_times.ttv_scatter = ttv_scatter
        star = Star("Kepler", "SYN", 1.01, 1, U1, U2)
        _, _, litecurve, _ = _synthetic_system()
        alg = DetrendVisits(star, [planet], litecurve)
        dt = 1e-4
        time = period + np.arange(-0.5, 0.5, dt)
        return np.sum(alg._transit_mask(time, LC, tight_mask)) * dt

    @pytest.mark.parametrize("tight_mask, k", [(False, 2.0), (True, 1.5)])
    @pytest.mark.parametrize("duration", [0.02, 0.1])
    def test_usp_full_width_is_k_t14_plus_exptime(self, tight_mask, k, duration):
        assert self._width(1.0, 0.02, duration, tight_mask) == pytest.approx(
            k * duration + EXPTIME[LC], abs=1e-3
        )

    def test_non_usp_keeps_three_t14_wide_mask(self):
        assert self._width(5.0, 0.02, 0.1, False) == pytest.approx(0.3, abs=1e-3)
        assert self._width(1.0, 0.06, 0.1, False) == pytest.approx(0.3, abs=1e-3)


class TestPerform:
    def test_rejects_unknown_method(self):
        star, planets, litecurve, _ = _synthetic_system()
        with pytest.raises(ValueError, match="wide_mask"):
            DetrendVisits(star, planets, litecurve).perform(method="none")

    @pytest.mark.parametrize("method", ["wide_mask", "tight_mask"])
    def test_removes_trend_and_preserves_transits(self, method):
        star, planets, litecurve, clean = _synthetic_system()
        _, detrended = DetrendVisits(star, planets, litecurve).perform(method=method)

        # detrended flux matches the trend-free flux, in and out of transit,
        # to a small fraction of the photometric noise
        residual = detrended.flux - clean.flux
        assert np.std(residual) < 0.3 * NOISE
        np.testing.assert_allclose(detrended.error, clean.error, rtol=1e-3)
        assert detrended.meta["DTKERNEL"] in ("RotationTerm", "SHOTerm")
