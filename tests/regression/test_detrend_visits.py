"""
Tests for alderaan/algorithms/detrend_visits.py

Fast tier: segment identification, GP trend model (mean function ramp origin,
finite log posterior), Lomb-Scargle period estimate, method check.
Slow tier: GP detrending of a synthetic two-visit litecurve with a slow stellar
trend, a data gap, and injected transits.
"""

import copy

import numpy as np
import pytest

from alderaan.algorithms import DetrendVisits
from alderaan.algorithms.detrend_visits import _GPTrendModel
from alderaan.constants import EXPTIME
from alderaan.data_models import LiteCurve, Planet, Star, TransitTimes
from alderaan.transit_model import exposure_offsets, transit_flux
from alderaan.utils.astro import approx_transit_depth

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
        approx_transit_depth(ROR, IMPACT),
        T14,
        IMPACT,
    )
    planet.set_transit_times(
        TransitTimes("Kepler", "SYN", index, EPOCH + PERIOD * index)
    )
    planet.set_static_period_and_epoch()

    return star, [planet], litecurve, clean


class TestIdentifyBreaks:
    @staticmethod
    def _series(n=1000, seed=0):
        rng = np.random.default_rng(seed)
        return np.arange(n), 1 + 1e-4 * rng.standard_normal(n), np.zeros(n, bool)

    def test_time_gap_splits_segments(self):
        cadno, flux, mask = self._series()
        cadno[500:] += 50
        seg_id = DetrendVisits._identify_breaks(cadno, flux, 13, 5.0, mask)
        np.testing.assert_array_equal(seg_id, np.arange(len(cadno)) >= 500)

    def test_flux_jump_splits_segments_outside_transit_mask_only(self):
        cadno, flux, mask = self._series()
        flux[300:] += 0.01

        seg_id = DetrendVisits._identify_breaks(cadno, flux, 13, 5.0, mask)
        np.testing.assert_array_equal(seg_id, np.arange(len(cadno)) >= 300)

        mask[295:305] = True
        seg_id = DetrendVisits._identify_breaks(cadno, flux, 13, 5.0, mask)
        assert not seg_id.any()

    def test_fully_masked_segment_merges_into_previous(self, caplog):
        cadno, flux, mask = self._series()
        cadno[500:] += 50
        cadno[530:] += 50
        mask[500:530] = True
        seg_id = DetrendVisits._identify_breaks(cadno, flux, 13, 5.0, mask)
        np.testing.assert_array_equal(seg_id, np.arange(len(cadno)) >= 530)
        assert "fully masked" in caplog.text


class TestGPTrendModel:
    def test_ramp_origin_independent_of_evaluated_cadences(self):
        # segment 1 starts at t=10; drop its first cadences, as a transit mask would
        time = np.arange(0.0, 20.0, 0.1)
        flux = 1 + 1e-3 * np.random.default_rng(1).normal(size=time.size)
        seg_id = (time >= 10).astype(int)
        oot = (time < 10) | (time > 12)
        model = _GPTrendModel(
            time, flux, oot, seg_id, np.array([0.0, 10.0]), "SHOTerm", True, 2.0, 1.0
        )
        p = {"flux0": np.ones(2), "ramp_amp": np.full(2, 0.1), "log_tau": np.zeros(2)}
        np.testing.assert_allclose(model._mean(p, oot), model._mean(p)[oot])

    @pytest.mark.parametrize(
        "gp_term, correct_ramp",
        [("RotationTerm", True), ("RotationTerm", False), ("SHOTerm", False)],
    )
    def test_finite_logp_and_full_length_prediction(self, gp_term, correct_ramp):
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
            correct_ramp,
            2.0,
            1.0,
        )

        assert np.isfinite(model.logp(model.x0))
        pred = model.predict(model.x0)
        assert pred.shape == time.shape and np.all(np.isfinite(pred))


class TestOscillationPeriod:
    @staticmethod
    def _sinusoid(period):
        time = np.arange(0, 60, EXPTIME[LC])
        return time, 1 + 1e-3 * np.sin(2 * np.pi * time / period)

    def test_recovers_injected_period(self):
        time, flux = self._sinusoid(9.0)
        period = DetrendVisits._estimate_oscillation_period(time, flux, 1.0)
        assert period == pytest.approx(9.0, rel=0.02)

    def test_never_below_minimum_period(self):
        time, flux = self._sinusoid(0.5)
        period = DetrendVisits._estimate_oscillation_period(time, flux, 2.0)
        assert period >= 1.001 * 2.0


class TestPerform:
    def test_rejects_unknown_method(self):
        star, planets, litecurve, _ = _synthetic_system()
        with pytest.raises(ValueError, match="wide_mask"):
            DetrendVisits(star, planets, litecurve).perform(method="none")

    @pytest.mark.slow
    @pytest.mark.parametrize("method", ["wide_mask", "tight_mask"])
    def test_removes_trend_and_preserves_transits(self, method):
        star, planets, litecurve, clean = _synthetic_system()
        _, detrended = DetrendVisits(star, planets, litecurve).perform(method=method)

        # detrended flux matches the trend-free flux, in and out of transit,
        # to a small fraction of the photometric noise
        residual = detrended.flux - clean.flux
        assert np.std(residual) < 0.3 * NOISE
        np.testing.assert_allclose(detrended.error, clean.error, rtol=1e-3)
