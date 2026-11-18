"""
Tests for alderaan/algorithms/quick_fit_transit_times.py

Recovery of injected per-transit time offsets (TTVs) from a synthetic detrended
long cadence litecurve, including a transit that overlaps a second planet's;
low quality and undetected transits stay unmeasured; high-SNR short cadence transits are
measured by cross-correlation; transits with a sibling in the window go to
the optimizer, including a planet whose every transit has a USP sibling in
the window.
"""

import numpy as np
import pytest

from alderaan.algorithms import QuickFitTransitTimes
from alderaan.constants import EXPTIME
from alderaan.data_models import LiteCurve, Planet, Star, TransitTimes
from alderaan.transit_model import exposure_offsets, transit_flux
from alderaan.utils.astro import approx_transit_depth

LC = "long cadence"
SC = "short cadence"
U1, U2 = 0.4, 0.25
NOISE = 1e-4
TTV_AMP = 10 / 1440  # days

# (period, epoch, ror, impact, T14) per planet; planet 1 transit 1 overlaps
# planet 0 transit 4
TRUTH = [(5.3, 102.1, 0.08, 0.3, 0.15), (11.7, 111.65, 0.07, 0.4, 0.2)]


def _synthetic_system(truth, seed=0, noise=NOISE, obsmode=LC):
    """
    Star, planets on linear ephemerides with known shapes, a 60-day detrended
    single-obsmode litecurve whose transits carry sinusoidal TTVs, and the true
    transit times
    """
    rng = np.random.default_rng(seed)
    cadno = np.arange(int(60 / EXPTIME[obsmode]))
    time = 100.0 + cadno * EXPTIME[obsmode]
    offsets = exposure_offsets(EXPTIME[obsmode], 15)
    star = Star("Kepler", "SYN", 1.01, 1, U1, U2)

    flux = np.ones_like(time)
    planets, ttime_true = [], []
    for n, (period, epoch, ror, impact, T14) in enumerate(truth):
        index = np.arange(int((time.max() - epoch) / period) + 1)
        linear = epoch + period * index
        ttime = linear + TTV_AMP * np.sin(2 * np.pi * index / (4.7 + n))
        for tc in ttime:
            near = np.abs(time - tc) < period / 2
            flux[near] += (
                transit_flux(time[near], tc, period, ror, impact, T14, U1, U2, offsets)
                - 1.0
            )

        p = Planet(
            "Kepler",
            "SYN",
            period,
            epoch,
            approx_transit_depth(ror, impact),
            T14,
            impact,
        )
        p.set_transit_shape(approx_transit_depth(ror, impact), T14, impact, ror)
        p.set_transit_times(TransitTimes("Kepler", "SYN", index, linear))
        p.set_static_period_and_epoch()
        planets.append(p)
        ttime_true.append(ttime)

    flux += noise * rng.standard_normal(len(time))
    litecurve = LiteCurve(
        "Kepler",
        "SYN",
        time,
        flux,
        np.full_like(time, noise),
        cadno,
        np.ones_like(cadno),
        np.full(len(time), obsmode),
    )
    return star, planets, litecurve, ttime_true


class TestQuickFitTransitTimes:
    @pytest.mark.parametrize("truth", [TRUTH[:1], TRUTH], ids=["1pl", "2pl"])
    def test_recovers_injected_ttvs(self, truth):
        star, planets, litecurve, ttime_true = _synthetic_system(truth)
        new_planets, _ = QuickFitTransitTimes(star, planets, litecurve).perform()

        for old, new, true in zip(planets, new_planets, ttime_true):
            tt = new.transit_times
            residual = tt.ttime_obs - true
            assert np.all(np.isfinite(tt.ttime_obs))
            assert np.all(np.abs(residual) < 4 * tt.error_obs)
            assert np.std(residual) < 0.2 * TTV_AMP
            # errors are calibrated: normalized residuals have unit scatter
            assert 0.5 < np.std(residual / tt.error_obs) < 1.5

            # measured times only; no model yet (that is RegularizeOMC)
            np.testing.assert_array_equal(tt.index, old.transit_times.index)
            np.testing.assert_array_equal(tt.ttime_mod, tt.ttime_obs)
            assert not np.any(tt.out_flag)

    def test_low_quality_transit_is_not_measured(self):
        star, planets, litecurve, _ = _synthetic_system(TRUTH[:1])
        tt = planets[0].transit_times
        quality = np.ones(len(tt.index), dtype=bool)
        quality[3] = False
        planets[0].set_transit_times(
            TransitTimes("Kepler", "SYN", tt.index, tt.ttime_obs, quality=quality)
        )
        new_planets, _ = QuickFitTransitTimes(star, planets, litecurve).perform()

        new = new_planets[0].transit_times
        np.testing.assert_array_equal(new.quality, quality)
        assert np.isnan(new.ttime_obs[3]) and np.isnan(new.error_obs[3])
        assert new.ttime_mod[3] == tt.ttime_mod[3]
        assert np.all(np.isfinite(new.ttime_obs[quality]))

    def test_undetected_transit_is_not_measured(self):
        # transit 3 removed from the photometry: no trial transit beats a flat line
        star, planets, litecurve, ttime_true = _synthetic_system(TRUTH[:1])
        near = np.abs(litecurve.time - ttime_true[0][3]) < TRUTH[0][4]
        rng = np.random.default_rng(1)
        litecurve.flux[near] = 1 + NOISE * rng.standard_normal(np.sum(near))
        tt = planets[0].transit_times
        new_planets, _ = QuickFitTransitTimes(star, planets, litecurve).perform()

        new = new_planets[0].transit_times
        assert np.isnan(new.ttime_obs[3]) and np.isnan(new.error_obs[3])
        assert new.ttime_mod[3] == tt.ttime_mod[3]
        assert np.sum(np.isnan(new.ttime_obs)) == 1

    def test_high_snr_transits_measured_by_cross_correlation(self):
        # short cadence (no supersampling) at per-transit SNR ~ 500, the highest
        # expected for a Kepler planet; the tc grid must resolve the tc error
        ror, impact, T14 = 0.1, 0.3, 0.1
        depth = approx_transit_depth(ror, impact)
        noise = depth * np.sqrt(T14 / EXPTIME[SC]) / 500
        star, planets, litecurve, ttime_true = _synthetic_system(
            [(5.3, 102.1, ror, impact, T14)], noise=noise, obsmode=SC
        )
        new_planets, _ = QuickFitTransitTimes(star, planets, litecurve).perform()

        tt = new_planets[0].transit_times
        assert np.all(np.abs(tt.ttime_obs - ttime_true[0]) < 4 * tt.error_obs)

    def test_cross_correlation_ignores_nearby_sibling_transit(self):
        # a deep sibling transit 4 hr before planet 0's transit 4: outside the
        # overlap criterion but inside the cross-correlation window
        tc4 = 102.1 + 4 * 5.3
        truth = [(5.3, 102.1, 0.04, 0.3, 2 / 24), (25.0, tc4 - 4 / 24, 0.12, 0.3, 0.15)]
        star, planets, litecurve, ttime_true = _synthetic_system(truth)
        new_planets, _ = QuickFitTransitTimes(star, planets, litecurve).perform()

        tt = new_planets[0].transit_times
        assert abs(tt.ttime_obs[4] - ttime_true[0][4]) < 4 * tt.error_obs[4]

    def test_cross_correlation_stays_out_of_data_gap(self):
        # transit 3 is absent and its window's data ends 1.5 T14 after it, with a
        # dip on the last two cadences: a trial transit must lie in covered data
        star, planets, litecurve, ttime_true = _synthetic_system(TRUTH[:1])
        tc3, T14 = ttime_true[0][3], TRUTH[0][4]
        rng = np.random.default_rng(1)
        near = np.abs(litecurve.time - tc3) < T14
        litecurve.flux[near] = 1 + NOISE * rng.standard_normal(np.sum(near))
        keep = ~((litecurve.time > tc3 + 1.5 * T14) & (litecurve.time < tc3 + 1.0))
        edge = np.flatnonzero(keep & (litecurve.time <= tc3 + 1.5 * T14))[-2:]
        litecurve.flux[edge] -= approx_transit_depth(*TRUTH[0][2:4])
        lc = LiteCurve(
            "Kepler",
            "SYN",
            *(
                getattr(litecurve, k)[keep]
                for k in ("time", "flux", "error", "cadno", "visit", "obsmode")
            ),
        )
        new_planets, _ = QuickFitTransitTimes(star, planets, lc).perform()

        ttime = new_planets[0].transit_times.ttime_obs[3]
        data_end = litecurve.time[edge[-1]]
        assert np.isnan(ttime) or ttime + T14 / 2 <= data_end

    def test_planet_with_usp_sibling_measured_by_optimizer(self):
        # a 0.84 d sibling lies inside every cross-correlation window of the
        # 12 d planet, so all its transits are measured by the optimizer
        truth = [(0.837, 100.3, 0.013, 0.3, 1.8 / 24), (12.0, 103.0, 0.02, 0.3, 7 / 24)]
        star, planets, litecurve, ttime_true = _synthetic_system(truth)
        new_planets, _ = QuickFitTransitTimes(star, planets, litecurve).perform()

        tt = new_planets[1].transit_times
        assert np.all(np.isfinite(tt.ttime_obs)) and np.all(np.isfinite(tt.error_obs))
        assert np.all(np.abs(tt.ttime_obs - ttime_true[1]) < 4 * tt.error_obs)
