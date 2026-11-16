"""
Tests for alderaan/algorithms/quick_fit_transit_times.py

Recovery of injected per-transit time offsets (TTVs) from a synthetic detrended
long cadence litecurve, including a transit that overlaps a second planet's;
no pytensor compile, fast tier only.
"""

import numpy as np
import pytest

from alderaan.algorithms import QuickFitTransitTimes
from alderaan.constants import EXPTIME
from alderaan.data_models.litecurve import LiteCurve
from alderaan.data_models.planet import Planet
from alderaan.data_models.star import Star
from alderaan.data_models.transit_times import TransitTimes
from alderaan.transit_model import exposure_offsets, transit_flux
from alderaan.utils.astro import approx_transit_depth

LC = "long cadence"
U1, U2 = 0.4, 0.25
NOISE = 1e-4
TTV_AMP = 10 / 1440  # days

# (period, epoch, ror, impact, T14) per planet; planet 1 transit 1 overlaps
# planet 0 transit 4
TRUTH = [(5.3, 102.1, 0.08, 0.3, 0.15), (11.7, 111.65, 0.07, 0.4, 0.2)]


def _synthetic_system(truth, seed=0):
    """
    Star, planets on linear ephemerides with known shapes, a 60-day detrended
    litecurve whose transits carry sinusoidal TTVs, and the true transit times
    """
    rng = np.random.default_rng(seed)
    cadno = np.arange(int(60 / EXPTIME[LC]))
    time = 100.0 + cadno * EXPTIME[LC]
    offsets = exposure_offsets(EXPTIME[LC], 15)
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

    flux += NOISE * rng.standard_normal(len(time))
    litecurve = LiteCurve(
        "Kepler",
        "SYN",
        time,
        flux,
        np.full_like(time, NOISE),
        cadno,
        np.ones_like(cadno),
        np.full(len(time), LC),
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
