"""Tests for alderaan.algorithms.base: input handling and make_transit_mask"""

import numpy as np
import pytest

from alderaan.algorithms.base import BaseAlg
from alderaan.data_models import LiteCurve, Planet, Star, TransitTimes


def _star():
    return Star("Kepler", "SYN", koi_id=1, kic_id=1, u1=0.4, u2=0.25)


def _planet(period, epoch, duration, ttv=0.0, ntransit=10):
    """Planet whose ttime_mod is a linear ephemeris plus a constant offset ttv"""
    p = Planet("Kepler", "SYN", period, epoch, 1e-3, duration, 0.3)
    index = np.arange(ntransit)
    ttime = epoch + period * index
    p.set_transit_times(
        TransitTimes("Kepler", "SYN", index, ttime, ttime_mod=ttime + ttv)
    )
    return p


def _litecurve(t_max=50.0):
    time = np.arange(0.0, t_max, 0.02)
    npts = len(time)
    return LiteCurve(
        "Kepler",
        "SYN",
        time,
        np.ones(npts),
        np.full(npts, 1e-4),
        np.arange(npts),
        np.ones(npts, dtype=int),
        np.full(npts, "long cadence"),
    )


def _mask_at(alg, tc, offsets, **kwargs):
    return alg.make_transit_mask(time=tc + np.asarray(offsets), **kwargs)


class TestBaseAlgInputs:
    def test_inputs_are_deep_copied(self):
        star, planets, lc = _star(), [_planet(5.0, 2.0, 0.1)], _litecurve()
        alg = BaseAlg(star, planets, lc)
        alg.litecurve.flux[:] = 0.0
        alg.planets[0].duration = 1.0
        assert np.all(lc.flux == 1.0)
        assert planets[0].duration == 0.1

    @pytest.mark.parametrize("arg", ["star", "planets", "litecurve"])
    def test_wrong_input_type_raises(self, arg):
        inputs = {
            "star": _star(),
            "planets": [_planet(5.0, 2.0, 0.1)],
            "litecurve": _litecurve(),
        }
        inputs[arg] = object()
        with pytest.raises(TypeError):
            BaseAlg(**inputs)


class TestMakeTransitMask:
    eps = 1e-6

    def _alg(self, duration=0.1, ttv=0.0):
        return BaseAlg(_star(), [_planet(5.0, 2.0, duration, ttv=ttv)], _litecurve())

    def test_rel_size_sets_full_width(self):
        # full width = 2 x 0.1 = 0.2 d, half width 0.1 d
        alg = self._alg()
        h = 0.1
        mask = _mask_at(
            alg,
            7.0,
            [-h - self.eps, -h + self.eps, 0, h - self.eps, h + self.eps],
            rel_size=2.0,
        )
        np.testing.assert_array_equal(mask, [False, True, True, True, False])

    def test_abs_size_is_a_floor_for_short_transits(self):
        # rel_size x T14 = 0.1 d < abs_size = 0.5 d: half width 0.25 d
        alg = self._alg()
        h = 0.25
        mask = _mask_at(
            alg, 7.0, [h - self.eps, h + self.eps], rel_size=1.0, abs_size=0.5
        )
        np.testing.assert_array_equal(mask, [True, False])

    def test_abs_size_ignored_when_rel_size_is_wider(self):
        # rel_size x T14 = 0.3 d > abs_size = 0.1 d: half width 0.15 d
        alg = self._alg()
        h = 0.15
        mask = _mask_at(
            alg, 7.0, [h - self.eps, h + self.eps], rel_size=3.0, abs_size=0.1
        )
        np.testing.assert_array_equal(mask, [True, False])

    def test_buffer_adds_to_full_width(self):
        # max(1 x 0.1, 0.5) + 0.2 = 0.7 d full width: half width 0.35 d
        alg = self._alg()
        h = 0.35
        mask = _mask_at(
            alg,
            7.0,
            [h - self.eps, h + self.eps],
            rel_size=1.0,
            abs_size=0.5,
            buffer=0.2,
        )
        np.testing.assert_array_equal(mask, [True, False])

    def test_centered_on_modeled_transit_times(self):
        # ttime_mod is offset 0.3 d from the linear ephemeris
        alg = self._alg(ttv=0.3)
        mask = alg.make_transit_mask(time=np.array([7.0, 7.3]), rel_size=2.0)
        np.testing.assert_array_equal(mask, [False, True])

    def test_masks_every_transit_of_every_planet(self):
        planets = [_planet(5.0, 2.0, 0.1), _planet(7.0, 4.5, 0.2, ntransit=7)]
        alg = BaseAlg(_star(), planets, _litecurve())
        mask = alg.make_transit_mask(rel_size=1.0)

        # one contiguous masked window per transit (10 + 7, none overlapping)
        n_windows = np.sum(np.diff(mask.astype(int)) == 1)
        assert n_windows == 17
        for p in planets:
            for tc in p.transit_times.ttime_mod:
                assert mask[np.argmin(np.abs(alg.litecurve.time - tc))]
