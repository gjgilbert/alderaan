"""Tests for the Planet data model (ephemeris ownership via TransitTimes)."""

import numpy as np
import pytest

from alderaan.data_models import Planet, TransitTimes

PERIOD = 3.5
EPOCH = 100.2


def _planet():
    return Planet("Kepler", "K00137", PERIOD, EPOCH, 0.03, 0.0, 0.1)


def _transit_times(period, epoch=EPOCH, n=10):
    index = np.arange(n)
    return TransitTimes("Kepler", "K00137", index, epoch + period * index)


class TestPlanetEphemeris:
    def test_ephemeris_follows_attached_transit_times(self):
        p = _planet()
        assert (p.period, p.epoch) == (PERIOD, EPOCH)
        p.set_transit_times(_transit_times(PERIOD * 1.05, epoch=EPOCH + 0.01))
        assert np.isclose(p.period, PERIOD * 1.05)
        assert np.isclose(p.epoch, EPOCH + 0.01)

    def test_rejects_period_change_over_10_percent(self):
        p = _planet()
        with pytest.raises(ValueError, match="10%"):
            p.set_transit_times(_transit_times(PERIOD * 1.2))
        assert p.transit_times is None


class TestStaticEphemeris:
    def test_static_ephemeris_frozen_against_later_transit_times(self):
        p = _planet()
        p.set_transit_times(_transit_times(PERIOD))
        p.set_static_period_and_epoch()
        p.set_transit_times(_transit_times(PERIOD * 1.02))
        assert np.isclose(p.static_period, PERIOD)
        assert np.isclose(p.static_epoch, EPOCH)
        assert np.isclose(p.period, PERIOD * 1.02)

    def test_static_ephemeris_can_only_be_set_once(self):
        p = _planet()
        p.set_static_period_and_epoch()
        with pytest.raises(ValueError, match="already set"):
            p.set_static_period_and_epoch()
