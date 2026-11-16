"""Tests for the TransitTimes data model. Synthetic transit times only."""

import numpy as np
import pytest
from astropy.io import fits
from astropy.stats import mad_std
from numpy.testing import assert_allclose, assert_array_equal

from alderaan.data_models.transit_times import TransitTimes
from alderaan.utils.stats import sigma_cutoff

PERIOD = 3.5
EPOCH = 100.2


def _tt(n=20, noise=0.002, seed=0):
    """Transit times on a linear ephemeris, observed with Gaussian noise"""
    rng = np.random.default_rng(seed)
    index = np.arange(n)
    ttime_mod = EPOCH + PERIOD * index
    ttime_obs = ttime_mod + rng.normal(0, noise, n)
    return TransitTimes(
        "Kepler",
        "K00137",
        index,
        ttime_obs,
        error_obs=np.full(n, noise),
        ttime_mod=ttime_mod,
    )


class TestConstructor:
    def test_defaults(self):
        index = np.arange(5)
        obs = EPOCH + PERIOD * index
        tt = TransitTimes("Kepler", "K00137", index, obs)
        assert_array_equal(tt.ttime_mod, tt.ttime_obs)
        assert not np.shares_memory(tt.ttime_mod, tt.ttime_obs)
        assert np.all(np.isnan(tt.error_obs))
        assert np.all(np.isnan(tt.out_prob))
        assert tt.out_flag.dtype == bool and not tt.out_flag.any()

    def test_ephemeris_is_least_squares_fit_to_ttime_mod(self):
        rng = np.random.default_rng(1)
        index = np.array([0, 1, 2, 4, 7, 8, 12])
        ttime_mod = EPOCH + PERIOD * index + rng.normal(0, 0.01, len(index))
        tt = TransitTimes(
            "Kepler", "K00137", index, np.full(len(index), np.nan), ttime_mod=ttime_mod
        )
        period, epoch = np.polyfit(index, ttime_mod, 1)
        assert np.isclose(tt.period, period, rtol=0, atol=1e-10)
        assert np.isclose(tt.epoch, epoch, rtol=0, atol=1e-10)
        assert_allclose(tt.linear_ephemeris(), epoch + period * index)

    def test_requires_two_transits(self):
        with pytest.raises(ValueError, match="at least 2"):
            TransitTimes("Kepler", "K00137", [0], [EPOCH])

    def test_rejects_unequal_lengths(self):
        with pytest.raises(ValueError, match="equal lengths"):
            TransitTimes(
                "Kepler", "K00137", np.arange(3), np.zeros(3), error_obs=np.zeros(2)
            )


class TestFromLinearEphemeris:
    @pytest.mark.parametrize("epoch", [EPOCH - 10 * PERIOD, EPOCH, EPOCH + 10 * PERIOD])
    def test_covers_window_regardless_of_epoch(self, epoch):
        t_min, t_max = 101.0, 150.0
        tt = TransitTimes.from_linear_ephemeris(
            "Kepler", "K00137", PERIOD, epoch, t_min, t_max
        )
        assert_array_equal(tt.index, np.arange(len(tt.index)))
        assert t_min <= tt.ttime_mod[0] < t_min + PERIOD
        assert tt.ttime_mod[-1] < t_max <= tt.ttime_mod[-1] + PERIOD
        assert np.isclose(tt.period, PERIOD)
        # predictions stay on the input ephemeris
        cycles = (tt.ttime_mod - epoch) / PERIOD
        assert_allclose(cycles, np.round(cycles), rtol=0, atol=1e-9)


class TestTTVScatter:
    def test_scatter_ignores_missing_transits(self):
        tt = _tt()
        obs = tt.ttime_obs.copy()
        obs[[3, 7]] = np.nan
        tt_missing = TransitTimes(
            "Kepler", "K00137", tt.index, obs, ttime_mod=tt.ttime_mod
        )
        expected = mad_std(np.delete(tt.ttime_obs - tt.ttime_mod, [3, 7]))
        assert np.isclose(tt_missing.ttv_scatter(), expected)
        assert np.isclose(tt_missing.ttv_buffer(), sigma_cutoff(18) * expected)


class TestSubsetAndClip:
    def test_subset_keeps_masked_transits(self):
        tt = _tt()
        mask = tt.index % 3 == 0
        sub = tt.subset(mask)
        assert_array_equal(sub.index, tt.index[mask])
        assert_array_equal(sub.ttime_obs, tt.ttime_obs[mask])
        assert_array_equal(sub.error_obs, tt.error_obs[mask])

    def test_clip_rezeroes_index_and_ephemeris(self):
        tt = _tt()
        t_min, t_max = tt.ttime_mod[5], tt.ttime_mod[12]
        clipped = tt.clip(t_min, t_max)
        assert_array_equal(clipped.index, np.arange(8))
        assert_array_equal(clipped.ttime_mod, tt.ttime_mod[5:13])
        assert np.isclose(clipped.epoch, tt.ttime_mod[5])
        assert np.isclose(clipped.period, PERIOD)

    def test_clip_requires_two_transits(self):
        tt = _tt()
        with pytest.raises(ValueError, match="at least 2"):
            tt.clip(tt.ttime_mod[5] - 0.1, tt.ttime_mod[5] + 0.1)


class TestFitsTable:
    def test_hdu_round_trip(self, tmp_path):
        tt = _tt()
        tt.ttime_obs[4] = np.nan
        tt.out_prob[:] = np.linspace(0, 1, len(tt.index))
        tt.out_flag[[2, 9]] = True
        path = tmp_path / "tt.fits"
        fits.HDUList([fits.PrimaryHDU(), tt._to_hdu(3)]).writeto(path)

        with fits.open(path) as hdul:
            out = TransitTimes._from_hdu(hdul["TTIMES_03"])

        assert (out.mission, out.target) == (tt.mission, tt.target)
        for k in [
            "index",
            "ttime_obs",
            "ttime_mod",
            "error_obs",
            "out_prob",
            "out_flag",
        ]:
            assert_array_equal(getattr(out, k), getattr(tt, k))
            assert getattr(out, k).dtype == getattr(tt, k).dtype
