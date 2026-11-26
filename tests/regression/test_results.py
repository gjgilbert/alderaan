"""Tests for the Results data model (packaging, consistency, FITS round trip).
Synthetic components only."""

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits
from numpy.testing import assert_allclose, assert_array_equal

from alderaan.data_models import LiteCurve, Results, TransitTimes

TARGET = "K00137"
# match the make_posteriors fixture's reference ephemerides and span
REF_PERIOD = np.array([3.5, 7.6])
REF_EPOCH = np.array([131.1, 133.4])
T_START, T_END = 130.0, 200.0


def _transit_times(npl=2, target=TARGET):
    return [
        TransitTimes.from_linear_ephemeris(
            "Kepler", target, REF_PERIOD[n], REF_EPOCH[n], T_START, T_END
        )
        for n in range(npl)
    ]


def _litecurve(mission="Kepler"):
    n = 200
    return LiteCurve(
        mission,
        TARGET,
        time=np.linspace(T_START, T_END, n),
        flux=np.ones(n),
        error=np.full(n, 1e-4),
        cadno=np.arange(n),
        visit=np.full(n, 4),
        obsmode=np.full(n, "long cadence"),
    )


class TestConsistency:
    def test_metadata_from_components(self, make_posteriors):
        res = Results(
            posteriors=make_posteriors(),
            transit_times=_transit_times(),
            litecurve=_litecurve(),
        )
        assert (res.mission, res.target, res.npl) == ("Kepler", TARGET, 2)

    def test_rejects_mismatched_target_or_mission(self, make_posteriors):
        with pytest.raises(ValueError, match="inconsistent metadata"):
            Results(
                posteriors=make_posteriors(),
                transit_times=_transit_times(target="K00138"),
            )
        with pytest.raises(ValueError, match="inconsistent metadata"):
            Results(
                transit_times=_transit_times(), litecurve=_litecurve(mission="TESS")
            )

    def test_rejects_planet_count_mismatch(self, make_posteriors):
        with pytest.raises(ValueError, match="planets"):
            Results(
                posteriors=make_posteriors(npl=2), transit_times=_transit_times(npl=1)
            )


class TestFits:
    def test_round_trip_all_components(self, tmp_path, make_posteriors):
        res = Results(
            posteriors=make_posteriors(),
            transit_times=_transit_times(),
            litecurve=_litecurve(),
        )
        res.litecurve.meta.update(RUNID=("r1", "run"), DTNOMP=(2.5, "[d] period"))
        res.meta.update(res.litecurve.meta)
        res.meta.update(PLNAME0=("K00137.01", "name"), CHI2RED=(None, "blank"))
        res.posteriors.meta.update(SEED=(7, "seed"), LDU1=(None, "blank"))
        path = str(tmp_path / "results.fits")
        lc_path = str(tmp_path / "detrended.fits")
        res.to_fits(path)
        res.litecurve.to_fits(lc_path)

        out = Results.from_fits(path, lc_path)
        assert (out.mission, out.target, out.npl) == (res.mission, res.target, res.npl)
        # meta round-trips with comments; None is a blank card
        assert out.meta == res.meta
        assert out.litecurve.meta == res.litecurve.meta
        assert out.posteriors.meta == res.posteriors.meta
        samples_header = fits.getheader(path, "SAMPLES")

        p_in, p_out = res.posteriors, out.posteriors
        pd.testing.assert_frame_equal(p_out.samples, p_in.samples)
        for k, v in p_in.sampler_info.items():
            assert_array_equal(p_out.sampler_info[k], v)
        assert p_out.sampler_info["samples_n"].dtype.kind == "i"
        info = p_in.sampler_info
        assert samples_header["NCALL"] == np.sum(info["ncall"])
        assert np.isclose(samples_header["NEFF"], p_in.neff())
        assert samples_header["LNZ"] == info["logz"][-1]
        assert samples_header["LNZERR"] == info["logzerr"][-1]
        assert samples_header["INFO"] == info["information"][-1]
        assert_array_equal(p_out.ref_period, p_in.ref_period)
        assert_array_equal(p_out.ref_epoch, p_in.ref_epoch)
        assert (p_out.t_start, p_out.t_end) == (p_in.t_start, p_in.t_end)
        assert p_out.supersample == p_in.supersample
        pd.testing.assert_frame_equal(out.summary(), res.summary())

        for tt_in, tt_out in zip(res.transit_times, out.transit_times):
            assert_array_equal(tt_out.index, tt_in.index)
            assert_array_equal(tt_out.ttime_mod, tt_in.ttime_mod)
        assert_array_equal(out.litecurve.flux, res.litecurve.flux)

    def test_warns_if_litecurve_cards_disagree(self, tmp_path, caplog):
        res = Results(transit_times=_transit_times(), litecurve=_litecurve())
        res.litecurve.meta["RUNID"] = "r1"
        res.meta.update(res.litecurve.meta)
        path = str(tmp_path / "results.fits")
        lc_path = str(tmp_path / "detrended.fits")
        res.to_fits(path)
        res.litecurve.to_fits(lc_path)

        Results.from_fits(path, lc_path)
        assert "disagree" not in caplog.text

        res.litecurve.meta["RUNID"] = "r2"
        res.litecurve.to_fits(lc_path)
        Results.from_fits(path, lc_path)
        assert "disagree on RUNID" in caplog.text

    def test_round_trip_transit_times_only(self, tmp_path):
        res = Results(transit_times=_transit_times())
        path = str(tmp_path / "results.fits")
        res.to_fits(path)
        out = Results.from_fits(path)
        assert out.posteriors is None and out.litecurve is None
        assert out.npl == 2
        assert_array_equal(
            out.transit_times[1].ttime_obs, res.transit_times[1].ttime_obs
        )


class TestSummary:
    def test_computed_once(self, make_posteriors, monkeypatch):
        """summary() is cached: the median helpers reuse one Posteriors.summary"""
        post = make_posteriors()
        res = Results(posteriors=post, transit_times=_transit_times())
        expected = post.summary()

        calls = []
        compute = post.summary
        monkeypatch.setattr(post, "summary", lambda: calls.append(1) or compute())
        res.median_transit_params(0)
        res.median_transit_times(1)

        assert len(calls) == 1
        pd.testing.assert_frame_equal(res.summary(), expected)


class TestMedianModel:
    def test_median_transit_times_agree_with_planet_samples(self, make_posteriors):
        """With every sample at the same (C0, C1), the median transit times lie on
        the planet_samples ephemeris"""
        post = make_posteriors()
        n = 1
        post.samples[f"C0_{n}"] = 0.003
        post.samples[f"C1_{n}"] = -0.002
        res = Results(posteriors=post, transit_times=_transit_times())

        ps = res.planet_samples(n)
        tt = res.transit_times[n]
        expected = ps["EPOCH"][0] + ps["PERIOD"][0] * tt.index
        assert_allclose(res.median_transit_times(n), expected, rtol=0, atol=1e-9)
