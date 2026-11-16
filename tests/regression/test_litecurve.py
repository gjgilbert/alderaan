"""Tests for the LiteCurve data model. Fast tests use synthetic photometry; slow
tests load Kepler-18 PDCSAP files from tests/data."""

import glob
import os
from pathlib import Path

import lightkurve as lk
import numpy as np
import pytest
from astropy.io import fits
from numpy.testing import assert_array_equal

from alderaan.data_models import LiteCurve

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "MAST_downloads"
TARGET = "K00137"
KIC_ID = 8644288
VISITS = [4, 5, 6]

FIELDS = ["time", "flux", "error", "cadno", "visit", "obsmode"]


def _lc(n=100, seed=0):
    rng = np.random.default_rng(seed)
    cadno = np.arange(1000, 1000 + n)
    return LiteCurve(
        "Kepler",
        "K00137",
        time=0.02 * (cadno - 1000) + 130.0,
        flux=1 + rng.normal(0, 1e-4, n),
        error=np.full(n, 1e-4),
        cadno=cadno,
        visit=np.where(np.arange(n) < n // 2, 4, 5),
        obsmode=np.where(np.arange(n) < n // 2, "long cadence", "short cadence"),
    )


class TestConstructor:
    def test_span_defaults_to_time_range(self):
        lc = _lc()
        assert lc.t_start == lc.time.min()
        assert lc.t_end == lc.time.max()

    def test_rejects_unequal_lengths(self):
        lc = _lc()
        with pytest.raises(ValueError, match="equal lengths"):
            LiteCurve(
                "Kepler",
                "K00137",
                lc.time,
                lc.flux[:-1],
                lc.error,
                lc.cadno,
                lc.visit,
                lc.obsmode,
            )

    @pytest.mark.parametrize("field", ["flux", "error"])
    def test_rejects_nonfinite(self, field):
        lc = _lc()
        arrays = {k: getattr(lc, k) for k in FIELDS}
        arrays[field] = arrays[field].copy()
        arrays[field][3] = np.nan
        with pytest.raises(ValueError, match="must be finite"):
            LiteCurve("Kepler", "K00137", **arrays)


class TestSelectAndReset:
    def test_select_removes_cadences_but_keeps_span(self):
        lc = _lc()
        keep = (lc.time > 130.5) & (lc.time < 131.5)
        expected = {k: getattr(lc, k)[keep] for k in FIELDS}
        lc.select(keep)
        for k in FIELDS:
            assert_array_equal(getattr(lc, k), expected[k])
        assert lc.t_start == 130.0
        assert np.isclose(lc.t_end, 130.0 + 0.02 * 99)

    def test_reset_restores_raw_flux_by_cadence(self):
        raw = _lc()
        lc = _lc()
        keep = np.arange(len(lc.time)) % 3 != 0
        lc.select(keep)
        lc.flux = lc.flux / 2
        lc.error = lc.error * 2
        lc.reset(raw)
        assert_array_equal(lc.flux, raw.flux[keep])
        assert_array_equal(lc.error, raw.error[keep])
        assert not np.shares_memory(lc.flux, raw.flux)

    def test_reset_fails_if_raw_lacks_cadences(self):
        raw = _lc()
        lc = _lc()
        raw.select(raw.cadno != 1050)
        with pytest.raises(ValueError, match="does not contain"):
            lc.reset(raw)


class TestObsmodeAt:
    def test_nearest_cadence_with_ties_earlier(self):
        lc = LiteCurve(
            "Kepler",
            "K00137",
            time=[0.0, 1.0, 2.0, 3.0],
            flux=np.ones(4),
            error=np.ones(4),
            cadno=np.arange(4),
            visit=np.ones(4),
            obsmode=["a", "b", "c", "d"],
        )
        times = np.array([-5.0, 0.4, 1.0, 1.5, 1.6, 10.0])
        assert_array_equal(lc.obsmode_at(times), ["a", "a", "b", "b", "c", "d"])

    def test_requires_sorted_time(self):
        lc = _lc()
        lc.time = lc.time[::-1]
        with pytest.raises(ValueError, match="sorted"):
            lc.obsmode_at(np.array([131.0]))


class TestFits:
    def test_round_trip(self, tmp_path):
        lc = _lc()
        lc.select(lc.time > 130.3)
        path = str(tmp_path / "lc.fits")
        lc.to_fits(path)
        out = LiteCurve.from_fits(path)

        assert (out.mission, out.target) == (lc.mission, lc.target)
        assert (out.t_start, out.t_end) == (lc.t_start, lc.t_end)
        for k in FIELDS:
            assert_array_equal(getattr(out, k), getattr(lc, k))
        assert os.listdir(tmp_path) == ["lc.fits"]


@pytest.fixture(scope="module")
def kplr_lc():
    return LiteCurve.from_kplr_pdcsap(
        str(DATA_DIR), TARGET, KIC_ID, "long cadence", visits=VISITS
    )


@pytest.mark.slow
class TestFromKplrPdcsap:
    def test_loads_requested_quarters(self, kplr_lc):
        lc = kplr_lc
        assert (lc.mission, lc.target) == ("Kepler", TARGET)
        assert_array_equal(np.unique(lc.visit), VISITS)
        assert np.all(lc.obsmode == "long cadence")
        assert np.all(np.diff(lc.time) > 0)
        assert np.all(np.diff(lc.cadno) > 0)
        assert (lc.t_start, lc.t_end) == (lc.time.min(), lc.time.max())

    def test_flux_is_finite_and_normalized_per_quarter(self, kplr_lc):
        lc = kplr_lc
        for k in ["time", "flux", "error"]:
            assert np.all(np.isfinite(getattr(lc, k)))
        for v in VISITS:
            assert np.isclose(np.median(lc.flux[lc.visit == v]), 1.0, atol=1e-3)

    def test_kepler_quality_flagged_cadences_removed(self, kplr_lc):
        flagged = []
        for mf in glob.glob(str(DATA_DIR / f"kplr{KIC_ID:09d}*_llc.fits")):
            with fits.open(mf) as hdul:
                if hdul[0].header["QUARTER"] not in VISITS:
                    continue
                data = hdul["LIGHTCURVE"].data
                good = lk.KeplerQualityFlags.create_quality_mask(
                    data["SAP_QUALITY"], bitmask="default"
                )
                flagged.append(data["CADENCENO"][~good])
        flagged = np.concatenate(flagged)

        assert len(flagged) > 0
        assert not np.any(np.isin(kplr_lc.cadno, flagged))

    def test_missing_obsmode_fails(self):
        with pytest.raises(FileNotFoundError, match="short cadence"):
            LiteCurve.from_kplr_pdcsap(
                str(DATA_DIR), TARGET, KIC_ID, "short cadence", visits=VISITS
            )
