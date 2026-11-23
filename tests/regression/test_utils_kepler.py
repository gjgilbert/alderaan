"""
Tests for alderaan/utils/kepler.py: KOI and Holczer+2016 catalogs, target loading

Catalog reads (KOI, full Holczer+2016) and synthetic Holczer tables are fast;
load_kepler_data on K00137 (Kepler-18) is slow.
"""

from pathlib import Path

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

from alderaan.constants import EXPTIME
from alderaan.data_models import LiteCurve, Planet, TransitTimes
from alderaan.errors import UnsupportedTargetError
from alderaan.utils.kepler import (
    drop_few_transit_planets,
    load_kepler_data,
    match_holczer16_transit_times,
    parse_holczer16_catalog,
    parse_koi_catalog,
)

CATALOG_DIR = Path(__file__).resolve().parents[2] / "reference"
TARGET = "K00137"
KIC_ID = 8644288

KOI_CSV = str(CATALOG_DIR / "alderaan" / "kepler_merged_alderaan.csv")
HOLCZER_TXT = str(CATALOG_DIR / "kepler" / "kepler_ttvs_holczer_2016.txt")

# Kepler-18 d, b, c in ascending period
K00137_PERIODS = [3.50471620, 7.64156004, 14.85892272]


class TestParseKoiCatalog:
    def test_k00137(self):
        catalog = parse_koi_catalog(KOI_CSV, TARGET)

        assert len(catalog) == 3
        assert (catalog.kic_id == KIC_ID).all()
        npt.assert_allclose(catalog.period, K00137_PERIODS)
        # ppm --> fraction, hours --> days (Kepler-18 b row)
        npt.assert_allclose(catalog.depth[1], 2269.7e-6)
        npt.assert_allclose(catalog.duration[1], 3.41018 / 24)

    def test_blank_limb_darkening_allowed(self, tmp_path):
        rows = pd.read_csv(KOI_CSV, index_col=0)
        rows = rows[rows.koi_id == TARGET].copy()
        rows[["limbdark_1", "limbdark_2"]] = np.nan
        path = tmp_path / "koi.csv"
        rows.to_csv(path)

        catalog = parse_koi_catalog(str(path), TARGET)
        assert catalog.limbdark_1.isna().all() and catalog.limbdark_2.isna().all()

        rows.iloc[0, rows.columns.get_loc("limbdark_1")] = 0.4
        rows.to_csv(path)
        with pytest.raises(ValueError, match="inconsistent limbdark_1"):
            parse_koi_catalog(str(path), TARGET)

    def test_unknown_koi_raises(self):
        with pytest.raises(ValueError, match="K99999 not found"):
            parse_koi_catalog(KOI_CSV, "K99999")


def _planet(period, epoch, t_min, t_max):
    p = Planet("Kepler", "SYN", period, epoch, 0.03, 0.0, 0.1)
    p.set_transit_times(
        TransitTimes.from_linear_ephemeris("Kepler", "SYN", period, epoch, t_min, t_max)
    )
    return p


class TestMatchHolczer16:
    T_MIN, T_MAX = 100.0, 200.0

    def test_assigns_tables_by_period(self):
        planets = [
            _planet(4.0, 101.0, self.T_MIN, self.T_MAX),
            _planet(9.0, 103.0, self.T_MIN, self.T_MAX),
        ]
        # tables in reverse planet order; the long-period table skips index 2
        idx_a = np.array([0, 1, 3, 4, 5, 6, 7, 8, 9, 10])
        tables = [
            (idx_a, 103.0 + 9.0 * idx_a + 0.01, np.full(idx_a.size, 1e-3)),
            (np.arange(25), 101.0 + 4.0 * np.arange(25), np.full(25, 2e-3)),
        ]

        planets = match_holczer16_transit_times(planets, tables, self.T_MIN, self.T_MAX)

        short, long = planets[0].transit_times, planets[1].transit_times
        npt.assert_allclose(short.ttime_obs, 101.0 + 4.0 * np.arange(25))
        npt.assert_allclose(short.ttime_err, 2e-3)

        assert long.ttime_obs.size == 11
        npt.assert_allclose(long.ttime_obs[idx_a], 103.0 + 9.0 * idx_a + 0.01)
        # unmeasured epochs are interpolated in OMC, with no error
        assert np.isnan(long.ttime_err[2])
        npt.assert_allclose(long.ttime_obs[2], 103.01 + 9.0 * 2)
        npt.assert_allclose(long.ttime_mod, long.ttime_obs)

    def test_unmeasured_epochs_interpolated_in_omc(self):
        # Holczer measures epochs 0, 2, 3 of 5 in the window, OMC 0, +2, +3 min
        planets = [_planet(10.0, 105.0, self.T_MIN, self.T_MAX)]
        index = np.array([0, 2, 3])
        omc = np.array([0.0, 2.0, 3.0]) / 24 / 60
        tables = [(index, 105.0 + 10.0 * index + omc, np.full(3, 1e-3))]

        planets = match_holczer16_transit_times(planets, tables, self.T_MIN, self.T_MAX)
        t = planets[0].transit_times.ttime_obs

        assert np.all(np.isfinite(t))
        # epoch 1 lies midway between Holczer epochs 0 and 2
        npt.assert_allclose(t[1], 0.5 * (t[0] + t[2]))
        # past the last Holczer epoch, OMC is held flat: equal spacing
        npt.assert_allclose(np.diff(t[3:]), np.diff(t[3:])[0])

    def test_table_matching_two_planets_raises(self):
        planets = [
            _planet(4.0, 101.0, self.T_MIN, self.T_MAX),
            _planet(4.001, 101.5, self.T_MIN, self.T_MAX),
        ]
        index = np.arange(25)
        tables = [(index, 101.0 + 4.0 * index, np.full(25, 1e-3))]
        with pytest.raises(ValueError, match="matches multiple planets"):
            match_holczer16_transit_times(planets, tables, self.T_MIN, self.T_MAX)


def _catalog(rows):
    """parse_koi_catalog-like table: (period, epoch, duration [days]) per planet"""
    period, epoch, duration = np.array(rows, dtype=float).T
    return pd.DataFrame(
        {"npl": len(rows), "period": period, "epoch": epoch, "duration": duration}
    )


def _litecurve(t_max=80.0):
    time = np.arange(0.0, t_max, EXPTIME["long cadence"])
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


class TestDropFewTransitPlanets:
    def test_keeps_planets_with_three_transits(self):
        catalog = _catalog([(3.1, 1.3, 0.2), (26.0, 10.0, 0.2)])  # 26, 3 transits
        lc = _litecurve()
        out = drop_few_transit_planets(catalog, lc)
        assert len(out) == 2
        assert len(lc.time) == len(_litecurve().time)

    def test_single_planet_with_two_transits_raises(self):
        with pytest.raises(
            UnsupportedTargetError, match="2 transits within litecurve span"
        ):
            drop_few_transit_planets(_catalog([(37.0, 21.5, 0.2)]), _litecurve())

    def test_drops_planet_and_its_transits_in_multiplanet_system(self):
        # planet 1: one transit, at 50.0
        catalog = _catalog([(3.1, 1.3, 0.2), (100.0, 50.0, 0.4)])
        lc = _litecurve()
        time = lc.time.copy()

        out = drop_few_transit_planets(catalog, lc)

        npt.assert_allclose(out.period, [3.1])
        assert (out.npl == 1).all()
        # 3 T14 window around the dropped transit
        npt.assert_array_equal(lc.time, time[np.abs(time - 50.0) >= 1.5 * 0.4])

    def test_all_planets_dropped_raises(self):
        catalog = _catalog([(100.0, 50.0, 0.2), (90.0, 30.0, 0.2)])
        with pytest.raises(
            UnsupportedTargetError, match="every planet has < 3 transits"
        ):
            drop_few_transit_planets(catalog, _litecurve())


class TestParseHolczer16:
    def test_k00137(self):
        tables = parse_holczer16_catalog(HOLCZER_TXT, TARGET)

        # planets 137.01, 137.02, 137.03 in planet-number order
        assert [len(index) for index, _, _ in tables] == [177, 94, 393]
        index, ttime, error = tables[0]
        # first 137.01 row: 60.767838 d, O-C 1.7 min, error 1.8 min, BJD-2454900
        npt.assert_allclose(ttime[0], 60.767838 + 1.7 / 1440 + 67)
        npt.assert_allclose(error[0], 1.8 / 1440)
        assert index[0] == 0


@pytest.mark.slow
class TestLoadKeplerData:
    def test_k00137(self, k00137):
        star, planets, litecurve = k00137

        assert star.mission == "Kepler"
        assert star.kic_id == KIC_ID
        assert star.koi_id == TARGET

        assert len(planets) == 3
        npt.assert_allclose([p.period for p in planets], K00137_PERIODS, rtol=1e-3)
        for p in planets:
            tt = p.transit_times
            assert p.static_period is not None and p.static_epoch is not None
            assert tt.index[0] == 0
            assert tt.ttime_mod.min() >= litecurve.t_start
            assert tt.ttime_mod.max() <= litecurve.t_end
            # Holczer+2016 measured most transits of all three planets
            assert np.mean(np.isfinite(tt.ttime_obs)) > 0.5

    def test_missing_requested_quarter_raises(self):
        data_dir = Path(__file__).resolve().parents[1] / "data" / "MAST_downloads"
        with pytest.raises(FileNotFoundError, match=r"quarters \[99\] not found"):
            load_kepler_data(TARGET, str(data_dir), KOI_CSV, visits=[4, 99])
