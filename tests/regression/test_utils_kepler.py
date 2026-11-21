"""
Tests for alderaan/utils/kepler.py: KOI and Holczer+2016 catalogs, target loading

The KOI catalog and synthetic Holczer tables are fast; the full Holczer+2016
table and load_kepler_data on K00137 (Kepler-18) are slow.
"""

from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest

from alderaan.data_models import Planet, TransitTimes
from alderaan.utils.kepler import (
    load_kepler_data,
    match_holczer16_transit_times,
    parse_holczer16_catalog,
    parse_koi_catalog,
)

CATALOG_DIR = Path(__file__).resolve().parents[1] / "catalogs"
TARGET = "K00137"
KIC_ID = 8644288

KOI_CSV = str(CATALOG_DIR / "kepler_dr25_gaia_dr2_crossmatch.csv")
HOLCZER_TXT = str(CATALOG_DIR / "holczer_2016_kepler_ttvs.txt")

# Kepler-18 d, b, c in ascending period
K00137_PERIODS = [3.504691252, 7.641567589, 14.85890852]


class TestParseKoiCatalog:
    def test_k00137(self):
        catalog = parse_koi_catalog(KOI_CSV, TARGET)

        assert len(catalog) == 3
        assert (catalog.kic_id == KIC_ID).all()
        npt.assert_allclose(catalog.period, K00137_PERIODS)
        # ppm --> fraction, hours --> days (Kepler-18 b row)
        npt.assert_allclose(catalog.depth[1], 2269.7e-6)
        npt.assert_allclose(catalog.duration[1], 3.41018 / 24)

    def test_unknown_koi_raises(self):
        with pytest.raises(ValueError, match="K99999 not found"):
            parse_koi_catalog(KOI_CSV, "K99999")


def _planet(period, epoch, t_min, t_max):
    p = Planet("Kepler", "SYN", period, epoch, 1e-3, 0.1, 0.3)
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
        assert np.isnan(long.ttime_obs[2]) and np.isnan(long.ttime_err[2])
        npt.assert_allclose(long.ttime_obs[idx_a], 103.0 + 9.0 * idx_a + 0.01)
        # unmeasured epochs keep the linear ephemeris prediction
        npt.assert_allclose(long.ttime_mod[2], 103.01 + 9.0 * 2)

    def test_table_matching_two_planets_raises(self):
        planets = [
            _planet(4.0, 101.0, self.T_MIN, self.T_MAX),
            _planet(4.001, 101.5, self.T_MIN, self.T_MAX),
        ]
        index = np.arange(25)
        tables = [(index, 101.0 + 4.0 * index, np.full(25, 1e-3))]
        with pytest.raises(ValueError, match="matches multiple planets"):
            match_holczer16_transit_times(planets, tables, self.T_MIN, self.T_MAX)


@pytest.mark.slow
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
            load_kepler_data(TARGET, str(data_dir), str(CATALOG_DIR), visits=[4, 99])
