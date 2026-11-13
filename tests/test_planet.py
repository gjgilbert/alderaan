import os
from pathlib import Path

import numpy as np
import pytest
from alderaan.data_models.planet import Planet
from alderaan.modules.omc import OMC
from alderaan.utils.io import parse_koi_catalog, parse_holczer16_catalog


base_path = Path(__file__).resolve().parents[0]
catalog_csv = os.path.join(base_path, 'catalogs/kepler_dr25_gaia_dr2_crossmatch.csv')
holczer_txt = os.path.join(base_path, 'catalogs/holczer_2016_kepler_ttvs.txt')
golden_npz = os.path.join(base_path, 'data/planet_golden.npz')

# golden values were generated from the pre-merge Planet + Ephemeris classes
KOI_IDS = ['K00137', 'K00070', 'K00377', 'K00001']
T_MIN = 131.51
T_MAX = 1591.0


def make_planets(koi_id):
    catalog = parse_koi_catalog(catalog_csv, koi_id)
    return [Planet(catalog.period[n], catalog.epoch[n], catalog.depth[n],
                   catalog.duration[n], catalog.impact[n], T_MIN, T_MAX)
            for n in range(int(catalog.npl[0]))]


def record(result, koi_id, n, step, p):
    fields = {
        'index': p.transit_index,
        'ttime': p.transit_times,
        'error': p.transit_errors,
        'quality': p.transit_quality,
        'period': p.period,
        'epoch': p.epoch,
        'planet_period': p.period,
        'planet_epoch': p.epoch,
        'reference_period': p.reference_period,
        'reference_epoch': p.reference_epoch,
    }
    for k, v in fields.items():
        if v is not None:
            result[f"{koi_id}_{n}_step{step}_{k}"] = np.copy(v)


def omc_update(p, full):
    omc = OMC(p)
    omc.quick_fit_trend()
    p.update_from_omc(omc, full=full)


def replay_pipeline_bookkeeping(koi_id):
    """Replay the sequence of ephemeris operations performed by the pipeline recipes"""
    result = {}

    # step 1 : linear ephemeris
    planets = make_planets(koi_id)
    for n, p in enumerate(planets):
        record(result, koi_id, n, 1, p)

    # step 2 : Holczer+2016 ephemerides
    for index, ttime, error in parse_holczer16_catalog(holczer_txt, koi_id, len(planets)):
        for n, p in enumerate(planets):
            if np.isclose(Planet.fit_linear_ephemeris(index, ttime)[0], p.period, rtol=0.01, atol=p.duration):
                planets[n] = p.set_transit_times(index, ttime, error)
    for n, p in enumerate(planets):
        record(result, koi_id, n, 2, p)

    for n, p in enumerate(planets):
        has_error = p.transit_errors is not None

        # step 3 : clip range and set reference period and epoch
        p.clip_range(T_MIN, T_MAX)
        p.set_reference_period_and_epoch()
        p.refit_period_and_epoch()
        record(result, koi_id, n, 3, p)

        # step 4 : OMC update with full interpolation
        if has_error:
            omc_update(p, full=True)
            record(result, koi_id, n, 4, p)

        # step 5 : quality flags and removal
        p.transit_quality = p.transit_index % 7 != 3
        p.remove_poor_quality_transits()
        record(result, koi_id, n, 5, p)

        # step 6 : in-place transit time update
        if has_error:
            q = p.transit_index % 5 != 0
            p.transit_times[q] = p.transit_times[q] + 2e-4 * np.cos(p.transit_index[q])
            p.transit_errors[q] = p.transit_errors[q] * 1.1
            p.refit_period_and_epoch()
            record(result, koi_id, n, 6, p)

        # step 7 : additive transit time update
        theta = 1e-3 * np.sin(p.transit_index)
        p.transit_times += theta
        p.refit_period_and_epoch()
        record(result, koi_id, n, 7, p)

        # step 8 : OMC update without full interpolation
        if has_error:
            omc_update(p, full=False)
            record(result, koi_id, n, 8, p)

    return result


@pytest.mark.parametrize('koi_id', KOI_IDS)
def test_matches_golden(koi_id):
    golden = np.load(golden_npz)
    expected_keys = sorted(k for k in golden.files if k.startswith(f"{koi_id}_"))

    result = replay_pipeline_bookkeeping(koi_id)

    assert sorted(result.keys()) == expected_keys
    for k in expected_keys:
        assert result[k].dtype == golden[k].dtype, k
        assert np.array_equal(result[k], golden[k]), k


def test_period_change_raises():
    p = make_planets('K00137')[0]
    index = np.arange(10)
    ttime = 140. + 2 * p.period * index

    with pytest.raises(ValueError):
        p.set_transit_times(index, ttime, 1e-3 * np.ones(10))


def test_reference_period_and_epoch_set_once():
    p = make_planets('K00137')[0]
    p.set_reference_period_and_epoch()

    with pytest.raises(AssertionError):
        p.set_reference_period_and_epoch()
