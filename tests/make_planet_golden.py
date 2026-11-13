"""Generate tests/data/planet_golden.npz from the Ephemeris/Planet API (pre-merge baseline)"""
import os
import warnings
from pathlib import Path

import numpy as np
from alderaan.data_models.ephemeris import Ephemeris
from alderaan.data_models.planet import Planet
from alderaan.modules.omc import OMC
from alderaan.utils.io import parse_koi_catalog, parse_holczer16_catalog

base_path = Path(__file__).resolve().parents[0]
catalog_csv = os.path.join(base_path, 'catalogs/kepler_dr25_gaia_dr2_crossmatch.csv')
holczer_txt = os.path.join(base_path, 'catalogs/holczer_2016_kepler_ttvs.txt')
output_npz = os.path.join(base_path, 'data/planet_golden.npz')

KOI_IDS = ['K00137', 'K00070', 'K00377', 'K00001']
T_MIN = 131.51
T_MAX = 1591.0


def record(golden, koi_id, n, step, p):
    e = p.ephemeris
    fields = {
        'index': e.index,
        'ttime': e.ttime,
        'error': e.error,
        'quality': e.quality,
        'period': e.period,
        'epoch': e.epoch,
        'planet_period': p.period,
        'planet_epoch': p.epoch,
        'reference_period': getattr(e, '_static_period', None),
        'reference_epoch': getattr(e, '_static_epoch', None),
    }
    for k, v in fields.items():
        if v is not None:
            golden[f"{koi_id}_{n}_step{step}_{k}"] = np.copy(v)


def omc_update(p, full):
    omc = OMC(p.ephemeris)
    omc.quick_fit_trend()
    new_ephemeris = p.ephemeris.update_from_omc(omc).interpolate(full=full)
    p.update_ephemeris(new_ephemeris)


golden = {}

for koi_id in KOI_IDS:
    catalog = parse_koi_catalog(catalog_csv, koi_id)
    NPL = int(catalog.npl[0])

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        planets = [Planet(catalog, koi_id, n) for n in range(NPL)]

    # step 1 : linear ephemeris
    for n, p in enumerate(planets):
        p.update_ephemeris(Ephemeris(period=p.period, epoch=p.epoch, t_min=T_MIN, t_max=T_MAX))
        record(golden, koi_id, n, 1, p)

    # step 2 : Holczer+2016 ephemerides
    for _ephem in parse_holczer16_catalog(holczer_txt, koi_id, NPL):
        for n, p in enumerate(planets):
            if np.isclose(_ephem.period, p.period, rtol=0.01, atol=p.duration):
                planets[n] = p.update_ephemeris(_ephem)
    for n, p in enumerate(planets):
        record(golden, koi_id, n, 2, p)

    for n, p in enumerate(planets):
        has_error = p.ephemeris.error is not None

        # step 3 : clip range and set static references
        p.ephemeris.clip_range(T_MIN, T_MAX)
        p.ephemeris.set_static_references()
        p.update_ephemeris(p.ephemeris)
        record(golden, koi_id, n, 3, p)

        # step 4 : OMC update with full interpolation
        if has_error:
            omc_update(p, full=True)
            record(golden, koi_id, n, 4, p)

        # step 5 : quality flags and removal
        p.ephemeris.quality = p.ephemeris.index % 7 != 3
        p.ephemeris = p.ephemeris.remove_poor_quality_transits()
        record(golden, koi_id, n, 5, p)

        # step 6 : in-place transit time update
        if has_error:
            q = p.ephemeris.index % 5 != 0
            p.ephemeris.ttime[q] = p.ephemeris.ttime[q] + 2e-4 * np.cos(p.ephemeris.index[q])
            p.ephemeris.error[q] = p.ephemeris.error[q] * 1.1
            p.update_ephemeris(p.ephemeris)
            record(golden, koi_id, n, 6, p)

        # step 7 : additive transit time update
        theta = 1e-3 * np.sin(p.ephemeris.index)
        p.ephemeris.ttime += theta
        p.update_ephemeris(p.ephemeris)
        record(golden, koi_id, n, 7, p)

        # step 8 : OMC update without full interpolation
        if has_error:
            omc_update(p, full=False)
            record(golden, koi_id, n, 8, p)

np.savez(output_npz, **golden)
print(f"wrote {len(golden)} arrays to {output_npz}")
