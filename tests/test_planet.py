import os
import sys

import numpy as np
from pathlib import Path
import warnings
from alderaan.ephemeris import Ephemeris
from alderaan.planet import Planet
from alderaan.utils.io import parse_koi_catalog

warnings.simplefilter('always', UserWarning)


base_path = Path(__file__).resolve().parents[1]
catalog_csv = os.path.join(base_path, 'alderaan/examples/catalogs/kepler_dr25_gaia_dr2_crossmatch.csv')

koi_id = 'K00137'   # Kepler-18
catalog = parse_koi_catalog(catalog_csv, koi_id)

assert np.all(np.diff(catalog.period) > 0), "Planets should be ordered by ascending period"

t_min = 0.
t_max = 1400.

NPL = int(catalog.npl[0])
planets = [None]*NPL

for n in range(NPL):
    planets[n] = Planet(catalog, koi_id, n)

for n, p in enumerate(planets):
    ephemeris = Ephemeris(period=p.period, epoch=p.epoch, t_min=t_min, t_max=t_max)
    p = p.update_ephemeris(ephemeris)

print("\npassing")