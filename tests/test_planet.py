import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from alderaan.schema.ephemeris import Ephemeris
from alderaan.schema.planet import Planet
from alderaan.utils.io import parse_koi_catalog
import warnings

warnings.simplefilter('always', UserWarning)

filepath = '/data/user/gjgilbert/projects/alderaan/Catalogs/kepler_dr25_gaia_dr2_crossmatch.csv'
koi_id = 'K00148'
catalog = parse_koi_catalog(filepath, koi_id)

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

print("passing")