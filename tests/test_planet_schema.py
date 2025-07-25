import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils.io import parse_catalog
from src.schema.planet import Planet

filepath = '/data/user/gjgilbert/projects/alderaan/Catalogs/kepler_dr25_gaia_dr2_crossmatch.csv'
koi_id = 'K00148'
catalog = parse_catalog(filepath, koi_id, mission='Kepler')

assert np.all(np.diff(catalog.period) > 0), "Planets should be ordered by ascending period"

NPL = int(catalog.npl[0])
planets = [None]*NPL

for n in range(NPL):
    planets[n] = Planet(catalog, n, 0., 1400.)

print("passed")