import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from src.schema.planet import Planet
from src.utils.io import parse_koi_catalog

filepath = '/data/user/gjgilbert/projects/alderaan/Catalogs/kepler_dr25_gaia_dr2_crossmatch.csv'
koi_id = 'K00148'
catalog = parse_koi_catalog(filepath, koi_id)

assert np.all(np.diff(catalog.period) > 0), "Planets should be ordered by ascending period"

NPL = int(catalog.npl[0])
planets = [None]*NPL

for n in range(NPL):
    planets[n] = Planet(catalog, koi_id, n, t_min=0., t_max=1400.)

    print(n, planets[n].period, planets[n].epoch)

print("passing")