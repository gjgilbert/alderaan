import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from src.utils.io import parse_koi_catalog, parse_holczer16_catalog
from src.schema.planet import Planet
from src.schema.ephemeris import Ephemeris
from src.modules.omc.omc import OMC
import warnings

warnings.simplefilter('always', UserWarning)

koi_id = 'K00148'

# load KOI catalog
filepath = '/data/user/gjgilbert/projects/alderaan/Catalogs/kepler_dr25_gaia_dr2_crossmatch.csv'
catalog = parse_koi_catalog(filepath, koi_id)

assert np.all(np.diff(catalog.period) > 0), "Planets should be ordered by ascending period"

NPL = int(catalog.npl[0])
planets = [None]*int(catalog.npl[0])

print(f"{NPL} planets loaded for {koi_id}")

for n in range(NPL):
    planets[n] = Planet(catalog, koi_id, n)
    print(n, planets[n].period)


# load Holczer+2016 catalog
filepath = '/data/user/gjgilbert/projects/alderaan/Catalogs/holczer_2016_kepler_ttvs.txt'
holczer_ephemerides = parse_holczer16_catalog(filepath, koi_id, NPL)

print(f"{len(holczer_ephemerides)} ephemerides found in Holczer+2016")

for n, ephem in enumerate(holczer_ephemerides):
    print(n, ephem.period)


# match Holczer ephemerides to Planets
count = 0

for n, p in enumerate(planets):
    for ephem in holczer_ephemerides:
        match = np.isclose(ephem.period, p.period, rtol=0.1, atol=p.duration)

        if match:
            planets[n] = p.update_ephemeris(ephem)
            count += 1

print(f"{count} matching ephemerides found for {NPL} planets")

for n, p in enumerate(planets):
    print(n, p.period)

for n, p in enumerate(planets):
    omc = OMC(p.ephemeris)

print("passing")