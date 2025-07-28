import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import astropy
import numpy as np
from src.schema.litecurve import LiteCurve
from src.schema.planet import Planet
from src.modules.detrend.detrend import Detrend
from src.utils.io import parse_koi_catalog, parse_holczer16_catalog
import warnings

warnings.simplefilter('always', UserWarning)
warnings.filterwarnings(
    action='ignore', category=astropy.units.UnitsWarning, module='astropy'
)

koi_id = 'K00148'

# load KOI catalog
filepath = '/data/user/gjgilbert/projects/alderaan/Catalogs/kepler_dr25_gaia_dr2_crossmatch.csv'
catalog = parse_koi_catalog(filepath, koi_id)

assert np.all(np.diff(catalog.period) > 0), "Planets should be ordered by ascending period"

NPL = int(catalog.npl[0])
kic_id = int(catalog.kic_id[0])
planets = [None]*NPL

print(f"{NPL} planets loaded for {koi_id}")

for i in range(NPL):
    planets[i] = Planet(catalog, koi_id, i)
    print(i, planets[i].period)

# load Holczer+2016 catalog
filepath = '/data/user/gjgilbert/projects/alderaan/Catalogs/holczer_2016_kepler_ttvs.txt'
holczer_ephemerides = parse_holczer16_catalog(filepath, koi_id, NPL)

print(f"{len(holczer_ephemerides)} ephemerides found in Holczer+2016")

for i, ephem in enumerate(holczer_ephemerides):
    print(i, ephem.period)

# match Holczer ephemerides to Planets
count = 0

for n, p in enumerate(planets):
    for ephem in holczer_ephemerides:
        match = np.isclose(ephem.period, p.period, rtol=0.01, atol=p.duration)

        if match:
            planets[n] = p.update_ephemeris(ephem)
            count += 1

print(f"{count} matching ephemerides found for {NPL} planets")

for n, p in enumerate(planets):
    print(n, p.period)


# load lightcurves
data_dir = '/data/user/gjgilbert/data/MAST_downloads/'
kic_id = catalog.kic_id[0]

litecurve_raw = LiteCurve().load_kplr_pdcsap(data_dir, kic_id, 'long cadence')
litecurve_list = litecurve_raw.split_quarters()

for i, lc in enumerate(litecurve_list):
    lc = lc.remove_flagged_cadences(bitmask='default')

litecurve_clean = LiteCurve().from_list(litecurve_list)

if np.min(litecurve_clean.time) < 0:
    raise ValueError("Lightcurve has negative timestamps...this will cause problems")

litecurves = litecurve_clean.split_quarters()

print(f"{len(litecurves)} litecurves loaded")


# detrend each quarter
for j, litecurve in enumerate(litecurves):
    print(f"Quarter {litecurve.quarter[0]}")

    detrend = Detrend(litecurve, planets)
    mask = detrend.make_transit_mask(rel_size=3.0, abs_size=2/24)
    mask = np.sum(mask, axis=0) > 0

    detrend.litecurve = detrend.clip_outliers(kernel_size=13,
                                              sigma_upper=5.0,
                                              sigma_lower=5.0,
                                              mask=mask
                                             )
    
    detrend.litecurve = detrend.clip_outliers(kernel_size=13,
                                              sigma_upper=5.,
                                              sigma_lower=1000.,
                                              mask=None
                                             )
        
print("passing")