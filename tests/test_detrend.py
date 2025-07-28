import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import astropy
import numpy as np
from src.schema.litecurve import LiteCurve
from src.schema.planet import Planet
from src.modules.detrend.detrend import SimpleDetrender
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


# load lightcurves
data_dir = '/data/user/gjgilbert/data/MAST_downloads/'
kic_id = catalog.kic_id[0]

litecurve_raw = LiteCurve().load_kplr_pdcsap(data_dir, kic_id, 'long cadence')
litecurve_list = litecurve_raw.split_quarters()

for i, lc in enumerate(litecurve_list):
    lc = lc.remove_flagged_cadences(bitmask='default')

litecurve_clean = LiteCurve().from_list(litecurve_list)

t_min = litecurve_clean.time.min()
t_max = litecurve_clean.time.max()

if t_min < 0:
    raise ValueError("Lightcurve has negative timestamps...this will cause problems")

litecurves = litecurve_clean.split_quarters()

print(f"{len(litecurves)} litecurves loaded")

# update planet ephemerides
for n, p in enumerate(planets):
    if p.ephemeris is None:
        planets[n].ephemeris = p.predict_ephemeris(t_min, t_max)

# detrend each quarter
for j, litecurve in enumerate(litecurves):
    assert len(np.unique(litecurve.quarter)) == 1, "expected one quarter per litecurve"
    assert len(np.unique(litecurve.obsmode)) == 1, "expected one obsmode per litecurve"

detrenders = []
for j, litecurve in enumerate(litecurves):
    print(f"Quarter {litecurve.quarter[0]}")

    detrender = SimpleDetrender(litecurve, planets)
    mask = detrender.make_transit_mask(rel_size=3.0, abs_size=2/24)
    mask = np.sum(mask, axis=0) > 0

    npts_initial = len(detrender.litecurve.time)

    detrender.litecurve = detrender.clip_outliers(kernel_size=13,
                                                  sigma_upper=5,
                                                  sigma_lower=5,
                                                  mask=mask
                                                )
    
    detrender.litecurve = detrender.clip_outliers(kernel_size=13,
                                                  sigma_upper=5,
                                                  sigma_lower=1000,
                                                  mask=None
                                                 )
    
    npts_final = len(detrender.litecurve.time)

    print(f"  {npts_initial-npts_final} outliers rejected")

    oscillation_period = detrender.estimate_oscillation_period(min_period=1.0)

    print(f"  oscillation period = {oscillation_period:.1f} days")


print("passing")