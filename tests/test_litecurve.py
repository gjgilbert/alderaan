import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import astropy
import numpy as np
from alderaan.schema.litecurve import LiteCurve
import warnings

warnings.simplefilter('always', UserWarning)

# supress UnitsWarnings (this code doesn't use astropy units)
warnings.filterwarnings(
    action='ignore', category=astropy.units.UnitsWarning, module='astropy'
)

data_dir = '/data/user/gjgilbert/data/MAST_downloads/'
kic_id = 5735762

litecurve_raw = LiteCurve().load_kplr_pdcsap(data_dir, kic_id, 'long cadence')
litecurve_list = litecurve_raw.split_quarters()

for i, lc in enumerate(litecurve_list):
    lc = lc.remove_flagged_cadences(bitmask='default')

litecurve_clean = LiteCurve().from_list(litecurve_list)

if np.min(litecurve_clean.time) < 0:
    raise ValueError("Lightcurve has negative timestamps...this will cause problems")

print("passing")