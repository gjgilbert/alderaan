import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import astropy
import numpy as np
from alderaan.schema.litecurve import LiteCurve
from alderaan.schema.litecurve import KeplerLiteCurve
import warnings

warnings.simplefilter('always', UserWarning)

# supress UnitsWarnings (this code doesn't use astropy units)
warnings.filterwarnings(
    action='ignore', category=astropy.units.UnitsWarning, module='astropy'
)

data_dir = 'testdata/MAST_downloads/'
kic_id = 5735762

# No quarters
litecurve = KeplerLiteCurve.load_kplr_pdcsap(data_dir, kic_id, 'long cadence')

# Integer quarter
litecurve = KeplerLiteCurve.load_kplr_pdcsap(data_dir, kic_id, 'long cadence', quarters=1)

# List quarters
litecurve = KeplerLiteCurve.load_kplr_pdcsap(data_dir, kic_id, 'long cadence', quarters=[1,2,3])

if np.min(litecurve.time) < 0:
    raise ValueError("Lightcurve has negative timestamps...this will cause problems")

print("passing")