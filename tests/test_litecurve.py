import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import astropy
import numpy as np
from src.schema.litecurve import LiteCurve
import warnings

warnings.simplefilter('always', UserWarning)

# supress UnitsWarnings (this code doesn't use astropy units)
warnings.filterwarnings(
    action='ignore', category=astropy.units.UnitsWarning, module='astropy'
)

data_dir = 'testdata/'
kic_id = 5735762

litecurve = LiteCurve()._from_kplr_pdcsap(data_dir, kic_id, 'long cadence')

if np.min(litecurve.time) < 0:
    raise ValueError("Lightcurve has negative timestamps...this will cause problems")

print("passing")