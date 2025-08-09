import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import astropy
import numpy as np
from alderaan.schema.litecurve import LiteCurve
import warnings

data_dir = 'tests/testdata/MAST_downloads/'
kic_id = 5735762

litecurve = LiteCurve().from_kplr_pdcsap(data_dir, kic_id, 'long cadence')


print("passing")