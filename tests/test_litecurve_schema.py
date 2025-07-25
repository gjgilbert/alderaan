import os
import sys
import warnings

import astropy
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.schema.litecurve import LiteCurve

# supress UnitsWarnings (this code doesn't use astropy units)
warnings.filterwarnings(
    action="ignore", category=astropy.units.UnitsWarning, module="astropy"
)

data_dir = '/data/user/gjgilbert/data/MAST_downloads/'
kic_id = 5735762

litecurve_raw = LiteCurve().load_kplr_pdcsap(data_dir, kic_id, 'long cadence')
litecurve_list = litecurve_raw.split_quarters()

for i, lc in enumerate(litecurve_list):
    lc = lc.remove_flagged_cadences(bitmask='default')
    lc = lc.clip_outliers(kernel_size=13, sigma_upper=5, sigma_lower=1000)

litecurve_clean = LiteCurve().from_list(litecurve_list)