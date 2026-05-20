import os
import sys

import numpy as np
from astropy.units import UnitsWarning
from pathlib import Path
import warnings

from alderaan.litecurve import LiteCurve
from alderaan.litecurve import KeplerLiteCurve
from alderaan.litecurve import TessLiteCurve
#from alderaan.litecurve import K2LiteCurve

warnings.simplefilter('always', UserWarning)
warnings.filterwarnings(
    action='ignore', category=UnitsWarning, module='astropy'
)

base_path = Path(__file__).resolve().parents[1]
data_dir = os.path.join(base_path, 'alderaan/examples/data/MAST_downloads/')

# KEPLER
# kic_id = 8644288  # KOI-137 (Kepler-18)

# # No quarters
# litecurve = KeplerLiteCurve.load_kplr_pdcsap(data_dir, kic_id, 'long cadence')

# # Integer quarter
# litecurve = KeplerLiteCurve.load_kplr_pdcsap(data_dir, kic_id, 'long cadence', quarters=1)

# # List quarters
# litecurve = KeplerLiteCurve.load_kplr_pdcsap(data_dir, kic_id, 'long cadence', quarters=[1,2,3])

# litecurves = litecurve.split_quarters()

# print(litecurves)


# K2
# epic_id = 211913977

# No campaign
#litecurve = K2LiteCurve.load_K2_everest(data_dir, epic_id, 'long cadence')

# Integer campaign
#litecurve = K2LiteCurve.load_K2_everest(data_dir, epic_id, 'long cadence', campaigns=5)

# List campaigns
#litecurve = K2LiteCurve.load_K2_everest(data_dir, epic_id, 'long cadence', campaigns=[5, 16, 18])


# TESS
tic_id = 120255950

# No sectors
tess_litecurve = TessLiteCurve.load_tess_pdcsap(tic_id, data_dir=data_dir)

# Integer sector
tess_litecurve = TessLiteCurve.load_tess_pdcsap(tic_id, data_dir=data_dir, sectors=40)

# List sectors
tess_litecurve = TessLiteCurve.load_tess_pdcsap(tic_id, data_dir=data_dir, sectors=[40, 41, 53, 74, 81])

print("\npassing")