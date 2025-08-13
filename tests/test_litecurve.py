import os
import sys

import numpy as np
from astropy.units import UnitsWarning
from pathlib import Path
import warnings

from alderaan.litecurve import LiteCurve
from alderaan.litecurve import KeplerLiteCurve
from alderaan.litecurve import K2LiteCurve


warnings.simplefilter('always', UserWarning)
warnings.filterwarnings(
    action='ignore', category=UnitsWarning, module='astropy'
)

base_path = Path(__file__).resolve().parents[1]
data_dir = os.path.join(base_path, 'alderaan/examples/data/MAST_downloads/')
kic_id = 8644288  # KOI-137 (Kepler-18)

# # No quarters
# litecurve = KeplerLiteCurve.load_kplr_pdcsap(data_dir, kic_id, 'long cadence')

# # Integer quarter
# litecurve = KeplerLiteCurve.load_kplr_pdcsap(data_dir, kic_id, 'long cadence', quarters=1)

# # List quarters
# litecurve = KeplerLiteCurve.load_kplr_pdcsap(data_dir, kic_id, 'long cadence', quarters=[1,2,3])

epic_id = 211913977

# K2: No quarters
litecurve = K2LiteCurve.load_K2_everest(data_dir, epic_id, 'long cadence')

print("\npassing")