import os
import sys


import numpy as np
from astropy.units import UnitsWarning
from pathlib import Path

from alderaan.schema.litecurve import LiteCurve
from alderaan.schema.litecurve import KeplerLiteCurve


warnings.simplefilter('always', UserWarning)
warnings.filterwarnings(
    action='ignore', category=UnitsWarning, module='astropy'
)

base_path = Path(__file__).resolve().parents[1]
data_dir = os.path.join(base_path, 'alderaan/examples/data/MAST_downloads/')
kic_id = 8644288  # KOI-137 (Kepler-18)

# No quarters
litecurve = KeplerLiteCurve.load_kplr_pdcsap(data_dir, kic_id, 'long cadence')

# Integer quarter
litecurve = KeplerLiteCurve.load_kplr_pdcsap(data_dir, kic_id, 'long cadence', quarters=1)

# List quarters
litecurve = KeplerLiteCurve.load_kplr_pdcsap(data_dir, kic_id, 'long cadence', quarters=[1,2,3])

print("\npassing")