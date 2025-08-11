import os
import sys

from astropy.units import UnitsWarning
from pathlib import Path
import warnings
from alderaan.litecurve import LiteCurve


warnings.simplefilter('always', UserWarning)
warnings.filterwarnings(
    action='ignore', category=UnitsWarning, module='astropy'
)

base_path = Path(__file__).resolve().parents[1]
data_dir = os.path.join(base_path, 'alderaan/examples/data/MAST_downloads/')
kic_id = 5735762

litecurve = LiteCurve().from_kplr_pdcsap(data_dir, kic_id, 'long cadence')

print("\npassing")