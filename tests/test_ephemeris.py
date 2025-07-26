import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from src.schema.ephemeris import Ephemeris


period = 11.3
epoch = 6.4

index = np.array([1,2,6,7,9])
ttime = epoch + period*index + 0.1*np.random.normal(size=len(index))
error = 0.1*np.ones_like(ttime)

ephemeris = Ephemeris(index, ttime, error)

full_index, full_ttime = ephemeris.full_ephemeris()

print(ephemeris.period, ephemeris.epoch)
print(full_index, full_ttime)

print("passing")