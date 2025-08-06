import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from alderaan.schema.ephemeris import Ephemeris, WarpEphemeris
import warnings

warnings.simplefilter('always', UserWarning)


period = 11.3
epoch = 6.4

index = np.array([1,2,6,7,9])
ttime = epoch + period*index + 0.1*np.random.normal(size=len(index))
error = 0.1*np.ones_like(ttime)

# TTV init
ephemeris = Ephemeris(index=index, ttime=ttime, error=error)

print(ephemeris.period, ephemeris.epoch)
print(len(ephemeris.index), len(ephemeris.ttime))

# linear init
ephemeris = Ephemeris(period=period, epoch=epoch, t_min=0., t_max=1400.)

print(ephemeris.period, ephemeris.epoch)
print(len(ephemeris.index), len(ephemeris.ttime))

# WarpEphemeris
ephemeris = WarpEphemeris(index=index, ttime=ttime)

print(ephemeris.period, ephemeris.epoch)
print(ephemeris._static_period, ephemeris._static_epoch)

print("passing")