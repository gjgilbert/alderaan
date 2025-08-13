import os
import sys

import numpy as np
import warnings
from alderaan.ephemeris import Ephemeris

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

print("\npassing")