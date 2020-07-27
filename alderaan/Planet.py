import numpy as np
import scipy.optimize as op
import scipy.signal as sig
from   scipy import stats
import astropy
import sys
import os
import warnings

from .constants import *

__all__ = ['Planet']

class Planet:
    def __init__(self, epoch=None, period=None, depth=None, duration=None, \
                 index=None, tts=None, quality=None, overlap=None, dtype=None):


        self.epoch            = epoch            # reference transit time in range (0, period)
        self.period           = period           # orbital period
        self.depth            = depth            # transit depth
        self.duration         = duration         # transit duration

        self.index            = index            # index of each transit in range (0,1600) -- Kepler baseline
        self.tts              = tts              # all midtransit times in range (0,1600) -- Kepler baseline
        self.quality          = quality          # boolean flag per transit; True=good
        self.overlap          = overlap          # 
        self.dtype            = dtype            #
        
        ###