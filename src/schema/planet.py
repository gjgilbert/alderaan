import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from src.schema.ephemeris import Ephemeris

__all__ = ['Planet']

class Planet:
    def __init__(self, 
                 dataframe,
                 koi_id,
                 planet_no, 
                 ephemeris=None, 
                 t_min=None,
                 t_max=None
                ):
        """
        Docstring
        """
        # read transit parameters from pandas dataframe
        self = self._from_dataframe(dataframe, planet_no)

        # set up ephemeris
        if ephemeris is not None:
            print("ephemeris")
            self.ephemeris = ephemeris

        elif (t_min is not None) & (t_max is not None):
            ttime = np.arange(self.epoch, t_max, self.period)
            index = np.array(np.round((ttime - self.epoch) / self.period), dtype=int)
            self.ephemeris = Ephemeris(index, ttime, t_min=t_min, t_max=t_max)

        else:
            raise ValueError("either an Ephemeris or (t_min, t_max) must be provided")

        # put epoch in range (t_min, t_min + period)
        #self.ephemeris.epoch = self.ephemeris._adjust_epoch(t_min)

        # ensure self-consistent ephemeris
        #self = self.update_ephemeris(self.ephemeris)


    def _from_dataframe(self, df, planet_no):
        self.period = float(df.at[planet_no, 'period'])
        self.epoch = float(df.at[planet_no, 'epoch'])
        self.depth = float(df.at[planet_no, 'depth'])
        self.duration = float(df.at[planet_no, 'duration'])
        self.impact = float(df.at[planet_no, 'impact'])

        return self
    

    def initialize_ephemeris(self, t_min, t_max):
        pass
    
    def update_ephemeris(self, ephemeris):
        self.ephemeris = ephemeris
        self.period = self.ephemeris.period
        self.epoch = self.ephemeris.epoch

        return self