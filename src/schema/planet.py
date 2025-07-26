import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from src.schema.ephemeris import Ephemeris
import warnings

__all__ = ['Planet']

class Planet:
    def __init__(self, 
                 catalog,
                 koi_id,
                 planet_no, 
                 ephemeris=None
                ):
        """
        Docstring
        """
        # read transit parameters from pandas dataframe
        self = self._from_dataframe(catalog, koi_id, planet_no)

        # set up ephemeris
        if ephemeris is not None:
            self = self.update_ephemeris(ephemeris)

        else:
            warnings.warn("WARNING: Planet initiated without Ephemeris")


    def _from_dataframe(self, catalog, koi_id, planet_no):
        df = catalog.loc[catalog.koi_id == koi_id].sort_values(by='period').reset_index(drop=True)

        self.period = float(df.at[planet_no, 'period'])
        self.epoch = float(df.at[planet_no, 'epoch'])
        self.depth = float(df.at[planet_no, 'depth'])
        self.duration = float(df.at[planet_no, 'duration'])
        self.impact = float(df.at[planet_no, 'impact'])

        return self
    

    def predict_ephemeris(self, t_min, t_max):
        ttime = np.arange(self.epoch, t_max, self.period)
        index = np.array(np.round((ttime - self.epoch) / self.period), dtype=int)
        
        return Ephemeris(index, ttime, t_min=t_min, t_max=t_max)


    def update_ephemeris(self, ephemeris):
        self.ephemeris = ephemeris
        self.period = self.ephemeris.period
        self.epoch = self.ephemeris.epoch

        return self