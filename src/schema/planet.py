__all__ = ['Planet']

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from src.schema.ephemeris import Ephemeris
import warnings


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
            self.ephemeris = None
            warnings.warn("WARNING: Planet initiated without Ephemeris")


    def _from_dataframe(self, catalog, koi_id, planet_no):
        df = catalog.loc[catalog.koi_id == koi_id].sort_values(by='period').reset_index(drop=True)

        self.koi_id = koi_id
        self.kic_id = str(df.at[planet_no, 'kic_id'])
        self.period = float(df.at[planet_no, 'period'])
        self.epoch = float(df.at[planet_no, 'epoch'])
        self.depth = float(df.at[planet_no, 'depth']) * 1e-6       # ppm
        self.duration = float(df.at[planet_no, 'duration']) / 24.  # hrs --> days
        self.impact = float(df.at[planet_no, 'impact'])

        return self
    

    def predict_ephemeris(self, t_min, t_max):
        ttime = np.arange(self.epoch, t_max, self.period)
        index = np.array(np.round((ttime - self.epoch) / self.period), dtype=int)
        
        return Ephemeris(index, ttime, t_min=t_min, t_max=t_max)


    def update_ephemeris(self, ephemeris):
        if not np.isclose(self.period, ephemeris.period, rtol=0.1):
            raise ValueError(f"New period ({ephemeris.period:.1f}) differs from old period ({self.period:.1f}) by more than 10%")

        self.ephemeris = ephemeris
        self.period = self.ephemeris.period
        self.epoch = self.ephemeris.epoch

        return self