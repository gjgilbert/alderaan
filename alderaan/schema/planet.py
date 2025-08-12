__all__ = ['Planet']

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from alderaan.schema.ephemeris import Ephemeris
import warnings


class Planet:
    """Planet
    """
    def __init__(self, 
                 catalog,
                 koi_id,
                 planet_no, 
                 ephemeris=None
                ):
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
    

    def update_ephemeris(self, ephemeris):
        """
        Update ephemeris and corresponding attributes (period & epoch)

        Args:
          ephemeris (Ephemeris)
        
        Returns:
          Planet : self
        """
        if not np.isclose(self.period, ephemeris.period, rtol=0.1):
            raise ValueError(f"New period ({ephemeris.period:.6f}) differs from old period ({self.period:.6f}) by more than 10%")

        self.ephemeris = ephemeris.update_period_and_epoch()
        self.period = self.ephemeris.period
        self.epoch = self.ephemeris.epoch

        return self