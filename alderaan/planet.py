__all__ = ['Planet']

import numpy as np
from alderaan.ephemeris import Ephemeris
import warnings


class Planet:
    """Planet
    """
    def __init__(self, 
                 catalog,
                 target,
                 planet_no, 
                 ephemeris=None
                ):
        # read transit parameters from pandas dataframe
        self = self._from_dataframe(catalog, target, planet_no)

        # set up ephemeris
        if ephemeris is not None:
            self = self.update_ephemeris(ephemeris)
        else:
            self.ephemeris = None
            warnings.warn("WARNING: Planet initiated without Ephemeris")


    def _from_dataframe(self, catalog, target, planet_no):
        # Detect mission from catalog columns and filter accordingly
        if 'koi_id' in catalog.columns:
            # Kepler catalog
            df = catalog.loc[catalog.koi_id == target].sort_values(by='period').reset_index(drop=True)
            self.koi_id = target
            self.kic_id = str(df.at[planet_no, 'kic_id'])
            self.target_id = target
            self.star_id = self.kic_id
        elif 'epic_id' in catalog.columns:
            # K2 catalog
            target_id = target.split('-')[-1]
            df = catalog.loc[catalog.epic_id == target_id].sort_values(by='period').reset_index(drop=True)
            self.epic_id = target
            # self.cand_id = str(df.at[planet_no, 'epic_id'])
            self.target_id = target
            self.star_id = self.epic_id
        elif 'toi_id' in catalog.columns:
            # TESS catalog
            df = catalog.loc[catalog.toi_id == target].sort_values(by='period').reset_index(drop=True)
            self.toi_id = target
            self.tic_id = str(df.at[planet_no, 'tic_id'])
            self.target_id = target
            self.star_id = self.tic_id
        else:
            raise ValueError("Catalog must contain either 'koi_id' or 'epic_id' or 'toi_id' column")

        self.period = np.float64(df.at[planet_no, 'period'])
        self.epoch = np.float64(df.at[planet_no, 'epoch'])
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
