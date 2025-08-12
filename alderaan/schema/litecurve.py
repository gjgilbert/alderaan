__all__ = ['LiteCurve']

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from astropy.io import fits
import glob
import lightkurve as lk
import numpy as np
from alderaan.constants import kepler_lcit, kepler_scit


class KeplerLiteCurve(LiteCurve):
    
    # def __init__(self):
    def __init__(self, data_dir, target_id, obsmode, quarters=None):
        
        super()

        self.quarter = visit.copy()
        # delete self.visit

        super().__init__()
        self.load_kplr_pdcsap(data_dir, target_id, obsmode, visits=self.quarter)


    def split_quarters(self):
        return super().split_visits()

    @classmethod
    def load_kplr_pdcsap(cls, data_dir, target_id, obsmode, visits=None):
        """
        Load photometric data from Kepler Project PDCSAP Flux lightcurves
        The raw fits files must be pre-downloaded from MAST servers and stored locally
        
        This function performs minimal detrending steps
         * remove_nans()
         * normalize()
                
        Args:
            data_dir (str) : path to where data are stored
            target_id (int) : KIC number
            obsmode (str) : 'short cadence' or 'long cadence'
            visits (list) : optional, list of visits (Kepler quarters) to load.
        Returns:
            LiteCurve : self
        """

        # create instance of litecurve
        lc_instance = cls()
        lc_instance.mission = "Kepler"

        # sanitize inputs
        if visits is None:
            visits = np.arange(18, dtype=int) # hard coded for Kepler
        if isinstance(visits, int):
            visits = [visits]

        # load the raw MAST files using lightcurve
        mast_files = glob.glob(data_dir + f"kplr{target_id:09d}*.fits") # hard-coded for Kepler
        mast_files.sort()
        
        mast_data_list = []
        for i, mf in enumerate(mast_files):
            with fits.open(mf) as hdu_list:
                if hdu_list[0].header["OBSMODE"] == obsmode and np.isin(
                    hdu_list[0].header["QUARTER"], visits # hard coded for Kepler
                ):
                    mast_data_list.append(lk.read(mf))

        lk_col_raw = lk.LightCurveCollection(mast_data_list)

        # clean up the Collection data structure
        visits = []
        for lkc in lk_col_raw:
            visits.append(lkc.quarter) # hard coded for Kepler

        lk_col_clean = []
        for v in np.unique(visits):
            lkc_list = []
            cadno = []

            for lkc in lk_col_raw:
                if (lkc.quarter == v) * (lkc.targetid == target_id): # hard coded for kepler
                    lkc_list.append(lkc)
                    cadno.append(lkc.cadenceno.min())

            order = np.argsort(cadno)
            lkc_list = [lkc_list[j] for j in order]

            # lk.stitch() also normalizes the lightkurves
            lkc = lk.LightCurveCollection(lkc_list).stitch().remove_nans()
            
            lkc.quarter = lkc.quarter*np.ones(len(lkc.time), dtype='int') # hard coded for kepler
            lkc.season = lkc.quarter % 4 # hard coded for Kepler
            
            lk_col_clean.append(lkc)

        lk_col_clean = lk.LightCurveCollection(lk_col_clean)

        # stitch into a single LightCurve
        lklc = lk_col_clean.stitch()

        # set LiteCurve attributes
        lc_instance.time = np.array(lklc.time.value, dtype=float)
        lc_instance.flux = np.array(lklc.flux.value, dtype=float)
        lc_instance.error = np.array(lklc.flux_err.value, dtype=float)
        lc_instance.cadno = np.array(lklc.cadenceno.value, dtype=int)
        lc_instance.visit = np.array(lklc.quarter, dtype=int) # hard coded for Kepler
        lc_instance.obsmode = np.array([obsmode]*len(lc_instance.cadno), dtype=str)
        lc_instance.quality = np.array(lklc.quality.value, dtype=int)
        lc_instance.season = np.array(lklc.season, dtype=int)
        
        # remove cadences flagged by Kepler project pipeline
        lc_instance = lc_instance._remove_flagged_cadences(lklc.quality)

        return lc_instance
        



class LiteCurve:
    """LiteCurve
    """
    def __init__(self, *args, **kwargs):

        self._set_empty_attributes()

    
    
    @classmethod
    def _set_empty_attribute_arrays(cls):
        lc_instance = cls.__new__(cls)
        lc_instance.time = np.array([]).astype(float)
        lc_instance.flux = np.array([]).astype(float)
        lc_instance.error = np.array([]).astype(float)
        lc_instance.cadno = np.array([]).astype(int)
        lc_instance.visit = np.array([]).astype(int)
        lc_instance.obsmode = np.array([]).astype(str)
        lc_instance.quality = np.array([]).astype(bool)
        return lc_instance

    
    @classmethod
    def from_list(cls, litecurve_list):
        
        lc_instance = cls()
        lc_instance = lc_instance._set_empty_attribute_arrays()

        for i, lc in enumerate(litecurve_list):
            for k in lc_instance.__dict__.keys():
                if type(lc_instance.__dict__[k]) is np.ndarray:
                    lc_instance.__setattr__(k, np.hstack([lc_instance.__dict__[k],lc.__dict__[k]]))

        return lc_instance
    

    @classmethod
    def from_k2(cls, data_dir, target_id, obsmode, visits=None):
        raise NotImplementedError("Loading K2 data not yet implemented")
    
    
    @classmethod
    def from_tess(cls, data_dir, target_id, obsmode, visits=None):
        raise NotImplementedError("Loading TESS data not yet implemented")

        
    @classmethod
    def from_alderaan(cls, data_dir, target_id):
        raise NotImplementedError("Loading ALDERAAN files not yet implemented")
    
    
    def split_visits(self):
        visits = np.unique(self.visit)

        litecurve_list = []
        for v in visits:
            litecurve = LiteCurve()
            for k in self.__dict__.keys():
            # for k in litecurve.__dict__.keys():
                # if type(litecurve.__dict__[k]) is np.ndarray:
                if type(self.__dict__[k]) is np.ndarray:
                    litecurve.__setattr__(k, self.__dict__[k][self.visit == v])
            litecurve_list.append(litecurve)

        return litecurve_list
      

    def _remove_flagged_cadences(self, quality_flags, bitmask='default'):
        qmask = lk.KeplerQualityFlags.create_quality_mask(
            quality_flags, bitmask=bitmask
        )
        for k in self.__dict__.keys():
            if type(self.__dict__[k]) is np.ndarray:
                self.__setattr__(k, self.__dict__[k][qmask])

        self.quality = np.ones(len(self.time), dtype=bool)

        return self
