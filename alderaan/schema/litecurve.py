__all__ = ['LiteCurve']

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from astropy.io import fits
import glob
import lightkurve as lk
import numpy as np
from alderaan.constants import kepler_lcit, kepler_scit


class LiteCurve:
    """LiteCurve
    """
    def __init__(self, *args, **kwargs):
        if len(args) == 0:
            self = self._set_empty_attribute_arrays()
        
        elif len(args) == 1 and isinstance(args[0], list):
            if all([isinstance(lc, LiteCurve) for lc in args[0]]):
                self = self._from_list(*args, **kwargs)
            else:
                raise TypeError("Unexpected input types in list")
        
        elif (len(args) > 1) and isinstance(args[0], str):
            if 'data_source' not in kwargs:
                raise ValueError("Missing required keyword argmument 'data_source")
            else:
                data_source = kwargs.pop('data_source')

            if data_source == 'Kepler PDCSAP':
                self = self._from_kplr_pdcsap(*args, **kwargs)
            elif data_source == 'ALDERAAN':
                self = self._from_alderaan(*args, **kwargs)
            else:
                raise ValueError(f"Unsupported data_source: {data_source}")      
        
        else:
            raise TypeError("Unsupported init signature")


    def _set_empty_attribute_arrays(self):
        self.time = np.array([]).astype(float)
        self.flux = np.array([]).astype(float)
        self.error = np.array([]).astype(float)
        self.cadno = np.array([]).astype(int)
        self.quarter = np.array([]).astype(int)
        self.obsmode = np.array([]).astype(str)
        self.quality = np.array([]).astype(bool)

        return self    


    def _from_list(self, litecurve_list):
        self = self._set_empty_attribute_arrays()

        for i, lc in enumerate(litecurve_list):
            for k in self.__dict__.keys():
                if type(self.__dict__[k]) is np.ndarray:
                    self.__setattr__(k, np.hstack([self.__dict__[k],lc.__dict__[k]]))

        return self
    

    def _from_kplr_pdcsap(self, data_dir, kic_id, obsmode, quarters=None):
        """
        Load photometric data from Kepler Project PDCSAP Flux lightcurves
        The raw fits files must be pre-downloaded from MAST servers and stored locally
        
        This function performs minimal detrending steps
         * remove_nans()
         * normalize()
                
        Args:
            data_dir (str) : path to where data are stored
            kic_id (int) : Kepler Input Catalog (KIC) identification number
            obsmode (str) : 'short cadence' or 'long cadence'
            quarters (list) : optional, list of quarters to load

        Returns:
            LiteCurve : self
        """
        # sanitize inputs
        if quarters is None:
            quarters = np.arange(18, dtype=int)
        if isinstance(quarters, int):
            quarters = [quarters]

        # load the raw MAST files using lightcurve
        mast_files = glob.glob(data_dir + f"kplr{kic_id:09d}*.fits")
        mast_files.sort()
        
        mast_data_list = []
        for i, mf in enumerate(mast_files):
            with fits.open(mf) as hdu_list:
                if hdu_list[0].header["OBSMODE"] == obsmode and np.isin(
                    hdu_list[0].header["QUARTER"], quarters
                ):
                    mast_data_list.append(lk.read(mf))

        lk_col_raw = lk.LightCurveCollection(mast_data_list)

        # clean up the Collection data structure
        quarters = []
        for lkc in lk_col_raw:
            quarters.append(lkc.quarter)

        lk_col_clean = []
        for q in np.unique(quarters):
            lkc_list = []
            cadno = []

            for lkc in lk_col_raw:
                if (lkc.quarter == q) * (lkc.targetid == kic_id):
                    lkc_list.append(lkc)
                    cadno.append(lkc.cadenceno.min())

            order = np.argsort(cadno)
            lkc_list = [lkc_list[j] for j in order]

            # lk.stitch() also normalizes the lightkurves
            lkc = lk.LightCurveCollection(lkc_list).stitch().remove_nans()
            
            lkc.quarter = lkc.quarter*np.ones(len(lkc.time), dtype='int')
            lkc.season = lkc.quarter % 4
            
            lk_col_clean.append(lkc)

        lk_col_clean = lk.LightCurveCollection(lk_col_clean)

        # stitch into a single LightCurve
        lklc = lk_col_clean.stitch()

        # set LiteCurve attributes
        self.time = np.array(lklc.time.value, dtype=float)
        self.flux = np.array(lklc.flux.value, dtype=float)
        self.error = np.array(lklc.flux_err.value, dtype=float)
        self.cadno = np.array(lklc.cadenceno.value, dtype=int)
        self.quarter = np.array(lklc.quarter, dtype=int)
        self.season = np.array(lklc.season, dtype=int)
        self.obsmode = np.array([obsmode]*len(self.cadno), dtype=str)
        
        # remove cadences flagged by Kepler project pipeline
        self = self._remove_flagged_cadences(lklc.quality)

        return self
        

    def _from_alderaan(self, data_dir, target_id):
        raise NotImplementedError("Loading ALDERAAN files not yet implemented")
    
    
    def _remove_flagged_cadences(self, quality_flags, bitmask='default'):
        qmask = lk.KeplerQualityFlags.create_quality_mask(
            quality_flags, bitmask=bitmask
        )
        for k in self.__dict__.keys():
            if type(self.__dict__[k]) is np.ndarray:
                self.__setattr__(k, self.__dict__[k][qmask])

        self.quality = np.ones(len(self.time), dtype=bool)

        return self
    

    def split_quarters(self):
        """Split a single LiteCurve into a list of LiteCurves by quarter

        Returns:
            list : a list of LiteCurve objects
        """
        quarters = np.unique(self.quarter)

        litecurve_list = []
        for q in quarters:
            litecurve = LiteCurve()
            for k in litecurve.__dict__.keys():
                if type(litecurve.__dict__[k]) is np.ndarray:
                    litecurve.__setattr__(k, self.__dict__[k][self.quarter == q])
            litecurve_list.append(litecurve)

        return litecurve_list
    