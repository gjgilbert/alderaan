__all__ = ['LiteCurve']

import glob
import numpy as np
import lightkurve as lk
from astropy.io import fits


class LiteCurve:
    def __init__(self):
        self.time = np.array([]).astype(float)
        self.flux = np.array([]).astype(float)
        self.error = np.array([]).astype(float)
        self.cadno = np.array([]).astype(int)
        self.quarter = np.array([]).astype(int)
        self.obsmode = np.array([]).astype(str)
        self.quality = np.array([]).astype(bool)


    def load_kplr_pdcsap(self, data_dir, kic_id, obsmode, quarters=None):
        """
        Load photometric data from Kepler Project PDCSAP Flux lightcurves
        The raw fits files must be pre-downloaded from MAST servers and stored locally

        Arguments:
            data_dir (str) : path to where data are stored
            kic_id (int) : Kepler Input Catalog (KIC) identification number
            obsmode (str) : 'short cadence' or 'long cadence'
            quarters (list) : optional, list of quarters to load
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
        self.quality = np.array(lklc.quality.value)

        return self
    

    def load_alderaan(self, data_dir, kic_id, obsmode):
        raise ValueError("Not yet implemented")
    
    
    def from_list(self, litecurve_list):
        for i, lc in enumerate(litecurve_list):
            for k in self.__dict__.keys():
                if type(self.__dict__[k]) is np.ndarray:
                    self.__setattr__(k, np.hstack([self.__dict__[k],lc.__dict__[k]]))

        return self
    

    def split_quarters(self):
        quarters = np.unique(self.quarter)

        litecurve_list = []
        for q in quarters:
            litecurve = LiteCurve()
            for k in litecurve.__dict__.keys():
                if type(litecurve.__dict__[k]) is np.ndarray:
                    litecurve.__setattr__(k, self.__dict__[k][self.quarter == q])
            litecurve_list.append(litecurve)

        return litecurve_list
    

    def remove_flagged_cadences(self, bitmask='default'):
        qmask = lk.KeplerQualityFlags.create_quality_mask(
            self.quality, bitmask=bitmask
        )
        for k in self.__dict__.keys():
            if type(self.__dict__[k]) is np.ndarray:
                self.__setattr__(k, self.__dict__[k][qmask])

        return self
    