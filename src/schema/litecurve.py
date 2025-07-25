import glob
import numpy as np
import lightkurve as lk
from astropy.io import fits
from astropy.stats import sigma_clip, mad_std
from scipy.signal import medfilt as median_filter
from scipy.signal import savgol_filter

__all__ = ["LiteCurve"]

class LiteCurve:
    def __init__(self):
        self.time = np.array([]).astype(float)
        self.flux = np.array([]).astype(float)
        self.error = np.array([]).astype(float)
        self.cadno = np.array([]).astype(int)
        self.quarter = np.array([]).astype(int)
        self.quality = np.array([]).astype(bool)


    def load_kplr_pdcsap(self, 
                         data_dir, 
                         kic_id, 
                         obsmode, 
                         quarters=None
                        ):
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

        # set ALDERAAN schema attributes
        self.time = np.array(lklc.time.value, dtype=float)
        self.flux = np.array(lklc.flux.value, dtype=float)
        self.error = np.array(lklc.flux_err.value, dtype=float)
        self.cadno = np.array(lklc.cadenceno.value, dtype=int)
        self.quarter = np.array(lklc.quarter, dtype=int)
        self.season = np.array(lklc.season, dtype=int)
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
    

    def clip_outliers(self, 
                      kernel_size, 
                      sigma_upper, 
                      sigma_lower, 
                      mask=None,
                      trend=None
                      ):
        """
        Sigma-clip outliers using astropy.stats.sigma_clip() and a median filtered trend
        Applies an additional iteration wrapper to allow for masked cadences

        Arguments:
            kernel_size : int
                size of the median filter kernel
            sigma_upper : float
                upper sigmga clipping threshold
            sigma_lower : float
                lower sigma clipping threshold
            mask : array-like, bool (optional)
                do not reject cadences within masked regions; useful for protecting transits
            trend : array-like, float (optional)
                precomputed model for the trend, e.g. a Keplerian transit lightcurve
        """
        if mask is None:
            mask = np.zeros(len(self.time), dtype=bool)
        
        if trend is not None:
            raise ValueError("Not yet configured for trend kwarg")

        loop = True
        count = 0
        while loop:
            # first pass: try median filter
            if loop:
                smoothed = median_filter(self.flux, kernel_size=kernel_size)

                bad = sigma_clip(self.flux - smoothed,
                                 sigma_upper=sigma_upper,
                                 sigma_lower=sigma_lower,
                                 stdfunc=mad_std
                                ).mask
                bad = bad * ~mask

            # second pass: try savgol filter
            # if over 1% of points were flagged with median filter
            if np.sum(bad)/len(bad) > 0.01:
                smoothed = savgol_filter(self.flux,
                                         window_length=kernel_size,
                                         polyorder=2
                                        )
                bad = sigma_clip(self.flux - smoothed,
                                 sigma_upper=sigma_upper,
                                 sigma_lower=sigma_lower,
                                 stdfunc=mad_std
                                ).mask
                bad = bad * ~mask

            # third pass: skip outlier rejection
            if np.sum(bad) / len(bad) > 0.01:
                bad = np.zeros_like(mask)

            # set attributes
            for k in self.__dict__.keys():
                if type(self.__dict__[k]) is np.ndarray:
                    self.__setattr__(k, self.__dict__[k][~bad])

            # update mask and iterate
            mask = mask[~bad]

            if np.sum(bad) == 0:
                loop = False
            else:
                count += 1

            if count >= 3:
                loop = False

        return self