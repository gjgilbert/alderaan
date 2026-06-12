__all__ = ['LiteCurve', 'KeplerLiteCurve', 'K2LiteCurve', 'TessLiteCurve']

from astropy.io import fits
import glob
import lightkurve as lk
import numpy as np
import os
from alderaan.constants import kepler_lcit, kepler_scit, tess_2min_it, tess_20sec_it


class LiteCurve:
    """
    Base class for inheritance
    """
    def __init__(self, *args, **kwargs):
        self.time = np.array([]).astype(float)
        self.flux = np.array([]).astype(float)
        self.error = np.array([]).astype(float)
        self.cadno = np.array([]).astype(int)
        self.visit = np.array([]).astype(int)
        self.obsmode = np.array([]).astype(str)
        self.quality = np.array([]).astype(bool)

    
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
        lc_instance.season = np.array([]).astype(int)
        return lc_instance

    
    @classmethod
    def from_list(cls, litecurve_list):
        
        """Concatenate a list of LiteCurve objects into a single LiteCurve instance."""
        
        lc_instance = cls()
        lc_instance = lc_instance._set_empty_attribute_arrays()

        for i, lc in enumerate(litecurve_list):
            for k in lc_instance.__dict__.keys():
                if type(lc_instance.__dict__[k]) is np.ndarray:
                    lc_instance.__setattr__(k, np.hstack([lc_instance.__dict__[k],lc.__dict__[k]]))

        return lc_instance
    

    def split_visits(self):
        visits = np.unique(self.visit)

        litecurve_list = []
        for v in visits:
            litecurve = LiteCurve()
            for k in self.__dict__.keys():
                if type(self.__dict__[k]) is np.ndarray:
                    litecurve.__setattr__(k, self.__dict__[k][self.visit == v])
            litecurve_list.append(litecurve)

        return litecurve_list
      



class KeplerLiteCurve(LiteCurve):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def _remove_flagged_cadences(self, quality_flags, bitmask='default'):
        qmask = lk.KeplerQualityFlags.create_quality_mask(
            quality_flags, bitmask=bitmask
        )
        for k in self.__dict__.keys():
            if type(self.__dict__[k]) is np.ndarray:
                self.__setattr__(k, self.__dict__[k][qmask])

        self.quality = np.ones(len(self.time), dtype=bool)

        return self
    

    def split_quarters(self, quarters=None):
        """
        Kepler wrapper for split_visits().
        
        Args:
            quarters (list) : optional, subset of quarters to return. If None, returns all.
        Returns:
            list of LiteCurve : one per quarter
        """
        litecurve_list = self.split_visits()
        if quarters is not None:
            litecurve_list = [lc for lc in litecurve_list if lc.quarter[0] in quarters]
        return litecurve_list
    

    @classmethod
    def load_kplr_pdcsap(cls, data_dir, target_id, obsmode, quarters=None):
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
            quarters (list) : optional, list of quarters (Kepler quarters) to load.
        Returns:
            KeplerLiteCurve : instance
        """

        # create instance of litecurve
        lc_instance = cls.__new__(cls)
        super(cls, lc_instance).__init__()  # initialize base attributes
        lc_instance.mission = "Kepler"

        # sanitize inputs
        if quarters is None:
            quarters = np.arange(18, dtype=int) # hard coded for Kepler
        if isinstance(quarters, int):
            quarters = [quarters]

        # load the raw MAST files using lightcurve
        mast_files = glob.glob(data_dir + f"kplr{target_id:09d}*.fits") # hard-coded for Kepler
        mast_files.sort()
        
        mast_data_list = []
        for i, mf in enumerate(mast_files):
            with fits.open(mf) as hdu_list:
                if hdu_list[0].header["OBSMODE"] == obsmode and np.isin(
                    hdu_list[0].header["QUARTER"], quarters # hard coded for Kepler
                    ):
                    mast_data_list.append(lk.read(mf))

        lk_col_raw = lk.LightCurveCollection(mast_data_list)

        # clean up the Collection data structure
        quarters = []
        for lkc in lk_col_raw:
            quarters.append(lkc.quarter) # hard coded for Kepler

        lk_col_clean = []
        for q in np.unique(quarters):
            lkc_list = []
            cadno = []

            for lkc in lk_col_raw:
                if (lkc.quarter == q) * (lkc.targetid == target_id): # hard coded for kepler
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

    def to_fits(self, target, filename, cadence):
        """
        Save LiteCurve object as a fits file

        Args:
            target (str) : name of target
            filename (str) : path to save the fits file to
            cadence (str) : cadence of lightcurve data; "LONG" or "SHORT" "
        """
        # make primary HDU
        primary_hdu = fits.PrimaryHDU()

        header = primary_hdu.header

        header["TARGET"] = target
        header["CADENCE"] = cadence

        primary_hdu.header = header

        # add it to HDU list
        hdulist = []
        hdulist.append(primary_hdu)

        hdulist.append(fits.ImageHDU(self.time, name="TIME"))
        hdulist.append(fits.ImageHDU(self.flux, name="FLUX"))
        hdulist.append(fits.ImageHDU(self.error, name="ERROR"))
        hdulist.append(fits.ImageHDU(self.cadno, name="CADNO"))
        hdulist.append(fits.ImageHDU(self.visit, name="visit"))

        hdulist = fits.HDUList(hdulist)
        hdulist.writeto(filename, overwrite=True)

        return None
    


class K2LiteCurve(LiteCurve):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def _remove_flagged_cadences(self, quality_flags, bitmask='default'):
        qmask = lk.K2QualityFlags.create_quality_mask(
            quality_flags, bitmask=bitmask
        )
        for k in self.__dict__.keys():
            if type(self.__dict__[k]) is np.ndarray:
                self.__setattr__(k, self.__dict__[k][qmask])

        self.quality = np.ones(len(self.time), dtype=bool)

        return self
    

    def split_quarters(self, quarters=None):
        """
        Kepler wrapper for split_visits().
        
        Args:
            quarters (list) : optional, subset of quarters to return. If None, returns all.
        Returns:
            list of LiteCurve : one per quarter
        """
        litecurve_list = self.split_visits()
        if quarters is not None:
            litecurve_list = [lc for lc in litecurve_list if lc.quarter[0] in quarters]
        return litecurve_list
    

    @classmethod
    def load_kplr_pdcsap(cls, data_dir, target_id, obsmode, quarters=None):
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
            quarters (list) : optional, list of quarters (Kepler quarters) to load.
        Returns:
            KeplerLiteCurve : instance
        """

        # create instance of litecurve
        lc_instance = cls.__new__(cls)
        super(cls, lc_instance).__init__()  # initialize base attributes
        lc_instance.mission = "Kepler"

        # sanitize inputs
        if quarters is None:
            quarters = np.arange(18, dtype=int) # hard coded for Kepler
        if isinstance(quarters, int):
            quarters = [quarters]

        # load the raw MAST files using lightcurve
        mast_files = glob.glob(data_dir + f"kplr{target_id:09d}*.fits") # hard-coded for Kepler
        mast_files.sort()
        
        mast_data_list = []
        for i, mf in enumerate(mast_files):
            with fits.open(mf) as hdu_list:
                if hdu_list[0].header["OBSMODE"] == obsmode and np.isin(
                    hdu_list[0].header["QUARTER"], quarters # hard coded for Kepler
                    ):
                    mast_data_list.append(lk.read(mf))

        lk_col_raw = lk.LightCurveCollection(mast_data_list)

        # clean up the Collection data structure
        quarters = []
        for lkc in lk_col_raw:
            quarters.append(lkc.quarter) # hard coded for Kepler

        lk_col_clean = []
        for q in np.unique(quarters):
            lkc_list = []
            cadno = []

            for lkc in lk_col_raw:
                if (lkc.quarter == q) * (lkc.targetid == target_id): # hard coded for kepler
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

    def to_fits(self, target, filename, cadence):
        """
        Save LiteCurve object as a fits file

        Args:
            target (str) : name of target
            filename (str) : path to save the fits file to
            cadence (str) : cadence of lightcurve data; "LONG" or "SHORT" "
        """
        # make primary HDU
        primary_hdu = fits.PrimaryHDU()

        header = primary_hdu.header

        header["TARGET"] = target
        header["CADENCE"] = cadence

        primary_hdu.header = header

        # add it to HDU list
        hdulist = []
        hdulist.append(primary_hdu)

        hdulist.append(fits.ImageHDU(self.time, name="TIME"))
        hdulist.append(fits.ImageHDU(self.flux, name="FLUX"))
        hdulist.append(fits.ImageHDU(self.error, name="ERROR"))
        hdulist.append(fits.ImageHDU(self.cadno, name="CADNO"))
        hdulist.append(fits.ImageHDU(self.visit, name="visit"))

        hdulist = fits.HDUList(hdulist)
        hdulist.writeto(filename, overwrite=True)

        return None
    


class TessLiteCurve(LiteCurve):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)



    def split_sectors(self, sectors=None):
        """
        TESS wrapper for split_visits().
        
        Args:
            sectors (list) : optional, subset of sectors to return. If None, returns all.
        Returns:
            list of LiteCurve : one per sector
        """
        litecurve_list = self.split_visits()
        if sectors is not None:
            litecurve_list = [lc for lc in litecurve_list if lc.sector[0] in sectors]
        return litecurve_list
    


    def _remove_flagged_cadences(self, quality_flags, bitmask='default'):
        """Override to use TESS quality flags instead of Kepler."""
        qmask = lk.TessQualityFlags.create_quality_mask(
            quality_flags, bitmask=bitmask
        )
        for k in self.__dict__.keys():
            if type(self.__dict__[k]) is np.ndarray:
                self.__setattr__(k, self.__dict__[k][qmask])

        self.quality = np.ones(len(self.time), dtype=bool)

        return self


    @classmethod
    def load_tess_pdcsap(cls, tic_id, data_dir=None, sectors=None,
                         prefer_short_cadence=False):
        """
        Load photometric data from TESS PDCSAP Flux lightcurves using lightkurve.

        Uses lightkurve.search_lightcurve() to find and download TESS data.
        Downloaded FITS files are cached by lightkurve (~/.lightkurve/cache/).
        If data_dir is provided, copies are also saved there for reproducibility.

        When multiple cadence modes exist for the same sector (e.g. 20-sec
        and 2-min), only one is kept to avoid overlapping time arrays.
        By default the longest cadence is kept (faster runtime).  Set
        prefer_short_cadence=True to keep the shortest cadence instead.

        This function performs minimal detrending steps:
         * remove_nans()
         * normalize()

        Args:
            tic_id (int) : TIC number
            data_dir (str) : optional path to cache downloaded FITS files
            sectors (list) : optional, list of TESS sectors to load.
                             If None, all available sectors are loaded.
            prefer_short_cadence (bool) : if True, keep the shortest cadence
                when multiple cadence modes exist for the same sector.
                Default False (keep longest cadence for faster runtime).
        Returns:
            TessLiteCurve : instance
        """
        # create instance of litecurve
        lc_instance = cls.__new__(cls)
        super(cls, lc_instance).__init__()  # initialize base attributes
        lc_instance.mission = "TESS"

        # Search for TESS lightcurves via lightkurve
        search_result = lk.search_lightcurve(
            f"TIC {tic_id}",
            mission="TESS",
            author="SPOC",
        )

        if len(search_result) == 0:
            raise ValueError(f"No TESS SPOC lightcurves found for TIC {tic_id}")

        # Filter by sectors if requested
        if sectors is not None:
            if isinstance(sectors, int):
                sectors = [sectors]
            # search_result.table has a 'sequence_number' column for sector
            mask = np.isin(search_result.table['sequence_number'], sectors)
            search_result = search_result[mask]
            if len(search_result) == 0:
                raise ValueError(
                    f"No TESS SPOC lightcurves found for TIC {tic_id} "
                    f"in sectors {sectors}"
                )

        # Download the lightcurves (lightkurve handles caching)
        lk_col_raw = search_result.download_all()

        # Optionally copy FITS files into data_dir for offline reproducibility
        if data_dir is not None:
            os.makedirs(data_dir, exist_ok=True)
            for lkc in lk_col_raw:
                if hasattr(lkc, 'filename') and lkc.filename is not None:
                    src = lkc.filename
                    dst = os.path.join(data_dir, os.path.basename(src))
                    if not os.path.exists(dst):
                        import shutil
                        shutil.copy2(src, dst)

        # Organize by sector and deduplicate cadence modes
        # TESS SPOC can deliver both 20-sec and 2-min cadence for the same
        # sector.  Keeping both would produce overlapping (unsorted) time
        # arrays that break the GP.  We keep only the longest-cadence product
        # per sector (faster runtime); users who want the short cadence can
        # pass prefer_short_cadence=True.
        sector_list = []
        exptime_list = []
        for lkc in lk_col_raw:
            # sector
            if hasattr(lkc, 'sector'):
                sector_list.append(int(lkc.sector))
            elif 'SECTOR' in lkc.meta:
                sector_list.append(int(lkc.meta['SECTOR']))
            else:
                raise ValueError("Cannot determine sector for lightcurve")
            # exposure time (TIMEDEL in days; fall back to median dt)
            timedel = lkc.meta.get('TIMEDEL', None)
            if timedel is None and len(lkc.time) > 1:
                timedel = float(np.median(np.diff(lkc.time.value)))
            exptime_list.append(timedel if timedel is not None else 0.0)

        sector_list = np.array(sector_list)
        exptime_list = np.array(exptime_list)

        lk_col_clean = []
        for s in np.unique(sector_list):
            in_sector = (sector_list == s)
            sector_exptimes = exptime_list[in_sector]

            # If multiple cadence modes exist, keep only the longest cadence
            # (largest TIMEDEL) to avoid overlapping time arrays and to
            # improve runtime.  When prefer_short_cadence is True, keep the
            # shortest instead.
            if len(np.unique(np.round(sector_exptimes, 8))) > 1:
                if prefer_short_cadence:
                    target_exptime = sector_exptimes.min()
                else:
                    target_exptime = sector_exptimes.max()
                keep_mask = np.abs(sector_exptimes - target_exptime) < 1e-8
            else:
                keep_mask = np.ones(len(sector_exptimes), dtype=bool)

            # indices into lk_col_raw for this sector
            sector_indices = np.where(in_sector)[0]

            lkc_list = []
            cadno = []
            for idx, keep in zip(sector_indices, keep_mask):
                lkc = lk_col_raw[int(idx)]
                if keep and int(lkc.targetid) == int(tic_id):
                    lkc_list.append(lkc)
                    cadno.append(lkc.cadenceno.min())

            if len(lkc_list) == 0:
                continue

            order = np.argsort(cadno)
            lkc_list = [lkc_list[j] for j in order]

            # stitch also normalizes the lightkurves
            lkc = lk.LightCurveCollection(lkc_list).stitch().remove_nans()
            lkc.sector = s * np.ones(len(lkc.time), dtype='int')
            lk_col_clean.append(lkc)

        lk_col_clean = lk.LightCurveCollection(lk_col_clean)

        # stitch into a single LightCurve
        lklc = lk_col_clean.stitch()

        # Determine obsmode from TIMEDEL header keyword
        # TIMEDEL is the time between cadences in days
        obsmode_arr = []
        for lkc_clean in lk_col_clean:
            n = len(lkc_clean.time)
            # Check TIMEDEL from meta
            timedel = lkc_clean.meta.get('TIMEDEL', None)
            if timedel is not None:
                if timedel < 60.0 / 86400.0:
                    mode = '20 sec cadence'
                else:
                    mode = '2 min cadence'
            else:
                # Fallback: estimate from median time spacing
                if n > 1:
                    dt = np.median(np.diff(lkc_clean.time.value))
                    if dt < 60.0 / 86400.0:
                        mode = '20 sec cadence'
                    else:
                        mode = '2 min cadence'
                else:
                    mode = '2 min cadence'
            obsmode_arr.extend([mode] * n)

        # set LiteCurve attributes
        lc_instance.time = np.array(lklc.time.value, dtype=float)
        lc_instance.flux = np.array(lklc.flux.value, dtype=float)
        lc_instance.error = np.array(lklc.flux_err.value, dtype=float)
        lc_instance.cadno = np.array(lklc.cadenceno.value, dtype=int)
        lc_instance.visit = np.array(lklc.sector, dtype=int)
        lc_instance.obsmode = np.array(obsmode_arr, dtype=str)
        lc_instance.quality = np.array(lklc.quality.value, dtype=int)

        # remove cadences flagged by TESS project pipeline
        lc_instance = lc_instance._remove_flagged_cadences(lklc.quality)

        return lc_instance
