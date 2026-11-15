__all__ = ["LiteCurve"]

import glob
import logging

import lightkurve as lk
import numpy as np
from astropy.io import fits

logger = logging.getLogger(__name__)

# per-cadence arrays carried by every LiteCurve, in __init__ order
ARRAY_FIELDS = (
    "time",
    "flux",
    "error",
    "cadno",
    "visit",
    "obsmode",
    "quality",
    "flux_raw",
    "error_raw",
)


class LiteCurve:
    """
    Photometric time series for one target

    Parameters
    ----------
    mission, target : str
    time, flux, error : ndarray
    cadno : ndarray
        cadence number
    visit : ndarray
        observing visit (Kepler quarter)
    obsmode : ndarray
        'long cadence' or 'short cadence'
    quality : ndarray
        True for good quality cadences
    flux_raw, error_raw : ndarray, optional
        photometry before detrending; None if not available (e.g. loaded from disk)
    """

    def __init__(
        self,
        mission,
        target,
        time,
        flux,
        error,
        cadno,
        visit,
        obsmode,
        quality,
        flux_raw=None,
        error_raw=None,
    ):
        self.mission = mission
        self.target = target
        self.time = np.asarray(time, dtype=float)
        self.flux = np.asarray(flux, dtype=float)
        self.error = np.asarray(error, dtype=float)
        self.cadno = np.asarray(cadno, dtype=int)
        self.visit = np.asarray(visit, dtype=int)
        self.obsmode = np.asarray(obsmode, dtype=str)
        self.quality = np.asarray(quality, dtype=bool)
        self.flux_raw = None if flux_raw is None else np.asarray(flux_raw, dtype=float)
        self.error_raw = (
            None if error_raw is None else np.asarray(error_raw, dtype=float)
        )

        lengths = {len(getattr(self, k)) for k in self._fields()}
        if len(lengths) != 1:
            raise ValueError(f"{', '.join(self._fields())} must have equal lengths")

    def _fields(self):
        """Names of array fields that are set (flux_raw/error_raw may be None)"""
        return [k for k in ARRAY_FIELDS if getattr(self, k) is not None]

    @classmethod
    def from_list(cls, litecurve_list):
        """Concatenate litecurves; flux_raw/error_raw are None if missing from any"""
        first = litecurve_list[0]
        for lc in litecurve_list:
            if (lc.mission, lc.target) != (first.mission, first.target):
                raise ValueError("litecurves have inconsistent mission/target")

        arrays = {}
        for k in ARRAY_FIELDS:
            parts = [getattr(lc, k) for lc in litecurve_list]
            arrays[k] = None if any(v is None for v in parts) else np.hstack(parts)

        return cls(first.mission, first.target, **arrays)

    @classmethod
    def from_hdu(cls, hdu):
        """Build from a LITECURVE table written by to_hdu (no flux_raw/error_raw)"""
        data = hdu.data
        return cls(
            hdu.header["MISSION"],
            hdu.header["TARGET"],
            data["TIME"],
            data["FLUX"],
            data["ERROR"],
            data["CADNO"],
            data["VISIT"],
            data["OBSMODE"],
            data["QUALITY"],
        )

    @classmethod
    def from_fits(cls, filepath):
        """Load a litecurve written by LiteCurve.to_fits"""
        logger.info("reading %s", filepath)
        with fits.open(filepath) as hdul:
            return cls.from_hdu(hdul["LITECURVE"])

    @classmethod
    def from_kplr_pdcsap(cls, data_dir, target, kic_id, obsmode, visits=None):
        """
        Load photometric data from Kepler Project PDCSAP Flux lightcurves
        The raw fits files must be pre-downloaded from MAST servers and stored locally

        This function performs minimal detrending steps
         * remove_nans()
         * normalize()
         * remove cadences flagged by the Kepler pipeline (default bitmask)

        Args:
            data_dir (str) : path to where data are stored
            target (str) : target name, e.g. K00137
            kic_id (int) : KIC number
            obsmode (str) : 'short cadence' or 'long cadence'
            visits (list) : optional, list of visits (Kepler quarters) to load.
        Returns:
            LiteCurve
        """
        # sanitize inputs
        if visits is None:
            visits = np.arange(18, dtype=int)  # hard coded for Kepler
        if isinstance(visits, int):
            visits = [visits]

        # load the raw MAST files using lightcurve
        mast_files = glob.glob(
            data_dir + f"kplr{kic_id:09d}*.fits"
        )  # hard-coded for Kepler
        mast_files.sort()

        mast_data_list = []
        logger.info("found %d MAST files for KIC %d", len(mast_files), kic_id)
        for mf in mast_files:
            with fits.open(mf) as hdu_list:
                if hdu_list[0].header["OBSMODE"] == obsmode and np.isin(
                    hdu_list[0].header["QUARTER"],
                    visits,  # hard coded for Kepler
                ):
                    mast_data_list.append(lk.read(mf))
                else:
                    logger.debug(
                        "skipped %s (OBSMODE = %s, QUARTER = %d not requested)",
                        mf,
                        hdu_list[0].header["OBSMODE"],
                        hdu_list[0].header["QUARTER"],
                    )

        if len(mast_data_list) == 0:
            raise FileNotFoundError(
                f"no {obsmode} files for KIC {kic_id} quarters {list(visits)} "
                f"in {data_dir}"
            )

        lk_col_raw = lk.LightCurveCollection(mast_data_list)

        # clean up the Collection data structure
        visits = []
        for lkc in lk_col_raw:
            visits.append(lkc.quarter)  # hard coded for Kepler

        lk_col_clean = []
        for v in np.unique(visits):
            lkc_list = []
            cadno = []

            for lkc in lk_col_raw:
                if (lkc.quarter == v) * (
                    lkc.targetid == kic_id
                ):  # hard coded for kepler
                    lkc_list.append(lkc)
                    cadno.append(lkc.cadenceno.min())

            nmismatch = sum(lkc.quarter == v for lkc in lk_col_raw) - len(lkc_list)
            if nmismatch > 0:
                logger.warning(
                    "KIC %d Q%d: dropped %d files with mismatched TARGETID",
                    kic_id,
                    v,
                    nmismatch,
                )

            order = np.argsort(cadno)
            lkc_list = [lkc_list[j] for j in order]

            # lk.stitch() also normalizes the lightkurves
            lkc = lk.LightCurveCollection(lkc_list).stitch()
            npts = len(lkc.time)
            lkc = lkc.remove_nans()
            logger.debug(
                "Q%d: removed %d NaN cadences of %d", v, npts - len(lkc.time), npts
            )

            lkc.quarter = lkc.quarter * np.ones(
                len(lkc.time), dtype="int"
            )  # hard coded for kepler

            lk_col_clean.append(lkc)

        lk_col_clean = lk.LightCurveCollection(lk_col_clean)

        # stitch into a single LightCurve
        lklc = lk_col_clean.stitch()

        # remove cadences flagged by Kepler project pipeline
        qmask = lk.KeplerQualityFlags.create_quality_mask(
            lklc.quality.value, bitmask="default"
        )
        logger.info(
            "removed %d of %d cadences with Kepler quality flags (default bitmask)",
            np.sum(~qmask),
            len(qmask),
        )

        flux = np.array(lklc.flux.value, dtype=float)[qmask]
        error = np.array(lklc.flux_err.value, dtype=float)[qmask]

        return cls(
            "Kepler",
            target,
            time=np.array(lklc.time.value, dtype=float)[qmask],
            flux=flux,
            error=error,
            cadno=np.array(lklc.cadenceno.value, dtype=int)[qmask],
            visit=np.array(lklc.quarter, dtype=int)[qmask],  # hard coded for Kepler
            obsmode=np.array([obsmode] * np.sum(qmask), dtype=str),
            quality=np.ones(np.sum(qmask), dtype=bool),
            flux_raw=flux.copy(),
            error_raw=error.copy(),
        )

    @classmethod
    def from_k2(cls, data_dir, target_id, obsmode, visits=None):
        raise NotImplementedError("Loading K2 data not yet implemented")

    @classmethod
    def from_tess(cls, data_dir, target_id, obsmode, visits=None):
        raise NotImplementedError("Loading TESS data not yet implemented")

    def select(self, mask):
        """Keep only cadences where mask is True (in place)"""
        for k in self._fields():
            setattr(self, k, getattr(self, k)[mask])

    def split_visits(self):
        """List of new litecurves, one per visit"""
        litecurve_list = []
        for v in np.unique(self.visit):
            use = self.visit == v
            arrays = {
                k: None if getattr(self, k) is None else getattr(self, k)[use]
                for k in ARRAY_FIELDS
            }
            litecurve_list.append(LiteCurve(self.mission, self.target, **arrays))

        return litecurve_list

    def reset(self):
        """Restore raw flux and error; mark all cadences as good quality (in place)"""
        if self.flux_raw is None or self.error_raw is None:
            raise ValueError("cannot reset: flux_raw/error_raw not available")

        self.flux = self.flux_raw.copy()
        self.error = self.error_raw.copy()
        self.quality = np.ones(len(self.time), dtype=bool)

    def to_hdu(self):
        """BinTableHDU named LITECURVE; flux_raw/error_raw are not stored"""
        width = max([1, *map(len, self.obsmode)])
        columns = [
            fits.Column(name="TIME", format="D", array=self.time),
            fits.Column(name="FLUX", format="D", array=self.flux),
            fits.Column(name="ERROR", format="D", array=self.error),
            fits.Column(name="CADNO", format="K", array=self.cadno),
            fits.Column(name="VISIT", format="K", array=self.visit),
            fits.Column(name="OBSMODE", format=f"{width}A", array=self.obsmode),
            fits.Column(name="QUALITY", format="L", array=self.quality),
        ]
        hdu = fits.BinTableHDU.from_columns(columns, name="LITECURVE")
        hdu.header["MISSION"] = self.mission
        hdu.header["TARGET"] = self.target
        return hdu

    def to_fits(self, filepath):
        """Save as a fits file: PRIMARY (MISSION, TARGET) + LITECURVE table"""
        primary_hdu = fits.PrimaryHDU()
        primary_hdu.header["MISSION"] = self.mission
        primary_hdu.header["TARGET"] = self.target

        fits.HDUList([primary_hdu, self.to_hdu()]).writeto(filepath, overwrite=True)
        logger.info("wrote %s", filepath)
