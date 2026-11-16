__all__ = ["LiteCurve"]

import glob
import logging
import os

import lightkurve as lk
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

from alderaan.utils.plotting import label_axes, save_and_close

logger = logging.getLogger(__name__)

# per-cadence arrays carried by every LiteCurve: name -> FITS column format;
# OBSMODE width is set at write time
FIELDS = {
    "time": "D",
    "flux": "D",
    "error": "D",
    "cadno": "K",
    "visit": "K",
    "obsmode": None,
}


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

    Bad cadences are removed (select), not flagged
    """

    def __init__(self, mission, target, time, flux, error, cadno, visit, obsmode):
        self.mission = mission
        self.target = target

        self.time = np.asarray(time, dtype=float)
        self.flux = np.asarray(flux, dtype=float)
        self.error = np.asarray(error, dtype=float)
        self.cadno = np.asarray(cadno, dtype=int)
        self.visit = np.asarray(visit, dtype=int)
        self.obsmode = np.asarray(obsmode, dtype=str)

        lengths = {len(getattr(self, k)) for k in FIELDS}
        if len(lengths) != 1:
            raise ValueError(f"{', '.join(FIELDS)} must have equal lengths")

    @classmethod
    def from_fits(cls, filepath):
        """Load a litecurve written by LiteCurve.to_fits"""
        logger.info("reading %s", filepath)
        with fits.open(filepath) as hdul:
            hdu = hdul["LITECURVE"]
            return cls(
                hdu.header["MISSION"],
                hdu.header["TARGET"],
                **{k: hdu.data[k.upper()] for k in FIELDS},
            )

    @classmethod
    def from_kplr_pdcsap(cls, data_dir, target, kic_id, obsmode, visits=None):
        """
        Load photometric data from Kepler Project PDCSAP Flux lightcurves
        The raw fits files must be pre-downloaded from MAST servers and stored locally

        This function performs minimal detrending steps
         * remove_nans()
         * normalize()
         * remove cadences flagged by the Kepler pipeline (default bitmask)

        Parameters
        ----------
        data_dir : str
            path to where data are stored
        target : str
            target name, e.g. K00137
        kic_id : int
            KIC number
        obsmode : str
            'short cadence' or 'long cadence'
        visits : list
            (optional) visits (Kepler quarters) to load

        Returns
        -------
        litecurve : LiteCurve
        """
        if visits is None:
            visits = np.arange(18, dtype=int)
        if isinstance(visits, int):
            visits = [visits]

        # read requested MAST files (visits are Kepler quarters)
        mast_files = sorted(
            glob.glob(os.path.join(data_dir, f"kplr{kic_id:09d}*.fits"))
        )
        logger.info("found %d MAST files for KIC %d", len(mast_files), kic_id)

        raw = []
        for mf in mast_files:
            header = fits.getheader(mf)
            if header["OBSMODE"] == obsmode and np.isin(header["QUARTER"], visits):
                raw.append(lk.read(mf))
            else:
                logger.debug(
                    "skipped %s (OBSMODE = %s, QUARTER = %d not requested)",
                    mf,
                    header["OBSMODE"],
                    header["QUARTER"],
                )

        if len(raw) == 0:
            raise FileNotFoundError(
                f"no {obsmode} files for KIC {kic_id} quarters {list(visits)} "
                f"in {data_dir}"
            )

        # one normalized, NaN-free light curve per quarter
        quarters = []
        for v in np.unique([lkc.quarter for lkc in raw]):
            in_q = [lkc for lkc in raw if lkc.quarter == v]
            keep = [lkc for lkc in in_q if lkc.targetid == kic_id]
            if len(in_q) > len(keep):
                logger.warning(
                    "KIC %d Q%d: dropped %d files with mismatched TARGETID",
                    kic_id,
                    v,
                    len(in_q) - len(keep),
                )
            keep.sort(key=lambda lkc: lkc.cadenceno.min())

            # lk.stitch() also normalizes the lightkurves
            lkc = lk.LightCurveCollection(keep).stitch()
            npts = len(lkc.time)
            lkc = lkc.remove_nans()
            logger.debug(
                "Q%d: removed %d NaN cadences of %d", v, npts - len(lkc.time), npts
            )
            lkc.quarter = np.full(len(lkc.time), lkc.quarter, dtype=int)
            quarters.append(lkc)

        lklc = lk.LightCurveCollection(quarters).stitch()

        # remove cadences flagged by Kepler project pipeline
        qmask = lk.KeplerQualityFlags.create_quality_mask(
            lklc.quality.value, bitmask="default"
        )
        logger.info(
            "removed %d of %d cadences with Kepler quality flags (default bitmask)",
            np.sum(~qmask),
            len(qmask),
        )

        litecurve = cls(
            "Kepler",
            target,
            time=lklc.time.value,
            flux=lklc.flux.value,
            error=lklc.flux_err.value,
            cadno=lklc.cadenceno.value,
            visit=lklc.quarter,
            obsmode=np.full(len(qmask), obsmode),
        )
        litecurve.select(qmask)
        return litecurve

    def select(self, mask):
        """Keep only cadences where mask is True (in place)"""
        for k in FIELDS:
            setattr(self, k, getattr(self, k)[mask])

    def obsmode_at(self, times):
        """Observing mode at the cadence nearest each time (ties go earlier)"""
        if np.any(np.diff(self.time) < 0):
            raise ValueError("litecurve.time must be sorted")

        i = np.clip(np.searchsorted(self.time, times), 1, len(self.time) - 1)
        earlier = (times - self.time[i - 1]) <= (self.time[i] - times)
        return self.obsmode[i - earlier]

    def reset(self, raw):
        """
        Restore raw flux and error for the current cadences, matched by cadence
        number (in place); removed cadences stay removed

        Parameters
        ----------
        raw : LiteCurve
            raw photometry containing every cadence of self, sorted by cadno
        """
        idx = np.searchsorted(raw.cadno, self.cadno)
        if (
            np.any(np.diff(raw.cadno) <= 0)
            or np.any(idx >= len(raw.cadno))
            or not np.array_equal(raw.cadno[idx], self.cadno)
        ):
            raise ValueError("raw litecurve does not contain all cadences")

        self.flux = raw.flux[idx].copy()
        self.error = raw.error[idx].copy()

    def to_fits(self, filepath):
        """Save as a fits file: PRIMARY (MISSION, TARGET) + LITECURVE table"""
        primary_hdu = fits.PrimaryHDU()
        primary_hdu.header["MISSION"] = self.mission
        primary_hdu.header["TARGET"] = self.target

        width = max([1, *map(len, self.obsmode)])
        columns = [
            fits.Column(
                name=k.upper(), format=fmt or f"{width}A", array=getattr(self, k)
            )
            for k, fmt in FIELDS.items()
        ]
        hdu = fits.BinTableHDU.from_columns(columns, name="LITECURVE")
        hdu.header["MISSION"] = self.mission
        hdu.header["TARGET"] = self.target

        fits.HDUList([primary_hdu, hdu]).writeto(filepath, overwrite=True)
        logger.info("wrote %s", filepath)

    def plot(self, visit=None, mark_times=None, ax=None, fig_path=None):
        """
        Plot flux vs time; returns the Figure

        Parameters
        ----------
        visit : int, optional
            plot one visit; default plots all
        mark_times : list of ndarray, optional
            one array of times per planet, marked below the photometry in C{n}
        ax : matplotlib Axes, optional
            draw here; default makes a new figure
        fig_path : str, optional
            save the figure here and close it; default leaves it open
        """
        use = (
            np.ones(len(self.time), dtype=bool)
            if visit is None
            else self.visit == visit
        )
        if not np.any(use):
            raise ValueError(f"no data for visit {visit}")
        time, flux = self.time[use], self.flux[use]

        new_figure = ax is None
        if new_figure:
            _, ax = plt.subplots(1, 1, figsize=(20, 4))

        ax.plot(time, flux, "k.", ms=1)
        ax.set_xlim(time.min(), time.max())

        yrange = 1.3 * np.max(np.abs(1 - np.percentile(flux, [0.1, 99.9])))
        ymin = 1 - yrange
        if mark_times is not None:
            ymin = 1 - yrange * (1.3 + 0.1 * len(mark_times))
            for n, tc in enumerate(mark_times):
                tc = tc[(tc >= time.min()) & (tc <= time.max())]
                y = 1 - (0.9 - 0.1 * n) * (1 - ymin)
                ax.plot(tc, np.full(len(tc), y), "^", c=f"C{n}")
        ax.set_ylim(ymin, 1 + yrange)

        label_axes(ax, "Time [BKJD]", "Flux")
        if new_figure:
            ax.figure.tight_layout()
        save_and_close(ax.figure, fig_path)

        return ax.figure
