"""LiteCurve: photometry time series with per-cadence visit and obsmode"""

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

# per-cadence arrays and their FITS formats; float32 flux and error err by
# <= 6e-8 relative (0.004 x error for a bright star), but time stays float64
# (float32 errs by seconds)
FIELDS = {
    "time": "D",
    "flux": "E",
    "error": "E",
    "cadno": "J",
    "visit": "I",
    "obsmode": "B",
}


class LiteCurve:
    """
    Photometric time series for one target

    Parameters
    ----------
    mission, target : str
    time, flux, error : ndarray
        time (days), flux and flux uncertainty; flux and error must be finite
    cadno : ndarray
        cadence number
    visit : ndarray
        observing visit (Kepler quarter)
    obsmode : ndarray
        'long cadence' or 'short cadence'
    t_start, t_end : float, optional
        photometry span; default time.min(), time.max()

    Notes
    -----
    Bad cadences are removed (select), not flagged. The span (t_start, t_end) is
    fixed at construction; select() and reset() do not change it
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
        t_start=None,
        t_end=None,
    ):
        self.mission = mission
        self.target = target
        self.meta = fits.Header()  # read-only tracking cards, written to PRIMARY

        self.time = np.asarray(time, dtype=float)
        self.t_start = float(self.time.min() if t_start is None else t_start)
        self.t_end = float(self.time.max() if t_end is None else t_end)
        self.flux = np.asarray(flux, dtype=float)
        self.error = np.asarray(error, dtype=float)
        self.cadno = np.asarray(cadno, dtype=int)
        self.visit = np.asarray(visit, dtype=int)
        self.obsmode = np.asarray(obsmode, dtype=str)

        lengths = {len(getattr(self, k)) for k in FIELDS}
        if len(lengths) != 1:
            raise ValueError(f"{', '.join(FIELDS)} must have equal lengths")

        if not (np.all(np.isfinite(self.flux)) and np.all(np.isfinite(self.error))):
            raise ValueError("flux and error must be finite")

    @classmethod
    def from_fits(cls, filepath):
        """Load a litecurve written by LiteCurve.to_fits"""
        logger.info("reading %s", filepath)
        with fits.open(filepath) as hdul:
            hdu = hdul["LITECURVE"]
            h = hdu.header
            arrays = {k: hdu.data[k.upper()] for k in FIELDS}
            names = np.array([h[f"OBSMODE{i}"] for i in range(h["NOBSMODE"])])
            arrays["obsmode"] = names[arrays["obsmode"]]
            lc = cls(
                h["MISSION"],
                h["TARGET"],
                **arrays,
                t_start=h["TSTART"],
                t_end=h["TSTOP"],
            )
            # meta: the PRIMARY cards that to_fits does not write itself
            cards = hdul[0].header.copy(strip=True).cards
            lc.meta = fits.Header(
                [c for c in cards if c.keyword not in ("MISSION", "TARGET")]
            )
        return lc

    @classmethod
    def from_kplr_pdcsap(cls, data_dir, target, kic_id, obsmode, visits=None):
        """
        Load Kepler PDCSAP lightcurves from pre-downloaded MAST fits files

        Removes NaNs, cadences flagged by the Kepler pipeline (default bitmask)
        and files whose TARGETID is not kic_id, and normalizes each quarter.

        Parameters
        ----------
        data_dir : str
            directory of MAST fits files
        target : str
            target name, e.g. K00137
        kic_id : int
            KIC number
        obsmode : str
            'short cadence' or 'long cadence'
        visits : int or list, optional
            visits (Kepler quarters) to load; default all

        Returns
        -------
        litecurve : LiteCurve

        Raises
        ------
        FileNotFoundError
            if no file matches kic_id, obsmode and visits
        """
        if visits is None:
            visits = np.arange(18, dtype=int)
        if isinstance(visits, int):
            visits = [visits]

        mast_files = sorted(
            glob.glob(os.path.join(data_dir, f"kplr{kic_id:09d}*.fits"))
        )
        logger.info("found %d MAST files for KIC %d", len(mast_files), kic_id)

        raw = []
        n_skipped = 0
        for mf in mast_files:
            header = fits.getheader(mf)
            if header["OBSMODE"] == obsmode and np.isin(header["QUARTER"], visits):
                # read from an open HDUList; lk.read(path) leaves the file open.
                # lightkurve drops Kepler-flagged cadences (default bitmask)
                with fits.open(mf) as hdul:
                    lkc = lk.KeplerLightCurve.read(hdul, format="kepler")
                    logger.info(
                        "Q%d: removed %d of %d cadences with Kepler quality flags "
                        "(default bitmask)",
                        header["QUARTER"],
                        len(hdul[1].data) - len(lkc.time),
                        len(hdul[1].data),
                    )
                raw.append(lkc)
            else:
                n_skipped += 1
        logger.debug(
            "skipped %d file(s) not %s in quarters %s", n_skipped, obsmode, list(visits)
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

        # span (t_start, t_end) is that of the quality-filtered cadences
        return cls(
            "Kepler",
            target,
            time=lklc.time.value,
            flux=lklc.flux.value,
            error=lklc.flux_err.value,
            cadno=lklc.cadenceno.value,
            visit=np.asarray(lklc.quarter),
            obsmode=np.full(len(lklc.time), obsmode),
        )

    def select(self, mask):
        """Keep only cadences where mask is True (in place)"""
        for k in FIELDS:
            setattr(self, k, getattr(self, k)[mask])

    def visit_rms(self, mask=None):
        """Flux std of each visit, excluding cadences where mask is True"""
        use = np.ones(len(self.time), dtype=bool) if mask is None else ~mask
        return {
            int(v): float(np.std(self.flux[use & (self.visit == v)]))
            for v in np.unique(self.visit)
        }

    def _nearest_cadence(self, times):
        """Index of the cadence nearest each time (ties go earlier)"""
        if np.any(np.diff(self.time) < 0):
            raise ValueError("litecurve.time must be sorted")

        i = np.clip(np.searchsorted(self.time, times), 1, len(self.time) - 1)
        earlier = (times - self.time[i - 1]) <= (self.time[i] - times)
        return i - earlier

    def obsmode_at(self, times):
        """Observing mode at the cadence nearest each time (ties go earlier)"""
        return self.obsmode[self._nearest_cadence(times)]

    def visit_at(self, times):
        """Visit at the cadence nearest each time (ties go earlier)"""
        return self.visit[self._nearest_cadence(times)]

    def reset(self, raw):
        """
        Restore raw flux and error for the current cadences (in place)

        Matched by cadence number; removed cadences stay removed.

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
        """
        Write PRIMARY (MISSION, TARGET, meta) and a LITECURVE table

        The table has one upper-case column per per-cadence array (formats in
        FIELDS) and header cards MISSION, TARGET, TSTART, TSTOP. OBSMODE is
        stored as a code: OBSMODEi names code i, for i < NOBSMODE.
        """
        primary_hdu = fits.PrimaryHDU()
        primary_hdu.header["MISSION"] = (self.mission, "observing mission")
        primary_hdu.header["TARGET"] = (self.target, "target ID")
        primary_hdu.header.update(self.meta)

        names, codes = np.unique(self.obsmode, return_inverse=True)
        arrays = {k: getattr(self, k) for k in FIELDS} | {"obsmode": codes}
        columns = [
            fits.Column(name=k.upper(), format=fmt, array=arrays[k])
            for k, fmt in FIELDS.items()
        ]
        hdu = fits.BinTableHDU.from_columns(columns, name="LITECURVE")
        hdu.header["MISSION"] = (self.mission, "observing mission")
        hdu.header["TARGET"] = (self.target, "target ID")
        hdu.header["TSTART"] = (self.t_start, "[d] start of data span")
        hdu.header["TSTOP"] = (self.t_end, "[d] end of data span")
        hdu.header["NOBSMODE"] = (len(names), "number of obsmodes")
        for i, name in enumerate(names):
            hdu.header[f"OBSMODE{i}"] = (name, f"obsmode {i}")

        # write then rename, so a crash never leaves a partial file at filepath
        fits.HDUList([primary_hdu, hdu]).writeto(filepath + ".tmp", overwrite=True)
        os.replace(filepath + ".tmp", filepath)
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
            save the figure here, closing it unless ax was given; default leaves
            it open
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
        save_and_close(ax.figure, fig_path, close=new_figure)

        return ax.figure
