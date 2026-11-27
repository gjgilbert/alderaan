"""Clip flux outliers visit-by-visit"""

import copy
import logging

import numpy as np
from astropy.stats import mad_std, sigma_clip
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter

from alderaan.algorithms.base import BaseAlg
from alderaan.constants import EXPTIME
from alderaan.transit_model import TransitModel

logger = logging.getLogger(__name__)

# smoothing kernel and CDPP timescale (cadences)
KERNEL_SIZE = 13

# cadence regions: out of the wide mask, in the wide mask out of T14, in T14
FAR, NEAR, IN = 0, 1, 2


class ClipOutliers(BaseAlg):
    """Remove outlier cadences from a litecurve visit-by-visit"""

    @staticmethod
    def _cdpp(time, flux, mask, exptime, windows):
        """
        sgCDPP proxy (Gilliland et al. 2011, ApJS 197, 6; Van Cleve et al. 2016,
        PASP 128, 075002) for the Kepler CDPP (Christiansen et al. 2012, PASP
        124, 1279), computed on raw flux out of the transit mask

        1. Flatten with a 2-day, quadratic Savitzky-Golay filter (masked flux
           replaced by a linear fill)
        2. Reject 5-sigma outliers
        3. CDPP(n) = std of the n-cadence running mean

        Parameters
        ----------
        windows : list of int
            running-mean lengths n (cadences)

        Returns
        -------
        cdpp : ndarray
            CDPP(n) for each window (relative flux)
        """
        f = flux.copy()
        f[mask] = np.interp(time[mask], time[~mask], flux[~mask])
        window = min(int(2.0 / exptime) // 2 * 2 + 1, len(f) // 2 * 2 - 1)
        x = (flux / savgol_filter(f, window_length=window, polyorder=2) - 1)[~mask]
        x = sigma_clip(x, sigma=5, stdfunc=mad_std).compressed()
        return np.array(
            [np.std(np.convolve(x, np.ones(n) / n, "valid")) for n in windows]
        )

    @staticmethod
    def _sigma_threshold(time, flux, mask, exptime):
        """
        Sigma threshold for clipping: 5, raised by the red-noise factor beta

        beta(n) = CDPP(n) sqrt(n) / CDPP(1) is 1 for white noise and > 1 for
        coherent variability on timescale n (Pont et al. 2006, MNRAS 373, 231);
        beta = max over n up to the trend kernel size, the variability a local
        trend may not follow

        Returns
        -------
        sigma, beta : float
        """
        windows = [1, 2, 4, 8, KERNEL_SIZE]
        cdpp = ClipOutliers._cdpp(time, flux, mask, exptime, windows)
        beta = np.max(cdpp[1:] * np.sqrt(windows[1:]) / cdpp[0])
        return 5.0 * max(1.0, beta), beta

    @staticmethod
    def _trend(time, flux, mask):
        """
        Local trend from out-of-mask cadences, interpolated across the mask

        Masked flux is replaced by a linear fill between the mask edges; a short
        median filter then removes spikes and a savgol filter follows curvature
        """
        f = flux.copy()
        f[mask] = np.interp(time[mask], time[~mask], flux[~mask])
        f = median_filter(f, size=5, mode="mirror")
        return savgol_filter(f, window_length=KERNEL_SIZE, polyorder=2)

    @staticmethod
    def _clip_outliers(resid, sigma, region, method):
        """
        Flag outliers in residuals; return True for cadences to keep

        region labels each cadence FAR (out of the wide mask), NEAR (in the
        wide mask, out of +/- 0.5 T14) or IN (+/- 0.5 T14); detrending gives
        each region its own noise, so each has its own median and mad_std

        'local_trend': FAR, both tails beyond sigma x mad_std(FAR); NEAR and IN,
            upper tail only (a transit cannot raise flux), on the FAR scale;
            NEAR uses 2 x sigma, for the error of the trend interpolated
            across the mask
        'transit_model': each region, both tails beyond sigma x its own mad_std
        """
        keep = np.ones(len(resid), dtype=bool)
        if method == "local_trend":
            far = region == FAR
            near = region == NEAR
            within = region == IN
            r = resid - np.median(resid[far])
            scale = mad_std(r[far])
            keep[far] = np.abs(r[far]) <= sigma * scale
            keep[near] = r[near] <= 2 * sigma * scale
            keep[within] = r[within] <= sigma * scale
            return keep

        for k in (FAR, NEAR, IN):
            sel = region == k
            if np.any(sel):
                r = resid[sel] - np.median(resid[sel])
                keep[sel] = np.abs(r) <= sigma * mad_std(r)
        return keep

    def perform(self, method, raw_litecurve=None):
        """
        Remove outlier cadences from each visit; flux is unchanged

        Parameters
        ----------
        method : str
            'local_trend' : clip flux about a local trend fit out of the transit
                mask (the larger of 3 x T14 or 2 x (exptime + 1 hour); 2 x T14 +
                exptime for ultra-short-period planets, P < 3 d and ror < 0.05);
                in the mask, clip upper outliers only (at 2 x sigma near transit)
            'transit_model' : clip (detrended) flux about the transit model;
                far from transit, near transit (in the wide mask) and in
                transit (+/- 0.5 T14) each use their own median and mad_std
        raw_litecurve : LiteCurve, optional
            raw photometry containing every cadence; required for
            'transit_model', whose flux is detrended, to measure CDPP

        Both methods clip at 5 sigma, raised by the CDPP red-noise factor of
        the raw flux (see _sigma_threshold)
        """
        if method not in ("local_trend", "transit_model"):
            raise ValueError(
                f"method must be 'local_trend' or 'transit_model', got {method!r}"
            )

        lc = self.litecurve
        visits = np.unique(lc.visit)

        # same mask as the DetrendVisits wide mask
        wide_mask = np.zeros(len(lc.time), dtype=bool)
        usp = np.array([p.period < 3 and p.ror < 0.05 for p in self.planets])
        for v in visits:
            in_visit = lc.visit == v
            exptime = EXPTIME[lc.obsmode[in_visit][0]]
            wide_mask[in_visit] = self.make_transit_mask(
                rel_size=np.where(usp, 2.0, 3.0),
                abs_size=np.where(usp, 0.0, 2 * (exptime + 1 / 24)),
                buffer=np.where(usp, exptime, 0.0),
                time=lc.time[in_visit],
            )

        in_transit = self.make_transit_mask(rel_size=1.0)
        region = np.where(in_transit, IN, np.where(wide_mask, NEAR, FAR))

        if method == "local_trend":
            raw_flux = lc.flux
        else:
            if raw_litecurve is None:
                raise ValueError("method 'transit_model' requires raw_litecurve")
            tm = TransitModel(self.star, self.planets, lc)
            f_mod = tm.model_flux(tm.theta_initial())
            raw = copy.deepcopy(lc)
            raw.reset(raw_litecurve)
            raw_flux = raw.flux

        keep = np.ones(len(lc.time), dtype=bool)
        for v in visits:
            use = lc.visit == v
            reg = region[use]
            if method == "local_trend":
                trend = self._trend(lc.time[use], lc.flux[use], wide_mask[use])
            else:
                trend = f_mod[use]
            resid = lc.flux[use] - trend

            sigma, beta = self._sigma_threshold(
                lc.time[use],
                raw_flux[use],
                wide_mask[use],
                EXPTIME[lc.obsmode[use][0]],
            )
            keep[use] = self._clip_outliers(resid, sigma, reg, method)

            npts = int(np.sum(use))
            nrej = npts - int(np.sum(keep[use]))
            n = [int(np.sum(reg == k)) for k in (FAR, NEAR, IN)]
            nr = [int(np.sum((reg == k) & ~keep[use])) for k in (FAR, NEAR, IN)]
            logger.debug("visit %d: beta %.2f, clip at %.1f sigma", v, beta, sigma)
            logger.info(
                "visit %d: %d of %d cadences rejected as outliers; far %d of %d, "
                "near %d of %d, in transit %d of %d (%s)",
                v,
                nrej,
                npts,
                nr[0],
                n[0],
                nr[1],
                n[1],
                nr[2],
                n[2],
                method,
            )
            if nrej > 0.01 * npts:
                logger.warning(
                    "visit %d: %d of %d cadences (> 1%%) rejected as outliers (%s)",
                    v,
                    nrej,
                    npts,
                    method,
                )

        lc.select(keep)

        return self.planets, lc
