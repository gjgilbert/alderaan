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
from alderaan.utils.stats import sigma_cutoff

logger = logging.getLogger(__name__)


class ClipOutliers(BaseAlg):
    """Remove outlier cadences from a litecurve visit-by-visit"""

    @staticmethod
    def _clip_outliers(flux, sigma_upper, sigma_lower, kernel_size, mask=None):
        """
        Sigma-clip flux about a median-filter trend, iterating up to 3 times

        If > 1% of cadences are flagged, retry with a savgol trend; if still
        > 1%, reject nothing that iteration.

        Parameters
        ----------
        flux : ndarray
            flux for one visit
        sigma_upper : float
            upper sigma clipping threshold
        sigma_lower : float
            lower sigma clipping threshold
        kernel_size : int
            size of the smoothing filter kernel
        mask : array-like, bool, optional
            never reject masked cadences (e.g. to protect transits)

        Returns
        -------
        keep : ndarray, bool
            True for cadences to keep
        """
        if mask is None:
            mask = np.zeros(len(flux), dtype=bool)
        npts = len(flux)
        idx = np.arange(npts)

        for _ in range(3):
            smoothed = median_filter(flux, size=kernel_size, mode="mirror")

            bad = sigma_clip(
                flux - smoothed,
                sigma_upper=sigma_upper,
                sigma_lower=sigma_lower,
                stdfunc=mad_std,
            ).mask
            bad = bad * ~mask

            if np.sum(bad) / len(bad) > 0.01:
                logger.debug(
                    "clip at %g/%g sigma (upper/lower): median filter flagged "
                    "%d of %d cadences; retrying with savgol filter",
                    sigma_upper,
                    sigma_lower,
                    np.sum(bad),
                    len(bad),
                )
                smoothed = savgol_filter(flux, window_length=kernel_size, polyorder=2)
                bad = sigma_clip(
                    flux - smoothed,
                    sigma_upper=sigma_upper,
                    sigma_lower=sigma_lower,
                    stdfunc=mad_std,
                ).mask
                bad = bad * ~mask

            if np.sum(bad) / len(bad) > 0.01:
                logger.warning(
                    "clip at %g/%g sigma (upper/lower): >1%% of cadences flagged "
                    "by median and savgol filters (%d of %d); skipping outlier "
                    "rejection this iteration",
                    sigma_upper,
                    sigma_lower,
                    np.sum(bad),
                    len(bad),
                )
                bad = np.zeros_like(mask)

            flux = flux[~bad]
            idx = idx[~bad]
            mask = mask[~bad]

            if not np.any(bad):
                break

        keep = np.zeros(npts, dtype=bool)
        keep[idx] = True
        return keep

    def perform(self, method, raw_litecurve=None):
        """
        Remove outlier cadences from each visit; flux is unchanged

        Parameters
        ----------
        method : str
            'local_trend' : clip flux about a smoothed trend, sparing transits
                (the larger of 3 x T14 or 2 x (exptime + 1 hour); 2 x T14 +
                exptime for ultra-short-period planets, P < 3 d and ror < 0.05);
                a second pass clips upper outliers everywhere, in transit too
            'transit_model' : clip (detrended) flux about the transit model, at
                the sigma beyond which one outlier is expected per visit (>= 5),
                raised for variable stars with deep transits, where detrending
                and model errors in transit are not outliers
        raw_litecurve : LiteCurve, optional
            raw photometry containing every cadence; required for
            'transit_model', which compares raw and detrended RMS out of transit
        """
        if method not in ("local_trend", "transit_model"):
            raise ValueError(
                f"method must be 'local_trend' or 'transit_model', got {method!r}"
            )

        lc = self.litecurve
        visits = np.unique(lc.visit)

        # same mask as the DetrendVisits wide mask
        transit_mask = np.zeros(len(lc.time), dtype=bool)
        usp = np.array([p.period < 3 and p.ror < 0.05 for p in self.planets])
        for v in visits:
            in_visit = lc.visit == v
            exptime = EXPTIME[lc.obsmode[in_visit][0]]
            transit_mask[in_visit] = self.make_transit_mask(
                rel_size=np.where(usp, 2.0, 3.0),
                abs_size=np.where(usp, 0.0, 2 * (exptime + 1 / 24)),
                buffer=np.where(usp, exptime, 0.0),
                time=lc.time[in_visit],
            )

        if method == "transit_model":
            tm = TransitModel(self.star, self.planets, lc)
            f_mod = tm.model_flux(tm.theta_initial())
            raw = copy.deepcopy(lc)
            raw.reset(raw_litecurve)
            rms_raw = raw.visit_rms(transit_mask)
            rms_det = lc.visit_rms(transit_mask)

        keep = np.ones(len(lc.time), dtype=bool)
        for v in visits:
            use = (lc.visit == v) & keep
            npts_initial = int(np.sum(use))

            if method == "local_trend":
                logger.info("visit %d: clip about local trend", v)
                keep[use] = self._clip_outliers(
                    lc.flux[use],
                    sigma_upper=5,
                    sigma_lower=5,
                    kernel_size=13,
                    mask=transit_mask[use],
                )
                # second pass: upper outliers only, including in transit
                use = (lc.visit == v) & keep
                keep[use] = self._clip_outliers(
                    lc.flux[use], sigma_upper=5, sigma_lower=1000, kernel_size=13
                )

            elif method == "transit_model":
                sigma = sigma_cutoff(int(np.sum(use)), sigma_min=5.0)
                rms = rms_det[v]
                stellar_variability = rms_raw[v] / rms
                snr_approx = max(p.ror**2 for p in self.planets) / rms
                sigma *= np.sqrt(
                    1 + (0.01 * min(stellar_variability, 10) * snr_approx) ** 2
                )
                logger.debug("visit %d: transit-model clipping at %.1f sigma", v, sigma)
                keep[use] = ~sigma_clip(
                    lc.flux[use] - f_mod[use],
                    sigma_upper=sigma,
                    sigma_lower=sigma,
                    stdfunc=mad_std,
                ).mask

            logger.info(
                "visit %d: %d of %d cadences rejected as outliers (%s)",
                v,
                npts_initial - int(np.sum((lc.visit == v) & keep)),
                npts_initial,
                method,
            )

        lc.select(keep)

        return self.planets, lc
