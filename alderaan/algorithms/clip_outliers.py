"""Clip flux outliers visit-by-visit"""

import logging

import numpy as np
from astropy.stats import mad_std, sigma_clip
from scipy.signal import medfilt as median_filter
from scipy.signal import savgol_filter

from alderaan.algorithms.base import BaseAlg
from alderaan.algorithms.transit_fitting import BaseTransitFittingAlg
from alderaan.models.transit_model import TransitModel
from alderaan.utils.stats import set_sigma_cutoff

logger = logging.getLogger(__name__)


class ClipOutliers(BaseAlg):
    """
    Remove outlier cadences from a litecurve visit-by-visit

    Parameters
    ----------
    star : Star
    planets : list of Planet
    litecurve : LiteCurve
    """

    @staticmethod
    def _clip_outliers(flux, visit, sigma_upper, sigma_lower, kernel_size, mask=None):
        """
        Sigma-clip outliers using astropy.stats.sigma_clip() and a smoothed trend
        Applies an additional iteration wrapper to allow for masked cadences

        Returns a boolean mask over flux; True for cadences to keep

        Arguments:
            flux : ndarray
                flux for one visit
            visit : int
                visit number (for logging)
            sigma_upper : float
                upper sigmga clipping threshold
            sigma_lower : float
                lower sigma clipping threshold
            kernel_size : int
                size of the smoothing filter kernel
            mask : array-like, bool (optional)
                do not reject cadences within masked regions;
                useful for protecting transits
        """
        if mask is None:
            mask = np.zeros(len(flux), dtype=bool)
        npts = len(flux)
        idx = np.arange(npts)

        loop = True
        count = 0
        while loop:
            # first pass: try median filter
            if loop:
                smoothed = median_filter(flux, kernel_size=kernel_size)

                bad = sigma_clip(
                    flux - smoothed,
                    sigma_upper=sigma_upper,
                    sigma_lower=sigma_lower,
                    stdfunc=mad_std,
                ).mask
                bad = bad * ~mask

            # second pass: try savgol filter
            # if over 1% of points were flagged with median filter
            if np.sum(bad) / len(bad) > 0.01:
                logger.debug(
                    "visit %d: median filter flagged %d of %d cadences; "
                    "retrying with savgol filter",
                    visit,
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

            # third pass: skip outlier rejection
            if np.sum(bad) / len(bad) > 0.01:
                logger.warning(
                    "visit %d: >1%% of cadences flagged by median and savgol filters "
                    "(%d of %d); skipping outlier rejection this iteration",
                    visit,
                    np.sum(bad),
                    len(bad),
                )
                bad = np.zeros_like(mask)

            flux = flux[~bad]
            idx = idx[~bad]

            # update mask and iterate
            mask = mask[~bad]

            if np.sum(bad) == 0:
                loop = False
            else:
                count += 1

            if count >= 3:
                loop = False

        keep = np.zeros(npts, dtype=bool)
        keep[idx] = True
        return keep

    def _transit_theta(self):
        """Transit model parameters using the shape from QuickFitTransitShape"""
        theta = BaseTransitFittingAlg._theta_initial(self.star, self.planets)
        for n, p in enumerate(self.planets):
            theta[5 * n + 2] = p.ror
            theta[5 * n + 3] = p.impact

        return theta

    def perform(self, method):
        """
        Parameters
        ----------
        method : str
            'local_trend' : clip flux about a smoothed trend, sparing transits
                (3 x T14 plus 1 hour)
            'transit_model' : clip (detrended) flux about the transit model

        Returns
        -------
        planets : list of Planet
        litecurve : LiteCurve
            outlier cadences removed; flux unchanged
        """
        if method not in ("local_trend", "transit_model"):
            raise ValueError(
                f"method must be 'local_trend' or 'transit_model', got {method!r}"
            )

        lc = self.litecurve
        visits = np.unique(lc.visit)

        # clip outliers from the incoming flux, one visit at a time
        if method == "transit_model":
            f_mod = TransitModel(self.star, self.planets, lc).model_flux(
                self._transit_theta()
            )

        keep = np.ones(len(lc.time), dtype=bool)
        for v in visits:
            use = (lc.visit == v) & keep
            npts_initial = int(np.sum(use))

            if method == "local_trend":
                transit_mask = self._transit_mask(
                    lc.time[use], lc.obsmode[use][0], tight_mask=False
                )
                keep[use] = self._clip_outliers(
                    lc.flux[use], v, 5, 5, 13, mask=transit_mask
                )
                use = (lc.visit == v) & keep
                keep[use] = self._clip_outliers(lc.flux[use], v, 5, 1000, 13)

            elif method == "transit_model":
                sigma = set_sigma_cutoff(int(np.sum(use)), sigma_min=5.0)
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
