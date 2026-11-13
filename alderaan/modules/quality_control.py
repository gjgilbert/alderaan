__all__ = ["QualityControl"]

import warnings

import numpy as np
from astropy.stats import mad_std

from alderaan.modules.base import BaseAlg


class QualityControl(BaseAlg):
    def __init__(self, litecurve, planets):
        super().__init__(litecurve, planets)

    def check_coverage(self):
        """
        Check that each transit has at least 50% coverage
          * number of cadences in transit (tc +/- 0.5*T14)
          * number of cadences near transit (tc +/- 1.5*T14)
        """
        lc = self.litecurve

        quality = [None] * self.npl
        obsmode = self.transit_obsmode

        for n, p in enumerate(self.planets):
            exptime = np.array([self.exptime[om] for om in obsmode[n]])
            count_expected = np.maximum(1, np.array(p.duration / exptime, dtype=int))

            quality[n] = np.zeros(len(p.transit_times), dtype="bool")

            for i, tc in enumerate(p.transit_times):
                in_transit = np.abs(lc.time - tc) / p.duration < 0.5
                near_transit = np.abs(lc.time - tc) / p.duration < 1.5

                enough_pts_in = np.sum(in_transit) >= 0.5 * count_expected[i]
                enough_pts_near = np.sum(near_transit) >= 1.5 * count_expected[i]

                quality[n][i] = enough_pts_in and enough_pts_near

            f_quality = np.sum(quality[n]) / len(quality[n])

            if f_quality < 0.8:
                warnings.warn(
                    f"WARNING: {int((1 - f_quality) * 100)}% of transits for "
                    f"Planet {n} have been flagged as low quality",
                    stacklevel=2,
                )

        return quality

    def check_rms(self, rel_size=None, abs_size=None, sigma_cut=5.0):
        if (rel_size is None) and (abs_size is None):
            raise ValueError("either rel_size or abs_size must be provided")

        if rel_size is None:
            rel_size = 0.0
        if abs_size is None:
            abs_size = 0.0

        lc = self.litecurve

        quality = [None] * self.npl
        overlap = self.identify_overlapping_transits(rtol=rel_size, atol=abs_size)
        obsmode = self.transit_obsmode

        for n, p in enumerate(self.planets):
            rms = np.zeros(len(p.transit_times))
            half_width = 0.5 * np.max([rel_size * p.duration, abs_size])

            for i, tc in enumerate(p.transit_times):
                flux = lc.flux[np.abs(lc.time - tc) < half_width]
                rms[i] = mad_std(flux, ignore_nan=True)

            quality[n] = np.ones(len(p.transit_times), dtype="bool")

            for om in np.unique(np.array(obsmode[n])):
                use = (obsmode[n] == om) & overlap[n]
                rms_mu = np.nanmedian(rms[use])
                rms_sd = mad_std(rms[use], ignore_nan=True)

                quality[n][use] &= np.abs(rms[use] - rms_mu) / rms_sd < sigma_cut

        return quality
