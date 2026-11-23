"""Flag transits with poor photometric coverage or high noise"""

import logging

import numpy as np
from astropy.stats import mad_std

from alderaan.algorithms.base import BaseAlg
from alderaan.constants import EXPTIME
from alderaan.data_models.transit_times import FIELDS, TransitTimes

logger = logging.getLogger(__name__)


class FlagBadTransits(BaseAlg):
    """
    Set photometry flags (TransitTimes.phot_coverage, phot_noise) for transits
    with poor photometric coverage or high noise; every transit is kept. Judged
    at ttime_mod on the current litecurve: run once after each detrend. A planet
    left with < 3 good transits is dropped and its transit windows removed from
    the litecurve (error if no planet would remain)

    Parameters
    ----------
    star : Star
    planets : list of Planet
    litecurve : LiteCurve
    """

    def _check_coverage(self):
        """
        Per transit, True if >= 50% of expected cadences are present both in
        transit (tc +/- 0.5 T14) and near transit (tc +/- 1.5 T14)
        """
        lc = self.litecurve

        quality = [None] * self.npl
        obsmode = [lc.obsmode_at(p.transit_times.ttime_mod) for p in self.planets]

        for n, p in enumerate(self.planets):
            exptime = np.array([EXPTIME[om] for om in obsmode[n]])
            count_expected = np.maximum(1, np.array(p.duration / exptime, dtype=int))

            quality[n] = np.zeros(len(p.transit_times.ttime_mod), dtype="bool")

            for i, tc in enumerate(p.transit_times.ttime_mod):
                in_transit = np.abs(lc.time - tc) / p.duration < 0.5
                near_transit = np.abs(lc.time - tc) / p.duration < 1.5

                enough_pts_in = np.sum(in_transit) >= 0.5 * count_expected[i]
                enough_pts_near = np.sum(near_transit) >= 1.5 * count_expected[i]

                quality[n][i] = enough_pts_in and enough_pts_near

        return quality

    def _check_rms(self, rel_size=0.0, abs_size=0.0, sigma_cut=5.0):
        if rel_size == 0 and abs_size == 0:
            raise ValueError("either rel_size or abs_size must be nonzero")

        lc = self.litecurve

        quality = [None] * self.npl
        overlap = self.identify_overlapping_transits(
            rel_size=rel_size, abs_size=abs_size
        )
        obsmode = [lc.obsmode_at(p.transit_times.ttime_mod) for p in self.planets]

        for n, p in enumerate(self.planets):
            # NaN = unmeasured: no cadences in the window (coverage flags these)
            rms = np.full(len(p.transit_times.ttime_mod), np.nan)
            half_width = 0.5 * np.max([rel_size * p.duration, abs_size])

            for i, tc in enumerate(p.transit_times.ttime_mod):
                flux = lc.flux[np.abs(lc.time - tc) < half_width]
                if len(flux) > 0:
                    rms[i] = mad_std(flux)

            quality[n] = np.ones(len(p.transit_times.ttime_mod), dtype="bool")

            for om in np.unique(np.array(obsmode[n])):
                use = (obsmode[n] == om) & ~overlap[n]
                measured = use & np.isfinite(rms)

                # too few transits to estimate the noise spread
                if np.sum(measured) < 8:
                    logger.info(
                        "planet %d (%s): %d transits (< 8); noise check skipped",
                        n,
                        om,
                        np.sum(measured),
                    )
                    continue

                rms_mu = np.median(rms[measured])
                rms_sd = mad_std(rms[measured])

                logger.debug(
                    "planet %d (%s): median in-window rms = %.2e, spread = %.2e, "
                    "cut = %.1f sigma; %d overlapping transits exempt",
                    n,
                    om,
                    rms_mu,
                    rms_sd,
                    sigma_cut,
                    np.sum((obsmode[n] == om) & overlap[n]),
                )
                quality[n][use] &= np.abs(rms[use] - rms_mu) / rms_sd < sigma_cut

        return quality

    def _drop_planets_with_few_transits(self):
        """
        Drop planets with < 3 good transits (too few for an ephemeris) and remove
        cadences within 3 T14 windows around their transits, so they cannot bias the
        detrending or the other planets; error if no planet would remain
        """
        lc = self.litecurve
        keep_planet = np.ones(self.npl, dtype=bool)
        keep_cadence = np.ones(len(lc.time), dtype=bool)
        for n, p in enumerate(self.planets):
            tt = p.transit_times
            ngood = np.sum(tt.phot_coverage & tt.phot_noise)
            if ngood >= 3:
                continue
            if self.npl == 1:
                raise ValueError(f"planet {n}: {ngood} good transits (< 3)")
            logger.warning(
                "planet %d: %d good transits (< 3); dropped and its transits "
                "removed from the litecurve",
                n,
                ngood,
            )
            keep_planet[n] = False
            for tc in tt.ttime_mod:
                keep_cadence &= np.abs(lc.time - tc) >= 1.5 * p.duration

        if not np.any(keep_planet):
            raise ValueError("every planet has < 3 good transits")
        lc.select(keep_cadence)

        return [p for p, k in zip(self.planets, keep_planet) if k], lc

    def perform(self):
        """
        Returns
        -------
        planets : list of Planet
        litecurve : LiteCurve
        """
        planets = self.planets

        coverage = self._check_coverage()

        good_rms = self._check_rms(rel_size=3.0, abs_size=2 / 24, sigma_cut=5.0)

        for n, p in enumerate(planets):
            assert len(coverage[n]) == len(p.transit_times.ttime_mod)
            assert len(good_rms[n]) == len(p.transit_times.ttime_mod)

            ntot = len(p.transit_times.ttime_mod)
            n_coverage = np.sum(~coverage[n])

            logger.log(
                logging.WARNING if n_coverage > 0.2 * ntot else logging.INFO,
                "planet %d: %d of %d transits flagged low quality "
                "(%d insufficient coverage, %d high noise)",
                n,
                n_coverage + np.sum(~good_rms[n] & coverage[n]),
                ntot,
                n_coverage,
                np.sum(~good_rms[n] & coverage[n]),
            )

            # flag rather than drop: rejected transits stay in transit masks,
            # overlap checks, and the transit model, but are not fit or measured
            tt = p.transit_times
            fields = {k: getattr(tt, k) for k in FIELDS}
            fields["phot_coverage"] = coverage[n]
            fields["phot_noise"] = good_rms[n]
            p.set_transit_times(
                TransitTimes(
                    tt.mission,
                    tt.target,
                    **fields,
                    model_name=tt.model_name,
                    model_dof=tt.model_dof,
                    ttv_scatter=tt.ttv_scatter,
                )
            )

        planets, litecurve = self._drop_planets_with_few_transits()

        return planets, litecurve
