"""Flag transits with poor photometric coverage or high noise"""

import logging

import numpy as np
from astropy.stats import mad_std

from alderaan.algorithms.base import BaseAlg
from alderaan.constants import EXPTIME
from alderaan.data_models.transit_times import FIELDS, TransitTimes
from alderaan.errors import UnsupportedTargetError

logger = logging.getLogger(__name__)


class FlagBadTransits(BaseAlg):
    """
    Flag transits with poor photometric coverage or high noise

    Sets TransitTimes.phot_coverage and phot_noise; every transit is kept. Judged
    at ttime_mod on the current litecurve: run once after each detrend. A planet
    left with < 3 good transits is dropped and its transit windows removed from
    the litecurve (UnsupportedTargetError if no planet would remain).
    """

    def _check_coverage(self):
        """
        Per planet, a bool array over transits: True for good coverage

        Good coverage: >= 50% of expected cadences are present both in transit
        (tc +/- 0.5 T14) and near transit (tc +/- 1.5 T14).
        """
        lc = self.litecurve

        good_coverage = [None] * self.npl
        obsmode = [lc.obsmode_at(p.transit_times.ttime_mod) for p in self.planets]

        for n, p in enumerate(self.planets):
            exptime = np.array([EXPTIME[om] for om in obsmode[n]])
            count_expected = np.maximum(1, np.array(p.duration / exptime, dtype=int))

            ntot = len(p.transit_times.ttime_mod)
            good_coverage[n] = np.zeros(ntot, dtype="bool")
            n_in_gap = 0

            for i, tc in enumerate(p.transit_times.ttime_mod):
                in_transit = np.abs(lc.time - tc) / p.duration < 0.5
                near_transit = np.abs(lc.time - tc) / p.duration < 1.5
                n_near = np.sum(near_transit)
                n_in_gap += n_near == 0

                enough_pts_in = np.sum(in_transit) >= 0.5 * count_expected[i]
                enough_pts_near = n_near >= 1.5 * count_expected[i]

                good_coverage[n][i] = enough_pts_in and enough_pts_near

            n_poor = np.sum(~good_coverage[n])
            n_partial = n_poor - n_in_gap
            # a transit in a data gap (e.g. a missing quarter) reflects the data;
            # partial coverage within observed data can come from the pipeline
            # (e.g. clipping in the transit window), so only it warns
            logger.log(
                logging.WARNING if n_partial > 0.2 * ntot else logging.INFO,
                "planet %d: %d of %d transits lack coverage "
                "(%d in data gaps, %d partially covered)",
                n,
                n_poor,
                ntot,
                n_in_gap,
                n_partial,
            )

        return good_coverage

    def _check_rms(self, rel_size=0.0, abs_size=0.0, sigma_cut=5.0):
        """
        Per planet, a bool array over transits: True for normal in-window noise

        A transit fails if its flux rms (MAD) in a window of full width
        max(rel_size * T14, abs_size) is > sigma_cut from the median over transits
        of the same obsmode, or if the window has no cadences. Transits that
        overlap another planet (same tolerances) and obsmodes with < 8 measured
        transits are not checked.
        """
        if rel_size == 0 and abs_size == 0:
            raise ValueError("either rel_size or abs_size must be nonzero")

        lc = self.litecurve

        good_noise = [None] * self.npl
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

            good_noise[n] = np.ones(len(p.transit_times.ttime_mod), dtype="bool")

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
                good_noise[n][use] &= np.abs(rms[use] - rms_mu) / rms_sd < sigma_cut

        return good_noise

    def _drop_planets_with_few_transits(self):
        """
        Drop planets with < 3 good transits (too few for an ephemeris)

        Also remove cadences within 1.5 T14 of their transits, so they cannot bias
        the detrending or the other planets. Returns (planets, litecurve); raises
        UnsupportedTargetError if no planet would remain.
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
                raise UnsupportedTargetError(f"planet {n}: {ngood} good transits (< 3)")
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
            raise UnsupportedTargetError("every planet has < 3 good transits")
        lc.select(keep_cadence)

        return [p for p, k in zip(self.planets, keep_planet) if k], lc

    def perform(self):
        """Set the photometry flags, then drop planets with < 3 good transits"""
        planets = self.planets

        good_coverage = self._check_coverage()

        good_noise = self._check_rms(rel_size=3.0, abs_size=2 / 24, sigma_cut=5.0)

        for n, p in enumerate(planets):
            assert len(good_coverage[n]) == len(p.transit_times.ttime_mod)
            assert len(good_noise[n]) == len(p.transit_times.ttime_mod)

            n_poor_coverage = np.sum(~good_coverage[n])
            n_high_noise = np.sum(~good_noise[n] & good_coverage[n])
            logger.info(
                "planet %d: %d of %d transits flagged low quality "
                "(%d poor coverage, %d high noise)",
                n,
                n_poor_coverage + n_high_noise,
                len(p.transit_times.ttime_mod),
                n_poor_coverage,
                n_high_noise,
            )

            # flag rather than drop: rejected transits stay in transit masks,
            # overlap checks, and the transit model, but are not fit or measured
            tt = p.transit_times
            fields = {k: getattr(tt, k) for k in FIELDS}
            fields["phot_coverage"] = good_coverage[n]
            fields["phot_noise"] = good_noise[n]
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
