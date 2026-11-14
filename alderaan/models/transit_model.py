__all__ = ["TransitModel"]

import logging
from copy import copy

import numpy as np
from batman import _quadratic_ld, _rsky

from alderaan.constants import KEPLER_LCIT, KEPLER_SCIT
from alderaan.utils.astro import transit_obsmode

logger = logging.getLogger(__name__)


class TransitModel:
    """
    Transit forward model for a star, planets, and litecurve

    model_flux(theta) uses the parameterization
        theta = num_planets * [C0, C1, rp, b, T14] + [q1, q2]
     * linear perturbation (C0,C1) on a fixed ephemeris
     * shape parameters (Rp/Rs, b, T14)
     * limb darkening (q1,q2) in the Kipping 2013 basis
    """

    def __init__(self, star, planets, litecurve):
        # trim() rebinds litecurve arrays, so a shallow copy protects the caller
        self.star = star
        self.planets = list(planets)
        self.litecurve = copy(litecurve)
        self.npl = len(planets)

        self.exptime = {
            "long cadence": KEPLER_LCIT,
            "short cadence": KEPLER_SCIT,
        }
        self.transit_obsmode, self.unique_obsmodes = transit_obsmode(
            self.planets, self.litecurve
        )

        self._init_supersampling()
        self._set_cadence_masks()
        self._init_time_warping()

    def _init_supersampling(self):
        lc = self.litecurve

        # ingress/egress timescale estimate following Winn 2010
        period = np.array([p.period for p in self.planets])
        depth = np.array([p.depth for p in self.planets])
        tau12 = (13 / 24) * (period / 365.25) ** (1 / 3) * np.sqrt(depth)

        # sigma so binning error is < 0.1% of photometric uncertainty
        sigma = np.nanmean(lc.error / lc.flux) * 0.04

        # supersample factor (forced odd, minimum 7) following Kipping 2010,
        # and exposure time integration offsets; short cadence is not supersampled
        self.supersample = {}
        self.exptime_offsets = {}
        for om in self.unique_obsmodes:
            if om == "short cadence":
                self.supersample[om] = 1
                self.exptime_offsets[om] = np.array([0.0])
                continue

            exptime = self.exptime[om]
            supersample = np.array(
                np.ceil(np.sqrt((depth / tau12) * (exptime / 8 / sigma))), dtype=int
            )
            self.supersample[om] = np.max(
                np.hstack([supersample + (supersample % 2 + 1), 7])
            )
            self.exptime_offsets[om] = np.linspace(
                -exptime / 2, exptime / 2, self.supersample[om]
            )
            logger.debug(
                "supersample (%s) = %d (min tau12 = %.2f min, sigma = %.2e)",
                om,
                self.supersample[om],
                np.min(tau12) * 1440,
                sigma,
            )

    def _set_cadence_masks(self):
        self.cadence_mask = {
            om: self.litecurve.obsmode == om for om in self.unique_obsmodes
        }

    def trim(self, mask):
        """
        Keep only cadences where mask is True; supersampling is not recomputed
        """
        lc = self.litecurve
        logger.debug("trimmed litecurve to %d of %d cadences", np.sum(mask), len(mask))

        for k in lc.__dict__.keys():
            if type(lc.__dict__[k]) is np.ndarray:
                lc.__setattr__(k, lc.__dict__[k][mask])

        self._set_cadence_masks()
        self._set_bins()
        self._set_warps()

        return self

    def _init_time_warping(self):
        """
        This function initializes warped time arrays
        * the litecurve.time vector is copied for each of the N planets
        * times are then "warped" to account for transit timing variations
        * the transit model assumes linear perturbations to a fixed ephemeris

        IMPORTANT!!! The warp functions assume zero-indexing on transit indexes
        """
        # track static ephemerides
        self.static_period = np.array([p.reference_period for p in self.planets])
        self.static_epoch = np.array([p.reference_epoch for p in self.planets])

        # set warping bins
        self._set_bins()
        self._set_warps()

    def _set_bins(self):
        self._bin_edges = [None] * self.npl
        self._bin_values = [None] * self.npl

        ttvs = [None] * self.npl
        for n, p in enumerate(self.planets):
            ttvs[n] = np.zeros_like(p.transit_times)

        self._update_bins(ttvs)

    def _update_bins(self, ttvs):
        for (
            n,
            p,
        ) in enumerate(self.planets):
            index_full = np.arange(0, p.transit_index.max() + 1, dtype=int)
            ttime_full = self.static_epoch[n] + self.static_period[n] * index_full
            ttime_full[p.transit_index] = p.transit_times + ttvs[n]

            self._bin_edges[n] = np.concatenate(
                [
                    [ttime_full[0] - 0.5 * self.static_period[n]],
                    0.5 * (ttime_full[1:] + ttime_full[:-1]),
                    [ttime_full[-1] + 0.5 * self.static_period[n]],
                ]
            )

            self._bin_values[n] = np.concatenate(
                [[ttime_full[0]], ttime_full, [ttime_full[-1]]]
            )

    def _set_warps(self):
        self.warped_time = [None] * self.npl  # warp lc.time vector for each planet
        self.warped_legx = [None] * self.npl  # legx is 1st-order Legendre polynomial

        self._update_warps()

    def _update_warps(self):
        for n, p in enumerate(self.planets):
            index_midpoint = p.transit_index[-1] // 2

            _t, _i = self._warp_times(self.litecurve.time, n, return_inds=True)
            _x = (_i - index_midpoint) / (p.transit_index[-1] / 2)

            self.warped_time[n] = _t  # len(self.litecurve.ttime)
            self.warped_legx[n] = _x  # len(self.litecurve.ttime)

        self._set_obsmode_slices()

    def _set_obsmode_slices(self):
        # precomputed so model_flux avoids boolean-mask copies on every call
        self._warped_time_om = {}
        self._warped_legx_om = {}
        for om in self.unique_obsmodes:
            mask = self.cadence_mask[om]
            self._warped_time_om[om] = [wt[mask] for wt in self.warped_time]
            self._warped_legx_om[om] = [wx[mask] for wx in self.warped_legx]

    def _get_model_dt(self, t, planet_no, return_inds=False):
        _inds = np.searchsorted(self._bin_edges[planet_no], t)
        _vals = self._bin_values[planet_no][_inds]

        if return_inds:
            return _vals, _inds
        return _vals

    def _warp_times(self, t, planet_no, return_inds=False):
        warps = self._get_model_dt(t, planet_no, return_inds=return_inds)

        if return_inds:
            return t - warps[0], warps[1]
        else:
            return t - warps

    @staticmethod
    def transit_flux(
        t, t0, period, ror, impact, duration, u1, u2, exptime_offsets=None
    ):
        """
        Quadratic limb darkened transit flux for a single planet

        If exptime_offsets is provided, flux is supersampled over each exposure
        and averaged back to the cadence times t
        """
        if exptime_offsets is not None:
            t = (exptime_offsets + t.reshape(t.size, 1)).ravel()

        nthreads = 1
        ds = _rsky._rsky(t, t0, period, ror, impact, duration, 1, nthreads)
        flux = _quadratic_ld._quadratic_ld(ds, np.abs(ror), u1, u2, nthreads)

        if exptime_offsets is not None:
            flux = np.mean(flux.reshape(-1, len(exptime_offsets)), axis=1)

        return flux

    def model_flux(self, theta):
        """
        theta : array-like
            num_planets * [C0, C1, rp, b, T14] + [q1, q2]

        Limb darkening is sampled in terms of [q1, q2] (see Kipping 2013)
        """
        q1, q2 = np.array(theta[-2:])
        u1 = 2 * np.sqrt(q1) * q2
        u2 = np.sqrt(q1) * (1 - 2 * q2)

        flux_mod = None
        for obsmode in self.unique_obsmodes:
            mask = self.cadence_mask[obsmode]
            flux_om = self._model_flux_obsmode(theta, obsmode, u1, u2)

            if mask.all():
                return flux_om

            if flux_mod is None:
                flux_mod = np.ones_like(self.litecurve.flux)
            flux_mod[mask] = flux_om

        return flux_mod

    def _model_flux_obsmode(self, theta, obsmode, u1, u2):
        warped_time = self._warped_time_om[obsmode]
        warped_legx = self._warped_legx_om[obsmode]

        flux_mod = np.ones(len(warped_time[0]))

        for n in range(self.npl):
            C0, C1, rp, b, T14 = np.array(theta[5 * n : 5 * (n + 1)])

            _t = warped_time[n] + C0 + C1 * warped_legx[n]

            qld_flux = self.transit_flux(
                _t,
                0.0,
                self.static_period[n],
                rp,
                b,
                T14,
                u1,
                u2,
                exptime_offsets=self.exptime_offsets[obsmode],
            )

            flux_mod += qld_flux - 1.0

        return flux_mod
