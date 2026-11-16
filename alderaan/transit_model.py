__all__ = [
    "PARAMETER_LABELS",
    "TransitModel",
    "transit_flux",
]

import logging
from copy import copy

import numpy as np
from batman import _quadratic_ld, _rsky

from alderaan.constants import EXPTIME
from alderaan.utils.astro import ld_q_to_u, ld_u_to_q

logger = logging.getLogger(__name__)

# plot labels for the five per-planet entries of theta
PARAMETER_LABELS = "C0 C1 r b T14".split()


def transit_flux(t, t0, period, ror, impact, duration, u1, u2, exptime_offsets=None):
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


class TransitModel:
    """
    Transit forward model for a star, planets, and litecurve

    model_flux(theta) uses the parameterization
        theta = npl * [C0, C1, ror, b, T14] + [q1, q2]
     * linear perturbation (C0,C1) on a fixed ephemeris
     * shape parameters (ror = Rp/Rs, impact b, duration T14)
     * limb darkening (q1,q2) in the Kipping 2013 basis
    """

    def __init__(self, star, planets, litecurve):
        # trim() rebinds litecurve arrays, so a shallow copy protects the caller
        self.star = star
        self.planets = list(planets)
        self.litecurve = copy(litecurve)
        self.npl = len(planets)

        for n, p in enumerate(self.planets):
            if p.static_period is None or p.static_epoch is None:
                raise ValueError(
                    f"planet {n}: static period/epoch not set; "
                    "call Planet.set_static_period_and_epoch() first"
                )
        self.static_period = np.array([p.static_period for p in self.planets])
        self.static_epoch = np.array([p.static_epoch for p in self.planets])

        # Legendre mapping of transit index, legx = (index - index_mid) / index_half
        self.index_mid = np.array(
            [p.transit_times.index[-1] // 2 for p in self.planets]
        )
        self.index_half = np.array(
            [p.transit_times.index[-1] / 2 for p in self.planets]
        )

        self.transit_obsmode = [
            self.litecurve.obsmode_at(p.transit_times.ttime_mod) for p in self.planets
        ]
        self.unique_obsmodes = np.unique(np.hstack(self.transit_obsmode))

        self._init_supersampling()
        self._set_cadence_masks()
        self._build_warps()

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

            exptime = EXPTIME[om]
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
        logger.debug("trimmed litecurve to %d of %d cadences", np.sum(mask), len(mask))
        self.litecurve.select(mask)

        self._set_cadence_masks()
        self._build_warps()

    def _build_warps(self):
        """
        Warp litecurve times so each planet's transits line up on a fixed ephemeris

        For each planet, every cadence is assigned to its nearest transit: the bin
        edges sit halfway between consecutive transit times. Within a bin,
          * warped_time = time - (that transit's time), so the transit is at 0
          * warped_legx = Legendre mapping of the bin number, used by the linear
            ephemeris perturbation C0 + C1 * legx; the bin number is the transit
            index + 1 (see notes/bugfixes.md S5)
        Missing transit indexes fall back to the static ephemeris, and cadences
        before the first or after the last transit join the end bins.

        Transit indexes must be zero-based.
        """
        time = self.litecurve.time

        self.warped_time = [None] * self.npl
        self.warped_legx = [None] * self.npl

        for n, p in enumerate(self.planets):
            period = self.static_period[n]

            # transit time for every index 0..max; observed where available
            index_full = np.arange(0, p.transit_times.index.max() + 1, dtype=int)
            ttime_full = self.static_epoch[n] + period * index_full
            ttime_full[p.transit_times.index] = p.transit_times.ttime_mod

            # bins halfway between transits, plus one end bin on each side
            bin_edges = np.concatenate(
                [
                    [ttime_full[0] - 0.5 * period],
                    0.5 * (ttime_full[1:] + ttime_full[:-1]),
                    [ttime_full[-1] + 0.5 * period],
                ]
            )
            bin_ttime = np.concatenate([[ttime_full[0]], ttime_full, [ttime_full[-1]]])

            # bin number of each cadence (bin 0 is the leading end bin)
            bin_number = np.searchsorted(bin_edges, time)

            self.warped_time[n] = time - bin_ttime[bin_number]
            self.warped_legx[n] = (bin_number - self.index_mid[n]) / self.index_half[n]

        # per-obsmode slices, precomputed so model_flux avoids boolean-mask copies
        self._warped_time_om = {}
        self._warped_legx_om = {}
        for om in self.unique_obsmodes:
            mask = self.cadence_mask[om]
            self._warped_time_om[om] = [wt[mask] for wt in self.warped_time]
            self._warped_legx_om[om] = [wx[mask] for wx in self.warped_legx]

    def theta_initial(self):
        """
        Starting parameters from the planets' depths and durations

        Returns
        -------
        theta : ndarray
            npl * [C0, C1, ror, b, T14] + [q1, q2]
        """
        theta = []
        for n, p in enumerate(self.planets):
            if not p.depth > 0:
                raise ValueError(
                    f"planet {n}: depth = {p.depth} must be positive to initialize ror"
                )
            theta.append(0.0)
            theta.append(0.0)
            theta.append(np.sqrt(p.depth))
            theta.append(0.5)
            theta.append(p.duration)

        # physical (u1, u2) --> sampling basis (q1, q2), see Kipping 2013
        theta.extend(ld_u_to_q(self.star.u1, self.star.u2))

        return np.array(theta)

    def model_flux(self, theta):
        """
        theta : array-like
            npl * [C0, C1, ror, b, T14] + [q1, q2]

        Limb darkening is sampled in terms of [q1, q2] (see Kipping 2013)
        """
        u1, u2 = ld_q_to_u(*np.array(theta[-2:]))

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
            C0, C1, ror, b, T14 = np.array(theta[5 * n : 5 * (n + 1)])

            _t = warped_time[n] + C0 + C1 * warped_legx[n]

            qld_flux = transit_flux(
                _t,
                0.0,
                self.static_period[n],
                ror,
                b,
                T14,
                u1,
                u2,
                exptime_offsets=self.exptime_offsets[obsmode],
            )

            flux_mod += qld_flux - 1.0

        return flux_mod

    def residuals(self, theta):
        """Normalized residuals (observed - model) / error over self.litecurve"""
        flux_mod = self.model_flux(theta)
        flux_obs = self.litecurve.flux
        flux_err = self.litecurve.error

        return (flux_obs - flux_mod) / flux_err

    def ln_likelihood(self, theta):
        """
        Gaussian log-likelihood of theta, plus a prior tying limb darkening to the
        star's (u1, u2); returns -1e300 where the result is not finite
        """
        lnlike = -0.5 * np.sum(self.residuals(theta) ** 2)

        # enforce prior on physical limb darkening (see Kipping 2013)
        u1, u2 = ld_q_to_u(theta[-2], theta[-1])

        sig_ld_sq = 0.01
        lnlike -= 1.0 / (2 * sig_ld_sq) * (u1 - self.star.u1) ** 2
        lnlike -= 1.0 / (2 * sig_ld_sq) * (u2 - self.star.u2) ** 2

        if not np.isfinite(lnlike):
            return -1e300

        return lnlike
