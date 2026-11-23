"""
Transit forward model (quadratic limb darkening via batman primitives)

Transit basis used across the pipeline (other docstrings refer here):
    period (P), epoch (t0) : linear ephemeris, days
    ror : planet-to-star radius ratio Rp/Rs
    impact (b) : impact parameter
    duration (T14) : first-to-fourth contact duration, days
    q1, q2 : quadratic limb darkening in the Kipping 2013 basis; u1, u2 are the
        standard quadratic coefficients (utils.astro.ld_q_to_u)
"""

__all__ = [
    "TRANSIT_PARAMETER_LABELS",
    "TransitModel",
    "exposure_offsets",
    "transit_flux",
]

import logging
from copy import copy

import numpy as np
from batman import _quadratic_ld, _rsky

from alderaan.constants import EXPTIME, KEPLER_LCIT
from alderaan.utils.astro import ld_q_to_u
from alderaan.utils.stats import legendre_p1

logger = logging.getLogger(__name__)

# plot labels for the five per-planet entries of theta
TRANSIT_PARAMETER_LABELS = "C0 C1 r b T14".split()


def exposure_offsets(exptime, n):
    """Midpoints of n equal sub-exposures, relative to the exposure center"""
    return (np.arange(n) + 0.5) / n * exptime - exptime / 2


def transit_flux(t, t0, period, ror, impact, duration, u1, u2, exptime_offsets=None):
    """
    Quadratic limb darkened transit flux of one planet at times t (days)

    The other arguments are the transit basis (module docstring). If
    exptime_offsets (days; see exposure_offsets) is given, flux is supersampled
    over each exposure and averaged back to the times t.
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

    Parameters
    ----------
    star : Star
    planets : list of Planet
        static period and epoch must be set (Planet.set_static_period_and_epoch)
    litecurve : LiteCurve
        shallow-copied, so trim() leaves the caller's litecurve unchanged

    Notes
    -----
    theta = npl * [C0, C1, ror, b, T14] + [q1, q2]
     * C0 + C1 * legendre_p1(T, (t_start, t_end)) shifts each transit time,
       where T is the transit's time on the planet's linear ephemeris and
       (t_start, t_end) is the photometry span: C0 is the offset (days) at
       mid-span, C1 the additional offset (days) at t_end
     * ror, b, T14, q1, q2: the transit basis (module docstring)
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

        # photometry span for the Legendre mapping of C0, C1 (fixed on the
        # litecurve, so outlier clipping and trim() do not change it)
        self.t_start = self.litecurve.t_start
        self.t_end = self.litecurve.t_end

        self.transit_obsmode = [
            self.litecurve.obsmode_at(p.transit_times.ttime_mod) for p in self.planets
        ]
        # from the cadences themselves, so every cadence is modeled even where a
        # transit straddles an obsmode boundary
        self.unique_obsmodes = np.unique(self.litecurve.obsmode)

        self._init_supersampling()
        self._set_cadence_masks()
        self._build_warps()

    def _init_supersampling(self):
        """Set the odd supersample factor and sub-exposure offsets per obsmode"""
        lc = self.litecurve

        # ingress/egress timescale estimate following Winn 2010
        period = np.array([p.period for p in self.planets])
        ror = np.array([p.ror for p in self.planets])
        depth = ror**2
        tau12 = (13 / 24) * (period / 365.25) ** (1 / 3) * ror

        self.supersample = {}
        self.exptime_offsets = {}
        for om in self.unique_obsmodes:
            exptime = EXPTIME[om]
            # Kipping 2010; 0.04 sigma inflates the effective noise < 0.1%
            use = lc.obsmode == om
            sigma = np.mean(lc.error[use] / lc.flux[use]) * 0.04
            n_noise = np.sqrt((depth / tau12) * (exptime / 8 / sigma))
            # integration error < 1% of depth at short periods; error ~ exptime / N^2
            n_min = np.maximum(7, 13 * period ** (-1 / 6)) * np.sqrt(
                exptime / KEPLER_LCIT
            )
            n = int(np.ceil(np.max(np.hstack([n_noise, n_min]))))
            self.supersample[om] = n + (1 - n % 2)  # odd
            self.exptime_offsets[om] = exposure_offsets(exptime, self.supersample[om])
            logger.debug(
                "supersample (%s) = %d (min tau12 = %.2f min, sigma = %.2e)",
                om,
                self.supersample[om],
                np.min(tau12) * 1440,
                sigma,
            )

    def _set_cadence_masks(self):
        """Set the boolean cadence mask of each obsmode"""
        self.cadence_mask = {
            om: self.litecurve.obsmode == om for om in self.unique_obsmodes
        }

    def trim(self, mask):
        """Keep only cadences where mask is True; supersampling is not recomputed"""
        logger.debug("trimmed litecurve to %d of %d cadences", np.sum(mask), len(mask))
        self.litecurve.select(mask)
        self.unique_obsmodes = np.unique(self.litecurve.obsmode)

        self._set_cadence_masks()
        self._build_warps()

    def _build_warps(self):
        """
        Warp litecurve times so each planet's transits line up on a fixed ephemeris

        Each cadence is assigned to its nearest transit k (bin edges halfway
        between consecutive transit times), then
          * warped_time = time - (transit k's time)
          * warped_legx = legendre_p1(ref_epoch + ref_period * k, (t_start, t_end)),
            used by the perturbation C0 + C1 * warped_legx
        Missing transit indexes use the least squares ephemeris; cadences
        outside the first/last transit join those transits.

        Transit indexes must be zero-based.
        """
        time = self.litecurve.time

        # least squares ephemeris of the transit times warped here; the Legendre
        # mapping and Posteriors use this same ephemeris
        self.ref_period = np.array([p.period for p in self.planets])
        self.ref_epoch = np.array([p.epoch for p in self.planets])

        self.warped_time = [None] * self.npl
        self.warped_legx = [None] * self.npl

        for n, p in enumerate(self.planets):
            period = self.ref_period[n]

            # transit time for every index 0..max; modeled where available
            index_full = np.arange(0, p.transit_times.index.max() + 1, dtype=int)
            ttime_full = self.ref_epoch[n] + period * index_full
            ttime_full[p.transit_times.index] = p.transit_times.ttime_mod

            bin_edges = np.concatenate(
                [
                    [ttime_full[0] - 0.5 * period],
                    0.5 * (ttime_full[1:] + ttime_full[:-1]),
                    [ttime_full[-1] + 0.5 * period],
                ]
            )

            # transit index of each cadence; bin 0 and the last bin are the end
            # bins, which join the first and last transits
            bin_number = np.searchsorted(bin_edges, time)
            transit_index = np.clip(bin_number - 1, 0, index_full[-1])

            self.warped_time[n] = time - ttime_full[transit_index]
            self.warped_legx[n] = legendre_p1(
                self.ref_epoch[n] + self.ref_period[n] * transit_index,
                (self.t_start, self.t_end),
            )

        # per-obsmode slices, precomputed so model_flux avoids boolean-mask copies
        self._warped_time_om = {}
        self._warped_legx_om = {}
        for om in self.unique_obsmodes:
            mask = self.cadence_mask[om]
            self._warped_time_om[om] = [wt[mask] for wt in self.warped_time]
            self._warped_legx_om[om] = [wx[mask] for wx in self.warped_legx]

    def theta_initial(self):
        """
        Starting theta: C0 = C1 = 0, planet ror, impact, duration, star q1, q2

        Returns
        -------
        theta : ndarray
            npl * [C0, C1, ror, b, T14] + [q1, q2]
        """
        theta = []
        for p in self.planets:
            theta.append(0.0)
            theta.append(0.0)
            theta.append(p.ror)
            theta.append(p.impact)
            theta.append(p.duration)

        theta.extend([self.star.q1, self.star.q2])

        return np.array(theta)

    def model_flux(self, theta):
        """Model flux at every litecurve cadence for theta (see class docstring)"""
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
        """Model flux at the cadences of one obsmode"""
        warped_time = self._warped_time_om[obsmode]
        warped_legx = self._warped_legx_om[obsmode]

        flux_mod = np.ones(len(warped_time[0]))

        for n in range(self.npl):
            C0, C1, ror, b, T14 = np.array(theta[5 * n : 5 * (n + 1)])

            # transit at ttime_mod + C0 + C1 * legx
            _t = warped_time[n] - (C0 + C1 * warped_legx[n])

            # _rsky gives flux exactly 1 beyond |t_phi| > T14, so skipping cadences
            # whose every sub-exposure lies there is exact
            period = self.static_period[n]
            t_phi = _t - period * np.round(_t / period)
            near = np.abs(t_phi) <= T14 + EXPTIME[obsmode]

            qld_flux = transit_flux(
                _t[near],
                0.0,
                self.static_period[n],
                ror,
                b,
                T14,
                u1,
                u2,
                exptime_offsets=self.exptime_offsets[obsmode],
            )

            flux_mod[near] += qld_flux - 1.0

        return flux_mod

    def residuals(self, theta):
        """Normalized residuals (observed - model) / error over self.litecurve"""
        flux_mod = self.model_flux(theta)
        flux_obs = self.litecurve.flux
        flux_err = self.litecurve.error

        return (flux_obs - flux_mod) / flux_err

    def ln_likelihood(self, theta):
        """
        Gaussian log-likelihood of theta, up to a constant

        If star.ld_prior, adds Gaussian priors of width 0.1 on u1 and u2, centered
        on the star's (u1, u2). Returns -1e300 if the result is not finite.
        """
        lnlike = -0.5 * np.sum(self.residuals(theta) ** 2)

        if self.star.ld_prior:
            u1, u2 = ld_q_to_u(theta[-2], theta[-1])
            sig_ld_sq = 0.01
            lnlike -= 1.0 / (2 * sig_ld_sq) * (u1 - self.star.u1) ** 2
            lnlike -= 1.0 / (2 * sig_ld_sq) * (u2 - self.star.u2) ** 2

        if not np.isfinite(lnlike):
            return -1e300

        return lnlike
