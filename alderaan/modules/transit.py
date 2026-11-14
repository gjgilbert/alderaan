__all__ = [
    "TransitModel",
    "RBDTransitModel",
    "TTVTransitModel",
]

import io
import sys
import time
from copy import deepcopy

import dynesty
import numpy as np
from batman import _quadratic_ld, _rsky
from dynesty.utils import print_fn
from scipy.optimize import least_squares

from alderaan.modules.base import BaseAlg
from alderaan.utils.astro import estimate_transit_depth
from alderaan.utils.stats import loguniform_ppf, norm_ppf, uniform_ppf


class TransitModel(BaseAlg):
    """
    Base transit model class, used for inheritance but not called directly
    """

    def __init__(self, litecurve, planets, star):
        super().__init__(litecurve, planets)

        self.star = deepcopy(star)
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

        # supersample factor (forced odd) following Kipping 2010,
        # and exposure time integration offsets
        self.supersample = {}
        self._exptime_offsets = {}
        for om in self.unique_obsmodes:
            exptime = self.exptime[om]
            supersample = np.array(
                np.ceil(np.sqrt((depth / tau12) * (exptime / 8 / sigma))), dtype=int
            )
            self.supersample[om] = np.max(supersample + (supersample % 2 + 1))
            self._exptime_offsets[om] = np.linspace(
                -exptime / 2, exptime / 2, self.supersample[om]
            )

    def _set_cadence_masks(self):
        self._cadence_mask = {
            om: self.litecurve.obsmode == om for om in self.unique_obsmodes
        }

    def trim_flux_arrays(self, rel_size=None, abs_size=None):
        lc = self.litecurve
        mask = self.make_transit_mask(
            rel_size=rel_size, abs_size=abs_size, mask_type="condensed"
        )

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
        self._static_period = np.array([p.reference_period for p in self.planets])
        self._static_epoch = np.array([p.reference_epoch for p in self.planets])

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
            ttime_full = self._static_epoch[n] + self._static_period[n] * index_full
            ttime_full[p.transit_index] = p.transit_times + ttvs[n]

            self._bin_edges[n] = np.concatenate(
                [
                    [ttime_full[0] - 0.5 * self._static_period[n]],
                    0.5 * (ttime_full[1:] + ttime_full[:-1]),
                    [ttime_full[-1] + 0.5 * self._static_period[n]],
                ]
            )

            self._bin_values[n] = np.concatenate(
                [[ttime_full[0]], ttime_full, [ttime_full[-1]]]
            )

    def _set_warps(self):
        self._warped_time = [None] * self.npl  # warp lc.time vector for each planet
        self._warped_legx = [None] * self.npl  # legx is 1st-order Legendre polynomial

        self._update_warps()

    def _update_warps(self):
        for n, p in enumerate(self.planets):
            index_midpoint = p.transit_index[-1] // 2

            _t, _i = self._warp_times(self.litecurve.time, n, return_inds=True)
            _x = (_i - index_midpoint) / index_midpoint

            self._warped_time[n] = _t.copy()  # len(self.litecurve.ttime)
            self._warped_legx[n] = _x.copy()  # len(self.litecurve.ttime)

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
    def throttled_print_fn(interval=10):
        last = [0]
        start = time.time()

        def callback(results, niter, ncall, *args, **kwargs):
            now = time.time()
            if now - last[0] >= interval:
                runtime = (now - start) / 60

                buf = io.StringIO()
                stdout = sys.stdout
                sys.stdout = buf
                print_fn(results, niter, ncall, *args, **kwargs)
                sys.stdout = stdout

                line = buf.getvalue().rstrip()
                print(f"{line} | runtime: {runtime:.1f} min")
                last[0] = now

        return callback


class RBDTransitModel(TransitModel):
    """
    Class for fitting transit shape parameters Rp/Rs, b, T14

    The transit model has 2 + 5 x num_planets free parameters
     * shape parameters of interest (Rp/Rs, b, T14)
     * linear perturbation (C0,C1) on a fixed ephemeris
     * limb darkening (u1,u2) is fit using informative priors (sigma_u = 0.1)

    Users will typically interact with the optimize() or sample() methods

    Methods:
        optimize : applies least squares optimization
        sample : samples from the posterior using dynamic nested sampling
    """

    def __init__(self, litecurve, planets, star):
        super().__init__(litecurve, planets, star)

    @staticmethod
    def model_flux(theta, transitmodel):
        """
        theta : array-like
            num_planets * [C0, C1, rp, b, T14] + [q1, q2]
        transitmodel : TransitModel
            instance of TransitModel, typically self

        Priors on planet parameters [C0, C1, rp, b, T14] are enforced outside
        this function

        Limb darkening is sampled in terms of [q1, q2] (see Kipping 2013)
        Limb darkening priors are enforced in terms of [u1, u2] inside this function
        """
        tm = transitmodel
        lc = tm.litecurve

        q1, q2 = np.array(theta[-2:])
        u1 = 2 * np.sqrt(q1) * q2
        u2 = np.sqrt(q1) * (1 - 2 * q2)

        flux_mod = np.ones_like(lc.flux)

        for obsmode in tm.unique_obsmodes:
            mask = tm._cadence_mask[obsmode]
            exptime_ioff = tm._exptime_offsets[obsmode]
            supersample = tm.supersample[obsmode]

            for n in range(tm.npl):
                C0, C1, rp, b, T14 = np.array(theta[5 * n : 5 * (n + 1)])

                _t = tm._warped_time[n][mask] + C0 + C1 * tm._warped_legx[n][mask]
                _t_supersample = (exptime_ioff + _t.reshape(_t.size, 1)).flatten()

                nthreads = 1
                ds = _rsky._rsky(
                    _t_supersample,
                    0.0,
                    tm._static_period[n],
                    rp,
                    b,
                    T14,
                    1,
                    nthreads,
                )

                qld_flux = _quadratic_ld._quadratic_ld(ds, np.abs(rp), u1, u2, nthreads)

                qld_flux = np.mean(qld_flux.reshape(-1, supersample), axis=1)

                flux_mod[mask] += qld_flux - 1.0

        return flux_mod

    @staticmethod
    def model_residuals(theta, transitmodel):
        flux_mod = transitmodel.model_flux(theta, transitmodel)
        flux_obs = transitmodel.litecurve.flux
        flux_err = transitmodel.litecurve.error

        return (flux_obs - flux_mod) / flux_err

    @staticmethod
    def ln_likelihood(theta, transitmodel):
        # calculate likelihood
        tm = transitmodel
        lnlike = -0.5 * np.sum(tm.model_residuals(theta, tm) ** 2)

        # enforce prior on physical limb darkening (see Kipping 2013)
        q1, q2 = theta[-2], theta[-1]
        u1 = 2 * np.sqrt(q1) * q2
        u2 = np.sqrt(q1) * (1 - 2 * q2)

        sig_ld_sq = 0.01
        lnlike -= 1.0 / (2 * sig_ld_sq) * (u1 - tm.star.u1) ** 2
        lnlike -= 1.0 / (2 * sig_ld_sq) * (u2 - tm.star.u2) ** 2

        if not np.isfinite(lnlike):
            return -1e300

        return lnlike

    @staticmethod
    def prior_transform(uniform_hypercube, transitmodel):
        """
        Prior transform over physical ALDERAAN basis {C0, C1, r, b, T14}...{q1,q2}

        Distributions are hard-coded to be:
            Normal on ephemeris perturbations {C0,C1}
            Log-uniform on radius ratio r and transit duration T14
            Uniform on impact parameter b
            Uniform on quadratic limb darkening coefficients {q1,q2}

        For motivation behind this prior-parameter choice see:
            Carter+ 2008 (2008ApJ...689..499C)
            Kipping 2013 (2013MNRAS.435.2152K)
            Gilbert, MacDougall, & Petigura 2022 (2022AJ....164...92G)
            MacDougal, Gilbert, & Petigura 2023 (2023AJ....166...61M)

        Assumes that transit durations are known with reasonable confidence
        * sampled T14 will be restricted between (0.01*fixed_duration, 3*fixed_duration)

        Parameters
        ----------
            uniform_hypercube : array-like, length 5 * N + 2
                list of parameters N * [C0, C1, r, b, T14] + [q1, q2]
            fixed_durations : array-like
                list of transit durations for N planets (used for setting prior limits)

        Returns
        -------
            transformed_hypercube : array_like, length 5 * N + 2
                transformed samples
        """
        tm = transitmodel

        u = np.array(uniform_hypercube)  # U(0,1) priors
        x = np.zeros_like(u)  # physical priors

        # 5 * npl + 2 parameters: [C0, C1, r, b, T14]...[q1, q2]
        for n in range(tm.npl):
            x[5 * n + 0] = norm_ppf(u[0 + n * 5], 0.0, 0.1)
            x[5 * n + 1] = norm_ppf(u[1 + n * 5], 0.0, 0.1)
            x[5 * n + 2] = loguniform_ppf(u[2 + n * 5], 1e-5, 0.99)
            x[5 * n + 3] = uniform_ppf(u[3 + n * 5], 0.0, 1 + x[5 * n + 2])
            x[5 * n + 4] = loguniform_ppf(
                u[4 + n * 5], 0.01 * tm.planets[n].duration, 3 * tm.planets[n].duration
            )

        # limb darkening coefficients (see Kipping 2013)
        x[-2] = uniform_ppf(u[-2], 0, 1)
        x[-1] = uniform_ppf(u[-1], 0, 1)

        return x

    def sample(self, checkpoint_file=None, checkpoint_every=60, progress_every=10):
        ndim = 5 * self.npl + 2
        sampler = dynesty.DynamicNestedSampler(
            self.ln_likelihood,
            self.prior_transform,
            ndim,
            bound="multi",
            sample="rwalk",
            logl_args=(self,),
            ptform_args=(self,),
        )
        sampler.run_nested(
            checkpoint_file=checkpoint_file,
            checkpoint_every=checkpoint_every,
            print_progress=True,
            print_func=self.throttled_print_fn(progress_every),
        )

        return sampler.results

    def optimize(self, fix_limbdark=True, niter=3):
        theta = self._theta_initial()
        bounds = self._theta_bounds()

        def _fxn(x, x0, fix, self):
            _x = x0.copy()
            _x[~fix] = x
            return self.model_residuals(_x, self)

        i_fix = np.arange(len(theta), dtype=int)
        var_names = np.array("C0 C1 r b T14".split())

        # fix ephemeris, vary [r, b, T14]
        fix_C0_C1 = np.zeros(len(i_fix), dtype=bool)
        fix_C0_C1[i_fix % 5 == 0] = True
        fix_C0_C1[i_fix % 5 == 1] = True

        # fix [r, b, T14], vary ephemeris
        fix_r_b_T14 = np.zeros(len(i_fix), dtype=bool)
        fix_r_b_T14[i_fix % 5 == 2] = True
        fix_r_b_T14[i_fix % 5 == 3] = True
        fix_r_b_T14[i_fix % 5 == 4] = True

        # limb darkening
        fix_C0_C1[-2:] = fix_limbdark
        fix_r_b_T14[-2:] = fix_limbdark

        # do the optimization
        for fix in [fix_r_b_T14, fix_C0_C1] * niter:
            print(f"optimizing logp for variables: [{', '.join(var_names[~fix[:5]])}]")

            logp_initial = self.ln_likelihood(theta, self).copy()

            result = least_squares(
                _fxn,
                theta[~fix],
                method="trf",
                bounds=bounds[~fix, :].T,
                args=[theta, fix, self],
            )

            theta[~fix] = result.x.copy()
            logp_final = self.ln_likelihood(theta, self).copy()

            print(f"logp: {logp_initial} -> {logp_final}")
            print(f"message: {result['message']}")

        return theta

    def _theta_initial(self):
        """
        theta : array-like
            num_planets * [C0, C1, rp, b, T14] + [q1, q2]
        """
        theta = []
        for p in self.planets:
            theta.append(0.0)
            theta.append(0.0)
            theta.append(np.sqrt(p.depth))
            theta.append(0.5)
            theta.append(p.duration)

        # physical (u1, u2) --> sampling basis (q1, q2), see Kipping 2013
        u1, u2 = self.star.u1, self.star.u2
        theta.append((u1 + u2) ** 2)
        theta.append(u1 / (2 * (u1 + u2)))

        return np.array(theta)

    def _theta_bounds(self):
        """
        theta : array-like
            num_planets * [C0, C1, rp, b, T14] + [q1, q2]
        """
        bounds = []
        for p in self.planets:
            bounds.append([-np.inf, np.inf])
            bounds.append([-np.inf, np.inf])
            bounds.append([1e-5, 0.99])
            bounds.append([0.0, 1.0])
            bounds.append([0.01 * p.duration, 3 * p.duration])

        bounds.append([0.0, 1.0])
        bounds.append([0.0, 1.0])

        return np.array(bounds)

    def update_planet_parameters(self, theta):
        for n, p in enumerate(self.planets):
            self.planets[n].ror = theta[2 + n * 5]
            self.planets[n].impact = theta[3 + n * 5]
            self.planets[n].duration = theta[4 + n * 5]
            self.planets[n].depth = estimate_transit_depth(p.ror, p.impact)

        return self.planets

    def update_star_parameters(self, theta):
        # sampling basis (q1, q2) --> physical (u1, u2), see Kipping 2013
        q1, q2 = theta[-2], theta[-1]
        self.star.u1 = 2 * np.sqrt(q1) * q2
        self.star.u2 = np.sqrt(q1) * (1 - 2 * q2)

        return self.star


class TTVTransitModel(TransitModel):
    """
    Class for fitting all transit times simultaneously
    Transit shape parameters and limb darkening are held fixed

    Users will typically interact with the optimize() or sample() methods

    Methods:
        optimize : applies least squares optimization
        sample : samples from the posterior using dynamic nested sampling
    """

    def __init__(self, litecurve, planets, star):
        super().__init__(litecurve, planets, star)

        self.num_transits = self._count_transits()

    def _count_transits(self):
        self.num_transits = np.zeros(self.npl, dtype=int)
        for n, p in enumerate(self.planets):
            self.num_transits[n] = len(p.transit_times)
        return self.num_transits

    @staticmethod
    def model_flux(theta, transitmodel):
        """
        theta : array-like
            ttvs
        transitmodel : TransitModel
            instance of TransitModel, typically self

        Priors on planet ttvs are enforced outside this function
        Assumes TTVs are small relative to period
        """
        tm = transitmodel
        lc = tm.litecurve

        iend = np.cumsum(tm.num_transits)
        istart = iend - tm.num_transits

        flux_mod = np.ones_like(lc.flux)

        for obsmode in tm.unique_obsmodes:
            mask = tm._cadence_mask[obsmode]
            exptime_ioff = tm._exptime_offsets[obsmode]
            supersample = tm.supersample[obsmode]

            for n, p in enumerate(tm.planets):
                ttvs = np.array(theta[istart[n] : iend[n]])

                tm._update_bins(ttvs)
                tm._update_warps()

                _t = tm._warped_time[n][mask]
                _t_supersample = (exptime_ioff + _t.reshape(_t.size, 1)).flatten()

                nthreads = 1
                ds = _rsky._rsky(
                    _t_supersample,
                    0.0,
                    p.period,
                    p.ror,
                    p.impact,
                    p.duration,
                    1,
                    nthreads,
                )

                qld_flux = _quadratic_ld._quadratic_ld(
                    ds, np.abs(p.ror), tm.star.u1, tm.star.u2, nthreads
                )

                qld_flux = np.mean(qld_flux.reshape(-1, supersample), axis=1)

                flux_mod[mask] += qld_flux - 1.0

        return flux_mod

    @staticmethod
    def model_residuals(theta, transitmodel):
        flux_mod = transitmodel.model_flux(theta, transitmodel)
        flux_obs = transitmodel.litecurve.flux
        flux_err = transitmodel.litecurve.error

        return (flux_obs - flux_mod) / flux_err

    @staticmethod
    def ln_likelihood(theta, transitmodel):
        tm = transitmodel
        lnlike = -0.5 * np.sum(tm.model_residuals(theta, tm) ** 2)

        if not np.isfinite(lnlike):
            return -1e300

        return lnlike

    @staticmethod
    def prior_transform(uniform_hypercube, transitmodel):
        """
        Docstring
        """
        tm = transitmodel

        u = np.array(uniform_hypercube)  # U(0,1) priors
        x = np.zeros_like(u)  # physical priors

        iend = np.cumsum(tm.num_transits)
        istart = iend - tm.num_transits

        for n in range(tm.npl):
            x[istart[n] : iend[n]] = uniform_ppf(
                u[istart[n] : iend[n]], -tm.planets[n].duration, tm.planets[n].duration
            )

        return x

    def sample(self, checkpoint_file=None, checkpoint_every=60, progress_every=10):
        ndim = np.sum(self.num_transits)
        sampler = dynesty.DynamicNestedSampler(
            self.ln_likelihood,
            self.prior_transform,
            ndim,
            bound="multi",
            sample="rwalk",
            logl_args=(self,),
            ptform_args=(self,),
        )
        sampler.run_nested(
            checkpoint_file=checkpoint_file,
            checkpoint_every=checkpoint_every,
            print_progress=True,
            print_func=self.throttled_print_fn(progress_every),
        )

        return sampler.results

    def optimize(self, niter=1):
        theta = self._theta_initial()
        bounds = self._theta_bounds()

        def _fxn(x, x0, fix, self):
            _x = x0.copy()
            _x[~fix] = x
            return self.model_residuals(_x, self)

        for i in range(np.sum(self.num_transits)):
            print(f"optimizing transit {i} of {len(theta)}")

            logp_initial = self.ln_likelihood(theta, self).copy()

            fix = np.ones(len(theta), dtype=bool)
            fix[i] = False

            _theta_free = theta[~fix].copy()
            _theta_copy = theta.copy()
            _theta_bounds = (bounds[~fix, 0], bounds[~fix, 1])

            result = least_squares(
                _fxn,
                _theta_free,
                method="trf",
                bounds=_theta_bounds,
                args=(_theta_copy, fix, self),
            )

            theta[~fix] = result.x.copy()
            logp_final = self.ln_likelihood(theta, self).copy()

            print(f"logp: {logp_initial} -> {logp_final}")
            print(f"message: {result['message']}")

            result = None

        return theta

    def _theta_initial(self):
        return np.zeros(np.sum(self.num_transits), dtype=float)

    def _theta_bounds(self):
        bounds = np.zeros((np.sum(self.num_transits), 2))

        iend = np.cumsum(self.num_transits)
        istart = iend - self.num_transits

        for n in range(self.npl):
            bounds[istart[n] : iend[n]] = (
                -self.planets[n].duration,
                self.planets[n].duration,
            )

        return bounds

    def update_transit_times(self, theta):
        iend = np.cumsum(self.num_transits)
        istart = iend - self.num_transits

        for n, p in enumerate(self.planets):
            p.transit_times += theta[istart[n] : iend[n]]
            self.planets[n] = p.refit_period_and_epoch()

        return self.planets
