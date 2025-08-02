__all__ = ['TransitModel']

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from copy import deepcopy
import dynesty
import numpy as np
from scipy.optimize import minimize, least_squares
from src.modules.base import BaseAlg
from src.modules.transit_model.dynesty import prior_transform
from src.schema.ephemeris import WarpEphemeris

from batman import _rsky
from batman import _quadratic_ld


class TransitModel(BaseAlg):
    def __init__(self, litecurve, planets, limbdark):
        super().__init__(litecurve, planets)
        
        self.limbdark = limbdark
        self._init_warp_ephemerides()


    def _init_warp_ephemerides(self):
        # inherit WarpEphemeris
        for n, p in enumerate(self.planets):
            q = p.ephemeris.quality
            w = WarpEphemeris(p.ephemeris.index[q], p.ephemeris.ttime[q])

            self.planets[n] = self.planets[n].update_ephemeris(w)

        # track static ephemerides
        self._static_period = np.array([p.ephemeris._static_period for p in self.planets])
        self._static_epoch = np.array([p.ephemeris._static_epoch for p in self.planets])

        # warp lc.time vector for ecah planet; legx is 1st-order Legendre polynomial
        self._warped_time = [None]*self.npl
        self._warped_legx = [None]*self.npl

        for n, p in enumerate(self.planets):
            index_midpoint = p.ephemeris.index[-1] // 2

            _t, _i = p.ephemeris._warp_times(self.litecurve.time, return_inds=True)            
            _x = (_i -index_midpoint) / index_midpoint

            self._warped_time[n] = _t   # len(self.litecurve.ttime)
            self._warped_legx[n] = _x   # len(self.litecurve.ttime)


    @staticmethod
    def model_flux(theta, transitmodel):
        """
        theta : array-like
            num_planets * [C0, C1, rp, b, T14] + [q1, q2]
        transitmodel : TransitModel
            instance of TransitModel, typically self

        Priors on planet parameters [C0, C1, rp, b, T14] are enforced outside this function
        
        Limb darkening is sampled in terms of [q1, q2] (see Kipping 2013)
        Limb darkening priors are enforced in terms of [u1, u2] inside this function
        """
        # shorthand
        tm = transitmodel
        lc = tm.litecurve

        # physical limb darkening (see Kipping 2013)
        q1, q2 = np.array(theta[-2:])
        u1 = 2 * np.sqrt(q1) * q2
        u2 = np.sqrt(q1) * (1 - 2 * q2)

        # calculate log-likelihood
        flux_mod = np.ones_like(lc.flux)

        for obsmode in tm.unique_obsmodes:
            exptime_ioff = tm._exptime_integration_offset_lookup[obsmode]
            supersample_factor = tm._supersample_lookup[obsmode]

            for n, p in enumerate(tm.planets):
                C0, C1, rp, b, T14 = np.array(theta[5 * n : 5 * (n + 1)])

                _t = tm._warped_time[n] + C0 + C1 * tm._warped_legx[n]
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

                qld_flux = _quadratic_ld._quadratic_ld(
                    ds, np.abs(rp), u1, u2, nthreads
                )

                qld_flux = np.mean(
                    qld_flux.reshape(-1, supersample_factor), axis=1
                )

                flux_mod[lc.obsmode == obsmode] += qld_flux - 1.0
                
        return flux_mod      
                
                
    @staticmethod
    def model_residuals(theta, transitmodel):
        flux_mod = transitmodel.model_flux(theta, transitmodel)
        flux_obs = transitmodel.litecurve.flux
        flux_err = transitmodel.litecurve.error

        return (flux_obs - flux_mod) / flux_err

    
    @staticmethod
    def ln_likelihood(theta, transitmodel):
        # shorthand
        tm = transitmodel

        # likelihood
        lnlike = -0.5 * np.sum(tm.model_residuals(theta, tm)**2)
               
        # enforce prior on limb darkening
        sig_ld_sq = 0.01
        lnlike -= 1.0 / (2 * sig_ld_sq) * (theta[-1] - tm.limbdark[0]) ** 2
        lnlike -= 1.0 / (2 * sig_ld_sq) * (theta[-2] - tm.limbdark[1]) ** 2

        if not np.isfinite(lnlike):
            return -1e300
        
        return lnlike
    

    def sample(self, checkpoint_file=None, checkpoint_every=60, progress=False):
        ndim = 5 * self.npl + 2
        sampler = dynesty.DynamicNestedSampler(
            self._lnlike, 
            prior_transform,
            ndim,
            bound='multi',
            sample='rwalk',
            logl_args=(self,),
            ptform_args=(self.durs,),
        )
        sampler.run_nested(
            checkpoint_file=checkpoint_file,
            checkpoint_every=checkpoint_every,
            print_progress=progress
        )
        
        return sampler.results
    

    def _theta_initial(self):
        """
        theta : array-like
            num_planets * [C0, C1, rp, b, T14] + [q1, q2]
        """
        theta = []
        for n, p in enumerate(self.planets):
            theta.append(0.0)
            theta.append(0.0)
            theta.append(np.sqrt(p.depth))
            theta.append(0.5)
            theta.append(p.duration)

        theta.append(self.limbdark[0])
        theta.append(self.limbdark[1])

        return np.array(theta)
    

    def _theta_bounds(self):
        """
        theta : array-like
            num_planets * [C0, C1, rp, b, T14] + [q1, q2]
        """
        bounds = []
        for n, p in enumerate(self.planets):
            bounds.append([-np.inf,np.inf])
            bounds.append([-np.inf,np.inf])
            bounds.append([1e-5,0.99])
            bounds.append([0.0,1.0])
            bounds.append([0.01*p.duration,3*p.duration])

        bounds.append([0.,1.])
        bounds.append([0.,1.])

        return np.array(bounds)

    
    def optimize(self, fix_limbdark=True, niter=3):
        theta_initial = self._theta_initial()
        bounds = self._theta_bounds()

        def _fxn(x, x0, fix, self):
            _x = x0.copy()
            _x[~fix] = x
            return self.model_residuals(_x, self)
        
        i_fix = np.arange(len(theta_initial), dtype=int)
        var_names = np.array('C0 C1 r b T14'.split())

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
        theta_final = theta_initial.copy()

        for fix in [fix_r_b_T14, fix_C0_C1] * niter:
            print(f"optimizing logp for variables: [{', '.join(var_names[~fix[:5]])}]")

            logp_initial = self.ln_likelihood(theta_final, self).copy()

            result = least_squares(
                _fxn,
                theta_final[~fix], 
                method='trf',
                bounds=bounds[~fix,:].T,
                args=[theta_initial, fix, self],
            )

            theta_final[~fix] = result.x.copy()
            logp_final = self.ln_likelihood(theta_final, self).copy()

            print(f"logp: {logp_initial} -> {logp_final}")
            print(f"message: {result['message']}")

        return theta_final