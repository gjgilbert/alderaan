__all__ = ['TransitModel']

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from copy import deepcopy
import dynesty
import numpy as np
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
        self._init_obsmode_tracking()        


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


    def _init_obsmode_tracking(self):
        self.transit_obsmode = self.get_transit_obsmode()
        self.unique_obsmodes = np.unique(np.hstack(self.transit_obsmode))

        self._define_supersample_lookup()
        self._define_exptime_integration_offset_lookup()
    

    def _compute_supersample_factor(self, obsmode):    
        # ingress/egress timescale estimate following Winn 2010
        tau12 = (13 / 24) * (np.array(self.periods) / 365.25) ** (1 / 3) * np.sqrt(np.array(self.depths)) 

        # sigma so binning error is < 0.1% of photometric uncertainty
        sigma = np.nanmean(self.litecurve.error / self.litecurve.flux) * 0.04

        # supersample factor following Kipping 2010
        exptime = self._obsmode_to_exptime(obsmode)
        supersample = np.array(np.ceil(np.sqrt((self.depths / tau12) * (exptime / 8 / sigma))), dtype=int)
        supersample = np.max([supersample + (supersample % 2 + 1)])

        return supersample
    

    def _define_supersample_lookup(self):
        self.__supersample_lookup = {}
        for om in np.unique(np.hstack(self.transit_obsmode)):
            self.__supersample_lookup[om] = self._compute_supersample_factor(om)
        

    def _obsmode_to_supersample(self, obsmode):
        return self.__supersample_lookup[obsmode]


    def _compute_exptime_integration_offset(self, obsmode):
        exptime = self._obsmode_to_exptime(obsmode)
        supersample = self._obsmode_to_supersample(obsmode)
        return np.linspace(-exptime/2, exptime/2, supersample)
    

    def _define_exptime_integration_offset_lookup(self):
        self.__exptime_integration_offset_lookup = {}
        for om in np.unique(np.hstack(self.transit_obsmode)):
            self.__exptime_integration_offset_lookup[om] = (
                self._compute_exptime_integration_offset(om)
            )
            

    def _obsmode_to_exptime_integration_offset(self, obsmode):
        return self.__exptime_integration_offset_lookup[obsmode]
    

    @staticmethod
    def _lnlike(theta, transitmodel):
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
        lnlike = 0.0

        for obsmode in tm.unique_obsmodes:
            flux_obs = lc.flux[lc.obsmode == obsmode]
            flux_err = lc.error[lc.obsmode == obsmode]
            flux_mod = np.ones_like(flux_obs)
            
            exptime_ioff = tm.__exptime_integration_offset_lookup[obsmode]
            supersample_factor = tm.__supersample_lookup[obsmode]

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

                flux_mod += qld_flux - 1.0
                lnlike += -0.5 * np.sum(((flux_obs - flux_mod) / flux_err) ** 2)

            # enforce prior on limb darkening
            sig_ld_sq = 0.01
            lnlike -= 1.0 / (2 * sig_ld_sq) * (u1 - tm.limbdark[0]) ** 2
            lnlike -= 1.0 / (2 * sig_ld_sq) * (u2 - tm.limbdark[1]) ** 2

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
    

    def optimize(self):
        pass