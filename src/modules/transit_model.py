__all__ = ['TransitModel']

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from copy import deepcopy
import numpy as np
from src.constants import kepler_lcit, kepler_scit
from src.modules.base import BaseAlg
from src.schema.ephemeris import WarpEphemeris

from batman import _rsky
from batman import _quadratic_ld




class TransitModel(BaseAlg):
    def __init__(self, litecurve, planets):
        super().__init__(litecurve, planets)

        self._init_warp_ephemerides()

        self.transit_obsmode = self.get_transit_obsmode()
        self.unique_obsmodes = np.unique(np.hstack(self.transit_obsmode))
        
        self._define_supersample_lookup()
        self._define_exptime_integration_offset_lookup()


    def _init_warp_ephemerides(self):
        for n, p in enumerate(self.planets):
            q = p.ephemeris.quality
            w = WarpEphemeris(p.ephemeris.index[q], p.ephemeris.ttime[q])

            self.planets[n] = self.planets[n].update_ephemeris(w)


    def _warp_time_for_each_planet(self):
        warped_time = [None]*self.npl
        warped_legx = [None]*self.npl

        for n, p in enumerate(self.planets):
            index_midpoint = p.ephemeris.index[n][-1] // 2

            _t, _i = p.ephemeris._warp_times(self.litecurve.ttime, return_inds=True)            
            _x = (_i -index_midpoint) / index_midpoint

            warped_time[n] = _t   # len(self.litecurve.ttime)
            warped_legx[n] = _x   # len(self.litecurve.ttime)

        return warped_time, warped_legx

    
    # GJG : I don't think these are needed
    def _compute_transit_time_ind_legx(self):
        transit_index_centered = [None]*self.npl
        transit_time = [None] * self.npl
        transit_legx = [None] * self.npl

        for n, p in enumerate(self.planets):
            transit_index_centered[n] = p.ephemeris.index[n] - (p.ephemeris.index[n][-1] // 2)
            transit_legx[n] = transit_index_centered[n] / (p.ephemeris.index[n] / 2)
        
        return transit_index_centered, transit_time, transit_legx
    

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
    



def dynesty_lnlike(x, transitmodel):
    """
    Arguments
    ---------
        x : array-like
            N x [C0, C1, r, b, T14] + [q1,q2]
    """
    # convenient shorthand
    npl = transitmodel.npl
    lc = transitmodel.litecurve

    # calculate physical limb darkening (see Kipping 2013)
    q1, q2 = np.array(x[-2:])
    u1 = 2 * np.sqrt(q1) * q2
    u2 = np.sqrt(q1) * (1 - 2 * q2)

    # calculate log-likelihood
    loglike = 0.0

    for obsmode in transitmodel.unique_obmodes:
        flux_obs = lc.flux[lc.obsmode == obsmode]
        flux_err = lc.error[lc.obsmode == obsmode]
        flux_mod = np.ones_like(flux_obs)
        
        exptime = transitmodel.__exptime_lookup[obsmode]
        exptime_ioff = transitmodel.__exptime_integration_offset_lookup[obsmode]
        supersample_factor = transitmodel.__supersample_lookup[obsmode]

        for n, p in enumerate(transitmodel.planets):
            C0, C1, rp, b, T14 = np.array(x[5 * npl : 5 * (npl + 1)])

            warped_time = transitmodel.warped_time[n]
            warped_legx = transitmodel.warped_legx[n]

            _t = warped_time + C0 + C1 * warped_legx
            _t_supersample = (exptime_ioff + _t.reshape(_t.size)).flatten()

            nthreads = 1
            ds = _rsky._rsky(
                _t_supersample,
                0.0,
                p.period,
                rp,
                b,
                T14,
                1,
                nthreads,
            )

            # look into the transit type argument
            qld_flux = _quadratic_ld._quadratic_ld(
                ds, np.abs(rp), u1, u2, nthreads
            )

            # PERF can probably speed this up
            qld_flux = np.mean(
                qld_flux.reshape(-1, supersample_factor), axis=1
            )

            flux_mod += qld_flux - 1.0
            loglike += -0.5 * np.sum(((flux_mod - flux_obs) / flux_err) ** 2)

    
        # enforce prior on limb darkening
        U1, U2 = transitmodel.limbdark_priors
        sig_ld_sq = 0.01
        loglike -= 1.0 / (2 * sig_ld_sq) * (u1 - U1) ** 2
        loglike -= 1.0 / (2 * sig_ld_sq) * (u2 - U2) ** 2

        if not np.isfinite(loglike):
            return -1e300
        
        return loglike