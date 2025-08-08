__all__ = ['TransitModel',
           'ShapeTransitModel',
           'TTimeTransitModel',
          ]

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from copy import deepcopy
import dynesty
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import numpy.polynomial.polynomial as poly
from scipy.optimize import least_squares
from alderaan.modules.base import BaseAlg
from alderaan.modules.transit_model.dynesty import prior_transform, throttled_print_fn
from alderaan.utils.astro import bin_data, estimate_transit_depth

from batman import _rsky
from batman import _quadratic_ld


class TransitModel(BaseAlg):
    def __init__(self, litecurve, planets, limbdark):
        super().__init__(litecurve, planets)
        
        self.limbdark = limbdark
        self._init_time_warping()


    def _init_time_warping(self):
        """
        This function initializes warped time arrays
        * the litecurve.time vector is copied for each of the N planets
        * times are then "warped" to account for transit timing variations
        * the transit model assumes linear perturbations to a fixed ephemeris
            
        IMPORTANT!!! The warp functions assume zero-indexing on transit indexes
        """
        # track static ephemerides
        self._static_period = np.array([p.ephemeris._static_period for p in self.planets])
        self._static_epoch = np.array([p.ephemeris._static_epoch for p in self.planets])

        # set warping bins
        self._set_bins()

        # warp lc.time vector for ecah planet; legx is 1st-order Legendre polynomial
        self._warped_time = [None]*self.npl
        self._warped_legx = [None]*self.npl

        for n, p in enumerate(self.planets):
            index_midpoint = p.ephemeris.index[-1] // 2

            _t, _i = self._warp_times(self.litecurve.time, n, return_inds=True)            
            _x = (_i -index_midpoint) / index_midpoint

            self._warped_time[n] = _t.copy()   # len(self.litecurve.ttime)
            self._warped_legx[n] = _x.copy()   # len(self.litecurve.ttime)


    def _set_bins(self):
        self._bin_edges = [None]*self.npl
        self._bin_values = [None]*self.npl

        for n, p, in enumerate(self.planets):
            index_full = np.arange(0, p.ephemeris.index.max()+1, dtype=int)
            ttime_full = self._static_epoch[n] + self._static_period[n] * index_full

            self._bin_edges[n] = np.concatenate(
                [
                    [ttime_full[0] - 0.5 * self._static_period[n]],
                    0.5 * (ttime_full[1:] + ttime_full[:-1]),
                    [ttime_full[-1] + 0.5 * self._static_period[n]],
                ]
            )

            self._bin_values[n] = np.concatenate([[ttime_full[0]], ttime_full, [ttime_full[-1]]])


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


class ShapeTransitModel(TransitModel):
    def __init__(self, litecurve, planets, limbdark):
        super().__init__(litecurve, planets, limbdark)
        

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
            supersample = tm._supersample_lookup[obsmode]

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
                    qld_flux.reshape(-1, supersample), axis=1
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
    

    def sample(self, checkpoint_file=None, checkpoint_every=60, progress_every=10):
        ndim = 5 * self.npl + 2
        sampler = dynesty.DynamicNestedSampler(
            self.ln_likelihood, 
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
            print_progress=True,
            print_func=throttled_print_fn(progress_every),
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
    

    def update_planet_parameters(self, theta):
        for n, p in enumerate(self.planets):
            #self.planets[n].period = self.planets[n].period
            #self.planets[n].epoch = self.planets[n].epoch
            self.planets[n].ror = theta[2 + n * 5]
            self.planets[n].impact = theta[3 + n * 5]
            self.planets[n].duration = theta[4 + n * 5]
            self.planets[n].depth = estimate_transit_depth(p.ror, p.impact)

        return self.planets
    

    def update_limbdark_parameters(self, theta):
        self.limbdark = theta[-2:]

        return self.limbdark
    

class TTimeTransitModel(TransitModel):
    def __init__(self, litecurve, planets, limbdark):
        super().__init__(litecurve, planets, limbdark)


    def _construct_template(self, planet_no, obsmode, transit_window_size):
        """
        planet_no (int) : planet number index

        IMPORTANT!!! Planet extensions must have desired [P, Rp/Rs, b, T14]
        """
        # shorthand
        p = self.planets[planet_no]

        exptime = self._exptime_lookup[obsmode]
        supersample = self._supersample_lookup[obsmode]

        time_template = np.arange(0, transit_window_size/2, exptime/supersample/2, dtype=float)
        time_template = np.hstack([-time_template[:-1][::-1], time_template])
        flux_template = np.zeros_like(time_template)

        nthreads = 1
        ds = _rsky._rsky(
            time_template,
            0.0,
            p.period,
            p.ror,
            p.impact,
            p.duration,
            1,
            nthreads,
        )

        qld_flux = _quadratic_ld._quadratic_ld(
            ds, np.abs(p.ror), self.limbdark[0], self.limbdark[1], nthreads
        )

        flux_template += qld_flux

        return time_template, flux_template
    

    def mazeh13_holczer16_method(
            self, planet_no, rel_window_size=5.0, abs_window_size_buffer=1/24, quicklook_dir=None,
        ):
        """
        Docstring
        """
        # shorthand
        lc = self.litecurve
        p = self.planets[planet_no]

        ttime = np.nan * np.ones(len(p.ephemeris.ttime))
        ttime_err = np.nan * np.ones(len(p.ephemeris.ttime))
        
        overlap = self.identify_overlapping_transits(rtol=1.0, atol=1.0)
       
        assert len(p.ephemeris.ttime) == len(overlap[planet_no]), (
            f"Mismatched sizes for ttime ({len(p.ephemeris.ttime)}) and overlap ({len(overlap[planet_no])})"
        )

        transit_obsmode = self.transit_obsmode[planet_no]

        for obsmode in self.unique_obsmodes:
            print(f"  Determining transit times using {obsmode} data")

            exptime = self._exptime_lookup[obsmode]
            supersample = self._supersample_lookup[obsmode]
            exptime_ioff = self._exptime_integration_offset_lookup[obsmode]

            transit_window_size = rel_window_size * p.duration + abs_window_size_buffer
            time_template, flux_template = self._construct_template(planet_no, obsmode, transit_window_size)

            
            for j, tc in enumerate(p.ephemeris.ttime):
                if (not overlap[planet_no][j]) and (p.ephemeris.quality[j]) and (transit_obsmode[j] == obsmode):
                    #print(f"  Transit {p.ephemeris.index[j]} : BKJD = {tc:.1f}")
                    
                    # STEP 0: pull data near a single transit
                    in_transit = np.abs(self.litecurve.time - tc) < p.duration / 2
                    in_window = np.abs(self.litecurve.time - tc) < transit_window_size / 2
                    
                    if np.sum(in_transit) > 0:
                        _t = lc.time[in_window]
                        _f_obs = lc.flux[in_window]
                        _f_err = lc.error[in_window]

                        _t_supersample = (exptime_ioff + _t.reshape(_t.size, 1)).flatten()

                        # remove any residual out-of-transit trend
                        use = ~in_transit[in_window]
                        try:
                            _f_obs /= poly.polyval(_t, poly.polyfit(_t[use], _f_obs[use], 1))
                        except TypeError:
                            pass

                        # STEP 1: generate tc_offset vs chisq vectors
                        gridstep = exptime / supersample / 1.618 / 3
                        tc_offset = np.arange(0, transit_window_size / 2, gridstep)
                        tc_offset = tc + np.hstack([-tc_offset[:-1][::-1], tc_offset])
                        chisq = np.zeros_like(tc_offset)

                        for i, tc_o in enumerate(tc_offset):
                            _f_mod = np.interp(_t_supersample - tc_o, time_template, flux_template)
                            _f_mod = bin_data(_t_supersample, _f_mod, exptime, bin_centers=_t)[1]

                            chisq[i] = np.sum(((_f_obs - _f_mod) / _f_err)**2)

                        # STEP 2: isolate relevant portions of {tc_offset, chisq} vectors
                        delta_chisq = 2.0

                        loop = True
                        while loop:
                            # grab data near chisq minimum
                            tc_fit = tc_offset[chisq < chisq.min() + delta_chisq]
                            x2_fit = chisq[chisq < chisq.min() + delta_chisq]

                            # eliminate points far from the local minimum
                            faraway = np.abs(tc_fit - np.median(tc_fit)) / np.median(np.diff(tc_fit)) > 1 + 0.5*len(tc_fit)

                            tc_fit = tc_fit[~faraway]
                            x2_fit = x2_fit[~faraway]

                            # check stopping conditions
                            if len(x2_fit) > 7:
                                loop = False
                            if delta_chisq >= 16:
                                loop = False

                            # increment chisq
                            delta_chisq *= np.sqrt(2)

                        # STEP 3: fit a parabola to local chisq minimum
                        if len(tc_fit) > 3:
                            # polynomial fitting
                            quad_coeffs = np.polyfit(tc_fit, x2_fit, 2)
                            quad_model = np.polyval(quad_coeffs, tc_fit)
                            qtc_min = -quad_coeffs[1] / (2 * quad_coeffs[0])
                            qx2_min = np.polyval(quad_coeffs, qtc_min)
                            qtc_err = np.sqrt(1 / quad_coeffs[0])

                            # transit time and scaled error
                            _ttj = np.nanmean([qtc_min, np.mean(tc_fit)])
                            _errj = qtc_err * (1 + np.std(x2_fit - quad_model))

                            # check that the fit is well-conditioned
                            convex_local_min = quad_coeffs[0] > 0
                            within_bounds = (_ttj > tc_fit.min()) and (_ttj < tc_fit.max())

                            #if convex_local_min and within_bounds:
                            ttime[j] = _ttj.copy()
                            ttime_err[j] = _errj.copy()

                        # STEP 4: make quicklook plot
                        if (quicklook_dir is not None) and (len(_f_obs) > 0):
                            target = 'K00148'
                            ttv_dir = os.path.join(quicklook_dir, 'ttvs')
                            os.makedirs(ttv_dir, exist_ok=True)
                            path = os.path.join(ttv_dir, f'{target}_{planet_no}_ttv_{j}.png')

                            _f_mod = np.interp(_t_supersample - _ttj, time_template, flux_template)
                            _f_mod = bin_data(_t_supersample, _f_mod, exptime, bin_centers=_t)[1]
                            
                            fig, ax = plt.subplots(1,2, figsize=(8,3))
                            
                            ax[0].plot(_t, _f_obs, 'ko')
                            ax[0].plot(_t, _f_mod, c=f'C{planet_no}', lw=3)

                            xticks = np.array([tc-transit_window_size/2, tc, tc+transit_window_size/2]).round(2)
                            ax[0].set_xticks(xticks)
                            ax[0].yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
                            ax[0].set_xlabel("Time [BJKD]", fontsize=14)
                            ax[0].set_ylabel("Flux", fontsize=14)

                            display = np.abs(chisq - qx2_min) < 2.5

                            _x = tc_offset[display]
                            _y_obs = (chisq-qx2_min)[display]
                            _y_mod = np.polyval(quad_coeffs, _x) - qx2_min

                            ax[1].plot(_x, _y_obs, 'o', mec='k', mfc='w')
                            ax[1].plot(_x, _y_mod, c=f'C{planet_no}', lw=3)
                            ax[1].axvline(qtc_min, color='k', ls=':')
                            ax[1].axvline(tc_fit[np.argmin(x2_fit)], color='k', ls=':')
                            ax[1].axvline(np.mean(tc_fit), color='k', ls=':')
                            ax[1].axvline(_ttj, color='k', lw=2)

                            xticks = np.array([_ttj - 1.5 * _errj, _ttj, _ttj + 1.5 * _errj])
                            ax[1].set_xticks(xticks, np.round(xticks - _ttj, 4))
                            ax[1].set_ylim(-0.5, 2.5)
                            ax[1].set_xlabel("$\Delta t_c$", fontsize=14)
                            ax[1].set_ylabel("$\Delta \chi^2$", fontsize=14)
                            
                            plt.suptitle(f"{target} - Planet {planet_no}", fontsize=18)
                            plt.tight_layout()
                            fig.savefig(path)
                            plt.close()

        return ttime, ttime_err