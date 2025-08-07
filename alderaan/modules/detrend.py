__all__ = ['SimpleDetrender',
           'GaussianProcessDetrender',
           'AutoCorrelationDetrender'
           ]

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import aesara_theano_fallback.tensor as T
from astropy.stats import sigma_clip, mad_std
from astropy.timeseries import LombScargle
from celerite2.theano import GaussianProcess
from celerite2.theano import terms as GPterms
import numpy as np
import pymc3 as pm
import pymc3_ext as pmx
from scipy.signal import medfilt as median_filter
from scipy.signal import savgol_filter
from alderaan.schema.planet import Planet
from alderaan.schema.litecurve import LiteCurve
from alderaan.modules.base import BaseAlg

class SimpleDetrender(BaseAlg):
    def __init__(self, litecurve, planets):
        super().__init__(litecurve, planets)


    def clip_outliers(self, 
                      kernel_size, 
                      sigma_upper, 
                      sigma_lower, 
                      mask=None,
                      trend=None
                      ):
        """
        Sigma-clip outliers using astropy.stats.sigma_clip() and a median filtered trend
        Applies an additional iteration wrapper to allow for masked cadences

        Arguments:
            kernel_size : int
                size of the smoothing filter kernel
            sigma_upper : float
                upper sigmga clipping threshold
            sigma_lower : float
                lower sigma clipping threshold
            mask : array-like, bool (optional)
                do not reject cadences within masked regions; useful for protecting transits
            trend : array-like, float (optional)
                precomputed model for the trend, e.g. a Keplerian transit lightcurve
        """
        lc = self.litecurve

        if mask is None:
            mask = np.zeros(len(lc.time), dtype=bool)
        
        if trend is not None:
            raise ValueError("Not yet configured for trend kwarg")

        loop = True
        count = 0
        while loop:
            # first pass: try median filter
            if loop:
                smoothed = median_filter(lc.flux, kernel_size=kernel_size)

                bad = sigma_clip(lc.flux - smoothed,
                                 sigma_upper=sigma_upper,
                                 sigma_lower=sigma_lower,
                                 stdfunc=mad_std
                                ).mask
                bad = bad * ~mask

            # second pass: try savgol filter
            # if over 1% of points were flagged with median filter
            if np.sum(bad)/len(bad) > 0.01:
                smoothed = savgol_filter(lc.flux,
                                         window_length=kernel_size,
                                         polyorder=2
                                        )
                bad = sigma_clip(lc.flux - smoothed,
                                 sigma_upper=sigma_upper,
                                 sigma_lower=sigma_lower,
                                 stdfunc=mad_std
                                ).mask
                bad = bad * ~mask

            # third pass: skip outlier rejection
            if np.sum(bad) / len(bad) > 0.01:
                bad = np.zeros_like(mask)

            # set attributes
            for k in lc.__dict__.keys():
                if type(lc.__dict__[k]) is np.ndarray:
                    lc.__setattr__(k, lc.__dict__[k][~bad])

            # update mask and iterate
            mask = mask[~bad]

            if np.sum(bad) == 0:
                loop = False
            else:
                count += 1

            if count >= 3:
                loop = False

        self.litecurve = lc


    def _identify_breaks(self, gap_tolerance, jump_tolerance, mask=None):
        """
        Identify gaps (breaks in times) and jumps (sudden flux changes) in a litecurve
            * GAPS are consecutive missing cadences in the TIME domain
            * JUMPS are large cadence-to-cadence FLUX variations

        Arguments
            gap_tolerance : int
                number of cadences to be considered a (time) gap
            jump_tolerance : float
                sigma threshold for identifying (flux) jumps
            mask : array-like, bool (optional)
                1 near transit, 0 far from transit
        """
        # identify GAPS in time
        gaps = np.pad(np.diff(self.litecurve.cadno), (1,0), 'constant', constant_values=(1,0))
        gap_locs = np.pad(
            np.where(gaps > gap_tolerance)[0], (1,1), 'constant', constant_values=(0, len(gaps)+1)
        )

        # identify JUMPS in flux
        jumps = np.pad(np.diff(self.litecurve.flux), (1,0), 'constant', constant_values=(0,0))
        jump_size = np.abs(jumps - np.median(jumps)) / mad_std(jumps)
        jump_locs = np.where(~mask.astype(bool) & (jump_size > jump_tolerance))[0]

        # combine breaks
        break_locs = np.sort(np.unique(np.hstack([gap_locs, jump_locs])))
        
        # fix edge behavior
        bad = np.hstack([False, np.diff(break_locs) < gap_tolerance])
        if bad[-1]:
            bad[-2:] = (True, False)
        break_locs = break_locs[~bad]
        break_locs[-1] -= 2

        assert break_locs[0] == 0, "unexpected edge behavior"
        assert break_locs[-1] == len(self.litecurve.cadno) - 1, "unexpected edge behavior"
        
        return break_locs


class GaussianProcessDetrender(SimpleDetrender):
    def __init__(self, litecurve, planets):
        super().__init__(litecurve, planets)


    def estimate_oscillation_period(self, min_period=None):
        """
        Docstring
        """
        lc = self.litecurve
        lombscargle = LombScargle(lc.time, lc.flux)

        if min_period is None:
            min_period = 3 * np.max(self.durs)

        min_freq = 1 / (lc.time.max() - lc.time.min())
        max_freq = 1 / min_period

        xf, yf = lombscargle.autopower(
            minimum_frequency=min_freq, maximum_frequency=max_freq
        )

        peak_freq = xf[np.argmax(yf)]
        peak_per = np.max([1.0 / peak_freq, 1.001 * min_period])

        return peak_per
    

    def detrend(self,
                gp_term,
                nominal_gp_period,
                minimum_gp_period,
                transit_mask=None,
                gap_tolerance=None,
                jump_tolerance=None,
                correct_ramp=True,
                return_trend=False, 
                progressbar=False
               ):
        """
        Arguments
            gp_term : str "RotationTerm" or "SHOTerm"
            nominal_gp_period : float
            minimum_gp_period : float
            transit_mask : ndarray
            gap_tolerance : int
            jump_tolerance : float
            correct_ramp : bool
            return_trend : bool
            progressbar : bool
        """
        # shorthand
        lc = self.litecurve

        # sanitize inputs
        if transit_mask is None:
            transit_mask = np.zeros(len(lc.time), dtype=bool)
        if gap_tolerance is None:
            gap_tolerance = len(lc.time)
        if jump_tolerance is None:
            jump_tolerance = 1000.

        # identify breaks (gaps in time, jumps in flux)
        # these will be used to identify contiguous "segments" of data
        # the model will fit all n_seg segments simultaneously
        # mean function (flux baseline + exponential trend) is independent for each segment
        # gp parameters are shared between all segments
        
        break_locs = self._identify_breaks(gap_tolerance, jump_tolerance, mask=transit_mask)
        
        n_seg = len(break_locs) - 1
        seg_id = np.zeros(len(lc.time), dtype=int)

        for i in range(n_seg):
            seg_id[break_locs[i]:break_locs[i+1]] = i

        approx_mean = [np.mean(lc.flux[seg_id == i]) for i in range(n_seg)]

        # convenience function
        def _mean_fxn(time, seg_id, flux0, ramp_amp, log_tau):
            nseg = len(np.unique(seg_id))
            mean = T.zeros(len(time))

            for i in range(nseg):
                t_min = time[seg_id == i].min()
                mean += (
                    flux0[i]
                    * (1 + ramp_amp[i] * T.exp(-(time - t_min) / T.exp(log_tau[i])))
                    * (seg_id == i)
                )

            return mean
        
        # pymc model
        with pm.Model() as model:
            # set up the kernel
            log_sigma = pm.Normal("log_sigma", mu=np.log(mad_std(lc.flux)), sd=5.0)
            sigma = pm.Deterministic("sigma", T.exp(log_sigma))

            log_P_off = pm.Normal("log_P", mu=np.log(nominal_gp_period - minimum_gp_period), sd=2.0)
            P = pm.Deterministic("P", minimum_gp_period + T.exp(log_P_off))

            log_Q0 = pm.Normal("log_Q0", mu=0.0, sd=5.0, testval=np.log(0.5))
            Q0 = pm.Deterministic("Q0", T.exp(log_Q0))

            if gp_term == 'RotationTerm':
                log_dQ = pm.Normal("log_dQ", mu=0.0, sd=5.0, testval=np.log(1e-3))
                mix = pm.Uniform("mix", lower=0, upper=1, testval=0.1)
                kernel = GPterms.RotationTerm(sigma=sigma, period=P, Q0=Q0, dQ=T.exp(log_dQ), f=mix)

            elif gp_term == 'SHOTerm':
                kernel = GPterms.SHOTerm(sigma=sigma, w0=2 * pi / P, Q=0.5 + Q0)
                
            else:
                raise ValueError(f"gp_term {gp_term} not recognized")
            
            # set up mean function and variance
            flux0 = pm.Normal("flux0", mu=approx_mean, sd=np.ones(n_seg), shape=n_seg)
            
            if correct_ramp:
                ramp_amp = pm.Normal("ramp_amp", mu=0, sd=np.std(lc.flux), shape=n_seg)
                log_tau = pm.Normal("log_tau", mu=0, sd=5, shape=n_seg)
            else:
                ramp_amp = T.zeros(n_seg)
                log_tau = T.zeros(n_seg)

            # set up the gp
            oot = ~transit_mask
            mean_oot = pm.Deterministic("mean_oot", _mean_fxn(lc.time[oot], seg_id[oot], flux0, ramp_amp, log_tau))
            mean_all = pm.Deterministic("mean_all", _mean_fxn(lc.time, seg_id, flux0, ramp_amp, log_tau))
            log_var = pm.Normal("log_var", mu=np.var(lc.flux - median_filter(lc.flux, 13)), sd=5.0)

            gp = GaussianProcess(kernel, mean=mean_oot)
            gp.compute(lc.time[oot], diag=T.exp(log_var)*T.ones(len(lc.time[oot])))
            
            # likelihood
            pm.Potential("lnlike", gp.log_likelihood(lc.flux[oot]))

            # predicted trend; if len(y)!=len(t) set include_mean=False
            pred = gp.predict(lc.flux[oot], t=lc.time, include_mean=False)
            pm.Deterministic("pred", pred + mean_all)

        # optimize hyperparameters
        with model:
            map_soln = pmx.optimize(start=model.test_point, vars=[flux0], progress=progressbar)
            map_soln = pmx.optimize(start=map_soln, vars=[flux0, log_var], progress=progressbar)

            for i in range(1 + correct_ramp):
                if gp_term == 'RotationTerm':
                    map_soln = pmx.optimize(
                        start=map_soln,
                        vars=[log_var, flux0, sigma, P, log_Q0, log_dQ, mix],
                        progress=progressbar,
                    )
                if gp_term == 'SHOTerm':
                    map_soln = pmx.optimize(
                        start=map_soln,
                        vars=[log_var, flux0, sigma, P, log_Q0],
                        progress=progressbar,
                    )
                if correct_ramp:
                    map_soln = pmx.optimize(
                        start=map_soln,
                        vars=[log_var, flux0, ramp_amp, log_tau],
                        progress=progressbar,
                    )

        self.litecurve.flux /= map_soln['pred']
        self.litecurve.error /= map_soln['pred']

        if return_trend:
            return self.litecurve, map_soln['pred']
        return self.litecurve


class AutoCorrelationDetrender(SimpleDetrender):
    def __init__(self, litecurve, planets):
        super().__init__(litecurve, planets)
