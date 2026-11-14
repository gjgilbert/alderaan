__all__ = ["DetrendVisits"]

import os
import warnings
from copy import deepcopy

import aesara_theano_fallback.tensor as T
import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
import pymc3_ext as pmx
from astropy.stats import mad_std, sigma_clip
from astropy.timeseries import LombScargle
from celerite2.backprop import LinAlgError
from celerite2.theano import GaussianProcess
from celerite2.theano import terms as GPterms
from scipy.signal import medfilt as median_filter
from scipy.signal import savgol_filter

from alderaan.constants import kepler_lcit, kepler_scit
from alderaan.data_models.litecurve import LiteCurve
from alderaan.data_models.shire import Shire
from alderaan.modules.base import BaseAlg
from alderaan.utils.plotting import plot_litecurve


class DetrendVisits(BaseAlg):
    def __init__(
        self,
        shire,
        target,
        results_dir,
        quicklook_dir,
        tight_mask,
        clip_outliers,
        ttv_buffer=None,
    ):
        super().__init__(shire.litecurve, shire.planets)
        self.star = deepcopy(shire.star)
        self.target = target
        self.results_dir = results_dir
        self.quicklook_dir = quicklook_dir
        self.tight_mask = tight_mask
        self.clip_outliers = clip_outliers
        self.ttv_buffer = ttv_buffer

        self.durations = np.array([p.duration for p in self.planets])

    def _transit_mask(self, lc):
        exptime = self.exptime[lc.obsmode[0]]

        if self.tight_mask:
            abs_size = 2 * (exptime + np.max(self.ttv_buffer))
            return self.make_transit_mask(
                rel_size=1.0, abs_size=abs_size, mask_type="condensed", time=lc.time
            )

        abs_size = 2 * (exptime + 1 / 24)
        return self.make_transit_mask(
            rel_size=3.0, abs_size=abs_size, mask_type="condensed", time=lc.time
        )

    def _min_period(self, obsmode):
        if obsmode == "short cadence":
            return np.max([5 * np.max(self.durations), 91 * kepler_scit])
        if obsmode == "long cadence":
            return np.max([5 * np.max(self.durations), 13 * kepler_lcit])
        raise ValueError(f"unsuported obsmode: {obsmode}")

    def _clip_outliers(self, lc, sigma_upper, sigma_lower, kernel_size, mask=None):
        """
        Sigma-clip outliers using astropy.stats.sigma_clip() and a smoothed trend
        Applies an additional iteration wrapper to allow for masked cadences

        Arguments:
            sigma_upper : float
                upper sigmga clipping threshold
            sigma_lower : float
                lower sigma clipping threshold
            kernel_size : int
                size of the smoothing filter kernel
            mask : array-like, bool (optional)
                do not reject cadences within masked regions;
                useful for protecting transits
        """
        if mask is None:
            mask = np.zeros(len(lc.time), dtype=bool)

        loop = True
        count = 0
        while loop:
            # first pass: try median filter
            if loop:
                smoothed = median_filter(lc.flux, kernel_size=kernel_size)

                bad = sigma_clip(
                    lc.flux - smoothed,
                    sigma_upper=sigma_upper,
                    sigma_lower=sigma_lower,
                    stdfunc=mad_std,
                ).mask
                bad = bad * ~mask

            # second pass: try savgol filter
            # if over 1% of points were flagged with median filter
            if np.sum(bad) / len(bad) > 0.01:
                smoothed = savgol_filter(
                    lc.flux, window_length=kernel_size, polyorder=2
                )
                bad = sigma_clip(
                    lc.flux - smoothed,
                    sigma_upper=sigma_upper,
                    sigma_lower=sigma_lower,
                    stdfunc=mad_std,
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

        return lc

    def _identify_breaks(self, lc, gap_tolerance, jump_tolerance, mask):
        """
        Identify gaps (breaks in times) and jumps (sudden flux changes) in a litecurve
            * GAPS are consecutive missing cadences in the TIME domain
            * JUMPS are large cadence-to-cadence FLUX variations

        Arguments
            gap_tolerance : int
                number of cadences to be considered a (time) gap
            jump_tolerance : float
                sigma threshold for identifying (flux) jumps
            mask : array-like, bool
                1 near transit, 0 far from transit
        """
        # identify GAPS in time
        gaps = np.pad(np.diff(lc.cadno), (1, 0), "constant", constant_values=(1, 0))
        gap_locs = np.pad(
            np.where(gaps > gap_tolerance)[0],
            (1, 1),
            "constant",
            constant_values=(0, len(gaps) + 1),
        )

        # identify JUMPS in flux
        jumps = np.pad(np.diff(lc.flux), (1, 0), "constant", constant_values=(0, 0))
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
        assert break_locs[-1] == len(lc.cadno) - 1, "unexpected edge behavior"

        return break_locs

    def _estimate_oscillation_period(self, lc, min_period):
        lombscargle = LombScargle(lc.time, lc.flux)

        min_freq = 1 / (lc.time.max() - lc.time.min())
        max_freq = 1 / min_period

        xf, yf = lombscargle.autopower(
            minimum_frequency=min_freq, maximum_frequency=max_freq
        )

        peak_freq = xf[np.argmax(yf)]
        peak_per = np.max([1.0 / peak_freq, 1.001 * min_period])

        return peak_per

    def _estimate_oscillation_periods(self, litecurves):
        oscillation_periods = np.zeros(len(litecurves))
        for j, lc in enumerate(litecurves):
            oscillation_periods[j] = self._estimate_oscillation_period(
                lc, self._min_period(lc.obsmode[0])
            )

        osc_per_mu = np.nanmedian(oscillation_periods)
        osc_per_sd = np.max(
            [mad_std(oscillation_periods, ignore_nan=True), 0.1 * osc_per_mu]
        )

        print(
            f"\nNominal stellar oscillation period = "
            f"{osc_per_mu:.1f} +/- {osc_per_sd:.1f} days"
        )

        print("\nSaving stellar oscillation periods...\n")
        filename = os.path.join(
            self.results_dir, f"{self.target}_stellar_oscillations.txt"
        )
        oscillation_period_by_quarter = np.ones(18) * np.nan
        quarter_indexes = [lc.visit[0] - 1 for lc in litecurves]
        oscillation_period_by_quarter[quarter_indexes] = oscillation_periods
        np.savetxt(filename, np.stack([np.arange(18), oscillation_period_by_quarter]).T)

        return oscillation_periods

    def _detrend_gp(
        self,
        lc,
        gp_term,
        nominal_gp_period,
        minimum_gp_period,
        transit_mask,
        gap_tolerance,
        jump_tolerance,
        correct_ramp,
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
        """
        # identify breaks (gaps in time, jumps in flux)
        # these will be used to identify contiguous "segments" of data
        # the model will fit all n_seg segments simultaneously
        # mean function (flux baseline + exponential trend) is independent
        # for each segment
        # gp parameters are shared between all segments

        break_locs = self._identify_breaks(
            lc, gap_tolerance, jump_tolerance, transit_mask
        )

        n_seg = len(break_locs) - 1
        seg_id = np.zeros(len(lc.time), dtype=int)

        for i in range(n_seg):
            seg_id[break_locs[i] : break_locs[i + 1]] = i

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

            log_P_off = pm.Normal(
                "log_P", mu=np.log(nominal_gp_period - minimum_gp_period), sd=2.0
            )
            P = pm.Deterministic("P", minimum_gp_period + T.exp(log_P_off))

            log_Q0 = pm.Normal("log_Q0", mu=0.0, sd=5.0, testval=np.log(0.5))
            Q0 = pm.Deterministic("Q0", T.exp(log_Q0))

            if gp_term == "RotationTerm":
                log_dQ = pm.Normal("log_dQ", mu=0.0, sd=5.0, testval=np.log(1e-3))
                mix = pm.Uniform("mix", lower=0, upper=1, testval=0.1)
                kernel = GPterms.RotationTerm(
                    sigma=sigma, period=P, Q0=Q0, dQ=T.exp(log_dQ), f=mix
                )

            elif gp_term == "SHOTerm":
                kernel = GPterms.SHOTerm(sigma=sigma, w0=2 * np.pi / P, Q=0.5 + Q0)

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
            mean_oot = pm.Deterministic(
                "mean_oot",
                _mean_fxn(lc.time[oot], seg_id[oot], flux0, ramp_amp, log_tau),
            )
            mean_all = pm.Deterministic(
                "mean_all", _mean_fxn(lc.time, seg_id, flux0, ramp_amp, log_tau)
            )
            log_var = pm.Normal(
                "log_var", mu=np.var(lc.flux - median_filter(lc.flux, 13)), sd=5.0
            )

            gp = GaussianProcess(kernel, mean=mean_oot)
            gp.compute(lc.time[oot], diag=T.exp(log_var) * T.ones(len(lc.time[oot])))

            # likelihood
            pm.Potential("lnlike", gp.log_likelihood(lc.flux[oot]))

            # predicted trend; if len(y)!=len(t) set include_mean=False
            pred = gp.predict(lc.flux[oot], t=lc.time, include_mean=False)
            pm.Deterministic("pred", pred + mean_all)

        # optimize hyperparameters
        with model:
            map_soln = pmx.optimize(
                start=model.test_point, vars=[flux0], progress=False
            )
            map_soln = pmx.optimize(
                start=map_soln, vars=[flux0, log_var], progress=False
            )

            for _ in range(1 + correct_ramp):
                if gp_term == "RotationTerm":
                    map_soln = pmx.optimize(
                        start=map_soln,
                        vars=[log_var, flux0, sigma, P, log_Q0, log_dQ, mix],
                        progress=False,
                    )
                if gp_term == "SHOTerm":
                    map_soln = pmx.optimize(
                        start=map_soln,
                        vars=[log_var, flux0, sigma, P, log_Q0],
                        progress=False,
                    )
                if correct_ramp:
                    map_soln = pmx.optimize(
                        start=map_soln,
                        vars=[log_var, flux0, ramp_amp, log_tau],
                        progress=False,
                    )

        lc.flux /= map_soln["pred"]
        lc.error /= map_soln["pred"]

        return lc

    def _detrend(self, lc, oscillation_periods):
        obsmode = lc.obsmode[0]

        if obsmode == "short cadence":
            gap_tolerance = np.max(
                [int(np.min(self.durations) / kepler_scit * 5 / 2), 91]
            )
        elif obsmode == "long cadence":
            gap_tolerance = np.max(
                [int(np.min(self.durations) / kepler_lcit * 5 / 2), 13]
            )
        else:
            raise ValueError(f"unsuported obsmode: {obsmode}")

        min_period = self._min_period(obsmode)
        mask = self._transit_mask(lc)

        # use successively simpler models as fallbacks
        try:
            return self._detrend_gp(
                lc,
                "RotationTerm",
                np.nanmedian(oscillation_periods),
                min_period,
                transit_mask=mask,
                gap_tolerance=gap_tolerance,
                jump_tolerance=5.0,
                correct_ramp=True,
            )
        except LinAlgError:
            warnings.warn(
                "Initial detrending failed..."
                "attempting to refit without exponential ramp component",
                stacklevel=1,
            )

        try:
            return self._detrend_gp(
                lc,
                "RotationTerm",
                np.nanmedian(oscillation_periods),
                min_period,
                transit_mask=mask,
                gap_tolerance=gap_tolerance,
                jump_tolerance=5.0,
                correct_ramp=False,
            )
        except LinAlgError:
            warnings.warn(
                "Detrending with RotationTerm failed..."
                "attempting to detrend with SHOTerm",
                stacklevel=1,
            )

        return self._detrend_gp(
            lc,
            "SHOTerm",
            np.nanmedian(oscillation_periods),
            min_period,
            transit_mask=mask,
            gap_tolerance=gap_tolerance,
            jump_tolerance=5.0,
            correct_ramp=False,
        )

    def perform(self):
        print("\n\nMODULE: DETREND VISITS\n")

        litecurves = self.litecurve.split_visits()

        if self.clip_outliers:
            for j, lc in enumerate(litecurves):
                npts_initial = len(lc.time)

                lc = self._clip_outliers(
                    lc,
                    sigma_upper=5,
                    sigma_lower=5,
                    kernel_size=13,
                    mask=self._transit_mask(lc),
                )
                lc = self._clip_outliers(
                    lc, sigma_upper=5, sigma_lower=1000, kernel_size=13
                )
                litecurves[j] = lc

                print(
                    f"  Quarter {lc.visit[0]} : "
                    f"{npts_initial - len(lc.time)} outliers rejected"
                )

        oscillation_periods = self._estimate_oscillation_periods(litecurves)

        for j, lc in enumerate(litecurves):
            print(f"\nDetrending {j + 1} of {len(litecurves)} litecurves", flush=True)
            litecurves[j] = self._detrend(lc, oscillation_periods)

        litecurve = LiteCurve().from_list(litecurves)

        filepath = os.path.join(
            self.quicklook_dir, f"{self.target}_litecurve_detrended.png"
        )
        plot_litecurve(litecurve, self.target, self.planets, filepath)
        plt.close("all")

        # output cadence follows the last visit
        obsmode = litecurves[-1].obsmode[0]
        if obsmode == "long cadence":
            filename = os.path.join(
                self.results_dir, f"{self.target}_lc_detrended.fits"
            )
            litecurve.to_fits(self.target, filename, cadence="LONG")
        elif obsmode == "short cadence":
            filename = os.path.join(
                self.results_dir, f"{self.target}_sc_detrended.fits"
            )
            litecurve.to_fits(self.target, filename, cadence="SHORT")

        return Shire(self.star, self.planets, litecurve)
