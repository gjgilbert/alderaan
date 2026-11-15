"""Detrend each visit of a litecurve with a GP, masking transits"""

import logging
import os

import aesara_theano_fallback.tensor as T
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

from alderaan.algorithms.base import BaseAlg
from alderaan.algorithms.transit_fitting import BaseTransitFittingAlg
from alderaan.constants import KEPLER_LCIT, KEPLER_SCIT
from alderaan.models.transit_model import TransitModel
from alderaan.utils.stats import set_sigma_cutoff

logger = logging.getLogger(__name__)


class DetrendVisits(BaseAlg):
    """
    Detrend a litecurve visit-by-visit

    Parameters
    ----------
    star : Star
    planets : list of Planet
    litecurve : LiteCurve
    config : ConfigParser
        pipeline configuration, including RUN and PATHS values set by main
    """

    def __init__(self, star, planets, litecurve, raw_litecurve, config):
        super().__init__(star, planets, litecurve, config)
        self.raw_litecurve = raw_litecurve

        self.durations = np.array([p.duration for p in self.planets])

    def _transit_mask(self, time, obsmode, tight_mask, ttv_buffer):
        exptime = self.exptime[obsmode]

        if tight_mask:
            abs_size = 2 * (exptime + np.max(ttv_buffer))
            return self.make_transit_mask(
                rel_size=1.0, abs_size=abs_size, mask_type="condensed", time=time
            )

        abs_size = 2 * (exptime + 1 / 24)
        return self.make_transit_mask(
            rel_size=3.0, abs_size=abs_size, mask_type="condensed", time=time
        )

    def _min_period(self, obsmode):
        if obsmode == "short cadence":
            return np.max([5 * np.max(self.durations), 91 * KEPLER_SCIT])
        if obsmode == "long cadence":
            return np.max([5 * np.max(self.durations), 13 * KEPLER_LCIT])
        raise ValueError(f"unsuported obsmode: {obsmode}")

    @staticmethod
    def _clip_outliers(flux, visit, sigma_upper, sigma_lower, kernel_size, mask=None):
        """
        Sigma-clip outliers using astropy.stats.sigma_clip() and a smoothed trend
        Applies an additional iteration wrapper to allow for masked cadences

        Returns a boolean mask over flux; True for cadences to keep

        Arguments:
            flux : ndarray
                flux for one visit
            visit : int
                visit number (for logging)
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
            mask = np.zeros(len(flux), dtype=bool)
        npts = len(flux)
        idx = np.arange(npts)

        loop = True
        count = 0
        while loop:
            # first pass: try median filter
            if loop:
                smoothed = median_filter(flux, kernel_size=kernel_size)

                bad = sigma_clip(
                    flux - smoothed,
                    sigma_upper=sigma_upper,
                    sigma_lower=sigma_lower,
                    stdfunc=mad_std,
                ).mask
                bad = bad * ~mask

            # second pass: try savgol filter
            # if over 1% of points were flagged with median filter
            if np.sum(bad) / len(bad) > 0.01:
                logger.debug(
                    "visit %d: median filter flagged %d of %d cadences; "
                    "retrying with savgol filter",
                    visit,
                    np.sum(bad),
                    len(bad),
                )
                smoothed = savgol_filter(flux, window_length=kernel_size, polyorder=2)
                bad = sigma_clip(
                    flux - smoothed,
                    sigma_upper=sigma_upper,
                    sigma_lower=sigma_lower,
                    stdfunc=mad_std,
                ).mask
                bad = bad * ~mask

            # third pass: skip outlier rejection
            if np.sum(bad) / len(bad) > 0.01:
                logger.warning(
                    "visit %d: >1%% of cadences flagged by median and savgol filters "
                    "(%d of %d); skipping outlier rejection this iteration",
                    visit,
                    np.sum(bad),
                    len(bad),
                )
                bad = np.zeros_like(mask)

            flux = flux[~bad]
            idx = idx[~bad]

            # update mask and iterate
            mask = mask[~bad]

            if np.sum(bad) == 0:
                loop = False
            else:
                count += 1

            if count >= 3:
                loop = False

        keep = np.zeros(npts, dtype=bool)
        keep[idx] = True
        return keep

    def _transit_theta(self):
        """Transit model parameters using the shape from QuickFitTransitShape"""
        theta = BaseTransitFittingAlg._theta_initial(self.star, self.planets)
        for n, p in enumerate(self.planets):
            theta[5 * n + 2] = p.ror
            theta[5 * n + 3] = p.impact

        return theta

    @staticmethod
    def _identify_breaks(cadno, flux, gap_tolerance, jump_tolerance, mask):
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
        gaps = np.pad(np.diff(cadno), (1, 0), "constant", constant_values=(1, 0))
        gap_locs = np.pad(
            np.where(gaps > gap_tolerance)[0],
            (1, 1),
            "constant",
            constant_values=(0, len(gaps) + 1),
        )

        # identify JUMPS in flux
        jumps = np.pad(np.diff(flux), (1, 0), "constant", constant_values=(0, 0))
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
        assert break_locs[-1] == len(cadno) - 1, "unexpected edge behavior"

        return break_locs

    @staticmethod
    def _estimate_oscillation_period(time, flux, min_period):
        lombscargle = LombScargle(time, flux)

        min_freq = 1 / (time.max() - time.min())
        max_freq = 1 / min_period

        xf, yf = lombscargle.autopower(
            minimum_frequency=min_freq, maximum_frequency=max_freq
        )

        peak_freq = xf[np.argmax(yf)]
        peak_per = np.max([1.0 / peak_freq, 1.001 * min_period])

        return peak_per

    def _estimate_oscillation_periods(self, lc):
        """One Lomb-Scargle oscillation period per visit, in np.unique(visit) order"""
        visits = np.unique(lc.visit)
        oscillation_periods = np.zeros(len(visits))
        for j, v in enumerate(visits):
            use = lc.visit == v
            min_period = self._min_period(lc.obsmode[use][0])
            oscillation_periods[j] = self._estimate_oscillation_period(
                lc.time[use], lc.flux[use], min_period
            )
            logger.debug(
                "visit %d: Lomb-Scargle oscillation period %.2f d (floor %.2f d)",
                v,
                oscillation_periods[j],
                1.001 * min_period,
            )

        osc_per_mu = np.nanmedian(oscillation_periods)
        osc_per_sd = np.max(
            [mad_std(oscillation_periods, ignore_nan=True), 0.1 * osc_per_mu]
        )

        logger.info(
            "nominal stellar oscillation period = %.1f +/- %.1f days",
            osc_per_mu,
            osc_per_sd,
        )

        filename = os.path.join(
            self.config["PATHS"]["results_dir"],
            f"{self.litecurve.target}_stellar_oscillations.txt",
        )
        oscillation_period_by_quarter = np.ones(18) * np.nan
        quarter_indexes = visits
        oscillation_period_by_quarter[quarter_indexes] = oscillation_periods
        np.savetxt(filename, np.stack([np.arange(18), oscillation_period_by_quarter]).T)
        logger.debug("wrote %s", filename)

        return oscillation_periods

    def _detrend_gp(
        self,
        time,
        flux,
        cadno,
        visit,
        gp_term,
        nominal_gp_period,
        minimum_gp_period,
        transit_mask,
        gap_tolerance,
        jump_tolerance,
        correct_ramp,
    ):
        """
        Fit a GP trend to one visit; returns the predicted trend (same length as time)

        Arguments
            time, flux, cadno : ndarray
                cadence times, fluxes, and cadence numbers for one visit
            visit : int
                visit number (for logging)
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
            cadno, flux, gap_tolerance, jump_tolerance, transit_mask
        )

        n_seg = len(break_locs) - 1
        seg_id = np.zeros(len(time), dtype=int)

        for i in range(n_seg):
            seg_id[break_locs[i] : break_locs[i + 1]] = i
        # break_locs[-1] is the last cadence, which belongs to the last segment
        seg_id[break_locs[-1]] = n_seg - 1

        approx_mean = [np.mean(flux[seg_id == i]) for i in range(n_seg)]

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
            log_sigma = pm.Normal("log_sigma", mu=np.log(mad_std(flux)), sd=5.0)
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
                ramp_amp = pm.Normal("ramp_amp", mu=0, sd=np.std(flux), shape=n_seg)
                log_tau = pm.Normal("log_tau", mu=0, sd=5, shape=n_seg)
            else:
                ramp_amp = T.zeros(n_seg)
                log_tau = T.zeros(n_seg)

            # set up the gp
            oot = ~transit_mask
            mean_oot = pm.Deterministic(
                "mean_oot",
                _mean_fxn(time[oot], seg_id[oot], flux0, ramp_amp, log_tau),
            )
            mean_all = pm.Deterministic(
                "mean_all", _mean_fxn(time, seg_id, flux0, ramp_amp, log_tau)
            )
            log_var = pm.Normal(
                "log_var", mu=np.var(flux - median_filter(flux, 13)), sd=5.0
            )

            gp = GaussianProcess(kernel, mean=mean_oot)
            gp.compute(time[oot], diag=T.exp(log_var) * T.ones(len(time[oot])))

            # likelihood
            pm.Potential("lnlike", gp.log_likelihood(flux[oot]))

            # predicted trend; if len(y)!=len(t) set include_mean=False
            pred = gp.predict(flux[oot], t=time, include_mean=False)
            pm.Deterministic("pred", pred + mean_all)

        # optimize hyperparameters
        def _optimize(start, vars):
            return pmx.optimize(
                start=start, vars=vars, progress=False, verbose=False, return_info=True
            )

        with model:
            map_soln, info = _optimize(model.test_point, [flux0])
            map_soln, info = _optimize(map_soln, [flux0, log_var])

            for _ in range(1 + correct_ramp):
                if gp_term == "RotationTerm":
                    map_soln, info = _optimize(
                        map_soln, [log_var, flux0, sigma, P, log_Q0, log_dQ, mix]
                    )
                if gp_term == "SHOTerm":
                    map_soln, info = _optimize(
                        map_soln, [log_var, flux0, sigma, P, log_Q0]
                    )
                if correct_ramp:
                    map_soln, info = _optimize(
                        map_soln, [log_var, flux0, ramp_amp, log_tau]
                    )

        # "precision loss" non-success is routine here; pmx warns on non-finite logp
        if info is not None and not info.success:
            logger.debug("visit %d: %s GP optimizer: %s", visit, gp_term, info.message)
        logger.info(
            "visit %d: %s GP, %d segments, ramp = %s, P = %.2f d (nominal %.2f d), "
            "Q0 = %.2f, sigma = %.2e",
            visit,
            gp_term,
            n_seg,
            correct_ramp,
            map_soln["P"],
            nominal_gp_period,
            map_soln["Q0"],
            map_soln["sigma"],
        )

        return map_soln["pred"]

    def _detrend(
        self,
        time,
        flux,
        cadno,
        visit,
        obsmode,
        oscillation_periods,
        tight_mask,
        ttv_buffer,
    ):
        """
        Detrend one visit, falling back to successively simpler GP models;
        returns the predicted trend (same length as time)
        """

        if obsmode == "short cadence":
            gap_tolerance = np.max(
                [int(np.min(self.durations) / KEPLER_SCIT * 5 / 2), 91]
            )
        elif obsmode == "long cadence":
            gap_tolerance = np.max(
                [int(np.min(self.durations) / KEPLER_LCIT * 5 / 2), 13]
            )
        else:
            raise ValueError(f"unsuported obsmode: {obsmode}")

        min_period = self._min_period(obsmode)
        mask = self._transit_mask(time, obsmode, tight_mask, ttv_buffer)
        logger.debug(
            "visit %d: %s transit mask covers %d of %d cadences; gap tolerance = %d",
            visit,
            "tight" if tight_mask else "wide",
            np.sum(mask),
            len(mask),
            gap_tolerance,
        )
        if np.mean(mask) > 0.5:
            logger.warning(
                "visit %d: transit mask covers %.0f%% of cadences",
                visit,
                100 * np.mean(mask),
            )

        # use successively simpler models as fallbacks
        try:
            return self._detrend_gp(
                time,
                flux,
                cadno,
                visit,
                "RotationTerm",
                np.nanmedian(oscillation_periods),
                min_period,
                transit_mask=mask,
                gap_tolerance=gap_tolerance,
                jump_tolerance=5.0,
                correct_ramp=True,
            )
        except LinAlgError as e:
            logger.warning(
                "visit %d: RotationTerm GP with ramp failed (%s); refitting w/o ramp",
                visit,
                e,
            )

        try:
            return self._detrend_gp(
                time,
                flux,
                cadno,
                visit,
                "RotationTerm",
                np.nanmedian(oscillation_periods),
                min_period,
                transit_mask=mask,
                gap_tolerance=gap_tolerance,
                jump_tolerance=5.0,
                correct_ramp=False,
            )
        except LinAlgError as e:
            logger.warning(
                "visit %d: RotationTerm GP failed (%s); falling back to SHOTerm",
                visit,
                e,
            )

        return self._detrend_gp(
            time,
            flux,
            cadno,
            visit,
            "SHOTerm",
            np.nanmedian(oscillation_periods),
            min_period,
            transit_mask=mask,
            gap_tolerance=gap_tolerance,
            jump_tolerance=5.0,
            correct_ramp=False,
        )

    def perform(self, tight_mask, clip_outliers, ttv_buffer=None):
        """
        Parameters
        ----------
        tight_mask : bool
            True to mask 1 x T14 plus ttv_buffer; False to mask 3 x T14 plus 1 hour
        clip_outliers : str
            'local_trend' : clip incoming flux about a smoothed trend, sparing transits
            'transit_model' : clip incoming (detrended) flux about the transit model
        ttv_buffer : list of float
            per-planet transit time uncertainty buffer; required if tight_mask

        Outliers are clipped from the incoming flux; the surviving cadences are
        then reset to raw photometry (LiteCurve.reset) before detrending

        Returns
        -------
        star : Star
        planets : list of Planet
        litecurve : LiteCurve
            detrended litecurve
        """
        target = self.litecurve.target
        results_dir = self.config["PATHS"]["results_dir"]

        if clip_outliers not in ("local_trend", "transit_model"):
            raise ValueError(
                "clip_outliers must be 'local_trend' or 'transit_model', "
                f"got {clip_outliers!r}"
            )
        if tight_mask and ttv_buffer is None:
            raise ValueError("tight_mask=True requires ttv_buffer (from RegularizeOMC)")

        lc = self.litecurve
        visits = np.unique(lc.visit)

        # clip outliers from the incoming flux, one visit at a time
        if clip_outliers == "transit_model":
            f_mod = TransitModel(self.star, self.planets, lc).model_flux(
                self._transit_theta()
            )

        keep = np.ones(len(lc.time), dtype=bool)
        for v in visits:
            use = (lc.visit == v) & keep
            npts_initial = int(np.sum(use))

            if clip_outliers == "local_trend":
                transit_mask = self._transit_mask(
                    lc.time[use], lc.obsmode[use][0], tight_mask, ttv_buffer
                )
                keep[use] = self._clip_outliers(
                    lc.flux[use], v, 5, 5, 13, mask=transit_mask
                )
                use = (lc.visit == v) & keep
                keep[use] = self._clip_outliers(lc.flux[use], v, 5, 1000, 13)

            elif clip_outliers == "transit_model":
                sigma = set_sigma_cutoff(int(np.sum(use)), sigma_min=5.0)
                logger.debug("visit %d: transit-model clipping at %.1f sigma", v, sigma)
                keep[use] = ~sigma_clip(
                    lc.flux[use] - f_mod[use],
                    sigma_upper=sigma,
                    sigma_lower=sigma,
                    stdfunc=mad_std,
                ).mask

            logger.info(
                "visit %d: %d of %d cadences rejected as outliers (%s)",
                v,
                npts_initial - int(np.sum((lc.visit == v) & keep)),
                npts_initial,
                clip_outliers,
            )

        # detrend raw photometry for surviving cadences
        lc.select(keep)
        lc.reset(self.raw_litecurve)

        oscillation_periods = self._estimate_oscillation_periods(lc)

        for v in visits:
            use = lc.visit == v
            pred = self._detrend(
                lc.time[use],
                lc.flux[use],
                lc.cadno[use],
                v,
                lc.obsmode[use][0],
                oscillation_periods,
                tight_mask,
                ttv_buffer,
            )
            lc.flux[use] /= pred
            lc.error[use] /= pred

        obsmodes = np.unique(lc.obsmode)
        if len(obsmodes) != 1 or obsmodes[0] not in ("long cadence", "short cadence"):
            raise ValueError(f"expected a single long or short cadence, got {obsmodes}")
        cadence = "lc" if obsmodes[0] == "long cadence" else "sc"
        lc.to_fits(os.path.join(results_dir, f"{target}_{cadence}_detrended.fits"))

        return self.star, self.planets, lc
