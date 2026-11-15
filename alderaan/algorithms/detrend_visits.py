"""Detrend each visit of a litecurve with a GP, masking transits"""

import logging

import aesara_theano_fallback.tensor as T
import numpy as np
import pymc3 as pm
import pymc3_ext as pmx
from astropy.stats import mad_std
from astropy.timeseries import LombScargle
from celerite2.backprop import LinAlgError
from celerite2.theano import GaussianProcess
from celerite2.theano import terms as GPterms
from scipy.signal import medfilt as median_filter

from alderaan.algorithms.base import BaseAlg
from alderaan.constants import KEPLER_LCIT, KEPLER_SCIT

logger = logging.getLogger(__name__)


class DetrendVisits(BaseAlg):
    """
    Detrend a litecurve visit-by-visit

    Parameters
    ----------
    star : Star
    planets : list of Planet
    litecurve : LiteCurve
    """

    def __init__(self, star, planets, litecurve):
        super().__init__(star, planets, litecurve)

        self.durations = np.array([p.duration for p in self.planets])

    def _min_period(self, obsmode):
        if obsmode == "short cadence":
            return np.max([5 * np.max(self.durations), 91 * KEPLER_SCIT])
        if obsmode == "long cadence":
            return np.max([5 * np.max(self.durations), 13 * KEPLER_LCIT])
        raise ValueError(f"unsuported obsmode: {obsmode}")

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
        mask = self._transit_mask(time, obsmode, tight_mask)
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

    def perform(self, tight_mask):
        """
        Detrend the incoming flux of each visit

        Parameters
        ----------
        tight_mask : bool
            True to mask 1 x T14 plus the largest TransitTimes.ttv_buffer();
            False to mask 3 x T14 plus 1 hour

        Returns
        -------
        planets : list of Planet
        litecurve : LiteCurve
            detrended litecurve
        """
        lc = self.litecurve
        visits = np.unique(lc.visit)

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
            )
            lc.flux[use] /= pred
            lc.error[use] /= pred

        return self.planets, lc
