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
from alderaan.constants import EXPTIME

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

    def _min_oscillation_period(self, obsmode):
        """
        Floor on the stellar oscillation period (days): 5 x the longest planet
        transit duration, or a fixed number of cadences, whichever is longer
        """
        if obsmode == "short cadence":
            return np.max([5 * np.max(self.durations), 91 * EXPTIME["short cadence"]])
        if obsmode == "long cadence":
            return np.max([5 * np.max(self.durations), 13 * EXPTIME["long cadence"]])
        raise ValueError(f"unsupported obsmode: {obsmode}")

    @staticmethod
    def _identify_breaks(cadno, flux, gap_tolerance_cadences, jump_tolerance, mask):
        """
        Identify gaps (breaks in times) and jumps (sudden flux changes) in a litecurve
            * GAPS are consecutive missing cadences in the TIME domain
            * JUMPS are large cadence-to-cadence FLUX variations

        Parameters
        ----------
        gap_tolerance_cadences : int
            number of cadences to be considered a (time) gap
        jump_tolerance : float
            sigma threshold for identifying (flux) jumps
        mask : array-like, bool
            1 near transit, 0 far from transit
        """
        # identify GAPS in time
        gaps = np.pad(np.diff(cadno), (1, 0), "constant", constant_values=(1, 0))
        gap_locs = np.pad(
            np.where(gaps > gap_tolerance_cadences)[0],
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
        bad = np.hstack([False, np.diff(break_locs) < gap_tolerance_cadences])
        if bad[-1]:
            bad[-2:] = (True, False)
        break_locs = break_locs[~bad]
        break_locs[-1] -= 2

        assert break_locs[0] == 0, "unexpected edge behavior"
        assert break_locs[-1] == len(cadno) - 1, "unexpected edge behavior"

        return break_locs

    @staticmethod
    def _estimate_oscillation_period(time, flux, min_oscillation_period):
        lombscargle = LombScargle(time, flux)

        min_freq = 1 / (time.max() - time.min())
        max_freq = 1 / min_oscillation_period

        xf, yf = lombscargle.autopower(
            minimum_frequency=min_freq, maximum_frequency=max_freq
        )

        peak_freq = xf[np.argmax(yf)]
        peak_per = np.max([1.0 / peak_freq, 1.001 * min_oscillation_period])

        return peak_per

    def _estimate_oscillation_periods(self, lc):
        """One Lomb-Scargle oscillation period per visit, in np.unique(visit) order"""
        visits = np.unique(lc.visit)
        oscillation_periods = np.zeros(len(visits))
        for j, v in enumerate(visits):
            use = lc.visit == v
            min_oscillation_period = self._min_oscillation_period(lc.obsmode[use][0])
            oscillation_periods[j] = self._estimate_oscillation_period(
                lc.time[use], lc.flux[use], min_oscillation_period
            )
            logger.debug(
                "visit %d: Lomb-Scargle oscillation period %.2f d (floor %.2f d)",
                v,
                oscillation_periods[j],
                1.001 * min_oscillation_period,
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
        gap_tolerance_cadences,
        jump_tolerance,
        correct_ramp,
    ):
        """
        Fit a GP trend to one visit; returns the predicted trend (same length as time)

        Parameters
        ----------
        time, flux, cadno : ndarray
            cadence times, fluxes, and cadence numbers for one visit
        visit : int
            visit number (for logging)
        gp_term : str
            "RotationTerm" or "SHOTerm"
        nominal_gp_period : float
            nominal stellar oscillation period (days); centers the GP period prior
        minimum_gp_period : float
            floor on the GP (stellar oscillation) period (days)
        transit_mask : ndarray
        gap_tolerance_cadences : int
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
            cadno, flux, gap_tolerance_cadences, jump_tolerance, transit_mask
        )

        n_seg = len(break_locs) - 1
        seg_id = np.zeros(len(time), dtype=int)

        for i in range(n_seg):
            seg_id[break_locs[i] : break_locs[i + 1]] = i
        # break_locs[-1] is the last cadence, which belongs to the last segment
        seg_id[break_locs[-1]] = n_seg - 1

        # priors from out-of-transit cadences only, as in the likelihood
        oot = ~transit_mask
        approx_mean = [np.mean(flux[oot & (seg_id == i)]) for i in range(n_seg)]

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
            log_sigma = pm.Normal("log_sigma", mu=np.log(mad_std(flux[oot])), sd=5.0)
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
                ramp_amp = pm.Normal(
                    "ramp_amp", mu=0, sd=np.std(flux[oot]), shape=n_seg
                )
                log_tau = pm.Normal("log_tau", mu=0, sd=5, shape=n_seg)
            else:
                ramp_amp = T.zeros(n_seg)
                log_tau = T.zeros(n_seg)

            # set up the gp
            mean_oot = pm.Deterministic(
                "mean_oot",
                _mean_fxn(time[oot], seg_id[oot], flux0, ramp_amp, log_tau),
            )
            mean_all = pm.Deterministic(
                "mean_all", _mean_fxn(time, seg_id, flux0, ramp_amp, log_tau)
            )
            log_var = pm.Normal(
                "log_var",
                mu=np.log(np.var(flux[oot] - median_filter(flux[oot], 13))),
                sd=5.0,
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
            "visit %d: %s GP, %d segments, ramp = %s, oscillation P = %.2f d "
            "(nominal %.2f d), "
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
        nominal_oscillation_period,
        tight_mask,
    ):
        """
        Detrend one visit, falling back to successively simpler GP models;
        returns the predicted trend (same length as time)

        nominal_oscillation_period is the stellar oscillation period (days)
        shared by all visits
        """

        if obsmode == "short cadence":
            gap_tolerance_cadences = np.max(
                [int(np.min(self.durations) / EXPTIME["short cadence"] * 5 / 2), 91]
            )
        elif obsmode == "long cadence":
            gap_tolerance_cadences = np.max(
                [int(np.min(self.durations) / EXPTIME["long cadence"] * 5 / 2), 13]
            )
        else:
            raise ValueError(f"unsupported obsmode: {obsmode}")

        min_oscillation_period = self._min_oscillation_period(obsmode)
        # tight is as narrow as possible for the best detrending;
        # wide is cautious, keeping well clear of transits
        exptime = EXPTIME[obsmode]
        if tight_mask:
            # full width = max(T14, 2 hours) + 2 x (exptime + largest ttv_buffer)
            ttv_buffer = max(p.transit_times.ttv_buffer() for p in self.planets)
            mask = self.make_transit_mask(
                rel_size=1.0,
                abs_size=2 / 24,
                buffer=2 * (exptime + ttv_buffer),
                time=time,
            )
        else:
            # full width = max(3 x T14, 2 x (exptime + 1 hour))
            mask = self.make_transit_mask(
                rel_size=3.0, abs_size=2 * (exptime + 1 / 24), time=time
            )
        logger.debug(
            "visit %d: %s transit mask covers %d of %d cadences; gap tolerance = %d",
            visit,
            "tight" if tight_mask else "wide",
            np.sum(mask),
            len(mask),
            gap_tolerance_cadences,
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
                nominal_oscillation_period,
                min_oscillation_period,
                transit_mask=mask,
                gap_tolerance_cadences=gap_tolerance_cadences,
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
                nominal_oscillation_period,
                min_oscillation_period,
                transit_mask=mask,
                gap_tolerance_cadences=gap_tolerance_cadences,
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
            nominal_oscillation_period,
            min_oscillation_period,
            transit_mask=mask,
            gap_tolerance_cadences=gap_tolerance_cadences,
            jump_tolerance=5.0,
            correct_ramp=False,
        )

    def perform(self, method):
        """
        Detrend the incoming flux of each visit

        Parameters
        ----------
        method : str
            'wide_mask' : mask transits over the larger of 3 x T14 or
                2 x (exptime + 1 hour)
            'tight_mask' : mask transits over max(T14, 2 hours) +
                2 x (exptime + largest TransitTimes.ttv_buffer())

        Returns
        -------
        planets : list of Planet
        litecurve : LiteCurve
            detrended litecurve
        """
        if method not in ("wide_mask", "tight_mask"):
            raise ValueError(
                f"method must be 'wide_mask' or 'tight_mask', got {method!r}"
            )
        tight_mask = method == "tight_mask"

        lc = self.litecurve
        visits = np.unique(lc.visit)

        # one stellar oscillation period per visit; all visits share the median
        oscillation_periods = self._estimate_oscillation_periods(lc)
        nominal_oscillation_period = np.nanmedian(oscillation_periods)

        for v in visits:
            use = lc.visit == v
            pred = self._detrend(
                lc.time[use],
                lc.flux[use],
                lc.cadno[use],
                v,
                lc.obsmode[use][0],
                nominal_oscillation_period,
                tight_mask,
            )
            lc.flux[use] /= pred
            lc.error[use] /= pred

        return self.planets, lc
