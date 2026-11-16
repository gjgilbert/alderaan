"""Detrend each visit of a litecurve with a GP, masking transits"""

import logging

import numpy as np
import pymc as pm
import pytensor.tensor as T
from astropy.stats import mad_std
from astropy.timeseries import LombScargle
from celerite2.backprop import LinAlgError
from celerite2.pymc import GaussianProcess
from celerite2.pymc import terms as GPterms
from scipy.signal import medfilt as median_filter

from alderaan.algorithms.base import BaseAlg
from alderaan.constants import EXPTIME
from alderaan.utils.optim import optimize

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
        Split a visit into contiguous segments at gaps (breaks in times) and jumps
        (sudden flux changes); returns an integer segment ID (0..n_seg-1) per cadence
            * GAPS are consecutive missing cadences in the TIME domain
            * JUMPS are large cadence-to-cadence FLUX variations outside the mask
        Breaks closer than gap_tolerance_cadences to the previous candidate or the
        end are dropped. A segment with no unmasked cadences is merged into the
        preceding segment (leading ones into the first segment with unmasked
        cadences), with a warning.

        Parameters
        ----------
        gap_tolerance_cadences : int
            number of cadences to be considered a (time) gap
        jump_tolerance : float
            sigma threshold for identifying (flux) jumps
        mask : ndarray, bool
            True near transit
        """
        # first cadence after each GAP in time
        gap_locs = np.where(np.diff(cadno) > gap_tolerance_cadences)[0] + 1

        # first cadence after each JUMP in flux
        jumps = np.pad(np.diff(flux), (1, 0), "constant", constant_values=(0, 0))
        jump_size = np.abs(jumps - np.median(jumps)) / mad_std(jumps)
        jump_locs = np.where(~mask & (jump_size > jump_tolerance))[0]

        # candidate segment starts; drop those too close to the previous or the end
        starts = np.unique(np.hstack([gap_locs, jump_locs]))
        starts = starts[(starts > 0) & (starts < len(cadno))]
        too_close = np.diff(np.hstack([0, starts])) < gap_tolerance_cadences
        if len(starts) and len(cadno) - starts[-1] < gap_tolerance_cadences:
            too_close[-1] = True
        seg_start = np.zeros(len(cadno), dtype=bool)
        seg_start[starts[~too_close]] = True
        seg_id = np.cumsum(seg_start)

        # a segment with no unmasked cadences is unconstrained; merge it
        has_oot = np.bincount(seg_id, weights=~mask) > 0
        if not has_oot.all():
            logger.warning(
                "merging %d fully masked segment(s) into a neighbor",
                np.sum(~has_oot),
            )
            seg_id = np.cumsum(has_oot & (np.cumsum(has_oot) > 1))[seg_id]

        return seg_id

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

    @staticmethod
    def _mean_fxn(time, seg_id, seg_t0, flux0, ramp_amp, log_tau):
        """
        GP mean: per-segment baseline flux0 x (1 + exponential ramp), the ramp
        starting at seg_t0 (segment start time from all cadences, so the mean is
        the same whichever cadences it is evaluated on)
        """
        return flux0[seg_id] * (
            1
            + ramp_amp[seg_id]
            * T.exp(-(time - seg_t0[seg_id]) / T.exp(log_tau[seg_id]))
        )

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

        seg_id = self._identify_breaks(
            cadno, flux, gap_tolerance_cadences, jump_tolerance, transit_mask
        )
        n_seg = int(seg_id.max()) + 1
        # ramp origin per segment, from all cadences, so fit and prediction agree
        seg_t0 = np.array([time[seg_id == i].min() for i in range(n_seg)])

        # priors from out-of-transit cadences only, as in the likelihood
        oot = ~transit_mask
        approx_mean = [np.mean(flux[oot & (seg_id == i)]) for i in range(n_seg)]

        # pymc model
        with pm.Model() as model:
            # set up the kernel
            log_sigma = pm.Normal("log_sigma", mu=np.log(mad_std(flux[oot])), sigma=5.0)
            sigma = pm.Deterministic("sigma", T.exp(log_sigma))

            log_P_off = pm.Normal(
                "log_P", mu=np.log(nominal_gp_period - minimum_gp_period), sigma=2.0
            )
            P = pm.Deterministic("P", minimum_gp_period + T.exp(log_P_off))

            log_Q0 = pm.Normal("log_Q0", mu=0.0, sigma=5.0, initval=np.log(0.5))
            Q0 = pm.Deterministic("Q0", T.exp(log_Q0))

            if gp_term == "RotationTerm":
                log_dQ = pm.Normal("log_dQ", mu=0.0, sigma=5.0, initval=np.log(1e-3))
                mix = pm.Uniform("mix", lower=0, upper=1, initval=0.1)
                kernel = GPterms.RotationTerm(
                    sigma=sigma, period=P, Q0=Q0, dQ=T.exp(log_dQ), f=mix
                )

            elif gp_term == "SHOTerm":
                kernel = GPterms.SHOTerm(sigma=sigma, w0=2 * np.pi / P, Q=0.5 + Q0)

            else:
                raise ValueError(f"gp_term {gp_term} not recognized")

            # set up mean function and variance
            flux0 = pm.Normal(
                "flux0", mu=approx_mean, sigma=np.ones(n_seg), shape=n_seg
            )

            if correct_ramp:
                ramp_amp = pm.Normal(
                    "ramp_amp", mu=0, sigma=np.std(flux[oot]), shape=n_seg
                )
                log_tau = pm.Normal("log_tau", mu=0, sigma=5, shape=n_seg)
            else:
                ramp_amp = T.zeros(n_seg)
                log_tau = T.zeros(n_seg)

            # set up the gp
            mean_oot = pm.Deterministic(
                "mean_oot",
                self._mean_fxn(
                    time[oot], seg_id[oot], seg_t0, flux0, ramp_amp, log_tau
                ),
            )
            mean_all = pm.Deterministic(
                "mean_all",
                self._mean_fxn(time, seg_id, seg_t0, flux0, ramp_amp, log_tau),
            )
            log_var = pm.Normal(
                "log_var",
                mu=np.log(np.var(flux[oot] - median_filter(flux[oot], 13))),
                sigma=5.0,
            )

            gp = GaussianProcess(kernel, mean=mean_oot)
            gp.compute(time[oot], diag=T.exp(log_var) * T.ones(len(time[oot])))

            # likelihood
            pm.Potential("lnlike", gp.log_likelihood(flux[oot]))

            # predicted trend; if len(y)!=len(t) set include_mean=False
            pred = gp.predict(flux[oot], t=time, include_mean=False)
            pm.Deterministic("pred", pred + mean_all)

        # optimize hyperparameters

        with model:
            map_soln, info = optimize(None, [flux0])
            map_soln, info = optimize(map_soln, [flux0, log_var])

            for _ in range(1 + correct_ramp):
                if gp_term == "RotationTerm":
                    map_soln, info = optimize(
                        map_soln,
                        [log_var, flux0, log_sigma, log_P_off, log_Q0, log_dQ, mix],
                    )
                if gp_term == "SHOTerm":
                    map_soln, info = optimize(
                        map_soln, [log_var, flux0, log_sigma, log_P_off, log_Q0]
                    )
                if correct_ramp:
                    map_soln, info = optimize(
                        map_soln, [log_var, flux0, ramp_amp, log_tau]
                    )

        # "precision loss" non-success is routine here
        if not info.success:
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
        models = [("RotationTerm", True), ("RotationTerm", False), ("SHOTerm", False)]
        for gp_term, correct_ramp in models:
            try:
                return self._detrend_gp(
                    time,
                    flux,
                    cadno,
                    visit,
                    gp_term,
                    nominal_oscillation_period,
                    min_oscillation_period,
                    transit_mask=mask,
                    gap_tolerance_cadences=gap_tolerance_cadences,
                    jump_tolerance=5.0,
                    correct_ramp=correct_ramp,
                )
            except LinAlgError as e:
                if (gp_term, correct_ramp) == models[-1]:
                    raise
                logger.warning(
                    "visit %d: %s GP (ramp = %s) failed (%s); trying a simpler model",
                    visit,
                    gp_term,
                    correct_ramp,
                    e,
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
