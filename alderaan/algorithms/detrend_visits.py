"""Detrend each visit of a litecurve with a GP, masking transits"""

import logging

import celerite2
import numpy as np
from astropy.stats import mad_std
from astropy.timeseries import LombScargle
from celerite2 import terms as GPterms
from celerite2.driver import LinAlgError
from scipy.ndimage import median_filter
from scipy.special import expit, logit

from alderaan.algorithms.base import BaseAlg
from alderaan.constants import EXPTIME
from alderaan.utils.optimize import optimize

logger = logging.getLogger(__name__)


class _GPTrendModel:
    """
    GP trend model for one visit, fit to out-of-transit cadences: a per-segment
    mean (baseline x exponential ramp) plus a celerite2 GP shared by all segments

    Parameters form a flat vector of unconstrained values with Normal priors;
    for RotationTerm, mix ~ Uniform(0, 1) is carried as logit(mix)

    Parameters
    ----------
    time, flux : ndarray
        cadence times and fluxes for one visit
    oot : ndarray, bool
        True for out-of-transit cadences (used in the likelihood)
    seg_id : ndarray, int
        segment ID per cadence (0..n_seg-1)
    seg_t0 : ndarray
        start time of each segment, from all cadences (ramp origin)
    gp_term : str
        "RotationTerm" or "SHOTerm"
    correct_ramp : bool
        fit a per-segment exponential ramp
    nominal_period : float
        nominal stellar oscillation period (days); centers the GP period prior
    min_period : float
        floor on the GP (stellar oscillation) period (days)
    """

    def __init__(
        self,
        time,
        flux,
        oot,
        seg_id,
        seg_t0,
        gp_term,
        correct_ramp,
        nominal_period,
        min_period,
    ):
        if gp_term not in ("RotationTerm", "SHOTerm"):
            raise ValueError(f"gp_term {gp_term} not recognized")

        self.time, self.flux, self.oot = time, flux, oot
        self.seg_id, self.seg_t0 = seg_id, seg_t0
        self.gp_term = gp_term
        self.rotation = gp_term == "RotationTerm"
        self.correct_ramp = correct_ramp
        self.nominal_period = nominal_period
        self.min_period = min_period
        self.n_seg = len(seg_t0)

        # Normal priors (mu, sigma), from out-of-transit cadences only
        flux_oot = flux[oot]
        approx_mean = [np.mean(flux[oot & (seg_id == i)]) for i in range(self.n_seg)]
        self.priors = {
            "log_sigma": (np.log(mad_std(flux_oot)), 5.0),
            "log_P": (np.log(nominal_period - min_period), 2.0),
            "log_Q0": (0.0, 5.0),
        }
        if self.rotation:
            self.priors["log_dQ"] = (0.0, 5.0)
        self.priors["flux0"] = (np.array(approx_mean), 1.0)
        if correct_ramp:
            self.priors["ramp_amp"] = (np.zeros(self.n_seg), np.std(flux_oot))
            self.priors["log_tau"] = (np.zeros(self.n_seg), 5.0)
        self.priors["log_var"] = (
            np.log(np.var(flux_oot - median_filter(flux_oot, size=13, mode="mirror"))),
            5.0,
        )

        # parameter vector layout: name -> indices
        sizes = {name: np.size(mu) for name, (mu, _) in self.priors.items()}
        if self.rotation:
            sizes["logit_mix"] = 1
        self.ix, start = {}, 0
        for name, size in sizes.items():
            self.ix[name] = np.arange(start, start + size)
            start += size

        # starting point: prior means, except as noted
        self.x0 = np.zeros(start)
        for name, (mu, _) in self.priors.items():
            self.x0[self.ix[name]] = mu
        self.x0[self.ix["log_Q0"]] = np.log(0.5)
        if self.rotation:
            self.x0[self.ix["log_dQ"]] = np.log(1e-3)
            self.x0[self.ix["logit_mix"]] = logit(0.1)

    def unpack(self, x):
        """Parameter vector x as a dict of named values"""
        p = {name: x[i] for name, i in self.ix.items()}
        if not self.correct_ramp:
            p["ramp_amp"] = np.zeros(self.n_seg)
            p["log_tau"] = np.zeros(self.n_seg)
        return p

    def hyperparameters(self, p):
        """GP oscillation period (days), Q0, and sigma"""
        period = self.min_period + np.exp(p["log_P"][0])
        return period, np.exp(p["log_Q0"][0]), np.exp(p["log_sigma"][0])

    def _mean(self, p, mask=slice(None)):
        """
        GP mean at the cadences selected by mask (default all): per-segment
        baseline flux0 x (1 + exponential ramp), the ramp starting at seg_t0
        (segment start from all cadences, so the mean does not depend on mask)
        """
        seg_id = self.seg_id[mask]
        ramp = np.exp(
            -(self.time[mask] - self.seg_t0[seg_id]) / np.exp(p["log_tau"][seg_id])
        )
        return p["flux0"][seg_id] * (1 + p["ramp_amp"][seg_id] * ramp)

    def _gp(self, p):
        period, Q0, sigma = self.hyperparameters(p)
        if self.rotation:
            kernel = GPterms.RotationTerm(
                sigma=sigma,
                period=period,
                Q0=Q0,
                dQ=np.exp(p["log_dQ"][0]),
                f=expit(p["logit_mix"][0]),
            )
        else:
            kernel = GPterms.SHOTerm(sigma=sigma, w0=2 * np.pi / period, Q=0.5 + Q0)
        gp = celerite2.GaussianProcess(kernel)
        diag = np.full(np.sum(self.oot), np.exp(p["log_var"][0]))
        gp.compute(self.time[self.oot], diag=diag)
        return gp

    def logp(self, x):
        """Log posterior (up to a constant) of parameter vector x"""
        p = self.unpack(x)
        resid = self.flux[self.oot] - self._mean(p, self.oot)
        lp = self._gp(p).log_likelihood(resid)
        for name, (mu, sd) in self.priors.items():
            lp += np.sum(-0.5 * ((p[name] - mu) / sd) ** 2 - np.log(sd))
        if self.rotation:
            # log-Jacobian of mix = expit(logit_mix)
            mix = expit(p["logit_mix"][0])
            lp += np.log(mix) + np.log1p(-mix)
        return lp

    def predict(self, x):
        """Predicted trend at every cadence for parameter vector x"""
        p = self.unpack(x)
        resid = self.flux[self.oot] - self._mean(p, self.oot)
        return self._gp(p).predict(resid, t=self.time) + self._mean(p)


class DetrendVisits(BaseAlg):
    """
    Detrend a litecurve visit-by-visit

    Parameters
    ----------
    star : Star
    planets : list of Planet
    litecurve : LiteCurve
    """

    # minimum gap tolerance and oscillation period, in cadences
    CADENCE_FLOOR = {"short cadence": 91, "long cadence": 13}

    def __init__(self, star, planets, litecurve):
        super().__init__(star, planets, litecurve)

        self.durations = np.array([p.duration for p in self.planets])

    def _min_oscillation_period(self, obsmode):
        """
        Floor on the stellar oscillation period (days): 5 x the longest planet
        transit duration, or a fixed number of cadences, whichever is longer
        """
        return np.max(
            [5 * np.max(self.durations), self.CADENCE_FLOOR[obsmode] * EXPTIME[obsmode]]
        )

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

    def _nominal_oscillation_period(self, lc):
        """
        Stellar oscillation period (days) shared by all visits: the median of
        one Lomb-Scargle estimate per visit
        """
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

        return osc_per_mu

    @staticmethod
    def _fit(model, visit):
        """
        Optimize a _GPTrendModel in stages, ending with all parameters jointly;
        returns the predicted trend (same length as model.time)
        """
        gp_vars = ["log_var", "flux0", "log_sigma", "log_P", "log_Q0"]
        if model.rotation:
            gp_vars += ["log_dQ", "logit_mix"]

        stages = [["flux0"], ["flux0", "log_var"]]
        for _ in range(1 + model.correct_ramp):
            stages.append(gp_vars)
            if model.correct_ramp:
                stages.append(["log_var", "flux0", "ramp_amp", "log_tau"])
        # final joint stage over every parameter, so correlated blocks (e.g.
        # ramp and GP) are not left at a block-coordinate optimum
        stages.append(list(model.ix))

        stage_ix = [np.concatenate([model.ix[n] for n in s]) for s in stages]
        x, info = optimize(model.logp, model.x0, stage_ix)

        # "precision loss" non-success is routine here
        if not info.success:
            logger.debug(
                "visit %d: %s GP optimizer: %s", visit, model.gp_term, info.message
            )
        period, Q0, sigma = model.hyperparameters(model.unpack(x))
        logger.info(
            "visit %d: %s GP, %d segments, ramp = %s, oscillation P = %.2f d "
            "(nominal %.2f d), Q0 = %.2f, sigma = %.2e",
            visit,
            model.gp_term,
            model.n_seg,
            model.correct_ramp,
            period,
            model.nominal_period,
            Q0,
            sigma,
        )

        return model.predict(x)

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

        The visit is split into contiguous segments at gaps and flux jumps;
        each segment has its own mean, and all share one GP.
        nominal_oscillation_period is the stellar oscillation period (days)
        shared by all visits
        """
        exptime = EXPTIME[obsmode]
        gap_tolerance_cadences = np.max(
            [int(np.min(self.durations) / exptime * 5 / 2), self.CADENCE_FLOOR[obsmode]]
        )

        # tight is as narrow as possible for the best detrending;
        # wide is cautious, keeping well clear of transits
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

        seg_id = self._identify_breaks(
            cadno, flux, gap_tolerance_cadences, jump_tolerance=5.0, mask=mask
        )
        # ramp origin per segment, from all cadences, so fit and prediction agree
        seg_t0 = np.array([time[seg_id == i].min() for i in range(seg_id.max() + 1)])

        # use successively simpler models as fallbacks
        models = [("RotationTerm", True), ("RotationTerm", False), ("SHOTerm", False)]
        for gp_term, correct_ramp in models:
            model = _GPTrendModel(
                time,
                flux,
                ~mask,
                seg_id,
                seg_t0,
                gp_term,
                correct_ramp,
                nominal_oscillation_period,
                self._min_oscillation_period(obsmode),
            )
            try:
                return self._fit(model, visit)
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
        nominal_oscillation_period = self._nominal_oscillation_period(lc)

        for v in np.unique(lc.visit):
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
