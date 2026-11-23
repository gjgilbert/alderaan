"""Detrend each visit of a litecurve with a GP, masking transits"""

import logging

import celerite2
import numpy as np
from astropy.stats import mad_std, sigma_clip
from celerite2 import terms as GPterms
from celerite2.driver import LinAlgError
from scipy.ndimage import median_filter
from scipy.special import expit, logit
from scipy.stats import norm

from alderaan.algorithms.base import BaseAlg
from alderaan.constants import EXPTIME
from alderaan.utils.optimize import optimize

logger = logging.getLogger(__name__)


class _GPTrendModel:
    """
    GP trend model for one visit, fit to out-of-transit cadences

    The trend is a per-segment mean (baseline x exponential ramp) plus a
    celerite2 GP shared by all segments.

    Parameters form a flat vector of unconstrained values with Normal priors
    (log-scale parameters truncated at 4 sd; logp is also -inf where the
    covariance cannot be factorized); RotationTerm's mix ~ Uniform(0, 1)
    is carried as logit(mix), truncated at |logit(mix)| <= 20.

    Quality: q = 1/2 + exp(log_Q0) is the SHOTerm Q (> 1/2) or the
    RotationTerm Q0 (> 1/2, so both of its terms have Q = 1/2 + Q0 (+ dQ) > 1).
    As Q -> 1/2, a RotationTerm term's w0 and damping rate diverge at fixed
    period (the timescale collapses); an SHOTerm at fixed w0 only gets slower.

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
    nominal_period : float
        nominal stellar oscillation period (days); sets the GP period prior
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
        nominal_period,
        min_period,
    ):
        if gp_term not in ("RotationTerm", "SHOTerm"):
            raise ValueError(f"gp_term {gp_term} not recognized")

        self.time, self.flux, self.oot = time, flux, oot
        self.seg_id, self.seg_t0 = seg_id, seg_t0
        self.rotation = gp_term == "RotationTerm"
        self.min_period = min_period
        self.n_seg = len(seg_t0)

        # Normal priors (mu, sigma), from out-of-transit cadences only; widths are
        # weakly informative, except log_P, kept near the vetted ACF period
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

        # starting point: prior means, except log_Q0, log_dQ and logit_mix
        self.x0 = np.zeros(start)
        for name, (mu, _) in self.priors.items():
            self.x0[self.ix[name]] = mu
        self.x0[self.ix["log_Q0"]] = np.log(2.5)  # q = 3 (one coherent cycle)
        if self.rotation:
            self.x0[self.ix["log_dQ"]] = np.log(1e-3)
            self.x0[self.ix["logit_mix"]] = logit(0.1)

    def unpack(self, x):
        """Parameter vector x as a dict of named values"""
        return {name: x[i] for name, i in self.ix.items()}

    def hyperparameters(self, p):
        """GP oscillation period (days), q (SHOTerm Q, RotationTerm Q0), sigma"""
        period = self.min_period + np.exp(p["log_P"][0])
        return period, 0.5 + np.exp(p["log_Q0"][0]), np.exp(p["log_sigma"][0])

    def _mean(self, p, mask=slice(None)):
        """
        GP mean at the cadences selected by mask (default all)

        Per-segment baseline flux0 x (1 + exponential ramp), the ramp starting at
        seg_t0 (segment start from all cadences, so the mean does not depend on
        mask).
        """
        seg_id = self.seg_id[mask]
        ramp = np.exp(
            -(self.time[mask] - self.seg_t0[seg_id]) / np.exp(p["log_tau"][seg_id])
        )
        return p["flux0"][seg_id] * (1 + p["ramp_amp"][seg_id] * ramp)

    def _gp(self, p):
        """celerite2 GP for named parameters p, computed at out-of-transit cadences"""
        period, q, sigma = self.hyperparameters(p)
        if self.rotation:
            kernel = GPterms.RotationTerm(
                sigma=sigma,
                period=period,
                Q0=q,
                dQ=np.exp(p["log_dQ"][0]),
                f=expit(p["logit_mix"][0]),
            )
        else:
            kernel = GPterms.SHOTerm(sigma=sigma, w0=2 * np.pi / period, Q=q)
        gp = celerite2.GaussianProcess(kernel)
        diag = np.full(np.sum(self.oot), np.exp(p["log_var"][0]))
        gp.compute(self.time[self.oot], diag=diag)
        return gp

    def logp(self, x):
        """Log posterior (up to a constant) of parameter vector x"""
        p = self.unpack(x)
        # priors on log-scale parameters are truncated at 4 sd, where celerite2,
        # the exp ramp, and the white-noise variance stay finite
        for name in ("log_sigma", "log_P", "log_Q0", "log_dQ", "log_tau", "log_var"):
            if name in self.priors:
                mu, sd = self.priors[name]
                if np.any(np.abs(p[name] - mu) > 4 * sd):
                    return -np.inf
        # |logit_mix| <= 20 keeps mix strictly inside (0, 1) in float64
        if self.rotation and abs(p["logit_mix"][0]) > 20:
            return -np.inf
        resid = self.flux[self.oot] - self._mean(p, self.oot)
        try:
            gp = self._gp(p)
        except LinAlgError:  # numerically singular covariance: reject like truncation
            return -np.inf
        lp = gp.log_likelihood(resid)
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
    """Detrend a litecurve visit-by-visit"""

    def __init__(self, star, planets, litecurve):
        super().__init__(star, planets, litecurve)

        self.durations = np.array([p.duration for p in self.planets])

    @staticmethod
    def _cadence_floor(obsmode):
        """Floor on gap tolerance and GP period: >= 13 cadences and >= 1.5 hr"""
        # 13 cadences give the GP enough points
        return max(13, int(1.5 / 24 / EXPTIME[obsmode]))

    def _min_period(self, obsmode):
        """
        Floor on the GP oscillation period (days)

        The larger of 3 x the longest T14 and _cadence_floor cadences.
        """
        return np.max(
            [
                3 * np.max(self.durations),
                self._cadence_floor(obsmode) * EXPTIME[obsmode],
            ]
        )

    @staticmethod
    def _identify_breaks(cadno, flux, mask, gap_tolerance, jump_tolerance):
        """
        Segment ID (0..n_seg-1) per cadence of one visit

        The visit splits at time gaps of more than gap_tolerance cadences and at
        flux jumps (> jump_tolerance sigma, outside the mask; mask is True near
        transit). Breaks within gap_tolerance cadences of the previous candidate
        break, the visit start, or the end are dropped. A segment with no
        unmasked cadences is merged into a neighbor.
        """
        # first cadence after each gap in time
        gap_locs = np.where(np.diff(cadno) > gap_tolerance)[0] + 1

        # first cadence after each jump in flux
        jumps = np.pad(np.diff(flux), (1, 0), "constant", constant_values=(0, 0))
        jump_size = np.abs(jumps - np.median(jumps)) / mad_std(jumps)
        jump_locs = np.where(~mask & (jump_size > jump_tolerance))[0]

        # candidate segment starts; drop those too close to the previous or the end
        starts = np.unique(np.hstack([gap_locs, jump_locs]))
        starts = starts[(starts > 0) & (starts < len(cadno))]
        too_close = np.diff(np.hstack([0, starts])) < gap_tolerance
        if len(starts) and len(cadno) - starts[-1] < gap_tolerance:
            too_close[-1] = True
        seg_start = np.zeros(len(cadno), dtype=bool)
        seg_start[starts[~too_close]] = True
        seg_id = np.cumsum(seg_start)

        # a segment with no unmasked cadences is unconstrained; merge it
        has_oot = np.bincount(seg_id, weights=~mask) > 0
        if not has_oot.all():
            logger.info(
                "merging %d fully masked segment(s) into a neighbor",
                np.sum(~has_oot),
            )
            seg_id = np.cumsum(has_oot & (np.cumsum(has_oot) > 1))[seg_id]

        return seg_id

    @staticmethod
    def _acf_period(cadno, flux, mask, exptime, min_period):
        """
        ACF comb test for quasi-periodic variability in one visit

        Null: Gaussian noise with the visit's smoothed spectrum. The comb C(P)
        is a Hann-tapered cosine over ACF lags [P/2, 3.5P), less the comb of the
        null ACF; S = C / sd(C) (Bartlett). Periodic if max S passes a
        Bonferroni threshold (alpha = 0.01) over the trial periods.

        Parameters
        ----------
        cadno, flux : ndarray
        mask : ndarray, bool
            True near transit (excluded)
        exptime : float
            cadence (days)
        min_period : float
            shortest trial period (days)

        Returns
        -------
        periodic : bool
        period_acf : float
            trial period at max S (days); NaN if no trial period fits the visit
        period_half : float
            2 pi tau_half / 1.4333 (days), clamped to [min_period, span]:
            tau_half is the lag where the ACF (normalized at lag 1, removing
            white noise) falls to 1/2; equals 2 pi / w0 for an SHOTerm with
            Q = 1/sqrt(2)
        """
        # native cadence grid; gaps and masked cadences have weight 0
        oot = ~mask
        k = cadno[oot] - cadno[oot].min()
        n = k.max() + 1
        span = n * exptime
        x, w = np.zeros(n), np.zeros(n)
        x[k] = flux[oot] - np.mean(flux[oot])
        w[k] = 1.0

        # periodogram and gap-aware ACF from one FFT (zero-padded: no wrap-around)
        m = 1 << int(np.ceil(np.log2(2 * n)))
        freq = np.fft.rfftfreq(m, exptime)
        power = np.abs(np.fft.rfft(x, m)) ** 2
        nlag = n // 2 + 1
        lagsum = np.fft.irfft(power, m)[:nlag]  # sum of x_i x_(i+k) over pairs
        pairs = np.round(np.fft.irfft(np.abs(np.fft.rfft(w, m)) ** 2, m)[:nlag])
        acov = lagsum / np.maximum(pairs, 1)
        rho = acov / acov[0]
        pgram = power / lagsum[0]  # normalized: 1 for white noise

        # noise spectrum: median of pgram over [f/2, 3f/2] (centred, so the median
        # of a monotonic spectrum is its value at f) / ln 2 (median -> mean of
        # chi2_2); on a log grid of 300 frequencies, interpolated (f = 0 takes f_1)
        f_grid = np.geomspace(freq[1], freq[-1], 300)
        lo = np.searchsorted(freq, f_grid / 2)
        hi = np.searchsorted(freq, 1.5 * f_grid, side="right")
        smooth = np.array([np.median(pgram[a:b]) for a, b in zip(lo, hi)]) / np.log(2)
        noise_spec = np.exp(
            np.interp(np.log(np.maximum(freq, freq[1])), np.log(f_grid), np.log(smooth))
        )
        rho_null = np.fft.irfft(noise_spec, m)[:nlag]
        rho_null /= rho_null[0]

        # half-decay lag (linear interpolation) -> SHOTerm (Q = 1/sqrt 2) period
        rho_red = rho / rho[1]
        below = np.nonzero(rho_red[1:] < 0.5)[0]
        if len(below):
            j = below[0] + 1
            frac = (rho_red[j - 1] - 0.5) / (rho_red[j - 1] - rho_red[j])
            tau_half = (j - 1 + frac) * exptime
            period_half = np.clip(2 * np.pi * tau_half / 1.4333, min_period, span)
        else:
            period_half = span

        max_period = (nlag - 1) * exptime / 3.5
        if max_period <= min_period:
            logger.info(
                "ACF: no trial period fits the visit; P_half = %.2f d", period_half
            )
            return False, np.nan, period_half
        periods = np.exp(np.arange(np.log(min_period), np.log(max_period), 0.02))

        # rfft bins -> sum over the full frequency band
        band = np.full(len(freq), 2.0)
        band[0] = band[-1] = 1.0
        S = np.zeros(len(periods))
        for i, period in enumerate(periods):
            lag = np.arange(round(period / 2 / exptime), round(3.5 * period / exptime))
            cos = np.cos(2 * np.pi * lag * exptime / period)
            hann = np.sin(np.pi * (lag - lag[0] + 0.5) / len(lag)) ** 2
            weight = hann * cos / np.sum(hann * cos**2)
            comb = np.zeros(m)
            comb[lag] = weight
            # Bartlett: var(C) = mean_f |W(f)|^2 noise_spec(f)^2 / (number of pairs)
            W2 = np.abs(np.fft.rfft(comb)) ** 2
            var = np.sum(band * W2 * noise_spec**2) / m / np.mean(pairs[lag])
            S[i] = np.sum(weight * (rho[lag] - rho_null[lag])) / np.sqrt(var)

        best = np.argmax(S)
        z_crit = norm.isf(0.01 / len(periods))
        periodic = bool(S[best] >= z_crit)
        logger.info(
            "ACF: S = %.2f (threshold %.2f), periodic = %s, P_acf = %.2f d, "
            "P_half = %.2f d",
            S[best],
            z_crit,
            periodic,
            periods[best],
            period_half,
        )
        return periodic, periods[best], period_half

    def _nominal_period(self, lc, masks):
        """
        Star-level kernel choice and nominal GP period (days)

        The star is periodic (RotationTerm) if >= half of its visits pass the ACF
        test, else aperiodic (SHOTerm). The nominal period is the 3-sigma (MAD)
        clipped geometric mean of period_acf over periodic visits, or of
        period_half over all visits for an aperiodic star, floored just above the
        largest _min_period over obsmodes.

        Parameters
        ----------
        lc : LiteCurve
        masks : dict
            visit -> transit mask (True near transit)

        Returns
        -------
        periodic : bool
        nominal_period : float
        """
        visits = np.unique(lc.visit)
        visit_periodic = np.zeros(len(visits), dtype=bool)
        period_acf, period_half = np.zeros(len(visits)), np.zeros(len(visits))
        for j, v in enumerate(visits):
            in_visit = lc.visit == v
            obsmode = lc.obsmode[in_visit][0]
            logger.info("visit %d: ACF test", v)
            visit_periodic[j], period_acf[j], period_half[j] = self._acf_period(
                lc.cadno[in_visit],
                lc.flux[in_visit],
                masks[v],
                EXPTIME[obsmode],
                self._min_period(obsmode),
            )

        periodic = np.mean(visit_periodic) >= 0.5
        log_P = np.log(period_acf[visit_periodic] if periodic else period_half)
        kept = sigma_clip(log_P, sigma=3, stdfunc="mad_std", masked=False)
        # strictly above every visit's floor, so the GP log_P prior is finite
        floor = max(self._min_period(m) for m in np.unique(lc.obsmode))
        nominal_period = max(np.exp(np.mean(kept)), 1.001 * floor)

        logger.info(
            "%s GP: %d of %d visits periodic; nominal period %.2f d "
            "(mean %s of %d visits, %d 3-sigma outliers excluded)",
            "RotationTerm" if periodic else "SHOTerm",
            np.sum(visit_periodic),
            len(visits),
            nominal_period,
            "P_acf" if periodic else "P_half",
            len(log_P),
            len(log_P) - len(kept),
        )

        return periodic, nominal_period

    @staticmethod
    def _fit(model):
        """
        Optimize a _GPTrendModel in stages, ending with all parameters jointly

        Returns the predicted trend (same length as model.time).
        """
        gp_vars = ["log_var", "flux0", "log_sigma", "log_P", "log_Q0"]
        if model.rotation:
            gp_vars += ["log_dQ", "logit_mix"]

        ramp_vars = ["log_var", "flux0", "ramp_amp", "log_tau"]
        stages = [["flux0"], ["flux0", "log_var"]] + [gp_vars, ramp_vars] * 2
        # final joint stage over every parameter, so correlated blocks (e.g.
        # ramp and GP) are not left at a block-coordinate optimum
        stages.append(list(model.ix))

        stage_ix = [np.concatenate([model.ix[n] for n in s]) for s in stages]
        x = optimize(model.logp, model.x0, stage_ix)

        period, q, sigma = model.hyperparameters(model.unpack(x))
        logger.info(
            "%s GP (%d segments): P = %.2f d, %s = %.2f, sigma = %.2e",
            "RotationTerm" if model.rotation else "SHOTerm",
            model.n_seg,
            period,
            "Q0" if model.rotation else "Q",
            q,
            sigma,
        )

        return model.predict(x)

    def _transit_mask(self, time, obsmode, tight_mask):
        """Transit mask (True near transit) for one visit"""
        exptime = EXPTIME[obsmode]
        # at short periods the general masks leave the GP too little data
        usp = np.array([p.period < 3 and p.ror < 0.05 for p in self.planets])

        # tight: as narrow as possible, for the best detrending;
        # wide: cautious, well clear of transits
        if tight_mask:
            # full width = max(T14, 2 hours) + 2 x (exptime + the planet's ttv_buffer)
            mask = self.make_transit_mask(
                rel_size=np.where(usp, 1.5, 1.0),
                abs_size=np.where(usp, 0.0, 2 / 24),
                buffer=[
                    exptime if u else 2 * (exptime + p.transit_times.ttv_buffer())
                    for u, p in zip(usp, self.planets)
                ],
                time=time,
            )
        else:
            # full width = max(3 x T14, 2 x (exptime + 1 hour))
            mask = self.make_transit_mask(
                rel_size=np.where(usp, 2.0, 3.0),
                abs_size=np.where(usp, 0.0, 2 * (exptime + 1 / 24)),
                buffer=np.where(usp, exptime, 0.0),
                time=time,
            )
        return mask

    def _detrend(self, time, flux, cadno, mask, obsmode, periodic, nominal_period):
        """
        Detrend one visit; returns the predicted trend (same length as time)

        Segments (split at gaps and flux jumps) each have their own mean and
        share one GP: RotationTerm if the star is periodic, else SHOTerm.
        """
        exptime = EXPTIME[obsmode]
        gap_tolerance = np.max(
            [
                int(np.min(self.durations) / exptime * 5 / 2),
                self._cadence_floor(obsmode),
            ]
        )
        logger.debug("gap tolerance = %d cadences", gap_tolerance)

        seg_id = self._identify_breaks(
            cadno, flux, mask, gap_tolerance, jump_tolerance=5.0
        )
        # ramp origin per segment, from all cadences, so fit and prediction agree
        seg_t0 = np.array([time[seg_id == i].min() for i in range(seg_id.max() + 1)])

        model = _GPTrendModel(
            time,
            flux,
            ~mask,
            seg_id,
            seg_t0,
            "RotationTerm" if periodic else "SHOTerm",
            nominal_period,
            self._min_period(obsmode),
        )
        return self._fit(model)

    def perform(self, method):
        """
        Detrend the incoming flux of each visit

        Parameters
        ----------
        method : str
            'wide_mask' : mask transits over the larger of 3 x T14 or
                2 x (exptime + 1 hour)
            'tight_mask' : mask transits over max(T14, 2 hours) +
                2 x (exptime + each planet's TransitTimes.ttv_buffer)
            Ultra-short-period planets (P < 3 d, ror < 0.05) use 2 x T14 + exptime
            (wide) and 1.5 x T14 + exptime (tight)

        Returns
        -------
        litecurve : LiteCurve
            flux and error divided by the trend; meta DTKERNEL and DTNOMP give
            the GP kernel and nominal period (days); RMSRAW<v> and RMSDET<v> give
            the flux std of visit v outside the transit mask, before and after
            detrending
        """
        if method not in ("wide_mask", "tight_mask"):
            raise ValueError(
                f"method must be 'wide_mask' or 'tight_mask', got {method!r}"
            )
        tight_mask = method == "tight_mask"

        lc = self.litecurve
        visits = np.unique(lc.visit)
        in_visit = {v: lc.visit == v for v in visits}
        obsmode = {v: lc.obsmode[in_visit[v]][0] for v in visits}

        # transit masks first: the star-level kernel choice needs every visit
        masks = {}
        for v in visits:
            masks[v] = self._transit_mask(lc.time[in_visit[v]], obsmode[v], tight_mask)
            logger.debug(
                "visit %d: %s transit mask covers %d of %d cadences",
                v,
                "tight" if tight_mask else "wide",
                np.sum(masks[v]),
                len(masks[v]),
            )
            if np.mean(masks[v]) > 0.5:
                logger.warning(
                    "visit %d: transit mask covers %.0f%% of cadences",
                    v,
                    100 * np.mean(masks[v]),
                )
        periodic, nominal_period = self._nominal_period(lc, masks)
        mask = np.zeros(len(lc.time), dtype=bool)
        for v in visits:
            mask[in_visit[v]] = masks[v]
        rms_raw = lc.visit_rms(mask)

        for v in visits:
            use = in_visit[v]
            logger.info("visit %d: detrend %d cadences", v, np.sum(use))
            pred = self._detrend(
                lc.time[use],
                lc.flux[use],
                lc.cadno[use],
                masks[v],
                obsmode[v],
                periodic,
                nominal_period,
            )
            lc.flux[use] /= pred
            lc.error[use] /= pred

        lc.meta["DTKERNEL"] = "RotationTerm" if periodic else "SHOTerm"
        lc.meta["DTNOMP"] = float(nominal_period)
        for v, rms in lc.visit_rms(mask).items():
            lc.meta[f"RMSRAW{v}"] = rms_raw[v]
            lc.meta[f"RMSDET{v}"] = rms
        return self.planets, lc
