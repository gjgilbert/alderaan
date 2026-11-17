"""Regularize transit times by modeling observed-minus-calculated (OMC) residuals"""

import logging
import time

import celerite2
import dynesty
import numpy as np
from astropy.stats import mad_std
from astropy.timeseries import LombScargle
from celerite2 import terms as GPterms
from scipy import signal, stats
from scipy.interpolate import make_interp_spline
from scipy.ndimage import median_filter, uniform_filter
from scipy.signal import savgol_filter

from alderaan.algorithms.base import BaseAlg
from alderaan.constants import SEED
from alderaan.data_models import TransitTimes
from alderaan.utils.stats import legendre_p1, norm_ppf, uniform_ppf

logger = logging.getLogger(__name__)


class RegularizeOMC(BaseAlg):
    """
    Replace each planet's transit times with a regularized OMC model

    Parameters
    ----------
    star : Star
    planets : list of Planet
    litecurve : LiteCurve
    """

    @staticmethod
    def _normalized_residuals(res, yerr):
        """Residuals over measurement error combined with excess (unmodeled) scatter"""
        excess_var = max(0.0, mad_std(res) ** 2 - np.median(yerr) ** 2)
        return res / np.sqrt(yerr**2 + excess_var)

    @staticmethod
    def _sigma_clip(yobs, yerr, sigma_cut=5.0):
        """Robust sigma clip about a smoothed trend; True for transits kept"""
        if len(yobs) > 16:
            ysmooth = uniform_filter(median_filter(yobs, size=5, mode="mirror"), size=5)
        else:
            ysmooth = np.median(yobs)

        # zero scatter (e.g. times from a linear ephemeris): nothing to flag
        z = RegularizeOMC._normalized_residuals(yobs - ysmooth, yerr)
        scatter = mad_std(z)
        if len(yobs) > 4 and scatter > 0:
            return np.abs(z) / scatter < sigma_cut
        return np.ones(len(yobs), dtype=bool)

    @staticmethod
    def _smooth(index, yobs, mask):
        """
        Savitzky-Golay smooth of OMC over consecutive epochs; returns values at index

        OMC is first linearly interpolated onto every epoch from index.min() to
        index.max(), so the filter window spans a fixed number of epochs
        """
        epochs = np.arange(index.min(), index.max() + 1)
        npts = len(epochs)

        ysmooth = make_interp_spline(index[mask], yobs[mask], k=1)(epochs)

        if npts >= 8:
            window_length = int(np.log2(npts)) + 1
            window_length += (window_length + 1) % 2
            ysmooth = savgol_filter(ysmooth, window_length, 2)

        return ysmooth[index - index.min()]

    @staticmethod
    def _poly_model(xtime, yobs, yerr, mask, polyorder):
        """
        Fit TTV observed-minus-calculated data with a polynomial

        With Normal(0, 10) priors on the coefficients and a Gaussian likelihood,
        the posterior is Gaussian, so its median is the closed-form MAP

        Parameters
        ----------
        polyorder : int
            polynomial order

        Returns
        -------
        fit : dict
            'pred' at all xtime; 'dof' = polyorder + 1
        """

        # create Vandermonde matrix on xtime mapped to [-1, 1], for conditioning
        x = legendre_p1(xtime, (xtime.min(), xtime.max()))
        X = np.vander(x, N=polyorder + 1, increasing=True)

        w = 1 / yerr[mask] ** 2
        A = X[mask].T @ (w[:, None] * X[mask]) + np.eye(polyorder + 1) / 10**2
        C = np.linalg.solve(A, X[mask].T @ (w * yobs[mask]))

        return {"pred": X @ C, "dof": polyorder + 1}

    @staticmethod
    def _sin_model(xtime, yobs, yerr, mask, ttv_period):
        """
        Fit TTV observed-minus-calculated data with a line plus a single-frequency
        sinusoid; the line absorbs any offset and slope left by the linear
        ephemeris, which is not independent of a sinusoid over a finite baseline

        Priors: f ~ N(1/ttv_period, df) with df = 1/baseline; offset and slope
        ~ N(0, 10) as in _poly_model; sine and cosine amplitudes ~ N(0, 5 std(yobs)).
        At fixed f the model is linear with a Gaussian posterior, solved in closed
        form; f is marginalized on a grid spanning +/- 3 df at df/500 spacing

        Parameters
        ----------
        ttv_period : float
            pre-estimated sinusoid period; the model places tight priors on this period

        Returns
        -------
        fit : dict
            posterior mean 'pred' at all xtime; 'dof' = 5
        """

        df = 1 / (xtime[mask].max() - xtime[mask].min())
        amp_sd = 5 * np.std(yobs[mask])

        # line on xtime mapped to [-1, 1], for conditioning, as in _poly_model
        x = legendre_p1(xtime, (xtime.min(), xtime.max()))
        w = 1 / yerr[mask] ** 2
        prior_prec = np.array([1 / 10**2, 1 / 10**2, 1 / amp_sd**2, 1 / amp_sd**2])

        freqs = 1 / ttv_period + df * np.linspace(-3, 3, 3001)
        freqs = freqs[freqs > 0]

        log_post = np.zeros(len(freqs))
        preds = np.zeros((len(freqs), len(xtime)))
        for i, f in enumerate(freqs):
            phase = 2 * np.pi * f * xtime
            X = np.column_stack([np.ones_like(x), x, np.sin(phase), np.cos(phase)])
            A = X[mask].T @ (w[:, None] * X[mask]) + np.diag(prior_prec)
            b = X[mask].T @ (w * yobs[mask])
            C = np.linalg.solve(A, b)
            # log marginal likelihood over the linear coefficients, up to a
            # constant in f, plus the log prior on f
            log_post[i] = (
                0.5 * b @ C
                - 0.5 * np.linalg.slogdet(A)[1]
                - 0.5 * ((f - 1 / ttv_period) / df) ** 2
            )
            preds[i] = X @ C

        weights = np.exp(log_post - log_post.max())
        weights /= weights.sum()
        if 1 / np.sum(weights**2) < 5:
            logger.warning(
                "sinusoid OMC model: frequency posterior spans < 5 grid points"
            )

        return {"pred": weights @ preds, "dof": 5}

    @staticmethod
    def _matern32_model(xtime, yobs, yerr, mask):
        """
        Fit TTV observed-minus-calculated data with a regularized Matern-3/2 GP

        Returns
        -------
        fit : dict
            posterior median 'pred' at all xtime and
            'dof' = sum(predictive variance / yerr^2) over mask
        """

        # delta between each transit time
        dx = np.mean(np.diff(xtime))

        # maximum GP amplitude
        log_ymax = np.log(np.sqrt(mad_std(yobs) ** 2 - np.median(yerr) ** 2))

        def make_gp(theta):
            log_sigma, rho, mean = theta
            kernel = GPterms.Matern32Term(sigma=np.exp(log_sigma), rho=rho)
            gp = celerite2.GaussianProcess(kernel, mean=mean)
            gp.compute(xtime[mask], yerr=yerr[mask])
            return gp

        # priors: log_sigma ~ N(log_ymax, 5) truncated above at log_ymax;
        # rho ~ U(2 dx, baseline); mean ~ N(mean(yobs), std(yobs))
        def prior_transform(u):
            return np.array(
                [
                    norm_ppf(u[0] / 2, log_ymax, 5.0),
                    uniform_ppf(u[1], 2 * dx, xtime[mask].max() - xtime[mask].min()),
                    norm_ppf(u[2], np.mean(yobs[mask]), np.std(yobs[mask])),
                ]
            )

        def loglike(theta):
            return make_gp(theta).log_likelihood(yobs[mask])

        draws = RegularizeOMC._sample(loglike, prior_transform, 3)

        # medians over (at most) the first 500 equal-weight draws
        pred, dof = [], []
        for d in draws[:500]:
            gp = make_gp(d)
            _, var = gp.predict(yobs[mask], xtime[mask], return_var=True)
            pred.append(gp.predict(yobs[mask], xtime))
            dof.append(np.sum(var / yerr[mask] ** 2))

        return {"pred": np.median(pred, 0), "dof": np.median(dof)}

    @staticmethod
    def _sample(loglike, prior_transform, ndim):
        """
        Seeded dynesty nested sampling; returns equal-weight posterior draws
        """
        t0 = time.perf_counter()
        sampler = dynesty.NestedSampler(
            loglike,
            prior_transform,
            ndim,
            nlive=200,
            sample="auto",
            rstate=np.random.default_rng(SEED),
        )
        sampler.run_nested(dlogz=0.5, print_progress=False)
        results = sampler.results
        logger.info(
            "dynesty ncall = %d, %.1f s",
            np.sum(results.ncall),
            time.perf_counter() - t0,
        )
        return results.samples_equal(rstate=np.random.default_rng(SEED))

    @staticmethod
    def _lomb_scargle_peak(xtime, yobs, mask):
        """
        Highest peak of a Lomb-Scargle periodogram of OMC data and its
        bootstrap false alarm probability
        * assumes that data are nearly evenly sampled
        * optimized for finding marginal periodic TTV signals in OMC data

        Returns
        -------
        freq : float or None
            peak frequency; None for < 8 transits, a peak at the lowest searched
            frequency, or a failed search
        fap : float
            false alarm probability of the peak (1.0 when freq is None)
        """
        npts = np.sum(mask)
        if npts < 8:
            return None, 1.0

        x = xtime[mask]
        y = yobs[mask]

        try:
            # sampling frequency
            fsamp = 1 / np.min(x[1:] - x[:-1])

            # Hann window to reduce ringing
            hann = signal.windows.hann(len(x))
            hann /= np.sum(hann)

            # identify any egregious outliers
            out = np.abs(y - np.median(y)) / mad_std(y - np.median(y)) > 5.0
            xt = x[~out]
            yt = y[~out]

            # frequency grid, shared by the search and its false alarm probability
            grid = dict(
                minimum_frequency=1.5 / (xt.max() - xt.min()),
                maximum_frequency=0.25 * fsamp,
                samples_per_peak=10,
            )

            lombscargle = LombScargle(xt, yt * hann[~out])
            xf, yf = lombscargle.autopower(**grid)

            peak_freq = xf[np.argmax(yf)]
            peak_fap = lombscargle.false_alarm_probability(
                yf.max(),
                method="bootstrap",
                method_kwds={"random_seed": SEED},
                **grid,
            )

        except Exception:
            logger.warning(
                "Lomb-Scargle TTV search failed (%d transits); "
                "assuming no periodic signal",
                npts,
                exc_info=True,
            )
            return None, 1.0

        # a peak at the lowest searched frequency is a rising edge, not a period
        if peak_freq <= xf.min():
            return None, 1.0

        return peak_freq, peak_fap

    @staticmethod
    def _fit_with_rejection(yobs, yerr, good_clip, fit_fn):
        """
        Fit an OMC model on good_clip, reject its mixture outliers, and refit once

        Parameters
        ----------
        good_clip : ndarray of bool
            transits passing the sigma clip
        fit_fn : callable
            fit_fn(mask) -> fit dict with 'pred' at all transits and 'dof'

        Returns
        -------
        result : dict
            'pred' and 'dof' of the final fit; 'good_mix' = good_clip minus
            mixture outliers, the transits of the final fit; 'out_prob';
            'lnlike', the mixture log-likelihood over good_clip
        """
        fit = fit_fn(good_clip)
        out_prob, out_flag, lnlike = RegularizeOMC._calculate_outlier_probability(
            yobs, yerr, fit["pred"], mask=good_clip
        )
        good_mix = good_clip & ~out_flag

        if np.any(good_mix != good_clip):
            fit = fit_fn(good_mix)
            out_prob, _, lnlike = RegularizeOMC._calculate_outlier_probability(
                yobs, yerr, fit["pred"], mask=good_clip
            )

        return {
            "pred": fit["pred"],
            "dof": fit["dof"],
            "good_mix": good_mix,
            "out_prob": out_prob,
            "lnlike": lnlike,
        }

    @staticmethod
    def _select_best_model(results, npts):
        """
        Parameters
        ----------
        results : dict
            _fit_with_rejection results keyed by model name; every 'lnlike' is
            summed over the same (good_clip) transits
        npts : int
            number of transits passing the sigma clip

        Returns
        -------
        best : str
            name (dict key) of selected best model
        """
        # AIC and BIC for each model
        aic = {}
        bic = {}

        for k, r in results.items():
            aic[k] = 2 * r["dof"] - 2 * r["lnlike"]
            bic[k] = r["dof"] * np.log(npts) - 2 * r["lnlike"]
            logger.debug(
                "%s OMC model: dof = %.1f, %d rejected, lnL = %.1f, "
                "AIC = %.1f, BIC = %.1f",
                k,
                r["dof"],
                npts - np.sum(r["good_mix"]),
                r["lnlike"],
                aic[k],
                bic[k],
            )

        aic_best = min(aic, key=aic.get)
        bic_best = min(bic, key=bic.get)

        # Case 1: AIC and BIC agree
        if aic_best == bic_best:
            best = aic_best
            reason = "AIC and BIC agree"

        # Case 2: select the parametric model over the Matern-3/2 GP
        elif "matern32" in (aic_best, bic_best):
            best = bic_best if aic_best == "matern32" else aic_best
            reason = "parametric model preferred over matern32"

        # Case 3: select the model with more degrees of freedom (AIC on ties)
        else:
            best = max((aic_best, bic_best), key=lambda k: results[k]["dof"])
            reason = "more degrees of freedom"

        logger.debug(
            "AIC prefers %s, BIC prefers %s; selecting %s (%s)",
            aic_best,
            bic_best,
            best,
            reason,
        )

        return best

    @staticmethod
    def _calculate_outlier_probability(
        yobs, yerr, ymod, mask=None, maxiter=200, tol=1e-6
    ):
        """
        Calculate the probability that each measured omc transit time is an outlier

        Residuals r = yobs - ymod follow a two-component Gaussian mixture, fit by
        expectation-maximization
            foreground (1 - f) : N(0, yerr**2 + jitter**2)
            background  f      : N(0, yerr**2 + jitter**2 + V)
        jitter is scatter shared by all transits beyond the measurement errors;
        V >= 9 (yerr**2 + jitter**2) keeps the background at least 3x broader.
        A transit is an outlier if it is more likely background than foreground

        Parameters
        ----------
        yobs : ndarray
            observed omc values
        yerr : ndarray
            measurement errors on yobs
        ymod : ndarray
            regularized model for yobs
        mask : ndarray of bool, optional
            transits the mixture is fit to and lnlike is summed over (default all);
            probabilities are returned for every transit

        Returns
        -------
        out_prob : ndarray
            probability that each transit time is an outlier
        out_flag : ndarray
            boolean flag that each transit time is an outlier
        lnlike : float
            mixture log-likelihood over mask
        """
        if mask is None:
            mask = np.ones(len(yobs), dtype=bool)

        res = yobs - ymod
        err2 = yerr**2

        def log_components(f, jit2, V):
            s2 = err2 + jit2
            ln_fg = np.log1p(-f) + stats.norm.logpdf(res, scale=np.sqrt(s2))
            ln_bg = np.log(f) + stats.norm.logpdf(res, scale=np.sqrt(s2 + V))
            return ln_fg, ln_bg

        # fit on mask only
        r2, e2 = res[mask] ** 2, err2[mask]

        # start: foreground within 3 sigma of the normalized residuals
        z = RegularizeOMC._normalized_residuals(res[mask], yerr[mask])
        fg_prob = (np.abs(z) < 3 * mad_std(z)).astype(float)
        f = np.clip(np.mean(1 - fg_prob), 1e-3, 0.5)
        jit2 = 0.0
        V = 100 * np.median(e2)

        for _ in range(maxiter):
            # E-step: foreground probability
            ln_fg, ln_bg = log_components(f, jit2, V)
            fg_new = np.exp(ln_fg - np.logaddexp(ln_fg, ln_bg))[mask]

            # M-step: closed form for equal errors, approximate otherwise
            f = np.clip(np.mean(1 - fg_new), 1e-3, 0.5)
            jit2 = max(0.0, np.sum(fg_new * (r2 - e2)) / np.sum(fg_new))
            s2 = e2 + jit2
            V = max(
                np.sum((1 - fg_new) * (r2 - s2)) / max(np.sum(1 - fg_new), 1e-9),
                9 * np.median(s2),
            )

            converged = np.max(np.abs(fg_new - fg_prob)) < tol
            fg_prob = fg_new
            if converged:
                break
        else:
            logger.warning("outlier mixture did not converge in %d iterations", maxiter)

        # probabilities for every transit at the fitted mixture parameters
        ln_fg, ln_bg = log_components(f, jit2, V)
        ln_tot = np.logaddexp(ln_fg, ln_bg)
        fg_prob = np.exp(ln_fg - ln_tot)

        return 1 - fg_prob, fg_prob < 0.5, np.sum(ln_tot[mask])

    def _identify_periodic_signals(self, xtime, yobs, mask):
        """
        Significant periodic TTV frequency for each planet (None if none):
        FAP < 0.03, or FAP < 0.3 with a matching frequency (within 1/baseline)
        in another planet of a multiplanet system
        """
        npl = len(self.planets)

        peaks = [
            self._lomb_scargle_peak(xtime[n], yobs[n], mask[n]) for n in range(npl)
        ]
        candidate = [f if fap < 0.3 else None for f, fap in peaks]

        close = np.zeros((npl, npl), dtype=bool)
        for i in range(npl):
            df = 1 / (xtime[i].max() - xtime[i].min())
            for j in range(i + 1, npl):
                if candidate[i] is not None and candidate[j] is not None:
                    if np.abs(candidate[i] - candidate[j]) < df:
                        close[i, j] = True
                        close[j, i] = True

        freq = [None] * npl
        for n, (f, fap) in enumerate(peaks):
            if f is not None and (fap < 0.03 or (fap < 0.3 and np.any(close[n]))):
                freq[n] = f
                logger.info(
                    "planet %d: periodic TTV signal at P = %.1f d (FAP = %.3g)",
                    n,
                    1 / f,
                    fap,
                )
                continue

            if npl > 1 and candidate[n] is not None:
                logger.debug(
                    "planet %d: candidate TTV period %.1f d (FAP = %.3g) "
                    "rejected; no matching frequency in other planets",
                    n,
                    1 / f,
                    fap,
                )
            logger.info("planet %d: no significant periodic TTV signal", n)

        return freq

    def _fit_candidates(self, xtime, yobs, yerr, good_clip, freq):
        """
        Fit each candidate OMC model with outlier rejection; the candidate set is
        set by the sigma clip and the Lomb-Scargle frequency (None if no sinusoid)

        Returns
        -------
        results : dict
            _fit_with_rejection results keyed by model name
        """
        npts = np.sum(good_clip)

        results = {}

        # Matern-3/2 model | don't use GP on very noisy data
        if (npts >= 8) & (np.median(yerr) <= 0.5 * mad_std(yobs)):
            results["matern32"] = self._fit_with_rejection(
                yobs,
                yerr,
                good_clip,
                lambda mask: self._matern32_model(xtime, yobs, yerr, mask),
            )
        else:
            logger.debug(
                "matern32 OMC model skipped: %d transits (need 8), median error "
                "%.2e d (need <= %.2e d = 0.5 mad_std(OMC))",
                npts,
                np.median(yerr),
                0.5 * mad_std(yobs),
            )

        # Polynomial model | require 2^N transits, at most do 3rd order
        max_polyorder = np.max([1, np.min([3, int(np.log2(npts)) - 1])])
        for polyorder in range(1, max_polyorder + 1):
            results[f"poly_{polyorder}"] = self._fit_with_rejection(
                yobs,
                yerr,
                good_clip,
                lambda mask, k=polyorder: self._poly_model(xtime, yobs, yerr, mask, k),
            )

        # Sinusoidal model | freq is None for < 8 sigma-clipped transits
        if freq is not None:
            results["sinusoid"] = self._fit_with_rejection(
                yobs,
                yerr,
                good_clip,
                lambda mask: self._sin_model(
                    xtime, yobs, yerr, mask, ttv_period=1 / freq
                ),
            )

        return results

    @staticmethod
    def _regularized_transit_times(tt, measured, ymod, good_mix, out_prob):
        """
        Regularized transit times at every transit of tt: OMC model added to the
        current linear ephemeris, linearly interpolated over transits that are
        unmeasured or not kept by the model fit; measured transits not kept are
        flagged as outliers

        Parameters
        ----------
        tt : TransitTimes
            current transit times; ttime_obs is NaN where unmeasured
        measured : ndarray, bool
            True where tt.ttime_obs is finite
        ymod : ndarray
            OMC model relative to tt.linear_ephemeris(), at measured transits
        good_mix : ndarray
            True for measured transits used in the final model fit; the rest
            are flagged as outliers
        out_prob : ndarray
            outlier probability at measured transits (NaN if not computed)

        Returns
        -------
        transit_times : TransitTimes
            same index, ttime_obs, and error_obs as tt
        """
        ttime_mod = tt.linear_ephemeris()[measured] + ymod

        interpolator = make_interp_spline(
            tt.index[measured][good_mix], ttime_mod[good_mix], k=1
        )

        out_prob_full = np.full(len(tt.index), np.nan)
        out_prob_full[measured] = out_prob
        out_flag_full = np.zeros(len(tt.index), dtype=bool)
        out_flag_full[measured] = ~good_mix

        return TransitTimes(
            tt.mission,
            tt.target,
            tt.index,
            ttime_obs=tt.ttime_obs,
            error_obs=tt.error_obs,
            ttime_mod=interpolator(tt.index),
            out_prob=out_prob_full,
            out_flag=out_flag_full,
            quality=tt.quality,
        )

    def _select(self, xtime, yobs, yerr, good_clip, check_periodic):
        """
        Fit candidate models with outlier rejection, select one per planet, then
        search for a periodic signal once more on the selected model's kept
        transits. The sinusoid candidate is kept only if its period is confirmed
        by the second search; otherwise it is removed, or refit at (or added
        at) the second-search period, and selection reruns

        Returns
        -------
        results : list of dict
            per planet, _fit_with_rejection results keyed by model name
        best : list of str
            per planet, the selected model name
        """
        npl = len(self.planets)

        if check_periodic:
            freq = self._identify_periodic_signals(xtime, yobs, good_clip)
        else:
            freq = [None] * npl

        results = [
            self._fit_candidates(xtime[n], yobs[n], yerr[n], good_clip[n], freq[n])
            for n in range(npl)
        ]
        best = [
            self._select_best_model(results[n], np.sum(good_clip[n]))
            for n in range(npl)
        ]

        if not check_periodic:
            return results, best

        # Lomb-Scargle once more, without the selected model's outliers; a
        # sinusoid is offered only at a period significant after rejection
        good_mix = [results[n][best[n]]["good_mix"] for n in range(npl)]
        freq2 = self._identify_periodic_signals(xtime, yobs, good_mix)

        for n in range(npl):
            df = 1 / (xtime[n].max() - xtime[n].min())
            had_sinusoid = "sinusoid" in results[n]

            if freq2[n] is None:
                if not had_sinusoid:
                    continue
                del results[n]["sinusoid"]
                logger.info(
                    "planet %d: periodic TTV signal at P = %.1f d not significant "
                    "after outlier rejection; sinusoid rejected",
                    n,
                    1 / freq[n],
                )

            elif had_sinusoid and np.abs(freq2[n] - freq[n]) < df:
                logger.info(
                    "planet %d: periodic TTV signal at P = %.1f d confirmed after "
                    "outlier rejection",
                    n,
                    1 / freq[n],
                )
                continue

            else:
                logger.info(
                    "planet %d: sinusoid %s at P = %.1f d after outlier rejection",
                    n,
                    "moved" if had_sinusoid else "added",
                    1 / freq2[n],
                )
                results[n]["sinusoid"] = self._fit_with_rejection(
                    yobs[n],
                    yerr[n],
                    good_clip[n],
                    lambda mask, n=n: self._sin_model(
                        xtime[n], yobs[n], yerr[n], mask, ttv_period=1 / freq2[n]
                    ),
                )

            best[n] = self._select_best_model(results[n], np.sum(good_clip[n]))

        return results, best

    def perform(self, method, check_periodic=True):
        """
        Parameters
        ----------
        method : str
            'smooth' : interpolate + Savitzky-Golay smoothing, filling unmeasured epochs
            'select' : fit candidate models, each rejecting its mixture outliers
                       and refitting once; select by AIC/BIC
        check_periodic : bool
            search for periodic TTV signals before model selection, and again
            after outlier rejection ('select' only)

        Returns
        -------
        planets : list of Planet
        litecurve : LiteCurve

        Each planet carries its regularized TransitTimes, with OMC diagnostics
        ttime_obs, out_prob, out_flag; out_prob is NaN for 'smooth'
        """
        if method not in ("smooth", "select"):
            raise ValueError(f"method must be 'smooth' or 'select', got {method!r}")

        # fit measured transit times only; unmeasured ones are filled by the model
        tts = [p.transit_times for p in self.planets]
        measured = [np.isfinite(tt.ttime_obs) for tt in tts]
        # OMC x-axis: the linear ephemeris (noise-free and sorted), as in legacy
        xtime = [tt.linear_ephemeris()[m] for tt, m in zip(tts, measured)]
        yobs = [
            (tt.ttime_obs - tt.linear_ephemeris())[m] for tt, m in zip(tts, measured)
        ]
        yerr = [tt.error_obs[m] for tt, m in zip(tts, measured)]
        # 'smooth' runs on catalog times whose errors may be missing: equal weights
        if method == "smooth":
            yerr = [np.ones_like(y) for y in yobs]
        good_clip = [self._sigma_clip(y, e) for y, e in zip(yobs, yerr)]

        if method == "select":
            results, best = self._select(xtime, yobs, yerr, good_clip, check_periodic)

        for n, p in enumerate(self.planets):
            old_period = p.period

            if method == "smooth":
                ymod = self._smooth(tts[n].index[measured[n]], yobs[n], good_clip[n])
                good_mix = good_clip[n]
                out_prob = np.full(len(yobs[n]), np.nan)
                model_name = "smooth"

            elif method == "select":
                model_name = best[n]
                r = results[n][model_name]
                ymod, good_mix, out_prob = r["pred"], r["good_mix"], r["out_prob"]

            transit_times = self._regularized_transit_times(
                p.transit_times, measured[n], ymod, good_mix, out_prob
            )
            try:
                p.set_transit_times(transit_times)
            except ValueError as e:
                raise ValueError(f"planet {n}, OMC method {method!r}: {e}") from e
            logger.info(
                "planet %d: %s OMC model; %d measured, %d sigma-clipped, "
                "%d rejected by model; P = %.6f -> %.6f d, ttv_buffer = %.1f min",
                n,
                model_name,
                len(good_clip[n]),
                np.sum(~good_clip[n]),
                np.sum(good_clip[n] & ~good_mix),
                old_period,
                p.period,
                transit_times.ttv_buffer() * 1440,
            )

        return self.planets, self.litecurve
