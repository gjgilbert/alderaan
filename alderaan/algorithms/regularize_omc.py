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
    def _flag_outliers(yobs, yerr, sigma_cut=5.0):
        if len(yobs) > 16:
            ysmooth = uniform_filter(median_filter(yobs, size=5, mode="mirror"), size=5)
        else:
            ysmooth = np.median(yobs)

        # zero scatter (e.g. times from a linear ephemeris): nothing to flag
        z = RegularizeOMC._normalized_residuals(yobs - ysmooth, yerr)
        scatter = mad_std(z)
        if len(yobs) > 4 and scatter > 0:
            quality = np.abs(z) / scatter < sigma_cut
        else:
            quality = np.ones(len(yobs), dtype=bool)

        return quality

    @staticmethod
    def _smooth(index, yobs, quality):
        """
        Savitzky-Golay smooth of OMC over consecutive epochs; returns values at index

        OMC is first linearly interpolated onto every epoch from index.min() to
        index.max(), so the filter window spans a fixed number of epochs
        """
        q = quality
        epochs = np.arange(index.min(), index.max() + 1)
        npts = len(epochs)

        interpolator = make_interp_spline(index[q], yobs[q], k=1)
        ysmooth = interpolator(epochs)

        if npts >= 8:
            window_length = int(np.log2(npts)) + 1
            window_length += (window_length + 1) % 2
            ysmooth = savgol_filter(ysmooth, window_length, 2)

        return ysmooth[index - index.min()]

    @staticmethod
    def _poly_model(xtime, yobs, yerr, quality, polyorder):
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
            'trend' at quality cadences and 'pred' at all xtime
        """
        q = quality

        # create Vandermonde matrices on xtime mapped to [-1, 1], for conditioning
        x = legendre_p1(xtime, (xtime.min(), xtime.max()))
        Xt = np.vander(x[q], N=polyorder + 1, increasing=True)
        Xp = np.vander(x, N=polyorder + 1, increasing=True)

        w = 1 / yerr[q] ** 2
        A = Xt.T @ (w[:, None] * Xt) + np.eye(polyorder + 1) / 10**2
        C = np.linalg.solve(A, Xt.T @ (w * yobs[q]))

        return {"trend": Xt @ C, "pred": Xp @ C}

    @staticmethod
    def _sin_model(xtime, yobs, yerr, quality, ttv_period=None):
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
            posterior mean 'trend' at quality cadences and 'pred' at all xtime
        """
        q = quality

        # nominal TTV period
        if ttv_period is None:
            ttv_period = 2 * (xtime[q].max() - xtime[q].min())

        df = 1 / (xtime[q].max() - xtime[q].min())
        amp_sd = 5 * np.std(yobs[q])

        # line on xtime mapped to [-1, 1], for conditioning, as in _poly_model
        x = legendre_p1(xtime, (xtime.min(), xtime.max()))
        w = 1 / yerr[q] ** 2
        prior_prec = np.array([1 / 10**2, 1 / 10**2, 1 / amp_sd**2, 1 / amp_sd**2])

        def design(f):
            phase = 2 * np.pi * f * xtime
            return np.column_stack([np.ones_like(x), x, np.sin(phase), np.cos(phase)])

        freqs = 1 / ttv_period + df * np.linspace(-3, 3, 3001)
        freqs = freqs[freqs > 0]

        log_post = np.zeros(len(freqs))
        preds = np.zeros((len(freqs), len(xtime)))
        for i, f in enumerate(freqs):
            X = design(f)
            A = X[q].T @ (w[:, None] * X[q]) + np.diag(prior_prec)
            b = X[q].T @ (w * yobs[q])
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
        pred = weights @ preds

        return {"trend": pred[q], "pred": pred}

    @staticmethod
    def _matern32_model(xtime, yobs, yerr, quality):
        """
        Fit TTV observed-minus-calculated data with a regularized Matern-3/2 GP

        Returns
        -------
        fit : dict
            posterior median 'trend' at quality cadences, 'pred' at all xtime,
            and 'dof' = sum(predictive variance / yerr^2)
        """
        q = quality

        # delta between each transit time
        dx = np.mean(np.diff(xtime))

        # maximum GP amplitude
        log_ymax = np.log(np.sqrt(mad_std(yobs) ** 2 - np.median(yerr) ** 2))

        def make_gp(theta):
            log_sigma, rho, mean = theta
            kernel = GPterms.Matern32Term(sigma=np.exp(log_sigma), rho=rho)
            gp = celerite2.GaussianProcess(kernel, mean=mean)
            gp.compute(xtime[q], yerr=yerr[q])
            return gp

        # priors: log_sigma ~ N(log_ymax, 5) truncated above at log_ymax;
        # rho ~ U(2 dx, baseline); mean ~ N(mean(yobs), std(yobs))
        def prior_transform(u):
            return np.array(
                [
                    norm_ppf(u[0] / 2, log_ymax, 5.0),
                    uniform_ppf(u[1], 2 * dx, xtime[q].max() - xtime[q].min()),
                    norm_ppf(u[2], np.mean(yobs[q]), np.std(yobs[q])),
                ]
            )

        def loglike(theta):
            return make_gp(theta).log_likelihood(yobs[q])

        draws = RegularizeOMC._sample(loglike, prior_transform, 3)

        # medians over (at most) the first 500 equal-weight draws
        trend, pred, dof = [], [], []
        for d in draws[:500]:
            gp = make_gp(d)
            mu, var = gp.predict(yobs[q], xtime[q], return_var=True)
            trend.append(mu)
            pred.append(gp.predict(yobs[q], xtime))
            dof.append(np.sum(var / yerr[q] ** 2))

        return {
            "trend": np.median(trend, 0),
            "pred": np.median(pred, 0),
            "dof": np.median(dof),
        }

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
    def _identify_significant_frequencies(xtime, yobs, quality, critical_fap):
        """
        Identify significant periodic frequencies using a Lomb-Scargle periodogram

        Parameters
        ----------
        critical_fap : float
            false alarm probability threshold to consider a frequency significant
        """
        q = quality
        npts = np.sum(quality)

        peakfreq = None
        peakfap = 1.0

        if npts >= 8:
            try:
                xf, yf, freqs, faps = RegularizeOMC._ls_estimator(
                    xtime[q], yobs[q], fap=critical_fap
                )

                if len(freqs) > 0:
                    if freqs[0] > xf.min():
                        peakfreq = freqs[0]
                        peakfap = faps[0]

            except Exception:
                logger.warning(
                    "Lomb-Scargle TTV search failed (%d transits); "
                    "assuming no periodic signal",
                    npts,
                    exc_info=True,
                )

        return peakfreq, peakfap

    @staticmethod
    def _ls_estimator(x, y, fap=0.1, max_peaks=2):
        """
        Generate a Lomb-Scargle periodogram and identify significant frequencies
        * assumes that data are nearly evenly sampled
        * optimized for finding marginal periodic TTV signals in OMC data
        * may not perform well for other applications

        Parameters
        ----------
        x : array-like
            1D array of x data values; should be monotonically increasing
        y : array-like
            1D array of corresponding y data values, len(x)
        fap : float
            false alarm probability threshold to consider a frequency significant
            (default=0.1)

        Returns
        -------
        xf : ndarray
            1D array of frequencies
        yf : ndarray
            1D array of corresponding response
        freqs : list
            signficant frequencies
        faps : list
            corresponding false alarm probabilities
        """
        # sampling frequency
        fsamp = 1 / np.min(x[1:] - x[:-1])

        # Hann window to reduce ringing
        hann = signal.windows.hann(len(x))
        hann /= np.sum(hann)

        # identify any egregious outliers
        out = np.abs(y - np.median(y)) / mad_std(y - np.median(y)) > 5.0
        xt = x[~out]
        yt = y[~out]

        freqs = []
        faps = []

        # frequency grid, shared by the search and its false alarm probability
        grid = dict(
            minimum_frequency=1.5 / (xt.max() - xt.min()),
            maximum_frequency=0.25 * fsamp,
            samples_per_peak=10,
        )

        loop = True
        while loop:
            lombscargle = LombScargle(xt, yt * hann[~out])
            xf, yf = lombscargle.autopower(**grid)

            peak_freq = xf[np.argmax(yf)]
            peak_fap = lombscargle.false_alarm_probability(
                yf.max(),
                method="bootstrap",
                method_kwds={"random_seed": SEED},
                **grid,
            )

            # output first iteration of LS periodogram
            if len(freqs) == 0:
                xf_out = xf
                yf_out = yf

            if peak_fap < fap:
                yt -= lombscargle.model(xt, peak_freq) * len(xt)
                freqs.append(peak_freq)
                faps.append(peak_fap)

            else:
                loop = False

            if len(freqs) >= max_peaks:
                loop = False

        return xf_out, yf_out, freqs, faps

    @staticmethod
    def _fit_with_rejection(fit_fn, yobs, yerr, screen):
        """
        Fit an OMC model on screen, reject its mixture outliers, and refit once

        Parameters
        ----------
        fit_fn : callable
            fit_fn(quality) -> fit dict with 'pred' at all transits
        screen : ndarray of bool
            transits passing the initial outlier screen

        Returns
        -------
        result : dict
            'pred' at all transits; 'keep' = screen minus mixture outliers, the
            transits of the final fit; 'out_prob'; 'lnlike', the mixture
            log-likelihood over screen; 'nchange', screen transits whose flag
            changes after the refit; 'fit', the final fit dict
        """
        fit = fit_fn(screen)
        out_prob, flag, lnlike = RegularizeOMC._calculate_outlier_probability(
            yobs, fit["pred"], yerr, mask=screen
        )
        keep = screen & ~flag

        nchange = 0
        if np.any(keep != screen):
            fit = fit_fn(keep)
            out_prob, flag_refit, lnlike = RegularizeOMC._calculate_outlier_probability(
                yobs, fit["pred"], yerr, mask=screen
            )
            nchange = np.sum(screen & (flag != flag_refit))

        return {
            "pred": fit["pred"],
            "keep": keep,
            "out_prob": out_prob,
            "lnlike": lnlike,
            "nchange": nchange,
            "fit": fit,
        }

    @staticmethod
    def _select_best_model(results, npts):
        """
        Parameters
        ----------
        results : dict
            _fit_with_rejection results with 'dof' added, keyed by model name;
            every 'lnlike' is summed over the same (screen) transits
        npts : int
            number of transits passing the screen

        Returns
        -------
        name : str
            name (dict key) of selected best model
        """
        dofs = {k: r["dof"] for k, r in results.items()}

        # AIC and BIC for each model
        aic = {}
        bic = {}

        for k, r in results.items():
            aic[k] = 2 * dofs[k] - 2 * r["lnlike"]
            bic[k] = dofs[k] * np.log(npts) - 2 * r["lnlike"]
            logger.debug(
                "%s OMC model: dof = %.1f, %d rejected (%d flags change after "
                "refit), lnL = %.1f, AIC = %.1f, BIC = %.1f",
                k,
                dofs[k],
                npts - np.sum(r["keep"]),
                r["nchange"],
                r["lnlike"],
                aic[k],
                bic[k],
            )

        preferred_by_aic = min(aic, key=aic.get)
        preferred_by_bic = min(bic, key=bic.get)

        # Case 1: AIC and BIC match recommendation
        if preferred_by_aic == preferred_by_bic:
            best_omc_model_name = preferred_by_aic
            reason = "AIC and BIC agree"

        # Case 2: Select parametric models over Matern-3/2 GP
        elif np.any(np.array([preferred_by_aic, preferred_by_bic]) == "matern32"):
            reason = "parametric model preferred over matern32"
            if preferred_by_aic == "matern32":
                best_omc_model_name = preferred_by_bic
            elif preferred_by_bic == "matern32":
                best_omc_model_name = preferred_by_aic

        # Case 3: Select model with more degrees of freedom
        else:
            reason = "more degrees of freedom"
            if dofs[preferred_by_aic] >= dofs[preferred_by_bic]:
                best_omc_model_name = preferred_by_aic
            elif dofs[preferred_by_aic] < dofs[preferred_by_bic]:
                best_omc_model_name = preferred_by_bic

        logger.debug(
            "AIC prefers %s, BIC prefers %s; selecting %s (%s)",
            preferred_by_aic,
            preferred_by_bic,
            best_omc_model_name,
            reason,
        )

        return best_omc_model_name

    @staticmethod
    def _calculate_outlier_probability(
        yobs, ymod, yerr, mask=None, maxiter=200, tol=1e-6
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
        ymod : ndarray
            regularized model for yobs
        yerr : ndarray
            measurement errors on yobs
        mask : ndarray of bool, optional
            transits the mixture is fit to and lnlike is summed over (default all);
            probabilities are returned for every transit

        Returns
        -------
        out_prob : ndarray
            probability that each transit time is an outlier
        out_class : ndarray
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
        r, r2, e2 = res[mask], res[mask] ** 2, err2[mask]

        # start: foreground within 3 sigma of the normalized residuals
        z = RegularizeOMC._normalized_residuals(r, yerr[mask])
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

    def _identify_periodic_signals(self, xtime, yobs, quality):
        planets = self.planets
        npl = len(planets)

        if npl == 1:
            critical_fap = 0.03
        elif npl > 1:
            critical_fap = 0.3

        freq = [None] * npl
        fap = [None] * npl

        for n in range(npl):
            freq[n], fap[n] = self._identify_significant_frequencies(
                xtime[n], yobs[n], quality[n], critical_fap
            )

        # single planet systems : use direct search output (FAP < 0.03)
        # multiplanet systems : FAP < 0.03, or FAP < 0.3 with a matching frequency
        # in another planet
        if npl > 1:
            close = np.zeros((npl, npl), dtype=bool)

            for i in range(npl):
                df_min = 1 / (xtime[i].max() - xtime[i].min())

                for j in range(i + 1, npl):
                    if freq[i] is not None and freq[j] is not None:
                        df_ij = np.abs(freq[i] - freq[j])

                        if df_ij < df_min:
                            close[i, j] = True
                            close[j, i] = True

            for i in range(npl):
                if not np.any(close[i]) and fap[i] > 0.03:
                    # freq is None (fap = 1) when no candidate was found
                    if freq[i] is not None:
                        logger.debug(
                            "planet %d: candidate TTV period %.1f d (FAP = %.3g) "
                            "rejected; no matching frequency in other planets",
                            i,
                            1 / freq[i],
                            fap[i],
                        )
                    freq[i] = None
                    fap[i] = None

        for n in range(npl):
            if freq[n] is not None:
                logger.info(
                    "planet %d: periodic TTV signal at P = %.1f d (FAP = %.3g)",
                    n,
                    1 / freq[n],
                    fap[n],
                )
            else:
                logger.info("planet %d: no significant periodic TTV signal", n)

        return freq

    def _fit_candidates(self, xtime, yobs, yerr, quality, freq):
        """
        Fit each candidate OMC model with outlier rejection; the candidate set is
        set by the screen (quality) and the Lomb-Scargle frequency

        Returns
        -------
        results : dict
            _fit_with_rejection results with 'dof', keyed by model name
        """
        npts = np.sum(quality)

        results = {}

        def fit(name, fit_fn):
            results[name] = self._fit_with_rejection(fit_fn, yobs, yerr, quality)

        # Matern-3/2 model | don't use GP on very noisy data
        if (npts >= 8) & (np.median(yerr) <= 0.5 * mad_std(yobs)):
            fit("matern32", lambda q: self._matern32_model(xtime, yobs, yerr, q))
            results["matern32"]["dof"] = results["matern32"]["fit"]["dof"]
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
            fit(
                f"poly_{polyorder}",
                lambda q, k=polyorder: self._poly_model(xtime, yobs, yerr, q, k),
            )
            results[f"poly_{polyorder}"]["dof"] = 1 + polyorder

        # Sinusoidal model
        if npts >= 8 and freq is not None:
            fit(
                "sinusoid",
                lambda q: self._sin_model(xtime, yobs, yerr, q, ttv_period=1 / freq),
            )
            results["sinusoid"]["dof"] = 5

        return results

    @staticmethod
    def _regularized_transit_times(tt, measured, ymod, keep, out_prob, out_flag):
        """
        Regularized transit times at every transit of tt: OMC model added to the
        current linear ephemeris, linearly interpolated over transits that are
        unmeasured or not kept by the model fit

        Parameters
        ----------
        tt : TransitTimes
            current transit times; ttime_obs is NaN where unmeasured
        measured : ndarray, bool
            True where tt.ttime_obs is finite
        ymod : ndarray
            OMC model relative to tt.linear_ephemeris(), at measured transits
        keep : ndarray
            True for measured transits used in the final model fit
        out_prob : ndarray
            outlier probability at measured transits (NaN if not computed)
        out_flag : ndarray
            True for measured transits flagged as outliers

        Returns
        -------
        transit_times : TransitTimes
            same index, ttime_obs, and error_obs as tt
        """
        ttime_mod = tt.linear_ephemeris()[measured] + ymod

        interpolator = make_interp_spline(
            tt.index[measured][keep], ttime_mod[keep], k=1
        )

        out_prob_full = np.full(len(tt.index), np.nan)
        out_prob_full[measured] = out_prob
        out_flag_full = np.zeros(len(tt.index), dtype=bool)
        out_flag_full[measured] = out_flag

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

    def _select(self, xtime, yobs, yerr, quality, check_periodic):
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
            freq = self._identify_periodic_signals(xtime, yobs, quality)
        else:
            freq = [None] * npl

        results = [
            self._fit_candidates(xtime[n], yobs[n], yerr[n], quality[n], freq[n])
            for n in range(npl)
        ]
        best = [
            self._select_best_model(results[n], np.sum(quality[n])) for n in range(npl)
        ]

        if not check_periodic:
            return results, best

        # Lomb-Scargle once more, without the selected model's outliers; a
        # sinusoid is offered only at a period significant after rejection
        keep = [results[n][best[n]]["keep"] for n in range(npl)]
        freq2 = self._identify_periodic_signals(xtime, yobs, keep)

        for n in range(npl):
            df_min = 1 / (xtime[n].max() - xtime[n].min())
            had_sinusoid = "sinusoid" in results[n]

            if freq2[n] is None or np.sum(quality[n]) < 8:
                if not had_sinusoid:
                    continue
                del results[n]["sinusoid"]
                logger.info(
                    "planet %d: periodic TTV signal at P = %.1f d not significant "
                    "after outlier rejection; sinusoid rejected",
                    n,
                    1 / freq[n],
                )

            elif had_sinusoid and np.abs(freq2[n] - freq[n]) < df_min:
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
                    lambda q, n=n: self._sin_model(
                        xtime[n], yobs[n], yerr[n], q, ttv_period=1 / freq2[n]
                    ),
                    yobs[n],
                    yerr[n],
                    quality[n],
                )
                results[n]["sinusoid"]["dof"] = 5

            best[n] = self._select_best_model(results[n], np.sum(quality[n]))

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

        planets = self.planets

        # fit measured transit times only; unmeasured ones are filled by the model
        tts = [p.transit_times for p in planets]
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
        quality = [self._flag_outliers(y, e) for y, e in zip(yobs, yerr)]

        if method == "select":
            results, best = self._select(xtime, yobs, yerr, quality, check_periodic)

        for n, p in enumerate(planets):
            old_period = p.period

            if method == "smooth":
                ymod = self._smooth(tts[n].index[measured[n]], yobs[n], quality[n])
                keep = quality[n]
                out_prob = np.full(len(yobs[n]), np.nan)
                model_name = "smooth"

            elif method == "select":
                model_name = best[n]
                r = results[n][model_name]
                ymod, keep, out_prob = r["pred"], r["keep"], r["out_prob"]

            transit_times = self._regularized_transit_times(
                p.transit_times, measured[n], ymod, keep, out_prob, ~keep
            )
            try:
                p.set_transit_times(transit_times)
            except ValueError as e:
                raise ValueError(f"planet {n}, OMC method {method!r}: {e}") from e
            logger.info(
                "planet %d: %s OMC model; %d measured, %d fail initial screen, "
                "%d rejected by model; P = %.6f -> %.6f d, ttv_buffer = %.1f min",
                n,
                model_name,
                len(quality[n]),
                np.sum(~quality[n]),
                np.sum(quality[n] & ~keep),
                old_period,
                planets[n].period,
                transit_times.ttv_buffer() * 1440,
            )

        return self.planets, self.litecurve
