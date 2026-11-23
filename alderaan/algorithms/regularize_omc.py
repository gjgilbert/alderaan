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
from alderaan.errors import UnsupportedTargetError
from alderaan.transit_model import TransitModel, transit_flux
from alderaan.utils.stats import legendre_p1, norm_ppf, uniform_ppf

logger = logging.getLogger(__name__)


class RegularizeOMC(BaseAlg):
    """
    Replace each planet's transit times with a regularized OMC model

    Every per-planet array spans all epochs of the planet's TransitTimes:
    tlin is the linear ephemeris; omc and err are NaN where unmeasured. Boolean
    masks select transits: measured (good photometry, finite time), unclipped
    (measured and passing the sigma clip), inlier (kept by the model fit)

    Parameters
    ----------
    star : Star
    planets : list of Planet
    litecurve : LiteCurve
    """

    # --- sigma clip and smoothing -------------------------------------------

    @staticmethod
    def _normalized_residuals(res, yerr):
        """Residuals over measurement error combined with excess (unmodeled) scatter"""
        excess_var = max(0.0, mad_std(res) ** 2 - np.median(yerr) ** 2)
        return res / np.sqrt(yerr**2 + excess_var)

    @staticmethod
    def _sigma_clip(yobs, yerr, sigma_cut=5.0):
        """
        Robust sigma clip about a smoothed trend of consecutive measured times;
        True for transits kept. With < 16 transits no trend can be estimated
        and nothing is clipped (a clip about the median removes real TTVs)
        """
        keep = np.ones(len(yobs), dtype=bool)
        if len(yobs) < 16:
            return keep

        ysmooth = uniform_filter(median_filter(yobs, size=5, mode="mirror"), size=5)

        # zero scatter (e.g. times from a linear ephemeris): nothing to flag
        z = RegularizeOMC._normalized_residuals(yobs - ysmooth, yerr)
        scatter = mad_std(z)
        if scatter > 0:
            return np.abs(z) / scatter < sigma_cut
        return keep

    @staticmethod
    def _smooth(index, omc, mask):
        """
        Savitzky-Golay smooth of OMC over consecutive epochs; returns values at
        every index

        OMC on mask is first linearly interpolated onto every epoch it spans, so
        the filter window spans a fixed number of epochs
        """
        epochs = np.arange(index[mask].min(), index[mask].max() + 1)
        npts = len(epochs)

        ysmooth = make_interp_spline(index[mask], omc[mask], k=1)(epochs)

        if npts >= 8:
            window_length = int(np.log2(npts)) + 1
            window_length += (window_length + 1) % 2
            ysmooth = savgol_filter(ysmooth, window_length, 2)

        return make_interp_spline(epochs, ysmooth, k=1)(index)

    # --- regression models --------------------------------------------------

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
            'pred' at all xtime; 'dof' = polyorder + 1; 'leverage', the diagonal
            of the smoother matrix on mask (NaN elsewhere)
        """

        # xtime mapped to [-1, 1] for conditioning
        x = legendre_p1(xtime, (xtime.min(), xtime.max()))
        X = np.vander(x, N=polyorder + 1, increasing=True)

        w = 1 / yerr[mask] ** 2
        A = X[mask].T @ (w[:, None] * X[mask]) + np.eye(polyorder + 1) / 10**2
        C = np.linalg.solve(A, X[mask].T @ (w * yobs[mask]))

        leverage = np.full(len(xtime), np.nan)
        leverage[mask] = w * np.sum(X[mask] * np.linalg.solve(A, X[mask].T).T, axis=1)

        return {"pred": X @ C, "dof": polyorder + 1, "leverage": leverage}

    @staticmethod
    def _lasso_poly_model(xtime, yobs, yerr, mask, polyorder):
        """
        Fit TTV observed-minus-calculated data with a lasso polynomial: an L1
        penalty on every coefficient on top of _poly_model's Normal(0, 10) prior.
        The penalty is chosen to minimize BIC, with dof = number of nonzero coefficients
        (Zou et al. 2007)

        Returns
        -------
        fit : dict
            'pred' at all xtime; 'dof'
        """
        # Legendre basis: orthogonal columns keep coordinate descent fast
        x = legendre_p1(xtime, (xtime.min(), xtime.max()))
        X = np.polynomial.legendre.legvander(x, polyorder)
        sw = 1 / yerr[mask]
        A, y = X[mask] * sw[:, None], yobs[mask] * sw
        G = A.T @ A + np.eye(polyorder + 1) / 10**2
        c = A.T @ y

        best = None
        coef = np.zeros(polyorder + 1)
        # warm-started path down from the penalty that zeroes every coefficient
        for lam in np.max(np.abs(c)) * np.logspace(0, -6, 60):
            for _ in range(20000):
                step = 0.0
                for j in range(polyorder + 1):
                    rho = c[j] - G[j] @ coef + G[j, j] * coef[j]
                    new = np.sign(rho) * max(abs(rho) - lam, 0) / G[j, j]
                    step = max(step, abs(new - coef[j]))
                    coef[j] = new
                if step < 1e-10 * (1 + np.max(np.abs(coef))):
                    break
            dof = int(np.sum(coef != 0))
            bic = np.sum((y - A @ coef) ** 2) + dof * np.log(np.sum(mask))
            if best is None or bic < best[0]:
                best = (bic, X @ coef, dof)

        return {"pred": best[1], "dof": best[2]}

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
            pre-estimated TTV period; centers the frequency prior

        Returns
        -------
        fit : dict
            posterior mean 'pred' at all xtime; 'dof' = 5; 'leverage', the
            posterior mean diagonal of the smoother matrix on mask (NaN elsewhere)
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
        levs = np.zeros((len(freqs), np.sum(mask)))
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
            levs[i] = w * np.sum(X[mask] * np.linalg.solve(A, X[mask].T).T, axis=1)

        weights = np.exp(log_post - log_post.max())
        weights /= weights.sum()
        n_eff = 1 / np.sum(weights**2)
        if n_eff < 5:
            logger.warning(
                "sinusoid OMC model: frequency grid under-resolves the posterior "
                "(%.1f effective grid points, need >= 5)",
                n_eff,
            )

        leverage = np.full(len(xtime), np.nan)
        leverage[mask] = weights @ levs

        return {"pred": weights @ preds, "dof": 5, "leverage": leverage}

    @staticmethod
    def _matern32_model(xtime, yobs, yerr, mask):
        """
        Fit TTV observed-minus-calculated data with a regularized Matern-3/2 GP

        Transits with NaN yobs are unmeasured: they get predictions but do not
        enter the priors

        Returns
        -------
        fit : dict
            posterior median 'pred' at all xtime;
            'dof' = sum(predictive variance / yerr^2) over mask; 'leverage', the
            posterior median of predictive variance / yerr^2 (the diagonal of the
            smoother matrix) on mask (NaN elsewhere)
        """
        measured = np.isfinite(yobs)

        dx = np.mean(np.diff(xtime[measured]))

        # maximum GP amplitude: the scatter in excess of the errors; with < 16
        # transits that excess is poorly determined, so the scatter itself
        if np.sum(measured) < 16:
            log_ymax = np.log(np.std(yobs[mask]))
        else:
            log_ymax = np.log(
                np.sqrt(mad_std(yobs[measured]) ** 2 - np.median(yerr[measured]) ** 2)
            )

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

        logger.info("matern32 OMC model: sample GP hyperparameters")
        draws = RegularizeOMC._sample(loglike, prior_transform, 3)

        # medians over (at most) the first 500 equal-weight draws
        pred, lev = [], []
        for d in draws[:500]:
            gp = make_gp(d)
            _, var = gp.predict(yobs[mask], xtime[mask], return_var=True)
            pred.append(gp.predict(yobs[mask], xtime))
            lev.append(var / yerr[mask] ** 2)

        leverage = np.full(len(xtime), np.nan)
        leverage[mask] = np.median(lev, 0)

        return {
            "pred": np.median(pred, 0),
            "dof": np.median(np.sum(lev, 1)),
            "leverage": leverage,
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
            "dynesty: ncall = %d, %.1f s",
            np.sum(results.ncall),
            time.perf_counter() - t0,
        )
        return results.samples_equal(rstate=np.random.default_rng(SEED))

    @staticmethod
    def _fit(model, tlin, omc, err, mask, period=None):
        """
        Fit OMC model 'poly_<k>', 'sinusoid' (at TTV period) or 'matern32' to
        the transits in mask

        Returns
        -------
        fit : dict
            'pred' at every transit, 'dof', 'leverage' (NaN off mask)
        """
        if model == "matern32":
            return RegularizeOMC._matern32_model(tlin, omc, err, mask)
        if model == "sinusoid":
            return RegularizeOMC._sin_model(tlin, omc, err, mask, period)
        polyorder = int(model.removeprefix("poly_"))
        return RegularizeOMC._poly_model(tlin, omc, err, mask, polyorder)

    # --- outlier rejection --------------------------------------------------

    @staticmethod
    def _robust_fit(model, tlin, omc, err, unclipped, period=None):
        """
        Fit an OMC model on unclipped transits and reject its outliers, by the
        number of transits N:
          * N < 8: nothing rejected
          * 8 <= N < 16: leave-one-out test; the worst studentized residual is
            flagged while it fails a Bonferroni-corrected alpha = 0.01 test, with
            a refit after each flag. A flagged transit enters lnlike as a
            z_crit residual, so each flag costs z_crit^2 in AIC/BIC
          * N >= 16: mixture outliers rejected, then one refit

        Returns
        -------
        fit : dict
            _fit result of the final fit, plus 'inlier' (unclipped minus
            outliers, the transits of the final fit) and 'lnlike', the
            log-likelihood over unclipped (Gaussian for N < 16, mixture for
            N >= 16)
        """
        npts = np.sum(unclipped)
        fit = RegularizeOMC._fit(model, tlin, omc, err, unclipped, period)
        inlier = unclipped.copy()

        z_crit = stats.norm.isf(0.01 / (2 * npts))
        if 8 <= npts < 16:
            # keep a majority of transits, and > dof + 1 so residuals stay defined
            while np.sum(inlier) - 1 > max(npts / 2, fit["dof"] + 1):
                z = np.abs(omc - fit["pred"]) / (err * np.sqrt(1 - fit["leverage"]))
                worst = np.nanargmax(np.where(inlier, z, np.nan))
                if z[worst] < z_crit:
                    break
                inlier[worst] = False
                fit = RegularizeOMC._fit(model, tlin, omc, err, inlier, period)

        if npts < 16:
            # a flagged transit counts as a z_crit residual: each flag costs the
            # z_crit^2 that justified it, so flags cannot buy a poor model a fit
            res = np.where(inlier, omc - fit["pred"], z_crit * err)
            lnlike = np.sum(stats.norm.logpdf(res, scale=err)[unclipped])
        else:
            out_flag, lnlike = RegularizeOMC._calculate_outlier_probability(
                omc, err, fit["pred"], mask=unclipped
            )
            inlier = unclipped & ~out_flag
            # outliers are a minority, as the mixture assumes (f <= 0.5)
            if np.sum(inlier) <= npts / 2:
                inlier = unclipped.copy()
            if np.any(inlier != unclipped):
                fit = RegularizeOMC._fit(model, tlin, omc, err, inlier, period)
                _, lnlike = RegularizeOMC._calculate_outlier_probability(
                    omc, err, fit["pred"], mask=unclipped
                )

        return {**fit, "inlier": inlier, "lnlike": lnlike}

    @staticmethod
    def _calculate_outlier_probability(
        yobs, yerr, ymod, mask=None, maxiter=200, tol=1e-6
    ):
        """
        Flag measured omc transit times that are more likely outliers than not

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
            ln_fg, ln_bg = ln_fg[mask], ln_bg[mask]
            fg_new = np.exp(ln_fg - np.logaddexp(ln_fg, ln_bg))

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
            logger.warning(
                "outlier mixture did not converge in %d iterations; last iterate used",
                maxiter,
            )

        # probabilities for every transit at the fitted mixture parameters
        ln_fg, ln_bg = log_components(f, jit2, V)
        # Suppress NaN-input warnings: unmeasured transits (outside mask) have
        # NaN residuals by design and get NaN probabilities. A NaN inside mask
        # still warns in the E-step above, which has no suppression.
        with np.errstate(invalid="ignore"):
            ln_tot = np.logaddexp(ln_fg, ln_bg)
        fg_prob = np.exp(ln_fg - ln_tot)

        return fg_prob < 0.5, np.sum(ln_tot[mask])

    # --- periodic signals ---------------------------------------------------

    @staticmethod
    def _lomb_scargle_peak(xtime, yobs, mask):
        """
        Highest peak of a Lomb-Scargle periodogram of OMC data and its
        bootstrap false alarm probability; assumes nearly even sampling

        Returns
        -------
        freq : float or None
            peak frequency; None for < 8 transits, a peak at the lowest searched
            frequency, or a baseline (after the outlier cut) too short to search
        fap : float
            false alarm probability of the peak (1.0 when freq is None)
        """
        npts = np.sum(mask)
        if npts < 8:
            return None, 1.0

        x = xtime[mask]
        y = yobs[mask]

        fsamp = 1 / np.min(x[1:] - x[:-1])

        # Hann window to reduce ringing
        hann = signal.windows.hann(len(x))
        hann /= np.sum(hann)

        # egregious outliers excluded from the search
        out = np.abs(y - np.median(y)) / mad_std(y - np.median(y)) > 5.0
        xt = x[~out]
        yt = y[~out]

        # frequency grid, shared by the search and its false alarm probability
        grid = dict(
            minimum_frequency=1.5 / (xt.max() - xt.min()),
            maximum_frequency=0.25 * fsamp,
            samples_per_peak=10,
        )
        # outliers cut from the ends can leave too short a baseline for any period
        if grid["minimum_frequency"] >= grid["maximum_frequency"]:
            return None, 1.0

        lombscargle = LombScargle(xt, yt * hann[~out])
        xf, yf = lombscargle.autopower(**grid)

        peak_freq = xf[np.argmax(yf)]
        peak_fap = lombscargle.false_alarm_probability(
            yf.max(),
            method="bootstrap",
            method_kwds={"random_seed": SEED},
            **grid,
        )

        # a peak at the lowest searched frequency is a rising edge, not a period
        if peak_freq <= xf.min():
            return None, 1.0

        return peak_freq, peak_fap

    def _periodic_signals(self, tlin, omc, mask):
        """
        Significant periodic TTV period for each planet (None if none):
        FAP < 0.03, or FAP < 0.3 with a matching frequency (within 1/baseline)
        in another planet of a multiplanet system
        """
        npl = len(self.planets)

        peaks = [self._lomb_scargle_peak(tlin[n], omc[n], mask[n]) for n in range(npl)]
        candidate = [f if fap < 0.3 else None for f, fap in peaks]

        close = np.zeros((npl, npl), dtype=bool)
        for i in range(npl):
            df = 1 / np.ptp(tlin[i][mask[i]])
            for j in range(i + 1, npl):
                if candidate[i] is not None and candidate[j] is not None:
                    if np.abs(candidate[i] - candidate[j]) < df:
                        close[i, j] = True
                        close[j, i] = True

        period = [None] * npl
        for n, (f, fap) in enumerate(peaks):
            if f is not None and (fap < 0.03 or (fap < 0.3 and np.any(close[n]))):
                period[n] = 1 / f
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

        return period

    # --- model selection ----------------------------------------------------

    @staticmethod
    def _candidates(omc, err, measured, unclipped, period, max_polyorder="auto"):
        """
        Names of the OMC models offered: poly_1 up to poly_3 (2^(k+1) transits
        for order k, or up to max_polyorder if an int); sinusoid if a TTV period
        was found; matern32 with >= 16 transits and errors <= 0.5 mad_std(OMC)
        """
        npts = np.sum(unclipped)
        models = []

        # with < 16 transits the GP nearly interpolates the measured times
        # (dof ~ N), leaving the chi2 test in _vet without power; on very noisy
        # data the GP has nothing to model
        noise = np.median(err[measured])
        if npts >= 16 and noise <= 0.5 * mad_std(omc[measured]):
            models.append("matern32")
        else:
            logger.debug(
                "matern32 OMC model skipped: %d transits (need 16), median error "
                "%.2e d (need <= %.2e d = 0.5 mad_std(OMC))",
                npts,
                noise,
                0.5 * mad_std(omc[measured]),
            )

        if max_polyorder == "auto":
            max_polyorder = np.max([1, np.min([3, int(np.log2(npts)) - 1])])
        models += [f"poly_{k}" for k in range(1, max_polyorder + 1)]

        if period is not None:
            models.append("sinusoid")

        return models

    @staticmethod
    def _compare(fits, npts):
        """
        Select a model by AIC and BIC

        Parameters
        ----------
        fits : dict
            _robust_fit results keyed by model name; every 'lnlike' is summed
            over the same (unclipped) transits
        npts : int
            number of unclipped transits

        Returns
        -------
        model : str
        """
        aic = {}
        bic = {}

        for k, r in fits.items():
            aic[k] = 2 * r["dof"] - 2 * r["lnlike"]
            bic[k] = r["dof"] * np.log(npts) - 2 * r["lnlike"]
            logger.debug(
                "%s OMC model: dof = %.1f, %d rejected, lnL = %.1f, "
                "AIC = %.1f, BIC = %.1f",
                k,
                r["dof"],
                npts - np.sum(r["inlier"]),
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
            best = max((aic_best, bic_best), key=lambda k: fits[k]["dof"])
            reason = "more degrees of freedom"

        logger.debug(
            "AIC prefers %s, BIC prefers %s; selecting %s (%s)",
            aic_best,
            bic_best,
            best,
            reason,
        )

        return best

    def _recheck_sinusoid(self, tlin, omc, err, unclipped, fits, period, period2):
        """
        Keep the sinusoid candidate only if a second search, without
        the selected model's outliers, confirms its period; otherwise remove it,
        or refit (or add) it at the second-search period. Updates fits in place

        Returns
        -------
        period : float or None
            sinusoid period after the recheck
        """
        if period2 is None:
            if period is not None:
                del fits["sinusoid"]
                logger.info(
                    "periodic TTV signal at P = %.1f d not significant "
                    "after outlier rejection; sinusoid rejected",
                    period,
                )
            return None

        df = 1 / np.ptp(tlin[unclipped])
        if period is not None and np.abs(1 / period2 - 1 / period) < df:
            logger.info(
                "periodic TTV signal at P = %.1f d confirmed after outlier rejection",
                period,
            )
            return period

        logger.info(
            "sinusoid %s at P = %.1f d after outlier rejection",
            "added" if period is None else "moved",
            period2,
        )
        fits["sinusoid"] = self._robust_fit(
            "sinusoid", tlin, omc, err, unclipped, period2
        )
        return period2

    # --- vetting the selected model -----------------------------------------

    @staticmethod
    def _chi2_pvalue(omc, err, pred, mask, dof):
        """Chi-squared p-value of the times in mask against the model prediction"""
        chi2 = np.sum(((omc - pred)[mask] / err[mask]) ** 2)
        return stats.chi2.sf(chi2, max(np.sum(mask) - dof, 1))

    def _transit_dchi2(self, n, t_obs, t_pred, exptime_offsets):
        """
        Light-curve preference for a transit of planet n at t_obs over t_pred

        In a window spanning both times, with the other planets' transits
        subtracted, the planet's transit template is fit at each time with a
        free depth scale and with the expected depth

        Returns
        -------
        dchi2_free, dchi2_fixed : float
            chi2(t_pred) - chi2(t_obs) with free and with expected depth,
            divided by the reduced chi2 at t_obs if > 1, floored at 0
        """
        lc = self.litecurve
        center = 0.5 * (t_obs + t_pred)
        half_width = 1.5 * self.planets[n].duration + 0.5 * np.abs(t_obs - t_pred)
        use = np.abs(lc.time - center) < half_width
        time, error = lc.time[use], lc.error[use]
        obsmode = np.asarray(lc.obsmode)[use]

        def dip(p, tc):
            out = np.ones(len(time))
            for om in np.unique(obsmode):
                k = obsmode == om
                out[k] = transit_flux(
                    time[k],
                    tc,
                    p.period,
                    p.ror,
                    p.impact,
                    p.duration,
                    self.star.u1,
                    self.star.u2,
                    exptime_offsets[om],
                )
            return out - 1

        flux = lc.flux[use] - 1
        for q, p in enumerate(self.planets):
            if q == n:
                continue
            tc = p.transit_times.ttime_mod
            for t in tc[np.abs(tc - center) < half_width + p.duration]:
                flux = flux - dip(p, t)

        w = 1 / error**2
        chi2_free, chi2_fixed = [], []
        for t in (t_obs, t_pred):
            template = dip(self.planets[n], t)
            scale = np.sum(w * template * flux) / np.sum(w * template**2)
            chi2_free.append(np.sum(w * (flux - scale * template) ** 2))
            chi2_fixed.append(np.sum(w * (flux - template) ** 2))

        # noise in excess of the photometric errors inflates dchi2; with the
        # depth fixed, a dip still present at t_pred (shift < duration) or
        # missing there is judged against the expected depth
        rchi2 = max(chi2_free[0] / max(len(flux) - 1, 1), 1)
        return (
            max((chi2_free[1] - chi2_free[0]) / rchi2, 0),
            max((chi2_fixed[1] - chi2_fixed[0]) / rchi2, 0),
        )

    def _photometric_vet(
        self, n, tlin, omc, err, measured, model, period, dof, exptime_offsets
    ):
        """
        Test the measured times of planet n, worst leave-one-out residual first,
        against its selected model. A time is supported if the light curve
        prefers it over the model prediction by dchi2 >= 9 with free and with
        expected depth; a supported time ends the test. An unsupported time is
        flagged and the model refit; the model is restored if the chi-squared
        test of the remaining times then passes

        Returns
        -------
        fit : dict or None
            _fit result of the restored model, plus 'inlier' (the transits it
            keeps); None if the measured times stand
        """
        npts = np.sum(measured)
        inlier = measured.copy()
        # majority guard, as in _robust_fit
        while np.sum(inlier) - 1 > max(npts / 2, dof + 1):
            z = np.full(len(omc), -np.inf)
            pred = np.full(len(omc), np.nan)
            for i in np.flatnonzero(inlier):
                loo = inlier.copy()
                loo[i] = False
                pred[i] = self._fit(model, tlin, omc, err, loo, period)["pred"][i]
                z[i] = np.abs(omc[i] - pred[i]) / err[i]

            i = int(np.argmax(z))
            if z[i] < 4:
                break

            dchi2_free, dchi2_fixed = self._transit_dchi2(
                n, tlin[i] + omc[i], tlin[i] + pred[i], exptime_offsets
            )
            # 3 sigma photometric evidence for the measured time
            supported = dchi2_free >= 9 and dchi2_fixed >= 9
            logger.info(
                "planet %d: transit at %.3f, |z| = %.1f, dt = %+.0f min; "
                "photometric dchi2 = %.1f (free depth), %.1f (expected depth); %s",
                n,
                tlin[i],
                z[i],
                (omc[i] - pred[i]) * 1440,
                dchi2_free,
                dchi2_fixed,
                "supported" if supported else "flagged",
            )
            if supported:
                break

            inlier[i] = False
            fit = self._fit(model, tlin, omc, err, inlier, period)
            pval = self._chi2_pvalue(omc, err, fit["pred"], inlier, dof)
            if pval >= 0.01:
                logger.info(
                    "planet %d: %s OMC model chi2 test p = %.2g without flagged "
                    "transits; model kept",
                    n,
                    model,
                    pval,
                )
                return {**fit, "inlier": inlier}

        return None

    @staticmethod
    def _wild_score(tlin, omc, err, mask, model, k):
        """
        Residual of transit k against the model fit to mask and k, in units of
        the largest of its error, the robust (MAD) scatter of the other
        transits, and their median error

          * the residual is studentized by leverage (Cook & Weisberg 1982),
            undoing the fit's pull toward transit k without the extrapolation
            of a leave-one-out fit
          * a wild transit must stand out from its error and from the others:
            scatter shared by all transits (TTVs, underestimated errors) is not
            wild; the median error floors a scatter below the errors
          * the score never exceeds the studentized residual |u_k| / err_k, so
            a Bonferroni bound on that residual bounds the score too
        """
        fit_mask = mask.copy()
        fit_mask[k] = True
        fit = RegularizeOMC._fit(model, tlin, omc, err, fit_mask)
        u = (omc - fit["pred"]) / np.sqrt(1 - fit["leverage"])
        others = fit_mask.copy()
        others[k] = False
        scale = max(err[k], mad_std(u[others]), np.median(err[others]))
        return np.abs(u[k]) / scale

    def _vet(self, n, tlin, omc, err, measured, unclipped, fits, model, period):
        """
        Keep the selected model of planet n, or adopt its measured times
          * matern32 is dropped for the next best model if its inliers scatter
            less than their errors: a GP has nothing to model beyond a trend
          * N < 16: measured times adopted (outlier flags cleared) if they fail a
            chi-squared test (p < 0.001) against the model, unless (N >= 8) the
            light curve rejects the times driving the failure; the model is then
            kept with those times flagged. Failing that, polynomials up to
            poly_3 are refit; the best is kept if every time it rejects is wild
            (_wild_score) and not photometrically trusted, and its inliers pass
            the chi-squared test
          * N >= 16: a matern32 model with dof/N > 0.5 mostly reproduces the
            measured times; they are adopted (mixture outlier flags kept)

        Returns
        -------
        label : str
            selected model, or 'independent' if the measured times are adopted
        fit : dict
            final fit: 'pred' at every transit (a matern32 GP if independent),
            'inlier' (transits kept; measured transits off inlier are
            outliers)
        """
        npts = np.sum(measured)

        if model == "matern32":
            inlier = fits[model]["inlier"]
            if mad_std(omc[inlier]) < np.median(err[inlier]):
                model = self._compare(
                    {k: v for k, v in fits.items() if k != "matern32"},
                    np.sum(unclipped),
                )
                logger.info(
                    "planet %d: matern32 OMC model scatter below measurement "
                    "errors; %s selected",
                    n,
                    model,
                )

        fit = fits[model]

        if npts >= 16:
            if model == "matern32" and fit["dof"] / np.sum(fit["inlier"]) > 0.5:
                logger.info(
                    "planet %d: matern32 OMC model dof/N = %.2f > 0.5; "
                    "measured times adopted",
                    n,
                    fit["dof"] / np.sum(fit["inlier"]),
                )
                return "independent", fit
            return model, fit

        pval = self._chi2_pvalue(omc, err, fit["pred"], measured, fit["dof"])
        logger.info(
            "planet %d: %s OMC model chi2 test p = %.2g; %s",
            n,
            model,
            pval,
            "measured times preferred" if pval < 0.001 else "model kept",
        )
        # alpha = 0.001: with few transits, errors underestimated by ~1.5x alone
        # fail alpha = 0.01, and adopting the measured times then fits noise
        if pval >= 0.001:
            return model, fit

        if npts >= 8:
            exptime_offsets = TransitModel(
                self.star, self.planets, self.litecurve
            ).exptime_offsets
            restored = self._photometric_vet(
                n,
                tlin,
                omc,
                err,
                measured,
                model,
                period,
                fit["dof"],
                exptime_offsets,
            )
            if restored is not None:
                return model, restored

            # polynomial refit: poly_3 at any N, as the wild and chi2 gates,
            # not the N-based order limit, guard against overfitting
            refits = {
                k: self._robust_fit(k, tlin, omc, err, measured)
                for k in self._candidates(
                    omc, err, measured, measured, None, max_polyorder=3
                )
            }
            refit_model = self._compare(refits, npts)
            refit = refits[refit_model]
            # Bonferroni-corrected (Dunn 1961) family-wise false-alarm
            # probability of a two-sided 3 sigma event: rejecting measured
            # times takes stronger evidence than _robust_fit's alpha = 0.01
            z_crit = stats.norm.isf(0.0027 / (2 * npts))
            wild = False
            for k in np.flatnonzero(measured & ~refit["inlier"]):
                score = self._wild_score(
                    tlin, omc, err, refit["inlier"], refit_model, k
                )
                dchi2 = self._transit_dchi2(
                    n, tlin[k] + omc[k], tlin[k] + refit["pred"][k], exptime_offsets
                )
                # dchi2 >= 50 (~7 sigma) is beyond the noise dips a timing
                # search over many trial positions finds (look-elsewhere
                # effect; Gross & Vitells 2010): such a time is real
                wild = score >= z_crit and min(dchi2) < 50
                logger.info(
                    "planet %d: %s rejects transit at %.3f, wild score %.1f "
                    "(need >= %.1f), photometric dchi2 = %.0f (free depth), "
                    "%.0f (expected depth) (need < 50); %s",
                    n,
                    refit_model,
                    tlin[k],
                    score,
                    z_crit,
                    *dchi2,
                    "wild" if wild else "not wild",
                )
                if not wild:
                    break
            if wild:
                pval = self._chi2_pvalue(
                    omc, err, refit["pred"], refit["inlier"], refit["dof"]
                )
                logger.info(
                    "planet %d: %s OMC model chi2 test p = %.2g; %s",
                    n,
                    refit_model,
                    pval,
                    "model kept" if pval >= 0.01 else "measured times adopted",
                )
                if pval >= 0.01:
                    return refit_model, refit

        gp = self._fit("matern32", tlin, omc, err, measured)
        return "independent", {**gp, "inlier": measured}

    def _lasso_upgrade(self, tlin, omc, err, label, fit):
        """
        Replace the model (not 'independent') with a lasso polynomial of
        order 4 or 5 (_lasso_poly_model) if its kept times fail a chi-squared
        test (p < 0.001) and the lasso fit beats it in an F-test (p < 0.001; the
        order with the smaller p). The F-test judges the chi2 gain against the
        residual scatter, not the timing errors, which the failed test shows are
        unreliable. Fit on the kept times; outlier flags are unchanged

        Returns
        -------
        label : str
        fit : dict
            'pred', 'dof', 'inlier'
        """
        if label == "independent":
            return label, fit

        inlier = fit["inlier"]
        npts = np.sum(inlier)
        pval = self._chi2_pvalue(omc, err, fit["pred"], inlier, fit["dof"])
        if pval >= 0.001:
            return label, fit

        def chi2(pred):
            return np.sum(((omc - pred)[inlier] / err[inlier]) ** 2)

        best_p, best_label, best_fit = 1.0, None, None
        for k in (4, 5):
            lasso = self._lasso_poly_model(tlin, omc, err, inlier, k)
            # lasso dof can fall below the model's; it is still the larger model
            ddof = max(lasso["dof"] - fit["dof"], 1)
            resid_dof = max(npts - lasso["dof"], 1)
            F = ((chi2(fit["pred"]) - chi2(lasso["pred"])) / ddof) / (
                chi2(lasso["pred"]) / resid_dof
            )
            p = stats.f.sf(F, ddof, resid_dof)
            if p < best_p:
                best_p, best_label, best_fit = p, f"lasso_{k}", lasso

        logger.info(
            "%s OMC model chi2 test p = %.2g; best lasso F-test p = %.2g; %s",
            label,
            pval,
            best_p,
            f"{best_label} selected" if best_p < 0.001 else "model kept",
        )
        if best_p >= 0.001:
            return label, fit

        return best_label, {**best_fit, "inlier": inlier}

    # --- entry point --------------------------------------------------------

    def perform(self, method, check_periodic=True):
        """
        Parameters
        ----------
        method : str
            'smooth' : interpolate + Savitzky-Golay smoothing, filling unmeasured epochs
            'select' : offer candidate models (_candidates), fit each with outlier
                       rejection (_robust_fit), select by AIC/BIC (_compare,
                       _recheck_sinusoid), then keep the model or adopt the
                       measured times (_vet); a model whose kept times fail a
                       chi2 test may be replaced by a lasso polynomial
                       (_lasso_upgrade). The final fit gives every transit
                       time; adopted measured times are filled by a matern32 GP
        check_periodic : bool
            search for periodic TTV signals before model selection, and again
            after outlier rejection ('select' only)

        Returns
        -------
        planets : list of Planet
        litecurve : LiteCurve

        Each planet carries its regularized TransitTimes, with OMC diagnostics
        ttime_obs, kept_ttv, model_name (the final label: 'smooth', the
        selected model, or 'independent') and model_dof
        """
        if method not in ("smooth", "select"):
            raise ValueError(f"method must be 'smooth' or 'select', got {method!r}")
        npl = len(self.planets)

        # fit measured times with good photometry (FlagBadTransits) only; the rest
        # are returned unmeasured and filled by the model
        tts = [p.transit_times for p in self.planets]
        measured = [
            tt.phot_coverage & tt.phot_noise & np.isfinite(tt.ttime_obs) for tt in tts
        ]
        for n, m in enumerate(measured):
            # 1-2 transit planets need special handling (not yet implemented)
            if np.sum(m) < 3:
                raise UnsupportedTargetError(
                    f"planet {n}: {np.sum(m)} measured transit times; "
                    "RegularizeOMC needs at least 3"
                )
        # OMC x-axis: the linear ephemeris (noise-free and sorted), as in legacy
        tlin = [tt.linear_ephemeris() for tt in tts]
        omc = [
            np.where(m, tt.ttime_obs - t, np.nan)
            for tt, t, m in zip(tts, tlin, measured)
        ]
        err = [np.where(m, tt.ttime_err, np.nan) for tt, m in zip(tts, measured)]
        # 'smooth' runs on catalog times whose errors may be missing: equal weights
        if method == "smooth":
            err = [np.where(m, 1.0, np.nan) for m in measured]

        unclipped = [m.copy() for m in measured]
        for n, m in enumerate(measured):
            unclipped[n][m] = self._sigma_clip(omc[n][m], err[n][m])

        if method == "select":
            if check_periodic:
                period = self._periodic_signals(tlin, omc, unclipped)
            else:
                period = [None] * npl
            fits, model = [], []
            for n in range(npl):
                logger.info("planet %d: OMC model selection", n)
                fits.append(
                    {
                        k: self._robust_fit(
                            k, tlin[n], omc[n], err[n], unclipped[n], period[n]
                        )
                        for k in self._candidates(
                            omc[n], err[n], measured[n], unclipped[n], period[n]
                        )
                    }
                )
                model.append(self._compare(fits[n], np.sum(unclipped[n])))

            # Lomb-Scargle once more, without the selected model's outliers; a
            # sinusoid is offered only at a period significant after rejection
            if check_periodic:
                logger.info("periodic TTV search after outlier rejection")
                period2 = self._periodic_signals(
                    tlin, omc, [fits[n][model[n]]["inlier"] for n in range(npl)]
                )
                for n in range(npl):
                    logger.info("planet %d: periodic TTV recheck", n)
                    period[n] = self._recheck_sinusoid(
                        tlin[n],
                        omc[n],
                        err[n],
                        unclipped[n],
                        fits[n],
                        period[n],
                        period2[n],
                    )
                    model[n] = self._compare(fits[n], np.sum(unclipped[n]))

        # every planet is regularized before any is updated: the photometric
        # vetting subtracts the other planets' transits at their input times
        regularized = []
        for n, tt in enumerate(tts):
            if method == "smooth":
                label, inlier, dof = "smooth", unclipped[n], None
                pred = self._smooth(tt.index, omc[n], inlier)

            elif method == "select":
                logger.info("planet %d: vet selected OMC model", n)
                label, fit = self._vet(
                    n,
                    tlin[n],
                    omc[n],
                    err[n],
                    measured[n],
                    unclipped[n],
                    fits[n],
                    model[n],
                    period[n],
                )
                logger.info("planet %d: lasso check", n)
                label, fit = self._lasso_upgrade(tlin[n], omc[n], err[n], label, fit)
                inlier, dof = fit["inlier"], float(fit["dof"])
                pred = fit["pred"].copy()

            # independent: scatter about the GP, before the measured times replace it
            ttv_scatter = mad_std(omc[n][inlier] - pred[inlier])
            if label == "independent":
                # adopted measured times; the GP fills the rest
                pred[inlier] = omc[n][inlier]
                dof = float(np.sum(inlier))

            transit_times = TransitTimes(
                tt.mission,
                tt.target,
                tt.index,
                ttime_obs=np.where(measured[n], tt.ttime_obs, np.nan),
                ttime_err=np.where(measured[n], tt.ttime_err, np.nan),
                ttime_mod=tlin[n] + pred,
                phot_coverage=tt.phot_coverage,
                phot_noise=tt.phot_noise,
                kept_ttv=~(measured[n] & ~inlier),
                model_name=label,
                model_dof=dof,
                ttv_scatter=ttv_scatter,
            )
            regularized.append(transit_times)

        for n, (p, transit_times) in enumerate(zip(self.planets, regularized)):
            old_period = p.period
            try:
                p.set_transit_times(transit_times)
            except ValueError as e:
                raise ValueError(f"planet {n}, OMC method {method!r}: {e}") from e
            logger.info(
                "planet %d: %s OMC model; %d measured, %d sigma-clipped, "
                "%d rejected by model; P = %.6f -> %.6f d, ttv_buffer = %.1f min",
                n,
                transit_times.model_name,
                np.sum(measured[n]),
                np.sum(measured[n] & ~unclipped[n]),
                np.sum(unclipped[n] & ~transit_times.kept_ttv),
                old_period,
                p.period,
                transit_times.ttv_buffer() * 1440,
            )

        return self.planets, self.litecurve
