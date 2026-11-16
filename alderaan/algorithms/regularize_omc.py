"""Regularize transit times by modeling observed-minus-calculated (OMC) residuals"""

import logging
import warnings

import numpy as np
import pymc as pm
import pytensor.tensor as T
from astropy.stats import mad_std
from astropy.timeseries import LombScargle
from celerite2.pymc import GaussianProcess
from celerite2.pymc import terms as GPterms
from scipy import signal, stats
from scipy.interpolate import interp1d
from scipy.ndimage import median_filter, uniform_filter
from scipy.signal import savgol_filter
from sklearn.cluster import KMeans

from alderaan import SEED
from alderaan.algorithms.base import BaseAlg
from alderaan.data_models.transit_times import TransitTimes
from alderaan.utils.optim import optimize
from alderaan.utils.stats import legendre_p1

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
    def _flag_outliers(yobs, sigma_cut=5.0):
        if len(yobs) > 16:
            ysmooth = uniform_filter(median_filter(yobs, size=5, mode="mirror"), size=5)
        else:
            ysmooth = np.median(yobs)

        # zero scatter (e.g. times from a linear ephemeris): nothing to flag
        scatter = mad_std(yobs - ysmooth)
        if len(yobs) > 4 and scatter > 0:
            quality = np.abs(yobs - ysmooth) / scatter < sigma_cut
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

        interpolator = interp1d(
            index[q], yobs[q], kind="linear", fill_value="extrapolate"
        )
        ysmooth = interpolator(epochs)

        if npts >= 8:
            window_length = int(np.log2(npts)) + 1
            window_length += (window_length + 1) % 2
            ysmooth = savgol_filter(ysmooth, window_length, 2)

        return ysmooth[index - index.min()]

    @staticmethod
    def _poly_model(xtime, yobs, yerr, quality, polyorder):
        """
        Build a PyMC model to fit TTV observed-minus-calculated data
        Fits data with a polynomial

        Parameters
        ----------
        polyorder : int
            polynomial order

        Returns
        -------
        model : pm.Model()
        """
        q = quality

        # create Vandermonde matrices on xtime mapped to [-1, 1], for conditioning
        x = legendre_p1(xtime, (xtime.min(), xtime.max()))
        Xt = np.vander(x[q], N=polyorder + 1, increasing=True)
        Xp = np.vander(x, N=polyorder + 1, increasing=True)

        # build pymc model
        with pm.Model() as model:
            C = pm.Normal("C", mu=0, sigma=10, shape=polyorder + 1)

            trend = pm.Deterministic("trend", pm.math.dot(Xt, C))
            pm.Deterministic("pred", pm.math.dot(Xp, C))

            pm.Normal("obs", mu=trend, sigma=yerr[q], observed=yobs[q])

        return model

    @staticmethod
    def _sin_model(xtime, yobs, yerr, quality, ttv_period=None):
        """
        Build a PyMC model to fit TTV observed-minus-calculated data
        Fits data with a single-frequency sinusoid

        Parameters
        ----------
        ttv_period : float
            pre-estimated sinusoid period; the model places tight priors on this period

        Returns
        -------
        model : pm.Model()
        """
        q = quality

        # nominal TTV period
        if ttv_period is None:
            ttv_period = 2 * (xtime[q].max() - xtime[q].min())

        # convenience function
        def _sin_fxn(A, B, f, xt):
            return A * T.sin(2 * np.pi * f * xt) + B * T.cos(2 * np.pi * f * xt)

        # build pymc model
        with pm.Model() as model:
            df = 1 / (xtime[q].max() - xtime[q].min())
            f = pm.Normal("f", mu=1 / ttv_period, sigma=df)
            Ah = pm.Normal("Ah", mu=0, sigma=5 * np.std(yobs[q]))
            Bk = pm.Normal("Bk", mu=0, sigma=5 * np.std(yobs[q]))

            trend = pm.Deterministic("trend", _sin_fxn(Ah, Bk, f, xtime[q]))
            pm.Deterministic("pred", _sin_fxn(Ah, Bk, f, xtime))

            pm.Normal("obs", mu=trend, sigma=yerr[q], observed=yobs[q])

        return model

    @staticmethod
    def _matern32_model(xtime, yobs, yerr, quality):
        """
        Build a PyMC model to fit TTV observed-minus-calculated data
        Fits data with a regularized Matern-3/2 GP kernel

        Returns
        -------
        model : pm.Model()
        """
        q = quality

        # delta between each transit time
        dx = np.mean(np.diff(xtime))

        # maximum GP amplitude
        ymax = np.sqrt(mad_std(yobs) ** 2 - np.median(yerr) ** 2)

        # pymc model
        with pm.Model() as model:
            # build the kernel
            log_sigma = pm.TruncatedNormal(
                "log_sigma", mu=np.log(ymax), sigma=5, upper=np.log(ymax)
            )

            rho = pm.Uniform("rho", lower=2 * dx, upper=xtime[q].max() - xtime[q].min())
            kernel = GPterms.Matern32Term(sigma=T.exp(log_sigma), rho=rho)
            mean = pm.Normal("mean", mu=np.mean(yobs[q]), sigma=np.std(yobs[q]))

            # define the GP and factorize the covariance matrix
            gp = GaussianProcess(kernel, t=xtime[q], yerr=yerr[q], mean=mean)
            gp.compute(xtime[q], yerr=yerr[q])

            # track GP prediction, covariance, and degrees of freedom
            trend, cov = gp.predict(yobs[q], xtime[q], return_cov=True)

            trend = pm.Deterministic("trend", trend)
            pm.Deterministic("pred", gp.predict(yobs[q], xtime))
            pm.Deterministic("dof", T.trace(cov / yerr[q] ** 2))

            # add marginal likelihood to model
            lnlike = gp.log_likelihood(yobs[q])
            pm.Potential("lnlike", lnlike)

        return model

    @staticmethod
    def _sample(model, name, progressbar=False):
        """
        MAP-initialized NUTS sampling of a PyMC model; RuntimeWarnings raised
        during sampling are counted and logged at DEBUG under name
        """
        logger.debug("sampling %s model", name)
        with warnings.catch_warnings(record=True) as catch:
            warnings.simplefilter("always", category=RuntimeWarning)
            with model:
                map_soln, _ = optimize()

                # dense mass matrix adaptation
                trace = pm.sample(
                    tune=8000,
                    draws=2000,
                    chains=2,
                    target_accept=0.95,
                    init="adapt_full",
                    initvals=map_soln,
                    progressbar=progressbar,
                    return_inferencedata=False,
                    random_seed=SEED,
                )
        runtime = [w for w in catch if issubclass(w.category, RuntimeWarning)]
        for w in catch:
            if not issubclass(w.category, RuntimeWarning):
                # only RuntimeWarnings are expected here; pass anything else on
                warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)
        if len(runtime) > 0:
            logger.debug(
                "%s model: %d RuntimeWarnings during sampling (first: %s)",
                name,
                len(runtime),
                runtime[0].message,
            )
        return trace

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

        loop = True
        while loop:
            lombscargle = LombScargle(xt, yt * hann[~out])
            xf, yf = lombscargle.autopower(
                minimum_frequency=1.5 / (xt.max() - xt.min()),
                maximum_frequency=0.25 * fsamp,
                samples_per_peak=10,
            )

            peak_freq = xf[np.argmax(yf)]
            peak_fap = lombscargle.false_alarm_probability(
                yf.max(), method="bootstrap", method_kwds={"random_seed": SEED}
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
    def _select_best_model(yobs, yerr, quality, traces, dofs):
        """
        Parameters
        ----------
        traces : dict
            PyMC MultiTraces, output from _sample()
        dofs : dict
            degrees-of-freedom corresponding to each model trace

        Returns
        -------
        ymod : ndarray
            best model prediction
        name : str
            name (dict key) of selected best model
        """
        q = quality
        npts = np.sum(quality)

        # AIC and BIC for each model
        aic = {}
        bic = {}

        for k in traces.keys():
            trend = np.nanmedian(traces[k]["trend"], 0)
            lnlike = -0.5 * np.sum((yobs[q] - trend) ** 2 / yerr[q] ** 2)

            aic[k] = 2 * dofs[k] - 2 * lnlike
            bic[k] = dofs[k] * np.log(npts) - 2 * lnlike
            logger.debug(
                "%s OMC model: dof = %.1f, AIC = %.1f, BIC = %.1f",
                k,
                dofs[k],
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

        # calculate best model prediction and return
        ymod = np.nanmedian(traces[best_omc_model_name]["pred"], 0)

        return ymod, best_omc_model_name

    @staticmethod
    def _calculate_outlier_probability(yobs, ymod):
        """
        Calculate the probability that each measured omc transit time is an outlier
        Assumes a two-component Gaussian mixture model to calculate outlier probabilty
        Applies K-means clustering to assign each point to foreground or background

        Parameters
        ----------
        yobs : ndarray
            observed omc values
        ymod : ndarray
            regularized model for yobs

        Returns
        -------
        out_prob : ndarray
            probability that each transit time is an outlier
        out_class : ndarray
            boolean flag that each transit time is an outlier
        """
        # normalize and center residuals
        res = yobs - ymod
        res = res / np.std(res)
        res -= np.mean(res)

        # fit gaussian mixture
        with pm.Model() as model:
            w = pm.Dirichlet("w", np.array([1.0, 1.0]))
            mu = pm.Normal("mu", mu=0.0, sigma=1.0, shape=1)
            tau_latent = pm.Gamma("tau_latent", 1.0, 1.0, shape=2)
            tau = pm.Deterministic("tau", T.sort(tau_latent))

            pm.NormalMixture("obs", w, mu=mu * T.ones(2), tau=tau, observed=res)

        trace = RegularizeOMC._sample(model, "outlier mixture")

        # calculate foreground/background probability
        loc = np.nanmedian(trace["mu"], axis=0)
        scales = np.nanmedian(1 / np.sqrt(trace["tau"]), axis=0)

        z_fg = stats.norm(loc=loc, scale=scales.min()).pdf(res)
        z_bg = stats.norm(loc=loc, scale=scales.max()).pdf(res)

        fg_prob = z_fg / (z_fg + z_bg)

        # use KMeans clustering to assign each point to foreground/background
        km = KMeans(n_clusters=2, random_state=SEED)
        group = km.fit_predict(fg_prob.reshape(-1, 1))
        centroids = np.array(
            [np.mean(fg_prob[group == 0]), np.mean(fg_prob[group == 1])]
        )
        out = group == np.argmin(centroids)

        # reduce fg_prob threshold so that fewer than 30% of points are flagged
        if np.sum(out) / len(out) > 0.3:
            logger.warning(
                "K-means flagged %.0f%% of transit times as outliers; "
                "capping at the 30th percentile of fg_prob",
                100 * np.mean(out),
            )
            out = fg_prob < np.percentile(fg_prob, 30)

        return 1 - fg_prob, out

    def _identify_periodic_signals(self, xtime, yobs, quality):
        planets = self.planets
        npl = len(planets)

        if npl == 1:
            critical_fap = 0.1
        elif npl > 1:
            critical_fap = 0.99

        freq = [None] * npl
        fap = [None] * npl

        for n in range(npl):
            freq[n], fap[n] = self._identify_significant_frequencies(
                xtime[n], yobs[n], quality[n], critical_fap
            )

        # single planet systems : use direct search output (FAP < 0.1)
        # multiplanet systems : compare marginal frequencies to find close matches
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
                if not np.any(close[i]) and fap[i] > 0.1:
                    # freq is None (fap = 1) when no candidate was found
                    if freq[i] is not None:
                        logger.info(
                            "planet %d: candidate TTV period %.1f d "
                            "(FAP = %.3g) rejected; no matching frequency "
                            "in other planets",
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

    def _select_model(self, xtime, yobs, yerr, quality, freq):
        npts = np.sum(quality)

        traces = {}
        dofs = {}

        # Matern-3/2 model | don't use GP on very noisy data
        if (npts >= 8) & (np.median(yerr) <= 0.5 * mad_std(yobs)):
            traces["matern32"] = self._sample(
                self._matern32_model(xtime, yobs, yerr, quality), "matern32 OMC"
            )
            dofs["matern32"] = np.nanmedian(traces["matern32"]["dof"])
        else:
            logger.info(
                "skipping matern32 OMC model (%d transits; median error %.2e d "
                "vs 0.5 * mad_std(OMC) = %.2e d)",
                npts,
                np.median(yerr),
                0.5 * mad_std(yobs),
            )

        # Polynomial model | require 2^N transits, at most do 3rd order
        max_polyorder = np.max([1, np.min([3, int(np.log2(npts)) - 1])])
        for polyorder in range(1, max_polyorder + 1):
            traces[f"poly_{polyorder}"] = self._sample(
                self._poly_model(xtime, yobs, yerr, quality, polyorder),
                f"poly_{polyorder} OMC",
            )
            dofs[f"poly_{polyorder}"] = 1 + polyorder

        # Sinusoidal model
        if npts >= 8 and freq is not None:
            traces["sinusoid"] = self._sample(
                self._sin_model(xtime, yobs, yerr, quality, ttv_period=1 / freq),
                "sinusoid OMC",
            )
            dofs["sinusoid"] = 3

        logger.info("OMC models sampled: %s", ", ".join(traces))

        return self._select_best_model(yobs, yerr, quality, traces, dofs)

    @staticmethod
    def _regularized_transit_times(tt, measured, ymod, screen, out_prob, out_class):
        """
        Regularized transit times at every transit of tt: OMC model added to the
        current linear ephemeris, linearly interpolated over transits that are
        unmeasured or fail the initial outlier screen

        Parameters
        ----------
        tt : TransitTimes
            current transit times; ttime_obs is NaN where unmeasured
        measured : ndarray, bool
            True where tt.ttime_obs is finite
        ymod : ndarray
            OMC model relative to tt.linear_ephemeris(), at measured transits
        screen : ndarray
            True for measured transits passing the initial outlier screen
        out_prob : ndarray
            outlier probability at measured transits (NaN if not computed)
        out_class : ndarray
            True for measured transits flagged as model-based outliers

        Returns
        -------
        transit_times : TransitTimes
            same index, ttime_obs, and error_obs as tt; out_flag is True for
            transits failing the screen or flagged by the model
        """
        ttime_mod = tt.linear_ephemeris()[measured] + ymod

        interpolator = interp1d(
            tt.index[measured][screen],
            ttime_mod[screen],
            kind="linear",
            fill_value="extrapolate",
        )

        out_prob_full = np.full(len(tt.index), np.nan)
        out_prob_full[measured] = out_prob
        out_flag_full = np.zeros(len(tt.index), dtype=bool)
        out_flag_full[measured] = ~screen | out_class

        return TransitTimes(
            tt.mission,
            tt.target,
            tt.index,
            ttime_obs=tt.ttime_obs,
            error_obs=tt.error_obs,
            ttime_mod=interpolator(tt.index),
            out_prob=out_prob_full,
            out_flag=out_flag_full,
        )

    def perform(self, method, check_periodic=True):
        """
        Parameters
        ----------
        method : str
            'smooth' : interpolate + Savitzky-Golay smoothing, filling unmeasured epochs
            'select' : sample candidate models, select by AIC/BIC, flag outliers
        check_periodic : bool
            search for periodic TTV signals before model selection ('select' only)

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
        npl = len(planets)

        # fit measured transit times only; unmeasured ones are filled by the model
        tts = [p.transit_times for p in planets]
        measured = [np.isfinite(tt.ttime_obs) for tt in tts]
        # OMC x-axis: the linear ephemeris (noise-free and sorted), as in legacy
        xtime = [tt.linear_ephemeris()[m] for tt, m in zip(tts, measured)]
        yobs = [
            (tt.ttime_obs - tt.linear_ephemeris())[m] for tt, m in zip(tts, measured)
        ]
        yerr = [tt.error_obs[m] for tt, m in zip(tts, measured)]
        quality = [self._flag_outliers(y) for y in yobs]

        ymod = [None] * npl

        if method == "select" and check_periodic:
            freq = self._identify_periodic_signals(xtime, yobs, quality)
        else:
            freq = [None] * npl

        for n, p in enumerate(planets):
            old_period = p.period

            logger.info(
                "planet %d: %d of %d measured transit times pass initial outlier "
                "screen",
                n,
                np.sum(quality[n]),
                len(quality[n]),
            )

            if method == "smooth":
                ymod[n] = self._smooth(tts[n].index[measured[n]], yobs[n], quality[n])
                out_prob = np.full(len(yobs[n]), np.nan)
                out_class = np.zeros(len(yobs[n]), dtype=bool)

            elif method == "select":
                ymod[n], best_omc_model_name = self._select_model(
                    xtime[n], yobs[n], yerr[n], quality[n], freq[n]
                )
                out_prob, out_class = self._calculate_outlier_probability(
                    yobs[n], ymod[n]
                )
                logger.info(
                    "planet %d: adopting %s OMC model; %d of %d (%.1f%%) transit "
                    "times flagged as outliers",
                    n,
                    best_omc_model_name,
                    np.sum(out_class),
                    len(out_class),
                    100 * np.mean(out_class),
                )

            transit_times = self._regularized_transit_times(
                p.transit_times,
                measured[n],
                ymod[n],
                quality[n],
                out_prob,
                out_class,
            )
            try:
                p.set_transit_times(transit_times)
            except ValueError as e:
                raise ValueError(f"planet {n}, OMC method {method!r}: {e}") from e
            logger.info(
                "planet %d: P = %.6f -> %.6f d; ttv_buffer = %.1f min",
                n,
                old_period,
                planets[n].period,
                transit_times.ttv_buffer() * 1440,
            )

        return self.planets, self.litecurve
