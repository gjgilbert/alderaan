"""Regularize transit times by modeling observed-minus-calculated (OMC) residuals"""

import os
import warnings

import aesara_theano_fallback.tensor as T
import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
import pymc3_ext as pmx
from astropy.stats import mad_std
from astropy.timeseries import LombScargle
from celerite2.theano import GaussianProcess
from celerite2.theano import terms as GPterms
from scipy import signal, stats
from scipy.interpolate import interp1d
from scipy.ndimage import median_filter, uniform_filter
from scipy.signal import savgol_filter
from sklearn.cluster import KMeans

from alderaan.algorithms.base import BaseAlg
from alderaan.utils.io import save_omc_ephemeris
from alderaan.utils.plotting import plot_omc
from alderaan.utils.stats import set_sigma_cutoff


class RegularizeOMC(BaseAlg):
    """
    Replace each planet's transit times with a regularized OMC model

    Parameters
    ----------
    star : Star
    planets : list of Planet
    litecurve : LiteCurve
    config : ConfigParser
        pipeline configuration, including RUN and PATHS values set by main
    """

    def __init__(self, star, planets, litecurve, config):
        super().__init__(star, planets, litecurve, config)

    @staticmethod
    def _flag_outliers(yobs, sigma_cut=5.0):
        if len(yobs) > 16:
            ysmooth = uniform_filter(median_filter(yobs, size=5, mode="mirror"), size=5)
        else:
            ysmooth = np.median(yobs)

        if len(yobs) > 4:
            quality = np.abs(yobs - ysmooth) / mad_std(yobs - ysmooth) < sigma_cut
        else:
            quality = np.ones(len(yobs), dtype=bool)

        return quality

    @staticmethod
    def _smooth(xtime, yobs, quality):
        q = quality
        npts = len(q)

        interpolator = interp1d(
            xtime[q], yobs[q], kind="linear", fill_value="extrapolate"
        )
        ysmooth = interpolator(xtime)

        if npts >= 8:
            window_length = int(np.log2(npts)) + 1
            window_length += (window_length + 1) % 2
            ysmooth = savgol_filter(ysmooth, window_length, 2)

        return ysmooth

    @staticmethod
    def _poly_model(
        xtime, yobs, yerr, quality, polyorder, ignore_bad=True, xt_predict=None
    ):
        """
        Build a PyMC3 model to fit TTV observed-minus-calculated data
        Fits data with a polynomial

        Arguments
        ----------
        polyorder : int
            polynomial order
        ignore_bad : bool
            True (default) to exclude bad quality times from model
        xt_predict : ndarray
            (optional) time values to predict OMC model

        Returns
        -------
        model : pm.Model()
        """
        # quality mask
        if ignore_bad and quality is not None:
            q = quality
        else:
            q = np.ones(len(xtime), dtype=bool)

        # times where trend will be predicted
        if xt_predict is None:
            xt_predict = xtime

        # create Vandermonde matrices
        Xt = np.vander(xtime[q], N=polyorder + 1, increasing=True)
        Xp = np.vander(xt_predict, N=polyorder + 1, increasing=True)

        # build pymc model
        with pm.Model() as model:
            C = pm.Normal("C", mu=0, sd=10, shape=polyorder + 1)

            trend = pm.Deterministic("trend", pm.math.dot(Xt, C))
            pm.Deterministic("pred", pm.math.dot(Xp, C))

            pm.Normal("obs", mu=trend, sd=yerr[q], observed=yobs[q])

        return model

    @staticmethod
    def _sin_model(
        xtime, yobs, yerr, quality, ttv_period=None, ignore_bad=True, xt_predict=None
    ):
        """
        Build a PyMC3 model to fit TTV observed-minus-calculated data
        Fits data with a single-frequency sinusoid

        Arguments
        ----------
        ttv_period : float
            pre-estimated sinusoid period; the model places tight priors on this period
        ignore_bad : bool
            True (default) to exclude bad quality times from model
        xt_predict : ndarray
            (optional) time values to predict OMC model

        Returns
        -------
        model : pm.Model()
        """
        # quality mask
        if ignore_bad and quality is not None:
            q = quality
        else:
            q = np.ones(len(xtime), dtype=bool)

        # times where trend will be predicted
        if xt_predict is None:
            xt_predict = xtime

        # nominal TTV period
        if ttv_period is None:
            ttv_period = 2 * (xtime[q].max() - xtime[q].min())

        # convenience function
        def _sin_fxn(A, B, f, xt):
            return A * T.sin(2 * np.pi * f * xt) + B * T.cos(2 * np.pi * f * xt)

        # build pymc model
        with pm.Model() as model:
            df = 1 / (xtime[q].max() - xtime[q].min())
            f = pm.Normal("f", mu=1 / ttv_period, sd=df)
            Ah = pm.Normal("Ah", mu=0, sd=5 * np.std(yobs[q]))
            Bk = pm.Normal("Bk", mu=0, sd=5 * np.std(yobs[q]))

            trend = pm.Deterministic("trend", _sin_fxn(Ah, Bk, f, xtime[q]))
            pm.Deterministic("pred", _sin_fxn(Ah, Bk, f, xt_predict))

            pm.Normal("obs", mu=trend, sd=yerr[q], observed=yobs[q])

        return model

    @staticmethod
    def _matern32_model(xtime, yobs, yerr, quality, ignore_bad=True, xt_predict=None):
        """
        Build a PyMC3 model to fit TTV observed-minus-calculated data
        Fits data with a regularized Matern-3/2 GP kernel

        Arguments
        ----------
        ignore_bad : bool
            True (default) to exclude bad quality times from model
        xt_predict : ndarray
            (optional) time values to predict OMC model

        Returns
        -------
        model : pm.Model()
        """
        # quality mask
        if ignore_bad and quality is not None:
            q = quality
        else:
            q = np.ones(len(xtime), dtype=bool)

        # times where trend will be predicted
        if xt_predict is None:
            xt_predict = xtime

        # delta between each transit time
        dx = np.mean(np.diff(xtime))

        # maximum GP amplitude
        ymax = np.sqrt(mad_std(yobs) ** 2 - np.median(yerr) ** 2)

        # pymc model
        with pm.Model() as model:
            # build the kernel
            log_sigma = pm.Bound(pm.Normal, upper=np.log(ymax))(
                "log_sigma", mu=np.log(ymax), sd=5
            )

            rho = pm.Uniform("rho", lower=2 * dx, upper=xtime[q].max() - xtime[q].min())
            kernel = GPterms.Matern32Term(sigma=T.exp(log_sigma), rho=rho)
            mean = pm.Normal("mean", mu=np.mean(yobs[q]), sd=np.std(yobs[q]))

            # define the GP and factorize the covariance matrix
            gp = GaussianProcess(kernel, t=xtime[q], yerr=yerr[q], mean=mean)
            gp.compute(xtime[q], yerr=yerr[q])

            # track GP prediction, covariance, and degrees of freedom
            trend, cov = gp.predict(yobs[q], xtime[q], return_cov=True)

            trend = pm.Deterministic("trend", trend)
            cov = pm.Deterministic("cov", cov)
            pm.Deterministic("pred", gp.predict(yobs[q], xt_predict))
            pm.Deterministic("dof", pm.math.trace(cov / yerr[q] ** 2))

            # add marginal likelihood to model
            lnlike = gp.log_likelihood(yobs[q])
            pm.Potential("lnlike", lnlike)

        return model

    @staticmethod
    def _sample(model, progressbar=False):
        with model:
            map_soln = pmx.optimize(start=model.test_point, progress=progressbar)

            trace = pmx.sample(
                tune=8000,
                draws=2000,
                chains=2,
                target_accept=0.95,
                start=map_soln,
                progressbar=progressbar,
                return_inferencedata=False,
            )
        return trace

    @staticmethod
    def _identify_significant_frequencies(xtime, yobs, quality, critical_fap):
        """
        Identify significant periodic frequencies using a Lomb-Scargle periodogram

        Arguments
        ---------
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
                pass

        return peakfreq, peakfap

    @staticmethod
    def _ls_estimator(x, y, fsamp=None, fap=0.1, return_levels=False, max_peaks=2):
        """
        Generate a Lomb-Scargle periodogram and identify significant frequencies
        * assumes that data are nearly evenly sampled
        * optimized for finding marginal periodic TTV signals in OMC data
        * may not perform well for other applications

        Arguments
        ---------
        x : array-like
            1D array of x data values; should be monotonically increasing
        y : array-like
            1D array of corresponding y data values, len(x)
        fsamp: float
            nominal sampling frequency;
            if not provided it will be calculated from the data
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
        # get sampling frequency
        if fsamp is None:
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
            peak_fap = lombscargle.false_alarm_probability(yf.max(), method="bootstrap")

            # output first iteration of LS periodogram
            if len(freqs) == 0:
                xf_out = xf.copy()
                yf_out = yf.copy()
                levels = lombscargle.false_alarm_level([0.1, 0.01, 0.001])

            if peak_fap < fap:
                yt -= lombscargle.model(xt, peak_freq) * len(xt)
                freqs.append(peak_freq)
                faps.append(peak_fap)

            else:
                loop = False

            if len(freqs) >= max_peaks:
                loop = False

        if return_levels:
            return xf_out, yf_out, freqs, faps, levels

        else:
            return xf_out, yf_out, freqs, faps

    @staticmethod
    def _select_best_model(
        yobs, yerr, quality, traces, dofs, return_name=True, verbose=True
    ):
        """
        Args:
            traces (dict): PyMC3 MultiTraces, output from _sample()
            dofs (dict): degrees-of-freedom corresponding to each model trace

        Returns:
            ndarray : best model prediction
            str : name (dict key) of selected best model
        """
        q = quality
        npts = np.sum(quality)

        # AIC and BIC for each model
        aic = {}
        bic = {}

        for k in traces.keys():
            trend = np.nanmedian(traces[k]["trend"], 0)
            lnlike = -np.sum((yobs[q] - trend) ** 2 / yerr[q] ** 2)

            aic[k] = 2 * dofs[k] - 2 * lnlike
            bic[k] = dofs[k] * np.log(npts) - 2 * lnlike

        preferred_by_aic = min(aic, key=aic.get)
        preferred_by_bic = min(bic, key=bic.get)

        # Case 1: AIC and BIC match recommendation
        if preferred_by_aic == preferred_by_bic:
            best_omc_model_name = preferred_by_aic

        # Case 2: Select parametric models over Matern-3/2 GP
        elif np.any(np.array([preferred_by_aic, preferred_by_bic]) == "matern32"):
            if preferred_by_aic == "matern32":
                best_omc_model_name = preferred_by_bic
            elif preferred_by_bic == "matern32":
                best_omc_model_name = preferred_by_aic

        # Case 3: Select model with more degrees of freedom
        else:
            if dofs[preferred_by_aic] >= dofs[preferred_by_bic]:
                best_omc_model_name = preferred_by_aic
            elif dofs[preferred_by_aic] < dofs[preferred_by_bic]:
                best_omc_model_name = preferred_by_bic

        if verbose:
            print("\bselecting best model")
            print(f"  AIC : {preferred_by_aic}")
            print(f"  BIC : {preferred_by_bic}")
            print(f"  returning: {best_omc_model_name}")

        # calculate best model prediction and return
        ymod = np.nanmedian(traces[best_omc_model_name]["pred"], 0)

        if return_name:
            return ymod, best_omc_model_name
        return ymod

    @staticmethod
    def _calculate_outlier_probability(yobs, ymod):
        """
        Calculate the probability that each measured omc transit time is an outlier
        Assumes a two-component Gaussian mixture model to calculate outlier probabilty
        Applies K-means clustering to assign each point to foreground or background

        Args:
            yobs (ndarray) : observed omc values
            ymod (ndarray) : regularized model for yobs

        Returns:
            tuple : (out_prob, out_class)
              * out_prob (ndarray) : probability that each transit time is an outlier
              * out_class (ndarray) : boolean flag that each transit time is an outlier
        """
        # normalize and center residuals
        res = yobs - ymod
        res = res / np.std(res)
        res -= np.mean(res)

        # fit gaussian mixture
        with pm.Model() as model:
            w = pm.Dirichlet("w", np.array([1.0, 1.0]))
            mu = pm.Normal("mu", mu=0.0, sd=1.0, shape=1)
            tau_latent = pm.Gamma("tau_latent", 1.0, 1.0, shape=2)
            tau = pm.Deterministic("tau", T.sort(tau_latent))

            pm.NormalMixture("obs", w, mu=mu * T.ones(2), tau=tau, observed=res)

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always", category=RuntimeWarning)
            with model:
                trace = RegularizeOMC._sample(model)

        # calculate foreground/background probability
        loc = np.nanmedian(trace["mu"], axis=0)
        scales = np.nanmedian(1 / np.sqrt(trace["tau"]), axis=0)

        z_fg = stats.norm(loc=loc, scale=scales.min()).pdf(res)
        z_bg = stats.norm(loc=loc, scale=scales.max()).pdf(res)

        fg_prob = z_fg / (z_fg + z_bg)

        # use KMeans clustering to assign each point to foreground/background
        km = KMeans(n_clusters=2)
        group = km.fit_predict(fg_prob.reshape(-1, 1))
        centroids = np.array(
            [np.mean(fg_prob[group == 0]), np.mean(fg_prob[group == 1])]
        )
        out = group == np.argmin(centroids)

        # reduce fg_prob threshold so that than 30% of points are flagged
        if np.sum(out) / len(out) > 0.3:
            out = fg_prob < np.percentile(fg_prob, 0.3)

        print(
            f"{np.sum(out)} of {len(out)} ({100 * np.sum(out) / len(out):.1f}%) "
            "of transit times flagged as outliers"
        )

        return 1 - fg_prob, out

    def _identify_periodic_signals(self, xtime, yobs, quality):
        planets = self.planets
        npl = len(planets)

        print("searching for periodic signals")
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
                df_min = 1 / (
                    planets[i].transit_times.max() - planets[i].transit_times.min()
                )

                for j in range(i + 1, npl):
                    if freq[i] is not None and freq[j] is not None:
                        df_ij = np.abs(freq[i] - freq[j])

                        if df_ij < df_min:
                            close[i, j] = True
                            close[j, i] = True

            for i in range(npl):
                if fap[i] is not None:
                    if not np.any(close[i]) and fap[i] > 0.1:
                        freq[i] = None
                        fap[i] = None

        for n in range(npl):
            if freq[n] is not None:
                print(
                    f"  Planet {n} : periodic signal found at "
                    f"P = {1 / freq[n]:.1f} days"
                )
            else:
                print(f"  Planet {n} : no significant periodic component found")

        return freq

    def _select_model(self, xtime, yobs, yerr, quality, freq):
        npts = np.sum(quality)

        traces = {}
        dofs = {}

        # Matern-3/2 model | don't use GP on very noisy data
        if (npts >= 8) & (np.median(yerr) <= 0.5 * mad_std(yobs)):
            print("\nmatern-3/2 model")
            with warnings.catch_warnings(record=True) as catch:
                warnings.simplefilter("always", category=RuntimeWarning)
                traces["matern32"] = self._sample(
                    self._matern32_model(xtime, yobs, yerr, quality)
                )
                dofs["matern32"] = np.nanmedian(traces["matern32"]["dof"])

        # Polynomial model | require 2^N transits, at most do 3rd order
        max_polyorder = np.max([1, np.min([3, int(np.log2(npts)) - 1])])
        for polyorder in range(1, max_polyorder + 1):
            print(f"\npolynomial (degree {polyorder}) model")
            with warnings.catch_warnings(record=True) as catch:
                warnings.simplefilter("always", category=RuntimeWarning)
                traces[f"poly_{polyorder}"] = self._sample(
                    self._poly_model(xtime, yobs, yerr, quality, polyorder)
                )
                dofs[f"poly_{polyorder}"] = 1 + polyorder

        # Sinusoidal model
        if npts >= 8 and freq is not None:
            print("\nsinusoid model")
            with warnings.catch_warnings(record=True) as catch:
                warnings.simplefilter("always", category=RuntimeWarning)
                traces["sinusoid"] = self._sample(
                    self._sin_model(xtime, yobs, yerr, quality, ttv_period=1 / freq)
                )
                dofs["sinusoid"] = 3

        if len(catch) > 0:
            print(f"{len(catch)} RuntimeWarnings caught during sampling")

        return self._select_best_model(yobs, yerr, quality, traces, dofs)

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
        star : Star
        planets : list of Planet
        litecurve : LiteCurve
        ttv_buffer : list of float
            per-planet scatter of OMC residuals about the model
        """
        if method not in ("smooth", "select"):
            raise ValueError(f"method must be 'smooth' or 'select', got {method!r}")

        print(f"\n\nALGORITHM: REGULARIZE OMC ({method})\n")

        target = self.config["RUN"]["target"]
        results_dir = self.config["PATHS"]["results_dir"]
        quicklook_dir = self.config["PATHS"]["quicklook_dir"]

        planets = self.planets
        npl = len(planets)

        xtime = [p.transit_times for p in planets]
        yobs = [p.transit_times - p.eval_linear_ephemeris() for p in planets]
        yerr = [p.transit_errors for p in planets]
        quality = [self._flag_outliers(y) for y in yobs]
        period = [p.reference_period for p in planets]

        ymod = [None] * npl
        out_prob = [None] * npl
        ttv_buffer = [None] * npl

        if method == "select" and check_periodic:
            freq = self._identify_periodic_signals(xtime, yobs, quality)
        else:
            freq = [None] * npl

        for n, p in enumerate(planets):
            old_period = p.period

            if method == "smooth":
                ymod[n] = self._smooth(xtime[n], yobs[n], quality[n])

            elif method == "select":
                print(
                    f"\nperforming omc model selection for Planet {n} "
                    f"({np.sum(quality[n])} transits)"
                )
                ymod[n], best_omc_model_name = self._select_model(
                    xtime[n], yobs[n], yerr[n], quality[n], freq[n]
                )
                print(
                    f"\nPlanet {n} : adopting {best_omc_model_name} as omc model "
                    "and flagging outliers"
                )

                out_prob[n], out_class = self._calculate_outlier_probability(
                    yobs[n], ymod[n]
                )

                # save transit times
                q = quality[n]
                index = p.transit_index[q]
                static_ephemeris = p.reference_epoch + p.reference_period * index
                filename = os.path.join(results_dir, f"{target}_{n:02d}_quick.ttvs")
                save_omc_ephemeris(
                    filename,
                    index,
                    yobs[n][q] + static_ephemeris,
                    ymod[n][q] + static_ephemeris,
                    out_prob[n][q],
                    out_class[q],
                )

            planets[n] = p.update_ephemeris(
                p.transit_index,
                ymod[n],
                yerr[n],
                quality[n],
                full=(method == "smooth"),
            )
            print(f"Planet {n} : {old_period:.6f} --> {p.period:.6f}")

            ttv_buffer[n] = set_sigma_cutoff(len(yobs[n])) * mad_std(yobs[n] - ymod[n])

        if method == "smooth":
            filename = f"{target}_omc_initial.png"
        else:
            filename = f"{target}_omc.png"

        plot_omc(
            xtime,
            yobs,
            yerr,
            ymod,
            quality,
            out_prob,
            period,
            target,
            os.path.join(quicklook_dir, filename),
        )
        plt.close("all")

        return self.star, self.planets, self.litecurve, ttv_buffer
