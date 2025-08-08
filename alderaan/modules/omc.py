__all__ = ['OMC']

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import aesara_theano_fallback.tensor as T
from astropy.stats import mad_std
from celerite2.theano import terms as GPterms
from celerite2.theano import GaussianProcess
import numpy as np
import pymc3 as pm
import pymc3_ext as pmx
from scipy import stats
from scipy.ndimage import uniform_filter, median_filter
from sklearn.cluster import KMeans
from alderaan.constants import pi

class OMC:
    def __init__(self, ephemeris):
        # set initial O-C estimates
        self.index = ephemeris.index
        self.xtime = ephemeris.ttime
        self.yobs = ephemeris.ttime - ephemeris.eval_linear_ephemeris()
        self.yerr = ephemeris.error
        self.ymod = None

        # flag outliers
        self.quality = self.quick_flag_outliers()
        self.out_prob = None

        # set static reference period, epoch, and linear ephemeris
        self._set_static_references(ephemeris)

        # set nominal peak frequency and false alarm probability
        self.peakfreq = None
        self.peakfap = None


    def _set_static_references(self, ephemeris):
        self._static_period = ephemeris._static_period.copy()
        self._static_epoch = ephemeris._static_epoch.copy()


    def quick_flag_outliers(self, sigma_cut=5.0):
        if len(self.yobs) > 16:
            ysmooth = uniform_filter(median_filter(self.yobs, size=5, mode='mirror'), size=5)
        else:
            ysmooth = np.median(self.yobs)

        if len(self.yobs) > 4:
            quality = np.abs(self.yobs - ysmooth)/mad_std(self.yobs - ysmooth) < sigma_cut
        else:
            quality = np.zeros(len(self.yobs), dtype=bool)

        return quality


    def poly_model(self, polyorder, ignore_bad=True, xt_predict=None):
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
        if ignore_bad:
            q = self.quality
        else:
            q = np.ones(len(self.xtime), dtype=bool)

        # times where trend will be predicted
        if xt_predict is None:
            xt_predict = self.xtime

        # create Vandermonde matrices
        Xt = np.vander(self.xtime[q], N=polyorder+1, increasing=True)
        Xp = np.vander(xt_predict, N=polyorder+1, increasing=True)
            
        # build pymc model
        with pm.Model() as model:
            C = pm.Normal("C", mu=0, sd=10, shape=polyorder+1)

            trend = pm.Deterministic("trend", pm.math.dot(Xt,C))
            pred = pm.Deterministic("pred", pm.math.dot(Xp,C))

            pm.Normal("obs", mu=trend, sd=self.yerr[q], observed=self.yobs[q])

        return model


    def sin_model(self, ttv_period=None, ignore_bad=True, xt_predict=None):
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
        if ignore_bad:
            q = self.quality
        else:
            q = np.ones(len(self.xtime), dtype=bool)

        # times where trend will be predicted
        if xt_predict is None:
            xt_predict = self.xtime

        # nominal TTV period
        if (ttv_period is None) & (self.peakfreq is None):
            ttv_period = 2 * (self.xtime[q].max() - self.xtime[q].min())
        elif (ttv_period is None) & (self.peakfreq is not None):
            ttv_period = 1 / self.peakfreq
        
        # convenience function
        def _sin_fxn(A, B, f, xt):
                return A * T.sin(2 * pi * f * xt) + B * T.cos(2 * pi * f * xt)

        # build pymc model
        with pm.Model() as model:
            df = 1 / (self.xtime[q].max() - self.xtime[q].min())
            f = pm.Normal("f", mu=1 / ttv_period, sd=df)
            Ah = pm.Normal("Ah", mu=0, sd=5 * np.std(self.yobs[q]))
            Bk = pm.Normal("Bk", mu=0, sd=5 * np.std(self.yobs[q]))

            trend = pm.Deterministic("trend", _sin_fxn(Ah, Bk, f, self.xtime[q]))
            pred = pm.Deterministic("pred", _sin_fxn(Ah, Bk, f, xt_predict))

            pm.Normal("obs", mu=trend, sd=self.yerr[q], observed=self.yobs[q])

        return model


    def matern32_model(self, ignore_bad=True, xt_predict=None):
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
        if ignore_bad:
            q = self.quality
        else:
            q = np.ones(len(self.xtime), dtype=bool)

        # times where trend will be predicted
        if xt_predict is None:
            xt_predict = self.xtime

        # delta between each transit time
        dx = np.mean(np.diff(self.xtime))

        # maximum GP amplitude
        ymax = np.sqrt(mad_std(self.yobs) ** 2 - np.median(self.yerr) ** 2)

        # pymc model
        with pm.Model() as model:
            # build the kernel
            log_sigma = pm.Bound(pm.Normal, upper=np.log(ymax))(
                "log_sigma", mu=np.log(ymax), sd=5
            )

            rho = pm.Uniform("rho", lower=2 * dx, upper=self.xtime[q].max() - self.xtime[q].min())
            kernel = GPterms.Matern32Term(sigma=T.exp(log_sigma), rho=rho)
            mean = pm.Normal("mean", mu=np.mean(self.yobs[q]), sd=np.std(self.yobs[q]))
            
            # define the GP and factorize the covariance matrix
            gp = GaussianProcess(kernel, t=self.xtime[q], yerr=self.yerr[q], mean=mean)
            gp.compute(self.xtime[q], yerr=self.yerr[q])

            # track GP prediction, covariance, and degrees of freedom
            trend, cov = gp.predict(self.yobs[q], self.xtime[q], return_cov=True)

            trend = pm.Deterministic("trend", trend)
            cov = pm.Deterministic("cov", cov)
            pred = pm.Deterministic("pred", gp.predict(self.yobs[q], xt_predict))
            dof = pm.Deterministic("dof", pm.math.trace(cov / self.yerr[q]**2))

            # add marginal likelihood to model
            lnlike = gp.log_likelihood(self.yobs[q])
            pm.Potential("lnlike", lnlike)

        return model


    def sample(self, model, progressbar=False):
        """
        Docstring
        """
        with model:
            map_soln = pmx.optimize(start=model.test_point, 
                                    progress=progressbar
                                    )
            
            trace = pmx.sample(tune=8000,
                               draws=2000,
                               chains=2,
                               target_accept=0.95,
                               start=map_soln,
                               progressbar=progressbar,
                               return_inferencedata=False
                              )
        return trace


    def identify_significant_frequencies(self, critical_fap):
        """
        Identify significant periodic frequencies using a Lomb-Scargle periodogram

        Arguments
        ---------
        critical_fap : float
            false alarm probability threshold to consider a frequency significant (default=0.1)
        """
        q = self.quality
        npts = np.sum(self.quality)

        peakfreq = None
        peakfap = None

        if npts > 8:
            try:
                xf, yf, freqs, faps = self.LS_estimator(self.xtime[q], self.yobs[q], fap=critical_fap)

                if len(freqs) > 0:
                    if freqs[0] > xf.min():
                        peakfreq = freqs[0]
                        peakfap = faps[0]

            except Exception:
                pass
        
        return peakfreq, peakfap
    

    @staticmethod
    def LS_estimator(x, y, fsamp=None, fap=0.1, return_levels=False, max_peaks=2):
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
            nominal sampling frequency; if not provided it will be calculated from the data
        fap : float
            false alarm probability threshold to consider a frequency significant (default=0.1)

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


    def select_best_model(self, traces, dofs, verbose=True):
        """
        Arguments
            traces : dict
              dictionary of PyMC3 MultiTraces, model output from self.sample()
            dofs : dict
              degrees-of-freedom corresponding to each model trace

        Returns
            best_omc_model (str)
        """
        q = self.quality
        npts = np.sum(self.quality)

        # AIC and BIC for each model
        aic = {}
        bic = {}

        for k in traces.keys():
            trend = np.nanmedian(traces[k]['trend'], 0)
            lnlike = -np.sum((self.yobs[q] - trend)**2 / self.yerr[q]**2)

            aic[k] = 2 * dofs[k] - 2 * lnlike
            bic[k] = dofs[k] * np.log(npts) - 2 * lnlike

        preferred_by_aic = min(aic, key=aic.get)
        preferred_by_bic = min(bic, key=bic.get)

        if verbose:
            print(f"AIC : {preferred_by_aic}, BIC : {preferred_by_bic}")

        # Case 1: AIC and BIC match recommendation
        if preferred_by_aic == preferred_by_bic:
            best_omc_model = preferred_by_aic

        # Case 2: Select parametric models over Matern-3/2 GP
        elif np.any(np.array([preferred_by_aic, preferred_by_bic]) == 'matern32'):
            if preferred_by_aic == 'matern32':
                best_omc_model = preferred_by_bic
            elif preferred_by_bic == 'matern32':
                best_omc_model = preferred_by_aic
        
        # Case 3: Select model with more degrees of freedom
        else:
            if dofs[preferred_by_aic] >= dofs[preferred_by_bic]:
                best_omc_model = preferred_by_aic
            elif dofs[preferred_by_aic] < dofs[preferred_by_bic]:
                best_omc_model = preferred_by_bic

        return best_omc_model


    def calculate_outlier_probability(self, ymod):
        """
        Arguments:
            ymod (ndarray) : regularized model for self.yobs
        """
        # normalize and center residuals
        res = self.yobs - ymod
        res = res / np.std(res)
        res -= np.mean(res)

        # fit gaussian mixture
        with pm.Model() as model:
            w = pm.Dirichlet("w", np.array([1.0, 1.0]))
            mu = pm.Normal("mu", mu=0.0, sd=1.0, shape=1)
            tau_latent = pm.Gamma("tau_latent", 1.0, 1.0, shape=2)
            tau = pm.Deterministic("tau", T.sort(tau_latent))

            obs = pm.NormalMixture("obs", w, mu=mu * T.ones(2), tau=tau, observed=res)

        with model:
            trace = self.sample(model)

        # calculate foreground/background probability
        loc = np.nanmedian(trace["mu"], axis=0)
        scales = np.nanmedian(1 / np.sqrt(trace["tau"]), axis=0)

        z_fg = stats.norm(loc=loc, scale=scales.min()).pdf(res)
        z_bg = stats.norm(loc=loc, scale=scales.max()).pdf(res)

        fg_prob = z_fg / (z_fg + z_bg)

        # use KMeans clustering to assign each point to foreground/background
        km = KMeans(n_clusters=2)
        group = km.fit_predict(fg_prob.reshape(-1, 1))
        centroids = np.array([np.mean(fg_prob[group == 0]), np.mean(fg_prob[group == 1])])
        out = group == np.argmin(centroids)

        # reduce fg_prob threshold until less than 30% of points are flagged
        if np.sum(out) / len(out) > 0.3:
            out = fg_prob < np.percentile(fg_prob, 0.3)

        # return quality vector
        return fg_prob, out
