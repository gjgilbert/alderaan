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
from scipy.ndimage import uniform_filter, median_filter

from src.constants import *

class OMC:
    def __init__(self, ephemeris):
        # set initial O-C estimates
        self.index = ephemeris.index
        self.xtime = ephemeris.ttime
        self.yomc = ephemeris.ttime - ephemeris.eval_linear_ephemeris()
        self.yerr = ephemeris.error

        # flag outliers
        self.quality = self._flag_outliers()

        # set static reference period, epoch, and linear ephemeris
        self = self._set_static_references(ephemeris)


    def _set_static_references(self, ephemeris):
        self._static_period = ephemeris.period.copy()
        self._static_epoch = ephemeris.epoch.copy()
        self._static_ephemeris = ephemeris.eval_linear_ephemeris()

        ephem_1 = self._static_epoch + self.index*self._static_period
        ephem_2 = self._static_ephemeris

        assert np.allclose(ephem_1, ephem_2, rtol=1e-10, atol=1e-10), "static reference ephemeris is not self-consistent"

        return self
    

    def _flag_outliers(self, sigma_cut=5.0):
        if len(self.yomc) > 16:
            ysmooth = uniform_filter(median_filter(self.yomc, size=5, mode='mirror'), size=5)
        else:
            ysmooth = np.median(self.yomc)

        if len(self.yomc) > 4:
            quality = np.abs(self.yomc - ysmooth)/mad_std(self.yomc - ysmooth) < sigma_cut
        else:
            quality = np.zeros(len(self.yomc), dtype=bool)

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

            pm.Normal("obs", mu=trend, sd=self.yerr[q], observed=self.yomc[q])

        return model


    def sin_model(self, period, ignore_bad=True, xt_predict=None):
        """
        Build a PyMC3 model to fit TTV observed-minus-calculated data
        Fits data with a single-frequency sinusoid

        Arguments
        ----------
        period : float
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
        
        # convenience function
        def _sin_fxn(A, B, f, xt):
                return A * T.sin(2 * pi * f * xt) + B * T.cos(2 * pi * f * xt)

        # build pymc model
        with pm.Model() as model:
            df = 1 / (self.xtime[q].max() - self.xtime[q].min())
            f = pm.Normal("f", mu=1 / period, sd=df)
            Ah = pm.Normal("Ah", mu=0, sd=5 * np.std(self.yomc[q]))
            Bk = pm.Normal("Bk", mu=0, sd=5 * np.std(self.yomc[q]))

            trend = pm.Deterministic("trend", _sin_fxn(Ah, Bk, f, self.xtime[q]))
            pred = pm.Deterministic("pred", _sin_fxn(Ah, Bk, f, xt_predict))

            pm.Normal("obs", mu=trend, sd=self.yerr[q], observed=self.yomc[q])

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
        ymax = np.sqrt(mad_std(self.yomc) ** 2 - np.median(self.yerr) ** 2)

        # pymc model
        with pm.Model() as model:
            # build the kernel
            log_sigma = pm.Bound(pm.Normal, upper=np.log(ymax))(
                "log_sigma", mu=np.log(ymax), sd=5
            )

            rho = pm.Uniform("rho", lower=2 * dx, upper=self.xtime[q].max() - self.xtime[q].min())
            kernel = GPterms.Matern32Term(sigma=T.exp(log_sigma), rho=rho)
            mean = pm.Normal("mean", mu=np.mean(self.yomc[q]), sd=np.std(self.yomc[q]))
            
            # define the GP and factorize the covariance matrix
            gp = GaussianProcess(kernel, t=self.xtime[q], yerr=self.yerr[q], mean=mean)
            gp.compute(self.xtime[q], yerr=self.yerr[q])

            # track GP prediction, covariance, and degrees of freedom
            trend, cov = gp.predict(self.yomc[q], self.xtime[q], return_cov=True)

            trend = pm.Deterministic("trend", trend)
            cov = pm.Deterministic("cov", cov)
            pred = pm.Deterministic("pred", gp.predict(self.yomc[q], xt_predict))
            dof = pm.Deterministic("dof", pm.math.trace(cov / self.yerr[q]**2))

            # add marginal likelihood to model
            lnlike = gp.log_likelihood(self.yomc[q])
            pm.Potential("lnlike", lnlike)

        return model

    
    def sample(self, model, progressbar=False):
        with model:
            map_soln = pmx.optimize(start=model.test_point, 
                                    progress=progressbar
                                    )
            
            trace = pmx.sample(tune=8000,
                               draws=2000,
                               chains=2,
                               target_accept=0.95,
                               start=map_soln,
                               progressbar=progressbar
                              )
        return trace
        
