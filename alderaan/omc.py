__all__ = [
    "load_holczer_ttvs",
    "clean_holczer_ttvs",
    "match_holczer_ttv",
    "matern32_model",
    "poly_model",
    "sin_model",
    "mix_model",
    "flag_outliers",
]

import aesara_theano_fallback.tensor as T
from astropy.stats import mad_std
from celerite2.theano import GaussianProcess
from celerite2.theano import terms as GPterms
import pymc3 as pm
import pymc3_ext as pmx
import numpy as np
import numpy.polynomial.polynomial as poly
from scipy import stats
from scipy.ndimage import median_filter
from sklearn.cluster import KMeans


from .constants import *
from .utils import boxcar_smooth


def load_holczer_ttvs(file, npl, koi_id):
    """
    Docstring
    """
    data = np.loadtxt(file, usecols=[0, 1, 2, 3, 4])

    output = {}
    output["inds"] = [None] * npl
    output["tts"] = [None] * npl
    output["err"] = [None] * npl

    for n in range(npl):
        koi = int(koi_id[1:]) + 0.01 * (1 + n)
        use = np.isclose(data[:, 0], koi, rtol=1e-10, atol=1e-10)

        # Holczer uses BJD - 2454900; BJKD = BJD - 2454833
        if np.sum(use) > 0:
            output["inds"][n] = np.array(data[use, 1], dtype="int")
            output["tts"][n] = data[use, 2] + data[use, 3] / 24 / 60 + 67
            output["err"][n] = data[use, 4] / 24 / 60

    return output


def clean_holczer_ttvs(data, time_start, time_end, verbose=False):
    """
    Docstring
    """
    npl = len(data["tts"])

    data["period"] = np.nan * np.ones(npl)
    data["epoch"] = np.nan * np.ones(npl)
    data["full_inds"] = [None] * npl
    data["full_tts"] = [None] * npl
    data["out"] = [None] * npl

    for n in range(npl):
        if data["tts"][n] is not None:
            # fit a linear ephemeris
            pfit = poly.polyfit(data["inds"][n], data["tts"][n], 1)
            ephem = poly.polyval(data["inds"][n], pfit)

            # put fitted epoch in range (TIME_START, TIME_START + PERIOD)
            epoch, period = pfit

            if epoch < time_start:
                adj = 1 + (time_start - epoch) // period
                epoch += adj * period

            if epoch > (time_start + period):
                adj = (epoch - time_start) // period
                epoch -= adj * period

            data["period"][n] = np.copy(period)
            data["epoch"][n] = np.copy(epoch)

            full_ephem = np.arange(epoch, time_end, period)
            full_inds = np.array(np.round((full_ephem - epoch) / period), dtype="int")

            # calculate OMC and flag outliers
            xtime = np.copy(data["tts"][n])
            yomc = np.copy(data["tts"][n] - ephem)
            yerr = np.copy(data["err"][n])

            if len(yomc) > 16:
                ymed = boxcar_smooth(
                    median_filter(yomc, size=5, mode="mirror"), winsize=5
                )
            else:
                ymed = np.median(yomc)

            if len(yomc) > 4:
                out = np.abs(yomc - ymed) / mad_std(yomc - ymed) > 3.0
            else:
                out = np.zeros(len(yomc), dtype="bool")

            data["out"][n] = np.copy(out)

            # estimate TTV signal with a regularized Matern-3/2 GP
            if np.sum(~out) > 4:
                model = matern32_model(
                    xtime[~out], yomc[~out], yerr[~out], xt_predict=full_ephem
                )
            else:
                model = poly_model(
                    xtime[~out], yomc[~out], yerr[~out], 1, xt_predict=full_ephem
                )

            with model:
                map_soln = pmx.optimize(verbose=verbose)

            full_tts = full_ephem + map_soln["pred"]

            # track model prediction
            data["full_tts"][n] = np.copy(full_tts)
            data["full_inds"][n] = np.copy(full_inds)

    return data


def match_holczer_ttvs(planets, data):
    durs = np.zeros(len(planets))

    for n, p in enumerate(planets):
        durs[n] = p.duration

    for n, p in enumerate(planets):
        match = np.isclose(data["period"], p.period, rtol=0.1, atol=durs.max())

        if not ((np.sum(match) == 0) or (np.sum(match) == 1)):
            raise ValueError(
                "Something has gone wrong matching periods between DR25 and Holczer+2016"
            )

        if np.sum(match) == 1:
            loc = np.squeeze(np.where(match))

            hinds = data["inds"][loc]
            htts = data["tts"][loc]

            p.epoch = np.copy(data["epoch"][loc])
            p.period = np.copy(data["period"][loc])
            p.tts = np.copy(data["full_tts"][loc])
            p.index = np.copy(data["full_inds"][loc])

    return planets


def matern32_model(xtime, yomc, yerr, ymax=None, xt_predict=None):
    """
    Build a PyMC3 model to fit TTV observed-minus-calculated data
    Fits data with a regularized Matern-3/2 GP kernel

    Units (time) should be the same on all inputs

    Parameters
    ----------
    xtime : ndarray
        time values (e.g. linear ephemeris)
    yomc : ndarray
        observed-minus-caculated TTVs
    yerr : ndarray
        corresponding uncertainties on yomc
    ymax : float (optional)
        maximum allowed amplitude on GP scale parameter
    xt_predict : ndarray (optional)
        time values to predict OMC model; if not provided xtime will be used

    Returns
    -------
    model : pm.Model()
    """
    with pm.Model() as model:
        # delta between each transit time
        dx = np.mean(np.diff(xtime))

        # times where trend will be predicted
        if xt_predict is None:
            xt_predict = xtime

        # build the kernel
        if ymax is None:
            log_sigma = pm.Normal("log_sigma", mu=np.log(np.mean(yerr)), sd=5)
        else:
            log_sigma = pm.Bound(pm.Normal, upper=np.log(ymax))(
                "log_sigma", mu=np.log(ymax), sd=5
            )

        rho = pm.Uniform("rho", lower=2 * dx, upper=xtime.max() - xtime.min())
        kernel = GPterms.Matern32Term(sigma=T.exp(log_sigma), rho=rho)

        # mean
        mean = pm.Normal("mean", mu=np.mean(yomc), sd=np.std(yomc))

        # define the GP
        gp = GaussianProcess(kernel, t=xtime, yerr=yerr, mean=mean)

        # factorize the covariance matrix
        gp.compute(xtime, yerr=yerr)

        # track GP prediction, covariance, and degrees of freedom
        trend, cov = gp.predict(yomc, xtime, return_cov=True)

        trend = pm.Deterministic("trend", trend)
        cov = pm.Deterministic("cov", cov)
        pred = pm.Deterministic("pred", gp.predict(yomc, xt_predict))

        dof = pm.Deterministic("dof", pm.math.trace(cov / yerr**2))

        # add marginal likelihood to model
        lnlike = gp.log_likelihood(yomc)
        pm.Potential("lnlike", lnlike)

    return model


def poly_model(xtime, yomc, yerr, polyorder, xt_predict=None):
    """
    Build a PyMC3 model to fit TTV observed-minus-calculated data
    Fits data with a polynomial (up to cubic)

    Parameters
    ----------
    xtime : ndarray
        time values (e.g. linear ephemeris)
    yomc : ndarray
        observed-minus-caculated TTVs
    yerr : ndarray
        corresponding uncertainties on yomc
    polyorder : int
        polynomial order
    xt_predict : ndarray
        time values to predict OMC model; if not provided xtime will be used

    Returns
    -------
    model : pm.Model()
    """
    with pm.Model() as model:
        # times where trend will be predicted
        if xt_predict is None:
            xt_predict = xtime

        C0 = pm.Normal("C0", mu=0, sd=10)
        C1 = 0.0
        C2 = 0.0
        C3 = 0.0

        if polyorder >= 1:
            C1 = pm.Normal("C1", mu=0, sd=10)
        if polyorder >= 2:
            C2 = pm.Normal("C2", mu=0, sd=10)
        if polyorder >= 3:
            C3 = pm.Normal("C3", mu=0, sd=10)
        if polyorder >= 4:
            raise ValueError("only configured for 3rd order polynomials")

        def poly_fxn(c0, c1, c2, c3, xt):
            return c0 + c1 * xt + c2 * xt**2 + c3 * xt**3

        # mean
        trend = pm.Deterministic("trend", poly_fxn(C0, C1, C2, C3, xtime))

        # likelihood
        pm.Normal("obs", mu=trend, sd=yerr, observed=yomc)

        # track predicted trend
        pred = pm.Deterministic("pred", poly_fxn(C0, C1, C2, C3, xt_predict))

    return model


def sin_model(xtime, yomc, yerr, period, xt_predict=None):
    """
    Build a PyMC3 model to fit TTV observed-minus-calculated data
    Fits data with a single-frequency sinusoid

    Parameters
    ----------
    xtime : ndarray
        time values (e.g. linear ephemeris)
    yomc : ndarray
        observed-minus-caculated TTVs
    yerr : ndarray
        corresponding uncertainties on yomc
    period : float
        pre-estimated sinusoid period; the model places tight priors on this period
    xt_predict : ndarray
        time values to predict OMC model; if not provided xtime will be used

    Returns
    -------
    model : pm.Model()
    """
    with pm.Model() as model:
        # times where trend will be predicted
        if xt_predict is None:
            xt_predict = xtime

        # periodic component
        df = 1 / (xtime.max() - xtime.min())
        f = pm.Normal("f", mu=1 / period, sd=df)
        Ah = pm.Normal("Ah", mu=0, sd=5 * np.std(yomc))
        Bk = pm.Normal("Bk", mu=0, sd=5 * np.std(yomc))

        def sin_fxn(A, B, f, xt):
            return A * T.sin(2 * pi * f * xt) + B * T.cos(2 * pi * f * xt)

        # mean
        trend = pm.Deterministic("trend", sin_fxn(Ah, Bk, f, xtime))

        # likelihood
        pm.Normal("obs", mu=trend, sd=yerr, observed=yomc)

        # track predicted trend
        pred = pm.Deterministic("pred", sin_fxn(Ah, Bk, f, xt_predict))

    return model


def mix_model(x):
    """
    Build a 1D PyMC3 mixture model
    The model is composed of two normal distributions with the same mean but different variances

    Parameters
    ----------
    x : ndarray
        vector of data values

    Returns
    -------
    model : pm.Model()
    """
    xnorm = x / np.std(x)
    xnorm -= np.mean(xnorm)

    with pm.Model() as model:
        # mixture parameters
        w = pm.Dirichlet("w", np.array([1.0, 1.0]))
        mu = pm.Normal("mu", mu=0.0, sd=1.0, shape=1)
        tau = pm.Gamma("tau", 1.0, 1.0, shape=2)

        # here's the potential
        obs = pm.NormalMixture("obs", w, mu=mu * T.ones(2), tau=tau, observed=xnorm)

    return model


def flag_outliers(res, loc, scales):
    """
    Flag outliers in a residuals vector using a mixture model
    Applies unsupervised K-means clustering to assign points to either a narrow foreground or wide background distribution

    Parameters
    ----------
    res : ndarray (N)
        vector of residuals
    loc : float
        normal mean inferred from mix_model()
    scales : tuple
        normal standard deviations inferred from mix_model()

    Returns
    -------
    fg_prob : ndarray (N)
        probability the each item in res belongs to the foreground distribution
    out : bool (N)
        binary classification of each item in res into foreground/background distribution
    """
    resnorm = res / np.std(res)
    resnorm -= np.mean(resnorm)

    order = np.argsort(scales)
    scales = scales[order]

    # calculate foreground/background probability from results of mixture model
    z_fg = stats.norm(loc=loc, scale=scales[0]).pdf(resnorm)
    z_bg = stats.norm(loc=loc, scale=scales[1]).pdf(resnorm)

    fg_prob = z_fg / (z_fg + z_bg)

    # use KMeans clustering to assign each point to the foreground or background
    km = KMeans(n_clusters=2)
    group = km.fit_predict(fg_prob.reshape(-1, 1))
    centroids = np.array([np.mean(fg_prob[group == 0]), np.mean(fg_prob[group == 1])])
    out = group == np.argmin(centroids)

    return fg_prob, out
