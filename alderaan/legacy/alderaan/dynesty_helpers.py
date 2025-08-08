__all__ = ["uniform_ppf", "loguniform_ppf", "norm_ppf", "prior_transform", "lnlike"]


from batman import _rsky
from batman import _quadratic_ld
from celerite2 import GaussianProcess
import numpy as np
from scipy.special import erfinv

from .constants import *


def uniform_ppf(u, a, b):
    """
    Transform from U(0,1) --> Uniform(a,b)

    Parameters
    ----------
        u : array-like
            samples from standard uniform distribution U(0,1)
        a : float
            lower bound on transformed distribution U(a,b)
        b : float
            upper bound on transformed distribution U(a,b)

    Returns
    -------
        u_t : array-like
            transformed samples from uniform distribution
    """
    return u * (b - a) + a


def loguniform_ppf(u, a, b):
    """
    Transform from U(0,1) --> LogUniform(a,b)

    Parameters
    ----------
        u : array-like
            samples from standard uniform distribution U(0,1)
        a : float
            lower bound on transformed distribution lnU(a,b)
        b : float
            upper bound on transformed distribution lnU(a,b)

    Returns
    -------
        u_t : array-like
            transformed samples from log-uniform distribution
    """
    return np.exp(u * np.log(b) + (1 - u) * np.log(a))


def norm_ppf(u, mu, sig, eps=1e-12):
    """
    Transform from U(0,1) --> Normal(mu,sig)

    Parameters
    ----------
        u : array-like
            samples from standard uniform distribution U(0,1)
        mu : float
            mean of transformed distribution N(mu,sig)
        sig : float
            standard deviation of transformed distribution N(mu,sig)
        eps : float (optional)
            provides numerical stability in the inverse error function at edges

    Returns
    -------
        u_t : array-like
            transformed samples from Normal distribution
    """
    return mu + sig * np.sqrt(2) * erfinv((2 * u - 1) / (1 + eps))


def prior_transform(uniform_hypercube, num_planets, durations):
    """
    Prior transform over physical ALDERAAN basis {C0, C1, r, b, T14}...{q1,q2}

    Distributions are hard-coded to be:
        Normal on ephemeris perturbations {C0,C1}
        Log-uniform on radius ratio r and transit duration T14
        Uniform on impact parameter b
        Uniform on quadratic limb darkening coefficients {q1,q2}

    For motivation behind this prior-parameter choice see:
        Carter+ 2008 (2008ApJ...689..499C)
        Kipping 2013 (2013MNRAS.435.2152K)
        Gilbert, MacDougall, & Petigura 2022 (2022AJ....164...92G)
        MacDougal, Gilbert, & Petigura 2023 (2023AJ....166...61M)

    Parameters
    ----------
        uniform_hypercube : array-like, length 5xN + 2
            list of parameters N x {C0, C1, r, b, T14} + {q1,q2}
        num_planets : int
            number of planets in system
        durations : array-like
            list of transit durations for N planets (used for setting prior limits)

    Returns
    -------
        transformed_hypercube : array_like, length 5xN + 2
            transformed samples
    """
    if num_planets != len(durations):
        raise ValueError("input durations must match provided num_planets")

    u_ = np.array(uniform_hypercube)
    x_ = np.zeros_like(u_)

    # 5*num_planets (+2) parameters: {C0, C1, r, b, T14}...{q1,q2}
    for npl in range(num_planets):
        x_[5 * npl + 0] = norm_ppf(u_[0 + npl * 5], 0.0, 0.1)
        x_[5 * npl + 1] = norm_ppf(u_[1 + npl * 5], 0.0, 0.1)
        x_[5 * npl + 2] = loguniform_ppf(u_[2 + npl * 5], 1e-5, 0.99)
        x_[5 * npl + 3] = uniform_ppf(u_[3 + npl * 5], 0.0, 1 + x_[5 * npl + 2])
        x_[5 * npl + 4] = loguniform_ppf(u_[4 + npl * 5], scit, 3 * durations[npl])

    # limb darkening coefficients (see Kipping 2013)
    x_[-2] = uniform_ppf(u_[-2], 0, 1)
    x_[-1] = uniform_ppf(u_[-1], 0, 1)

    return x_


def lnlike(x, num_planets, theta, ephem_args, phot_args, ld_priors, gp_kernel=None):
    """
    Log-likelihood function to be passed to dynesty sampler

    *** Much of this function is redundant and can be cleaned up ***

    Parameters
    ----------
        x : array-like
            N x [C0, C1, r, b, T14] + [q1,q2]
        num_planets : int
            number of planets in the system
        theta : batman.TransitParams() object

        ephem_args : dict
            arguments related to the ephemeris
        phot_args : dict
            arguments related to the photometry
        ld_priors : tuple
            precomputed (U1,U2) values for limb darkening; used to set priors
        gp_kernel : celerite kernel
            pre-initialized celerite Gaussian Process kernel (optional)

    Returns
    -------
        loglike : float
            log-likelihood

    """
    # extract ephemeris kwargs
    inds = ephem_args["transit_inds"]
    tts = ephem_args["transit_times"]
    legx = ephem_args["transit_legx"]

    # calculate physical limb darkening (see Kipping 2013)
    q1, q2 = np.array(x[-2:])
    u1 = 2 * np.sqrt(q1) * q2
    u2 = np.sqrt(q1) * (1 - 2 * q2)

    # set planet paramters
    for npl in range(num_planets):
        C0, C1, rp, b, T14 = np.array(x[5 * npl : 5 * (npl + 1)])

        theta[npl].rp = rp
        theta[npl].b = b
        theta[npl].T14 = T14
        theta[npl].u = [u1, u2]
        theta[npl].limb_dark = "quadratic"

    # calculate likelihood
    loglike = 0.0

    for j, q in enumerate(phot_args["quarters"]):
        f_ = phot_args["flux"][j]
        e_ = phot_args["error"][j]
        light_curve = np.ones(len(f_), dtype="float")

        for npl in range(num_planets):
            t_ = phot_args["warped_t"][npl][j]
            x_ = phot_args["warped_x"][npl][j]
            C0 = x[5 * npl]
            C1 = x[5 * npl + 1]

            t_ = t_ + C0 + C1 * x_

            exp_time = phot_args["exptime"][q]
            texp_offsets = phot_args["texp_offsets"][q]
            supersample_factor = phot_args["oversample"][q]
            t_supersample = (texp_offsets + t_.reshape(t_.size, 1)).flatten()

            nthreads = 1
            ds = _rsky._rsky(
                t_supersample,
                theta[npl].t0,
                theta[npl].per,
                theta[npl].rp,
                theta[npl].b,
                theta[npl].T14,
                1,
                nthreads,
            )

            # look into the transit type argument
            qld_flux = _quadratic_ld._quadratic_ld(
                ds, np.abs(theta[npl].rp), theta[npl].u[0], theta[npl].u[1], nthreads
            )
            qld_flux = np.mean(
                qld_flux.reshape(-1, supersample_factor), axis=1
            )  # PERF can probably speed this up.
            light_curve += qld_flux - 1.0

        USE_GP = False
        if USE_GP:
            gp = GaussianProcess(gp_kernel[q % 4], mean=light_curve)
            gp.compute(t_, yerr=e_)
            loglike += gp.log_likelihood(f_)
        else:
            loglike += -0.5 * np.sum(((light_curve - f_) / e_) ** 2)

    # enforce prior on limb darkening
    U1, U2 = ld_priors
    sig_ld_sq = 0.01
    loglike -= 1.0 / (2 * sig_ld_sq) * (u1 - U1) ** 2
    loglike -= 1.0 / (2 * sig_ld_sq) * (u2 - U2) ** 2

    if not np.isfinite(loglike):
        return -1e300

    return loglike
