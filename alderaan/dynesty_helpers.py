import batman
from   celerite2 import GaussianProcess
from   celerite2 import terms as GPterms
import numpy as np
from scipy import stats
from scipy.special import erfinv

from .Ephemeris import Ephemeris
from .constants import *

__all__ = ['prior_transform',
           'lnlike'
          ]


def uniform_ppf(u, a, b):
    return u*(b-a) + a


def loguniform_ppf(u, a, b):
    return np.exp(u*np.log(b) + (1-u)*np.log(a))


def norm_ppf(u, mu, sig):
    return mu + sig*np.sqrt(2)*erfinv(2*u-1)


# functions for dynesty (hyperparameters are currently hard-coded)
def prior_transform(uniform_hypercube, num_planets, durations):
    if num_planets != len(durations):
        raise ValueError("input durations must match provided num_planets")
    
    u_ = np.array(uniform_hypercube)
    x_ = np.zeros_like(u_)
    
    # 5*num_planets (+2) parameters: {C0, C1, r, b, T14}...{q1,q2}
    for npl in range(num_planets):
        x_[5*npl+0] = norm_ppf(u_[0+npl*5], 0., 0.1)
        x_[5*npl+1] = norm_ppf(u_[1+npl*5], 0., 0.1)
        x_[5*npl+2] = loguniform_ppf(u_[2+npl*5], 1e-5, 0.99)
        x_[5*npl+3] = uniform_ppf(u_[3+npl*5], 0., 1+x_[5*npl+2])
        x_[5*npl+4] = loguniform_ppf(u_[4+npl*5], scit, 3*durations[npl])
                 
    # limb darkening coefficients (see Kipping 2013)
    x_[-2] = uniform_ppf(u_[-2], 0, 1)
    x_[-1] = uniform_ppf(u_[-1], 0, 1)
    
    return x_


def lnlike(x, num_planets, theta, transit_model, quarters, ephem_args, phot_args, gp_kernel, ld_priors):
    # extract ephemeris data
    ephem  = ephem_args['ephem']
    inds   = ephem_args['transit_inds']
    ttimes = ephem_args['transit_times']
    Leg0   = ephem_args['Leg0']
    Leg1   = ephem_args['Leg1']
    
    # extract photometry data
    t_ = phot_args['time']
    f_ = phot_args['flux']
    e_ = phot_args['error']
    m_ = phot_args['mask']
        
    # set limb darkening (see Kipping 2013)
    q1, q2 = np.array(x[-2:])

    u1 = 2*np.sqrt(q1)*q2
    u2 = np.sqrt(q1)*(1-2*q2)
    
    # set planet paramters
    for npl in range(num_planets):
        C0, C1, rp, b, T14 = np.array(x[5*npl:5*(npl+1)])

        # update ephemeris
        ephem[npl] = Ephemeris(inds[npl], ttimes[npl] + C0*Leg0[npl] + C1*Leg1[npl])
    
        # update transit parameters
        theta[npl].per = ephem[npl].period
        theta[npl].t0  = 0.                     # t0 must be set to zero b/c we are warping TTVs
        theta[npl].rp  = rp
        theta[npl].b   = b
        theta[npl].T14 = T14
        theta[npl].u   = [u1,u2]

    # calculate likelihood
    loglike = 0.

    for j, q in enumerate(quarters):
        light_curve = np.ones(len(t_[j]), dtype='float')
        
        for npl in range(num_planets):
            transit_model[npl][j].t = ephem[npl]._warp_times(t_[j])
            light_curve += transit_model[npl][j].light_curve(theta[npl]) - 1.0

        USE_GP = False
        if USE_GP:
            gp = GaussianProcess(gp_kernel[q%4], mean=light_curve)
            gp.compute(t_[j], yerr=e_[j])
            loglike += gp.log_likelihood(f_[j])
        else:
            loglike += np.sum(-np.log(e_[j]) - 0.5*((light_curve - f_[j]) / e_[j])**2)
        
        # enforce prior on limb darkening
        U1, U2 = ld_priors
        sig_ld_sq = 0.1**2
        loglike += -0.5*np.log(2*pi*sig_ld_sq) - 1./(2*sig_ld_sq) * (u1 - U1)**2
        loglike += -0.5*np.log(2*pi*sig_ld_sq) - 1./(2*sig_ld_sq) * (u2 - U2)**2

    if not np.isfinite(loglike):
        return -1e300

    return loglike