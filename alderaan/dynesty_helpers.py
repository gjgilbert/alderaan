import batman
from   batman import _rsky
from   batman import _quadratic_ld
from   celerite2 import GaussianProcess
from   celerite2 import terms as GPterms
import matplotlib.pyplot as plt
import numpy as np
from   scipy import stats
from   scipy.special import erfinv

from .Ephemeris import Ephemeris
from .constants import scit


__all__ = ['prior_transform',
           'lnlike'
          ]


def uniform_ppf(u, a, b):
    return u*(b-a) + a


def loguniform_ppf(u, a, b):
    return np.exp(u*np.log(b) + (1-u)*np.log(a))


def norm_ppf(u, mu, sig, eps=1e-12):
    '''
    eps provides numerical stability in the error function at edges u=[0,1]
    '''
    return mu + sig*np.sqrt(2)*erfinv((2*u-1)/(1+eps))


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


def lnlike(x, num_planets, theta, ephem_args, phot_args, ld_priors, gp_kernel=None):
    # extract ephemeris kwargs
    inds = ephem_args['transit_inds']
    tts  = ephem_args['transit_times']
    legx = ephem_args['transit_legx']
        
    # calculate physical limb darkening (see Kipping 2013)
    q1, q2 = np.array(x[-2:])
    u1 = 2*np.sqrt(q1)*q2
    u2 = np.sqrt(q1)*(1-2*q2)
    
    # set planet paramters
    for npl in range(num_planets):
        C0, C1, rp, b, T14 = np.array(x[5*npl:5*(npl+1)])

        # update ephemeris
        #A = np.ones((len(inds[npl]),2))
        #A[:,0] = inds[npl]
        
        #P, t0 = np.linalg.lstsq(A, tts[npl] + C0 + C1*legx[npl], rcond=None)[0]
        P = theta[npl].per
        
        # update transit parameters
        theta[npl].per = P
        theta[npl].t0  = 0.            # t0 must be set to zero b/c we are warping TTVs
        theta[npl].rp  = rp
        theta[npl].b   = b
        theta[npl].T14 = T14
        theta[npl].u   = [u1,u2]
        theta[npl].limb_dark = 'quadratic'

    # calculate likelihood
    loglike = 0.

    for j, q in enumerate(phot_args['quarters']):
        f_ = phot_args['flux'][j]
        e_ = phot_args['error'][j]
        light_curve = np.ones(len(f_), dtype='float')
        
        for npl in range(num_planets):
            t_ = phot_args['warped_t'][npl][j]
            x_ = phot_args['warped_x'][npl][j]
            C0 = x[5*npl]
            C1 = x[5*npl+1]
            
            t_ = t_ + C0 + C1*x_

            exp_time = phot_args['exptime'][q]
            texp_offsets = phot_args['texp_offsets'][q]
            supersample_factor = phot_args['oversample'][q]
            t_supersample = (texp_offsets + t_.reshape(t_.size, 1)).flatten()

            nthreads = 1
            ds = _rsky._rsky(t_supersample, 
                             theta[npl].t0, 
                             theta[npl].per, 
                             theta[npl].rp,
                             theta[npl].b, 
                             theta[npl].T14, 
                             1, 
                             nthreads)
            
            # look into the transit type argument
            qld_flux = _quadratic_ld._quadratic_ld(ds, np.abs(theta[npl].rp), theta[npl].u[0], theta[npl].u[1], nthreads)
            qld_flux = np.mean(qld_flux.reshape(-1, supersample_factor), axis=1) # PERF can probably speed this up.
            light_curve += qld_flux - 1.0


        USE_GP = False
        if USE_GP:
            gp = GaussianProcess(gp_kernel[q%4], mean=light_curve)
            gp.compute(t_, yerr=e_)
            loglike += gp.log_likelihood(f_)
        else:
            loglike += -0.5*np.sum( ((light_curve - f_)/e_)**2 )
        
    # enforce prior on limb darkening
    U1, U2 = ld_priors
    sig_ld_sq = 0.01
    loglike -= 1./(2*sig_ld_sq) * (u1 - U1)**2
    loglike -= 1./(2*sig_ld_sq) * (u2 - U2)**2

    if not np.isfinite(loglike):
        return -1e300

    return loglike
