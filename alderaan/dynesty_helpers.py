import batman
from   celerite2 import GaussianProcess
from   celerite2 import terms as GPterms
import numpy as np
from   scipy import stats

from .Ephemeris import Ephemeris
from .constants import *

__all__ = ['prior_transform',
           'lnlike'
          ]


# functions for dynesty (hyperparameters are currently hard-coded)
def prior_transform(uniform_hypercube, num_planets, durations):
    if num_planets != len(durations):
        raise ValueError("input durations must match provided num_planets")
    
    x_ = np.array(uniform_hypercube)

    # 5*NPL (+2) parameters: {C0, C1, r, b, T14}...{q1,q2}
    dists = []
    for npl in range(num_planets):
        C0  = stats.norm(loc=0., scale=0.1).ppf(x_[0+npl*5])
        C1  = stats.norm(loc=0., scale=0.1).ppf(x_[1+npl*5])
        r   = stats.loguniform(1e-5, 0.99).ppf(x_[2+npl*5])
        b   = stats.uniform(0., 1+r).ppf(x_[3+npl*5])
        T14 = stats.loguniform(scit, 3*durations[npl]).ppf(x_[4+npl*5])
        
        dists = np.hstack([dists, [C0, C1, r, b, T14]])
         
    # limb darkening coefficients (see Kipping 2013)
    q1 = stats.uniform(0., 1.).ppf(x_[5*num_planets+0])
    q2 = stats.uniform(0., 1.).ppf(x_[5*num_planets+1])
    
    return np.hstack([dists, [q1,q2]])



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