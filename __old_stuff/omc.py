import numpy as np
import scipy.optimize as op
import scipy.signal as sig
from   scipy import stats
from   scipy import fftpack
from   scipy import ndimage
import astropy.stats
import warnings
import theano.tensor as T

from .constants import *
from .utils import LS_estimator

__all__ = ["omc_model",
           "choose_omc_model"]


def omc_model(theta, x, n, k):
    """
    Calculates y = f(x) as a sum of an order n polynomial + k sinuosids
    
    y = SUM_n{ c_n*x**n } + SUM_k{ A_k*sin(2*pi*f_k*t) + B_k*cos(2*pi*f_k*t)
    
    Parameters
    ----------
    theta : list
        model parameters given in the order [c_n..., A_k, B_k, f_k...]
    x : ndarray
        1D data array of x-values for y = f(x)
    n : int
        polynomial order
    k : int
        number of sinusoidal frequences
    
    """
    if len(theta) != 1 + n + 3*k:
        raise ValueError("Inconsistent length = {0} of vector theta for specified values n = {1} and k = {2}" \
                         .format(len(theta), n, k))
        
    y = np.zeros_like(x)
    
    i = 0
    while i <= n:
        y += theta[i]*x**i
        i += 1
    
    if k > 0:
        j = 0
        while j < k:
            A = theta[n+j+1]
            B = theta[n+j+2]
            f = theta[n+j+3]
            
            y += A*np.sin(2*pi*f*x) + B*np.cos(2*pi*f*x)
            j += 1
    
    return y



def choose_omc_model(xtime, yomc, n=3):
    """
    Automatically select best OMC model using the Bayesian Information Criterion (BIC)
    
    y = f(x) is a sum of an order n polynomial + k sinuosids (k <= 2)
    
    y = SUM_n{ c_n*x**n } + SUM_k{ A_k*sin(2*pi*f_k*t) + B_k*cos(2*pi*f_k*t)
    
    Parameters
    ----------
    xtime : ndarray
        1D data array of time for y = f(x)
    yomc : ndarray
        1D data array of OMC for y = f(x)
    n : int
        maximum polynomial order (optional; default=3)
    
    """    # flag outliers
    ymed = ndimage.median_filter(yomc, size=5, mode="mirror")
    out  = np.abs(yomc-ymed)/astropy.stats.mad_std(yomc-ymed) > 5.0
    npts = np.sum(~out)
    
    # identify significant frequencies via Lomb-Scargle periodigram    
    xf, yf, freqs, faps = LS_estimator(xtime[~out], yomc[~out])
    
    BICs = []
    NKs  = []
    
    
    # residual function for op.leastsq
    def res_fxn(theta, x, y, n, k):
        return y - omc_model(theta, x, n, k)
    

    # models with Nth order polynomial and 0 sinusoidal frequencies
    for p in range(1,n+1):
        theta, success = op.leastsq(res_fxn, [0]*(p+1), args=(xtime[~out], yomc[~out], p, 0))
        rss = np.sum((yomc[~out] - omc_model(theta, xtime[~out], p, 0))**2)
        
        BICs.append(npts*np.log(rss/npts) + (1+p)*np.log(npts))
        NKs.append((p,0))
    
        # models with Nth order polynomials and 1 sinusoidal frequencies
        if (len(freqs) > 0)*(p <= 3):
            theta_guess = [0]*(p+1) + [np.std(yomc[~out]), np.std(yomc[~out]), freqs[0]]
            theta, success = op.leastsq(res_fxn, theta_guess, args=(xtime[~out], yomc[~out], p, 1))
            rss = np.sum((yomc[~out] - omc_model(theta, xtime[~out], p, 1))**2)

            BICs.append(npts*np.log(rss/npts) + (4+p)*np.log(npts))
            NKs.append((p,1))
            
    
    # identify the lowest BIC and select best model
    N,K = NKs[np.argmin(BICs)]
    
    theta_guess = [0]*(N+1) 
    if K == 1:
        theta_guess += [np.std(yomc[~out]), np.std(yomc[~out]), freqs[0]]
    theta_fit, success = op.leastsq(res_fxn, theta_guess, args=(xtime[~out], yomc[~out], N, K))
    
    
    # check for any remaining periodic signals in the residuals; if found fit new models w/ two frequencies
    yres = yomc-omc_model(theta_fit, xtime, N, K)
    
    res_xf, res_yf, res_freqs, res_faps = LS_estimator(xtime[~out], yres[~out])
    
    if len(res_freqs) > 0:
        for p in range(1,n+1):
            theta_guess = [0]*(p+1)
            theta_guess += [np.std(yomc[~out]), np.std(yomc[~out]), freqs[0]]
            theta_guess += [np.std(yres[~out]), np.std(yres[~out]), res_freqs[0]]
            
            theta, success = op.leastsq(res_fxn, theta_guess, args=(xtime[~out], yomc[~out], p, 2))
            rss = np.sum((yomc[~out] - omc_model(theta, xtime[~out], p, 2))**2)

            BICs.append(npts*np.log(rss/npts) + (7+p)*np.log(npts))
            NKs.append((p,2))
            
        
        N,K = NKs[np.argmin(BICs)]

        theta_guess = [0]*(N+1) 
        if K == 1:
            theta_guess += [np.std(yomc[~out]), np.std(yomc[~out]), freqs[0]]
        if K == 2:
            theta_guess += [np.std(yres[~out]), np.std(yres[~out]), res_freqs[0]]
            
        theta_fit, success = op.leastsq(res_fxn, theta_guess, args=(xtime[~out], yomc[~out], N, K))
    
    
    # output array ordered as [N, Kc0, c1,...c_n, A1, B1, f1, A2, B2, f2]
    return np.hstack([N, K, theta_fit])