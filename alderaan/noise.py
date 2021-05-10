import numpy as np
import scipy.optimize as op
import scipy.signal as sig
from   scipy import stats
import astropy
import warnings

import lightkurve as lk
import exoplanet as exo
import theano.tensor as T
import pymc3 as pm
import corner

import matplotlib.pyplot as plt

from .constants import *
from .utils import *


__all__ = ["make_chunklist",
           "generate_acf",
           "convert_frequency",
           "build_sho_model",
           "make_covariance_matrix",
           "model_acf",
           "generate_synthetic_noise",
           "make_gp_prior_dict"]



def make_chunklist(time, flux, cadno, Npts, sigma_reject=5.0, gap_tolerance=15, interpolate=True, cover=0.95):
    """
    Make an array of 'chunks' of data uninterupted by transits or data gaps
    
    Parameters
    ----------
    time : array-like
        out-of-transit time values (i.e. transits masked BEFORE passing array into make_chucklist)
    flux : array-like
        out-of-transit flux values
    cadno : array-like
        out-of-transit cadence numbers
    Npts : int
        number of points to use in each chunk; should be ~3x max transit duration
    sigma_reject : float
        sigma threshold for rejection of noisy chunks (default=5.0)
    gap_tolerance : int
        maximum number of consecutive missing cadences allowed (default=15)
    interpolate : bool
        True to perform linear interpolation over small gaps (default=True)
    cover : float between (0,1)
        fractional coverage required to consider a chunk "good"
        
    Returns
    -------
    chunklist : list
        M x N list of data 'chunks' uninterrupted by transits or data gaps
    """
    chunklist = []
    
    i = 0
    loop = True
    while(loop):
        # mark start and end cadence for this chunk
        cad_low  = cadno[i]
        cad_high = cad_low + Npts + 1

        use = (cadno >= cad_low)*(cadno < cad_high)

        # check if there are gaps
        index = cadno[use] - cad_low
        gaps  = np.hstack([1,index[1:]-index[:-1]])
                
        # make sure there are no long stretches of consecutive missing cadences
        if gaps.max() <= gap_tolerance:
            # pull time and flux chunks
            t_chunk = np.ones(Npts+1)*99      # use '99' to mark missing cadences
            f_chunk = np.ones(Npts+1)*99

            t_chunk[index] = time[use]
            f_chunk[index] = flux[use]

            # interpolate over missing cadences
            empty = t_chunk == 99

            t_chunk[empty] = np.interp(np.arange(Npts+1)[empty], index, t_chunk[~empty])
            f_chunk[empty] = np.interp(np.arange(Npts+1)[empty], index, f_chunk[~empty])
            f_chunk[empty] += np.random.normal(size=np.sum(empty))*np.std(f_chunk[~empty])
                        
            # require at least X% coverage (default = 95%)
            if np.sum(~empty)/len(empty) > cover:
                chunklist.append(f_chunk)

            i += Npts
            
        # if there are missing cadences, move forward past any gaps
        else:
            i += np.argmax(gaps > gap_tolerance)
        
        # finish the loop if there is no more unusued flux
        if i > (len(time)-Npts-1):
            loop = False
    
    
    # convert list to array
    chunklist = np.array(chunklist)
        
    
    # reject any chunks with unusually high medians or variability
    loop = True
    while(loop):
        mad_chunk = astropy.stats.mad_std(chunklist, axis=1)
        med_chunk = np.median(chunklist, axis=1)

        bad = np.abs(mad_chunk - np.median(mad_chunk))/astropy.stats.mad_std(mad_chunk) > sigma_reject
        bad += np.abs(med_chunk - np.median(med_chunk))/astropy.stats.mad_std(med_chunk) > sigma_reject

        chunklist = chunklist[~bad]

        if np.sum(bad) == 0:
            loop = False
            
            
    return chunklist



def generate_acf(time, flux, cadno, Npts, sigma_reject=5.0):
    """
    Generate an autocorrelation function from a collection of out-of-transit data 'chunks'
    
    Parameters
    ----------
    time : array-like
        out-of-transit time values (i.e. transits masked BEFORE passing array into make_chucklist)
    flux : array-like
        out-of-transit flux values
    cadno : array-like
        out-of-transit cadence numbers
    Npts : int
        number of points to use in each chunk; should be ~3x max transit duration
    sigma_reject : float
        sigma threshold for rejection of noisy chunks (default=5.0)
        
    Returns
    -------
    xcor : ndarray
        time-lag used to generate autocorrelation function with lag-zero value removed
    acor : ndarray
        autocorrelation function with lag-zero value removed
    wcor : ndarray
        corresponding weights (for now, all equal to the MAD_STD of acor)
    acf_stats : dict
        statistics describing the ACF       
    """
    chunklist = make_chunklist(time, flux, cadno, Npts, sigma_reject=sigma_reject)
    
    Nsamples = chunklist.shape[0]

    # generate the autocorrelation function
    acor = np.zeros((Nsamples, 2*Npts+1))

    for i in range(Nsamples):
        acor[i] = np.correlate(1-chunklist[i], 1-chunklist[i], mode='full')

    acor = np.median(acor, axis=0)
    acor = acor[Npts:]/acor[Npts]

    xcor = np.arange(Npts+1)
    wcor = np.ones(Npts+1)*astropy.stats.mad_std(acor[1:])

    # pull off the zero-lag value
    acor0 = acor[0]

    xcor = xcor[1:]
    acor = acor[1:]
    wcor = wcor[1:]
    
    
    # log statistics
    acf_stats = {}
    acf_stats['Nsamples'] = Nsamples
    acf_stats['points_per_sample'] = Npts
    acf_stats['acor0'] = acor0
    
    return xcor, acor, wcor, acf_stats



def convert_frequency(freq, Q):
    """
    Convert characteristic oscillation frequency to undamped oscillator frequency
    Follows Eq. 21 of Foreman-Mackey et al. 2017
    
    Parameters
    ----------
    freq : float
        characteristic frequency
    Q : theano variable
        quality factor in celerite SHOTerm; must have Q > 1/2
        
    Returns
    -------
    w0 : theano variable
        undamped oscillator frequency
    """
    return 2*Q*freq/T.sqrt(4*Q**2 - 1)



def build_sho_model(x, y, var_method, test_freq=None, fmin=None, fmax=None, fixed_Q=None):
    """
    
    Must specify either var or var_method
    Build PyMC3/exoplanet model for correlated noise using a sum of SHOTerms
    
    Parameters
    ----------
    x : array-like
        independent variable data (e.g. time)
    y : array-like
        corresponding dependent variable data (e.g. empirical ACF or flux)
    var_method : string
        automatic method for selecting y data variance
        'global' --> var = np.var(y)
        'local' --> var = np.var(y - local_trend)
        'fit' --> logvar is a free hyperparameter in the GP model
    test_freq : float (optional)
        an (ordinary, not angular) frequency to initialize the model
    fmin : float (optional)
        lower bound on (ordinary, not angular) frequency
    fmax : float (optional)
        upper bound on (ordinary, not angular) frequency
    fixed_Q : float (optional)
        a fixed value for Q
        
    Returns
    -------
    model : a pymc3 model
    
    """
    with pm.Model() as model:
        
        # amplitude parameter
        logSw4 = pm.Normal('logSw4', mu=np.log(np.var(y)), sd=15.0)
        
        
        # qualify factor
        if fixed_Q is not None:
            logQ = pm.Deterministic('logQ', T.log(fixed_Q))
        else:
            logQ = pm.Uniform('logQ',  lower=np.log(1/np.sqrt(2)), upper=np.log(100))
        
        
        # frequency; for Q > 0.7, the difference between standard and damped frequency is minimal
        if fmin is None and fmax is None:
            if test_freq is None:
                logw0 = pm.Normal('logw0', mu=0.0, sd=15.0)
            else:
                test_w0 = convert_frequency(2*pi*test_freq[0], T.exp(logQ))
                logw0 = pm.Normal('logw0', mu=np.log(test_w0), sd=np.log(1.1))
                
                
        if fmin is not None or fmax is not None:
            if fmin is None: logwmin = None
            else: logwmin = np.log(2*pi*fmin)
                
            if fmax is None: logwmax = None
            else: logwmax = np.log(2*pi*fmax)
            
            if test_freq is None:
                logw0 = pm.Bound(pm.Normal, lower=logwmin, upper=logwmax)('logw0', mu=0.0, sd=15.0)
                
            else:
                test_w0 = convert_frequency(2*pi*test_freq[0], T.exp(logQ))
                logw0 = pm.Bound(pm.Normal, lower=logwmin, upper=logwmax)('logw0', mu=np.log(test_w0), sd=np.log(1.1))
        
    
        # here's the kernel
        kernel = exo.gp.terms.SHOTerm(log_Sw4=logSw4, log_w0=logw0, log_Q=logQ)

            
        # set the variance
        if var_method == 'global':
            var = np.var(y)

        elif var_method == 'local':
            var = np.var(y - boxcar_smooth(y,29))
            
        elif var_method == 'fit':
            logvar = pm.Normal('logvar', mu=np.log(astropy.stats.mad_std(y)**2), sd=10.0)
            var = T.exp(logvar)
            
        else:
            raise ValueError("Must specify var_method as 'global', 'local', or 'fit'")


        # set up the GP
        gp = exo.gp.GP(kernel, x, var*T.ones(len(x)))

        # add custom potential (log-prob fxn) with the GP likelihood
        pm.Potential('obs', gp.log_likelihood(y))

        # track GP prediction
        gp_pred = pm.Deterministic('gp_pred', gp.predict())
        
        
    return model
    


def make_covariance_matrix(acf, size=None):
    """
    Generate a square 2D covariance matrix from a 1D autocorrelation function
    
    Parameters
    ----------
    acf : array-like
        1D autocorrelation function not including lag-zero term (length N)
    size : int
        size of output covariance matrix (optional; if not given size --> N+1 x N+1)
        
    Returns
    -------
    covmatrix : ndarray
        n x n array; diagonal terms all equal 1.0
    """
    N = len(acf)+1
    
    if size is None:
        n = N
    else:
        n = size
    
    if n > N:
        acf = np.hstack([acf, np.zeros(n-N)])

    covmatrix = np.zeros((n,n))
    
    for i in range(n):
        covmatrix[i,i+1:] = acf[:n-i-1]
    
    covmatrix += covmatrix.swapaxes(0,1)
    covmatrix += np.eye(n)
    
    return covmatrix



def model_acf(xcor, acor, fcut, fmin=None, fmax=None, crit_fap=0.003, method='smooth', window_length=None):
    """
    Model an empirical autocorrelation function (ACF) using one of several methods
    
    
    Parameters
    ----------
    xcor : array-like
        lag time values
    acor : array-like
        empirical autocorrelation function power at each time lag
    fcut : float
        cutoff value for seperating high vs. low frequencies
    fmin : float (optional)
        minimum frequency to check; if not provided this will be set to 1/baseline
    fmax : float (optional)
        maximum frequency to check; if not provided this will be set to the Nyquist frequency
    crit_fap : float
        critical false alarm probability for significant signals (default=0.003)
    method : string
        method to model low frequency component; either 'smooth', 'shoterm', or 'savgol'  (default='smooth')
    window_length : int
        size of boxcar smoothing window if method='smooth'; set automatically if not specified by user
        
        
    Returns
    -------
    acor_emp, acor_mod, xf, yf, freqs
    """
    # arrays to hold empirical and model ACF
    acor_mod = np.zeros_like(acor)
    acor_emp = acor.copy()
    
    # 1st model component (low frequency)
    xf_L, yf_L, freqs_L, faps_L = FFT_estimator(xcor, acor_emp, fmin=fmin, fmax=fmax, crit_fap=crit_fap)
    
    low_freqs = freqs_L[freqs_L < fcut]
    

    # model the ACF with chosen method
    if method == 'shoterm':
        xcor_mirror = np.hstack([-xcor[::-1], xcor])
        acor_mirror = np.hstack([acor_emp[::-1], acor_emp])
        
        model = build_sho_model(xcor_mirror, acor_mirror, var_method='fit')
        
        with model:
            map_soln = exo.optimize(start=model.test_point)
            
        acor_mod = map_soln['gp_pred'][len(acor_emp):]
        
    
    elif method == 'smooth':
        if window_length is None:
            if len(low_freqs) > 0:
                window_length = int(24*60/np.max(low_freqs)/2)
            else:
                window_length = int(len(acor_emp)/6)

            window_length = window_length + (window_length % 2) + 1
        
        acor_mirror = np.hstack([acor_emp[::-1], acor_emp])
        acor_mod = boxcar_smooth(acor_mirror, window_length)[len(acor_emp):]
            
    
    elif method == 'savgol':
        if len(low_freqs) > 0:
            window_length = int(24*60/np.max(low_freqs)/2)
        else:
            window_length = 59

        window_length = window_length + (window_length % 2) + 1
        
        acor_mirror = np.hstack([acor_emp[::-1], acor_emp])
        acor_mod = sig.savgol_filter(acor_mirror, polyorder=2, window_length=window_length)[len(acor_emp):]
        acor_mod = boxcar_smooth(acor_mod, 5)
    
    else:
        raise ValueError("method must be either 'shoterm', 'smooth', or 'savgol'")

      
    # check for any high-frequency components
    xf_H, yf_H, freqs_H, faps_H = FFT_estimator(xcor, acor_emp-acor_mod, fmin=fmin, fmax=fmax, crit_fap=crit_fap, max_peaks=5)
    
    high_freqs  = freqs_H[freqs_H > fcut]
    
    
    # combine freqs, return ACF and power spectrum
    freqs = np.hstack([low_freqs, high_freqs])
    
    return acor_emp, acor_mod, xf_L, yf_L, freqs



def generate_synthetic_noise(xcor, acor, n, sigma):
    """
    Generate synthetic correlated noise given a specified autorrelation function
    
    
    Parameters
    ----------
    xcor : array-like
        lag time values
    acor : array-like
        autocorrelation function power at each time lag
    n : int
        size of n x n covariance matrix
    sigma : float
        scale of white noise
    
    Returns
    -------
    x : ndarray
        1D array of time values (or some general independent coordinate)
    red_noise : ndarray
        synthetic correlated noise
    white_noise: ndarray
        gaussian noise vector used to generate red noise
    """
    # first make the covariance matrix
    covmatrix = make_covariance_matrix(acor, n)
    
    # decompose it
    try:
        L = np.linalg.cholesky(covmatrix)
    
    except:
        try:
            warnings.warn('Covariance matrix not positive definite...adjusting weights')
            
            # decompose for eigenvalues and eigenvectors
            # matrix was constructed to be symmetric - if eigh doesn't work there is a serious problem
            eigenvalues, eigenvectors = np.linalg.eigh(covmatrix)
            
            # diagonalize
            D = np.diag(eigenvalues)

            # elementwise comparison with zero
            Z = np.zeros_like(D)
            Dz = np.maximum(D,Z)

            # generate a positive semi-definite matrix
            psdm = np.dot(np.dot(eigenvectors,Dz),eigenvectors.T)

            # now make it positive definite
            eps = np.min(np.abs(acor[acor != 0]))*1e-6
            covmatrix = psdm + np.eye(n)*eps
            
            # renormalize
            covmatrix = covmatrix / covmatrix.max()
            
            # do Cholesky decomposition
            L = np.linalg.cholesky(covmatrix)
            

        except:
            warnings.warn('Covariance matrix fatally broken...returning identity matrix')
            covmatrix = np.eye(n)
            L = np.linalg.cholesky(covmatrix)
        
        
    # generate a vector of gaussian noise and remove any random covariance
    z = np.random.normal(size=n)*sigma
    
    # make correlated noise
    x = np.arange(n)*(xcor[1]-xcor[0])
    y = np.dot(L,z)
    
    return x, y-z, z



def make_gp_prior_dict(sho_trace, percs=[0.2, 2.3, 15.9, 50.0, 84.1, 97.7, 99.8]):
    """
    Generates a list of percentiles from posteriors for each hyperparameter of a GP noise model
    The expected sho_trace should be the output of a PyMC3/Exoplanet model built with noise.build_sho_model()
    
    Assumes a specific set of input variable names from sho_trace:
      - ['logw0', 'logSw4', 'logQ'] OR ['logw0_x', 'logSw4_x', 'logQ_x']
      - cannot have, e.g. both logw0 & logw0_x; both will be mapped to logw0
      
    Parameters
    ----------
    sho_trace : PyMC3 multitrace
        trace output of a PyMC3/Exoplanet model built with noise.build_sho_model()
    percs : list
        list of percentiles to return, by default 1- 2- 3-sigma and median
        
    Returns
    -------
    priors : dict
        Dictionary keys can be any combination of ['logw0', 'logSw4', 'logQ']
        Each key gives a list of values corresponding to specified percentiles from sho_trace
    """
    priors = {}
    priors['percentiles'] = percs
    
    varnames = sho_trace.varnames
    
    # check for redundancies
    if np.isin('logw0', varnames) & np.isin('logw0_x', varnames):
        raise ValueError('Expected only one of logw0 or logw0_x')
    if np.isin('logSw4', varnames) & np.isin('logSw4_x', varnames):
        raise ValueError('Expected only one of logSw4 or logSw4_x')
    if np.isin('logQ', varnames) & np.isin('logQ_x', varnames):
        raise ValueError('Expected only one of logQ or logQ_x')

        
    # get posterior percentiles of each hyperparameter and assign to a dictionary
    if np.isin('logw0', varnames):
        priors['logw0'] = np.percentile(sho_trace['logw0'], percs)
        
    if np.isin('logSw4', varnames):
        priors['logSw4'] = np.percentile(sho_trace['logSw4'], percs)

    if np.isin('logQ', varnames):
        priors['logQ'] = np.percentile(sho_trace['logQ'], percs)

    if np.isin('logw0_x', varnames):
        priors['logw0'] = np.percentile(sho_trace['logw0_x'], percs)
    
    if np.isin('logSw4_x', varnames):
        priors['logSw4'] = np.percentile(sho_trace['logSw4_x'], percs)
    
    if np.isin('logQ_x', varnames):
        priors['logQ'] = np.percentile(sho_trace['logQ_x'], percs)    
    
    
    return priors