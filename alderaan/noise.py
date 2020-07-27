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



def make_chunklist(time, flux, cadno, Npts, sigma_reject=5.0, gap_tolerance=15, interpolate=True):
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
        maximum number of consecutive missing cadences allowed (default=2)
    interpolate : bool
        True to perform linear interpolation over small gaps (default=True)
        
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

            # require at least 99% coverage
            if np.sum(empty)/Npts < 0.01:
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

    xcor = np.arange(Npts+1)*SCIT/3600/24
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



def build_sho_model(x, y, var_method, low_freqs=None, high_freqs=None, extra_term=None, match_Q=True):
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
    low_freqs : list (optional)
        list of at most one (ordinary, not angular) frequency
    high_freqs : list (optional)
        list of at most three (ordinary, not angular) frequencies
    extra_term : string (optional)
        'fixed' --> add an extra SHOTerm with fixed Q=1/sqrt(2)
        'free' --> add an extra SHOTerm with logQ as a free hyperparameter in the GP model
    match_Q : bool (default=True)
        True to use the same Q value for all high frequency terms
        
    Returns
    -------
    model : a pymc3 model
    
    """
    with pm.Model() as model:
        
        # set up the low frequency terms
        if low_freqs is not None:
            logSw4 = pm.Normal('logSw4', mu=np.log(np.var(y)), sd=15.0)
            logQ   = pm.Uniform('logQ',  lower=np.log(1/np.sqrt(2)), upper=np.log(100))

            w0 = convert_frequency(2*pi*low_freqs[0], T.exp(logQ))
            logw0 = pm.Normal('logw0', mu=np.log(w0), sd=np.log(1.1))

            kernel = exo.gp.terms.SHOTerm(log_Sw4=logSw4, log_w0=logw0, log_Q=logQ)
            

        if high_freqs is not None:
            logS1 = pm.Normal('logS1', mu=np.log(np.var(y)), sd=15.0)
            logQ1 = pm.Uniform('logQ1', lower=2.0, upper=10.0)

            w1 = convert_frequency(2*pi*high_freqs[0], T.exp(logQ1))
            
            try:
                kernel += exo.gp.terms.SHOTerm(log_S0=logS1, w0=w1, log_Q=logQ1)
            except:
                kernel = exo.gp.terms.SHOTerm(log_S0=logS1, w0=w1, log_Q=logQ1)


            if len(high_freqs) > 1:
                logS2 = pm.Normal('logS2', mu=np.log(np.var(y)), sd=15.0)

                if match_Q:
                    logQ2 = pm.Deterministic('logQ2', logQ1)
                else:
                    logQ2 = pm.Uniform('logQ2', lower=2.0, upper=10.0)

                w2 = convert_frequency(2*pi*high_freqs[1], T.exp(logQ2))

                kernel += exo.gp.terms.SHOTerm(log_S0=logS2, w0=w2, log_Q=logQ2)


            if len(high_freqs) > 2:
                logS3 = pm.Normal('logS3', mu=np.log(np.var(y)), sd=15.0)

                if match_Q:
                    logQ3 = pm.Deterministic('logQ3', logQ1)
                else:
                    logQ3 = pm.Uniform('logQ3', lower=2.0, upper=10.0)

                w3 = convert_frequency(2*pi*high_freqs[2], T.exp(logQ3))

                kernel += exo.gp.terms.SHOTerm(log_S0=logS3, w0=w3, log_Q=logQ3)


            if len(high_freqs) > 3:
                warnings.warn('%d frequencies given but only modeling the first 3' %len(high_freqs))



        if extra_term == 'free':
            logSw4_x = pm.Normal('logSw4_x', mu=np.log(np.var(y)), sd=15.0)
            logw0_x  = pm.Normal('logw0_x', mu=0.0, sd=15.0)
            logQ_x   = pm.Normal('logQ_x',  mu=0.0, sd=5.0)

            try:
                kernel += exo.gp.terms.SHOTerm(log_Sw4=logSw4_x, log_w0=logw0_x, log_Q=logQ_x)
            except:
                kernel = exo.gp.terms.SHOTerm(log_Sw4=logSw4_x, log_w0=logw0_x, log_Q=logQ_x)

        elif extra_term == 'fixed':
            logSw4_x = pm.Normal('logSw4_x', mu=np.log(np.var(y)), sd=15.0)
            logw0_x  = pm.Normal('logw0_x', mu=0.0, sd=15.0)

            try:
                kernel += exo.gp.terms.SHOTerm(log_Sw4=logSw4_x, log_w0=logw0_x, Q=1/np.sqrt(2))
            except:
                kernel = exo.gp.terms.SHOTerm(log_Sw4=logSw4_x, log_w0=logw0_x, Q=1/np.sqrt(2))

            
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



def model_acf(xcor, acor, fcut, sigma=5.0, method='smooth'):
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
    sigma : float
        threshold for finding significant frequencies in FFT of autocorrelation function
    method : string
        method to model low frequency component; either 'smooth', 'shoterm', or 'savgol'  (default='smooth')
        
        
    Returns
    -------
    acor_emp, acor_mod, xf, yf, freqs
    """
    # identify signficant frequencies
    xf, yf, freqs = FFT_estimator(xcor, acor, sigma=sigma)

    low_freqs  = freqs[freqs < fcut]
    high_freqs = freqs[freqs >= fcut]

    if len(low_freqs) > 2:
        low_freqs = low_freqs[[0,1]]

    if len(high_freqs) > 3:
        high_freqs = high_freqs[[0,1,2]]
        
    # arrays to hold empirical and model ACF with each subsequent term removed
    acor_emp = np.zeros((3,len(acor)))
    acor_mod = np.zeros_like(acor_emp)

    acor_emp[0] = acor.copy()

    # 1st model component (low frequency) 
    if method == 'shoterm':
        xcor_mirror = np.hstack([-xcor[::-1], xcor])
        acor_mirror = np.hstack([acor_emp[0][::-1], acor_emp[0]])
        
        if len(low_freqs) == 0:
            extra_term = 'free'
        else:
            extra_term = 'fixed'
        
        model = build_sho_model(xcor_mirror, acor_mirror, var_method='fit', extra_term='fixed')
        
        with model:
            map_soln = exo.optimize(start=model.test_point)
            
        acor_mod[0] = map_soln['gp_pred'][len(acor_emp[0]):]
        
    elif method == 'smooth':
        if len(low_freqs) > 0:
            window_length = int(24*60/np.max(low_freqs)/2)
        else:
            window_length = int(len(acor_emp[0])/6)

        window_length = window_length + (window_length % 2) + 1

        acor_mirror = np.hstack([acor_emp[0][::-1], acor_emp[0]])
        acor_mod[0] = boxcar_smooth(acor_mirror, window_length)[len(acor_emp[0]):]
            
    elif method == 'savgol':
        if len(low_freqs) > 0:
            window_length = int(24*60/np.max(low_freqs)/2)
        else:
            window_length = 59

        window_length = window_length + (window_length % 2) + 1
        
        
        acor_mirror = np.hstack([acor_emp[0][::-1], acor_emp[0]])
        acor_mirror = sig.savgol_filter(acor_mirror, polyorder=2, window_length=window_length)[len(acor_emp[0]):]
        acor_mod[0] = boxcar_smooth(acor_mirror, 5)
    
    else:
        raise ValueError("method must be either 'shoterm', 'smooth', or 'savgol'")
    
    acor_emp[1] = acor_emp[0] - acor_mod[0]

    
    # 2nd model component (high frequency)
    if len(high_freqs) > 0:
        model = build_sho_model(xcor, acor_emp[1], high_freqs=high_freqs, var_method='fit', match_Q=True)

        with model:
            map_soln = exo.optimize(start=model.test_point)    

        acor_mod[1] = map_soln['gp_pred']
        acor_emp[2] = acor_emp[1] - acor_mod[1]

    else:
        acor_emp[2] = acor_emp[1] - acor_mod[1]
        
        
    return acor_emp, acor_mod, xf, yf, freqs



def generate_synthetic_noise(xcor, acor_emp, acor_mod, high_freqs, fcut, n, sigma):
    """
    Generate synthetic correlated noise given a specified autorrelation function
    
    
    Parameters
    ----------
    xcor :
    
    acor_emp :
    
    acor_mod :
    
    high_freqs :
    
    n :
    
    sigma :
    
    
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
    covmatrix = make_covariance_matrix(acor_mod[0]+acor_mod[1], n)

    # decompose it
    try:
        L = np.linalg.cholesky(covmatrix)
    
    except:
        try:
            warnings.warn('Covariance matrix not positive definite...adjusting weights')
            
            diagbool = np.eye(n) == 1
            
            loop = True
            while loop:
                try:
                    covmatrix[~diagbool] /= 1.02
                    L = np.linalg.cholesky(covmatrix)
                    loop = False
                    
                except:
                    covmatrix[~diagbool] /= 1.02

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