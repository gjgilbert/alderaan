import numpy as np
from   scipy.interpolate import interp1d
from   sklearn.neighbors import KernelDensity
from   sklearn.model_selection import GridSearchCV
from   sklearn.covariance import EmpiricalCovariance
import warnings

from .constants import *


__all__ = ['draw_random_samples',
           'get_bw',
           'generate_synthetic_samples'
          ]


def draw_random_samples(pdf, domain, N, *args, **kwargs):
    """
    Draw random samples from a given pdf using inverse transform sampling
    
    Paramters
    --------
    pdf : univarite function
        outputs pdf as a function of input value
    domain : tuple 
        domain over which pdf is defined
    N : int
        number of random samples to draw
        
    Returns
    -------
    samples : ndarray
        vector of random samples drawn
    """
    x = np.linspace(domain[0], domain[1], int(1e5))
    y = pdf(x, *args, **kwargs)
    cdf_y = np.cumsum(y)
    cdf_y = cdf_y/cdf_y.max()
    inverse_cdf = interp1d(cdf_y,x)
        
    return inverse_cdf(np.random.uniform(cdf_y.min(), cdf_y.max(), N))


def get_bw(samples, weights=None, max_draws=1000):
    """
    Use cross-validation to determine KDE bandwidth for a set of 1D data samples
    Iteratively performs first coarse then fine estimation   
    
    Parameters
    ----------
    samples : array-like
        1D data samples
    weights : array-like (optional)
        weights corresponding to samples
    max_draws : int (default=1000)
        maximum number of samples to use during estimation
        
    Returns
    -------
    bw : float
        estimated bandwidth
    """
    N = len(samples)
    x = np.random.choice(samples, p=weights, size=np.min([max_draws,N]), replace=True)
    
    coarse_mesh = np.linspace((x.max()-x.min())/N, np.std(x), int(np.sqrt(N)))
    grid = GridSearchCV(KernelDensity(), {'bandwidth': coarse_mesh}, cv=5)
    grid.fit(x[:, None])
    
    fine_mesh = grid.best_params_['bandwidth'] + np.linspace(-1,1,int(np.sqrt(N)))*(coarse_mesh[1]-coarse_mesh[0])
    grid = GridSearchCV(KernelDensity(), {'bandwidth': fine_mesh}, cv=5)
    grid.fit(x[:, None])

    return grid.best_params_['bandwidth']


def generate_synthetic_samples(samples, bandwidths, n_up, weights=None):
    """
    Use PDF Over-Sampling (PDFOS - Gao+ 2014) to generate synthetic samples
    
    Parameters
    ----------
    samples : ndarray, (N x M)
        array of data samples arranged N_samples, M_parameters
    bandwitdhs : array-like, (M)
        pre-estimated KDE bandwidths for each of M parameters
    n_up : int
        number of upsampled synthetic data points to generate
    weights : ndarray, (N x M)
        array of weights corresponding to samples
        
    Returns
    -------
    new_samples : ndarray
        array containing synthetic samples
    """
    n_samp, n_dim = samples.shape
    index = np.arange(0, n_samp, dtype='int')
    
    # we'll generate a few more samples than needed in anticipation of rejecting a few
    n_up101 = int(1.01*n_up)

    # naive resampling (used only to estimate covariance matrix)
    naive_resamples = samples[np.random.choice(index, p=weights, size=3*n_samp)]

    # compute empirical covariance matrix and lower Cholesky decomposition
    emp_cov = EmpiricalCovariance().fit(naive_resamples).covariance_
    L = np.linalg.cholesky(emp_cov)

    # scale each parameter by precomputed bandwidths so they have similar variance
    samples_scaled = (samples - np.mean(samples, axis=0)) / bandwidths

    # calculate synthetic samples following PDFOS (Gao+ 2014)
    random_index = np.random.choice(index, p=weights, size=n_up101)
    random_samples = samples_scaled[random_index]
    random_jitter = np.random.normal(0, 1, n_up101*n_dim).reshape(n_up101, n_dim)
    new_samples = random_samples + np.dot(L.T, random_jitter.T).T

    # rescale each parameter to invert previous scaling
    new_samples = new_samples*bandwidths + np.mean(samples, axis=0)
    
    # reject any synthetic samples pushed out of bounds of original samples
    bad = np.zeros(n_up101, dtype='bool')
    
    for i in range(n_dim):
        bad += (new_samples[:,i] < samples[:,i].min())
        bad += (new_samples[:,i] > samples[:,i].max())
        
    if np.sum(bad)/len(bad) > 0.01:
        warnings.warn("More than 1% of PDFOS generated samples were beyond min/max values of original samples")
    
    new_samples = new_samples[~bad]
    
    # only return n_up samples
    if new_samples.shape[0] >= n_up:
        return new_samples[:n_up]
    
    else:
        # use naive resampling to replace rejected samples
        replacement_samples = samples[np.random.choice(index, p=weights, size=n_up-new_samples.shape[0])]
    
        return np.vstack([new_samples, replacement_samples])