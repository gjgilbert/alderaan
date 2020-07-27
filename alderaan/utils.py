import numpy as np
import scipy.optimize as op
import scipy.signal as sig
from   scipy import stats
from   scipy import fftpack
import astropy.stats
import warnings
import theano.tensor as T

from .constants import *

__all__ = ['boxcar_smooth',
           "get_sma",
           "get_dur_tot",
           'lorentzian',
           'heavyside',
           'notch_filter',
           'FFT_estimator',
           'LS_estimator']


def boxcar_smooth(x, winsize):
    """
    Docstring
    """
    win = sig.boxcar(winsize)/winsize
    xsmooth = np.pad(x, (winsize, winsize), mode='reflect')

    for i in range(2):
        xsmooth = sig.convolve(xsmooth, win, mode='same')
    
    xsmooth = xsmooth[winsize:-winsize]
    
    return xsmooth


def get_sma(P, Ms):
    """
    Parameters
    ----------
    P : period [days]
    Ms : stellar mass [Solar masses]
    
    Returns
    -------
    sma : semimajor axis [Solar radii]
    """
    return Ms**(1/3)*(P/365.24)**(2/3)/RSAU


def get_dur_tot(P, rp, Rs, b, sma):
    """
    P : period [days]
    rp : planet radius [Solar radii]
    Rs : stellar radius [Solar radii]
    b : impact parameter
    sma : semimajor axis [Solar radii]
    """
    k = rp
    inc = pi/2 - np.arctan2(b, sma)
    badj = np.sqrt((1+k)**2 - b**2)/np.sin(inc)
    
    return (P/pi)*np.arcsin(Rs/sma * badj)


def lorentzian(theta, x):
    """
    Model a Lorentzian (Cauchy) function
    
    Parameters
    ----------
    theta : array-like
        parameters for Lorentzian = [loc, scale, height, baseline]
    x : array-like
        1D array of x data values
        
    Returns
    -------
    pdf : array-like
        Lorentzian probability density function
    """
    loc, scale, height, baseline = theta
    
    return height*stats.cauchy(loc=loc, scale=scale).pdf(x) + baseline



def heavyside(x, x0=0., k=1000.):
    """
    Approximates the Heavyside step function with a smooth distribution
    Implemented using theano tensors for easily setting bounds on PyMC3 deterministic variables
    
    H0 = 1/(1+T.exp(-2*k*(x-x0)))
    
    Parameters
    ----------
    x : array-like
        input values at which to compute function
    x0 : float
        location of step
    k : float (optional)
        exponent factor in approximation (default=1000.)
    """
    return 1/(1+T.exp(-2*k*(x-x0)))



def notch_filter(data, f0, fsamp, Q):
    """
    Apply a 2nd-order notch filter (i.e. a narrow stopband filter) to a data array
    See scipy.signal.iirnotch & scipy.signal.filtfilt for details of implementation
    
    Parameters
    ----------
    data : array-like
        data to be filtered
    f0 : float
        center frequency of stopband
    fsamp: float
        sampling frequency, same units as f0
    Q : float
        quality factor
        
    Returns
    -------
    data_filtered: array-like
        data array with selcted frequency filtered out
    """
    w0 = f0/(fsamp/2)
    
    b, a = sig.iirnotch(w0, Q)
    
    data_filtered = sig.filtfilt(b, a, data)
    
    return data_filtered



def FFT_estimator(x, y, sigma=5.0):
    """
    Identify significant frequencies in a (uniformly sampled) data series
    Fits a Lorentzian around each peak
    
    Parameters
    ----------
    x : array-like
        1D array of x data values; should be monotonically increasing
    y : array-like
        1D array of corresponding y data values, len(x)
    sigma : float
        sigma threshold for selecting significant frequencies (default=5.0)
        
    Returns
    -------
    xf : ndarray
        1D array of frequency values
    yf : ndarray
        1D array of response values, len(xf)
    freqs : ndarray
        array of significant frequencies
    """
    # min/max testable time deltas (conservative low-freq cutoff)
    Tmin = 2*(x[1]-x[0])
    Tmax = (x.max()-x.min())/4

    N = len(x)//2

    # FFT convolved with a hann windown (to reduce spectral leakage)
    window = sig.hann(len(x))

    xf = np.linspace(0, 1/Tmin, N)
    yf = np.abs(fftpack.fft(window*y)[:N])
    
    yf -= np.median(yf)
        
    keep = xf > 1/Tmax
    
    xf = xf[keep]
    yf = yf[keep]
    
    
    # make a copy of raw xf and yf data
    xf_all = xf.copy()
    yf_all = yf.copy()

    
    # now search for significant frequencies
    freqs = []

    loop = True
    while loop:
        yf_noise = astropy.stats.mad_std(yf)
        peakfreq = xf[np.argmax(yf)]
        
        if (yf[xf==peakfreq]/yf_noise > sigma) and (yf[xf==peakfreq] > 1/xf[xf==peakfreq]):
            res_fxn = lambda theta, x, y: y - lorentzian(theta, x)
            
            theta_in = np.array([peakfreq, 1/Tmax, yf.max(), np.median(yf)])
            theta_out, success = op.leastsq(res_fxn, theta_in, args=(xf, yf))

            width = np.max(5*[theta_out[1], 3*(xf[1]-xf[0])])
            mask = np.abs(xf-theta_out[0])/width < 1

            yf[mask] = theta_out[3]

            freqs.append(theta_out[0])

        else:
            loop = False

        
    freqs = np.array(freqs)    
    
    return xf_all, yf_all, freqs



def LS_estimator(x, y, fsamp=None, fap=0.1, return_levels=False):
    """
    Generates a Lomb-Scargle periodogram and identifies significant frequencies from a data series
    Assumes that data are nearly evenly sampled
    
    Parameters
    ----------
    x : array-like
        1D array of x data values; should be monotonically increasing
    y : array-like
        1D array of corresponding y data values, len(x)
    fsamp: float
        nominal sampling frequency; if not provided it will be calculated from the data
    fap : float
        false alarm probability threshold to consider a frequency significate (default=0.1)
        
    Returns
    -------
    xf :
   
    yf :
   
    freqs :
   
    """
    # get sampling frequency
    if fsamp is None:
        fsamp = 1/np.min(x[1:]-x[:-1])
    
    # Hann window to reduce ringing
    hann = sig.windows.hann(len(x))
    hann /= np.sum(hann)
    
    # identify any egregious outliers
    out = np.abs(y - np.median(y))/astropy.stats.mad_std(y) > 5.0
    
    xt = x[~out]
    yt = y[~out]
    
    freqs = []
    faps  = []
    
    loop = True
    while loop:
        lombscargle = astropy.stats.LombScargle(xt, yt*hann[~out])
        xf, yf = lombscargle.autopower(minimum_frequency=2.0/(xt.max()-xt.min()), \
                                       maximum_frequency=0.25*fsamp, \
                                       samples_per_peak=10)
    
        peak_freq = xf[np.argmax(yf)]
        peak_fap  = lombscargle.false_alarm_probability(yf.max(), method='bootstrap')

     
        # output first iteration of LS periodogram
        if len(freqs) == 0:
            xf_out = xf.copy()
            yf_out = yf.copy()
            levels = lombscargle.false_alarm_level([0.1, 0.01, 0.001])
        
        if peak_fap < fap:
            yt -= lombscargle.model(xt, peak_freq)*len(xt)
            freqs.append(peak_freq)
            faps.append(peak_fap)
            
        else:
            loop = False
            
        if len(freqs) > 5:
            loop = False
    
    if return_levels:
        return xf_out, yf_out, freqs, faps, levels
    
    else:
        return xf_out, yf_out, freqs, faps



