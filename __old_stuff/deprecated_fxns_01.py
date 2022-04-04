# import relevant modules
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op
import scipy.signal as sig
from   scipy import stats
import astropy
from   astropy.io import fits as pyfits

import csv
import sys
import os
import warnings
import imp

import lightkurve as lk
import exoplanet as exo
import theano.tensor as T
import pymc3 as pm
import corner


# define constants
pi = np.pi

RJRE = 10.973      # (Rjup/Rearth)
RSRE = 109.2       # (Rsun/Rearth)
RSRJ = RSRE/RJRE   # (Rsun/Rjup)

MJME = 317.828     # (Mjup/Mearth)
MSME = 332948.6    # (Msun/Mearth)
MSMJ = MSME/MJME   # (Msun/Mjup)

RSAU = 0.00465     # solar radius in AU

LCIT = 29.4244     # long cadence integration time (min)
SCIT = 58.84876    # short cadence integration time (sec)

# turn off FutureWarnings
warnings.filterwarnings('ignore', category=FutureWarning)


###############################################################################################################################


class Planet:
    def __init__(self, epoch=None, period=None, depth=None, duration=None, index=None, tts=None, tts_err=None, quality=None, \
                 pttv=None, time_stamps=None, flux_stamps=None, error_stamps=None, mask_stamps=None, model_stamps=None, \
                 stamp_cadence=None, stamp_coverage=None, stamp_chisq=None, icov=None):


        self.epoch            = epoch            # reference transit time in range (0, period)
        self.period           = period           # orbital period
        self.depth            = depth            # transit depth
        self.duration         = duration         # transit duration

        self.index            = index            # index of each transit in range (0,1600) -- Kepler baseline
        self.tts              = tts              # all midtransit times in range (0,1600) -- Kepler baseline
        self.tts_err          = tts_err          # corresponding 1-sigma error bars on transit times
        self.quality          = quality          # boolean flag per transit; True=good

        self.pttv             = pttv             # [Amp, Pttv, phi, Q0, Q1, Q2, Q3]

        self.time_stamps      = time_stamps      # list of time stamps (one per transit) centered on midtransit
        self.flux_stamps      = flux_stamps      # list of flux stamps
        self.error_stamps     = error_stamps     # list of error stamps
        self.mask_stamps      = mask_stamps      # list of mask stamps, (mask=1 where OTHER planets transit)
        self.model_stamps     = model_stamps     # list of model stamps

        self.stamp_cadence    = stamp_cadence    # 'short', 'long', or 'none'
        self.stamp_coverage   = stamp_coverage   # fraction of transit/baseline covered by useable cadences
        self.stamp_chisq      = stamp_chisq      # chi-sq per transit

        self.icov             = icov             # inverse covariance matrix
        
        
        ###
        
        
    def mask_overlapping_transits(self):
        '''
        Remove cadences from stamps where other planets transit

        -- automatically updates time_, flux_, error_, mask_, and cadno_stamps
        '''
        for i, m in enumerate(self.mask_stamps):
            if len(m) > 0:
                self.time_stamps[i]  = self.time_stamps[i][~m]
                self.flux_stamps[i]  = self.flux_stamps[i][~m]
                self.error_stamps[i] = self.error_stamps[i][~m]
                self.mask_stamps[i]  = self.mask_stamps[i][~m]
                self.cadno_stamps[i] = self.cadno_stamps[i][~m]

        return None

    
    def clip_outlier_cadences(self, sigma=5.0, kernel_size=7):
        '''
        Do some iterative sigma rejection on each stamp

        sigma: rejection threshold for clipping (default=5.0)
        kernel_size: size of window for median filter (default=7)

        -- automatically updates time_, flux_, error_, mask_, and cadno_stamps
        '''
        for i, f in enumerate(self.flux_stamps):
            if len(f) > 0:
                loop = True
                while loop:
                    smoothed = sig.medfilt(self.flux_stamps[i], kernel_size=kernel_size)
                    outliers = np.abs(self.flux_stamps[i]-smoothed)/self.error_stamps[i] > sigma

                    if np.sum(outliers) > 0:
                        self.time_stamps[i]  = self.time_stamps[i][~outliers]
                        self.flux_stamps[i]  = self.flux_stamps[i][~outliers]
                        self.error_stamps[i] = self.error_stamps[i][~outliers]
                        self.cadno_stamps[i] = self.cadno_stamps[i][~outliers]
                    else:
                        loop = False

        return None


    def flatten_stamps(self, jitter=0.1):
        '''
        Fit a linear polynomial to out-of-transit flux to flatten data flux stamps

        jitter: fudge factor to avoid fitting in-transit flux if there are unresolved TTVs (default=0.1)

        -- automatically updates flux_stamps on Planet object
        '''
        for i, flux in enumerate(self.flux_stamps):
            if len(flux) > 0:
                time = self.time_stamps[i]

                intransit = np.abs(time-self.tts[i])/self.duration < 0.5+jitter

                if np.sum(~intransit) > 0:
                    coeffs = np.polyfit(time[~intransit],flux[~intransit],1)
                    linfit = np.polyval(coeffs, time)
                else:
                    linfit = 1.0

                self.flux_stamps[i] = flux/linfit

        return None



    def calculate_stamp_coverage(self, stampsize=1.5):
        '''
        Flag stamps with insufficient in-transit points

        stampsize: distance from each transit center to consider, in transit durations (default=1.5)
        '''
        # determine locations of SC and LC data
        sc_loc = self.stamp_cadence == 'short'
        lc_loc = self.stamp_cadence == 'long'

        # expected number of points in stamp if none are missing
        expected_sc_pts = 2*stampsize*self.duration/(SCIT/3600/24)
        expected_lc_pts = 2*stampsize*self.duration/(LCIT/60/24)

        # count up points per stamp overall
        pts_overall = []
        for t in self.time_stamps:
            pts_overall.append(len(t))
        pts_overall = np.array(pts_overall)

        # count up points per stamp in transit
        pts_in_transit = []
        for i, t0 in enumerate(self.tts):
            pts_in_transit.append(np.sum(np.abs(self.time_stamps[i]-t0) < self.duration/2))
        pts_in_transit = np.array(pts_in_transit)

        # calculate cover fraction        
        overall_fraction = np.zeros_like(self.tts)
        overall_fraction[sc_loc] = pts_overall[sc_loc]/expected_sc_pts
        overall_fraction[lc_loc] = pts_overall[lc_loc]/expected_lc_pts

        in_transit_fraction = np.zeros_like(self.tts)
        in_transit_fraction[sc_loc] = pts_in_transit[sc_loc]/(expected_sc_pts/2/stampsize)
        in_transit_fraction[lc_loc] = pts_in_transit[lc_loc]/(expected_lc_pts/2/stampsize)

        # use the smaller value as the coverage
        self.stamp_coverage = np.minimum(overall_fraction, in_transit_fraction)

        return None


    def calculate_stamp_chisq(self):
        '''
        Compare model_stamps, flux_stamps, and error_stamps to calcualte chisq for each transit
        '''
        mstamps = self.grab_stamps('model')
        fstamps = self.grab_stamps('flux')
        icov    = self.grab_icov()

        stamp_chisq = []
        j = 0
        for i, good in enumerate(self.quality):
            if good:
                y = mstamps[j]-fstamps[j]
                stamp_chisq.append(np.dot(y.T,np.dot(icov[j],y)))
                j += 1
            else:
                stamp_chisq.append(np.inf)

        self.stamp_chisq = np.array(stamp_chisq)

        return None


    def identify_good_transits(self, cover_fraction=0.7, chisq_sigma=5.0, verbose=True):
        '''
        Identify transits with sufficient coverage and non-outlier chisq

        cover_fraction: coverage threshold; eg. 0.7 will reject stamps with more than 70% of cadences missing (default=0.7)
        chisq_sigma: sigma threshold to reject stamps as poorly fit (default=5.0)
        verbose: boolean flag; 'True' to print results
        '''
        # determine locations of SC and LC data
        sc_loc = self.stamp_cadence == 'short'
        lc_loc = self.stamp_cadence == 'long'

        # flag stamps with sufficient coverage
        self.calculate_stamp_coverage()
        enough_pts = self.stamp_coverage > cover_fraction
        nonempty   = self.stamp_coverage > 0

        # count up points per stamp
        pts_per_stamp = []
        for t in self.time_stamps:
            pts_per_stamp.append(len(t))
        pts_per_stamp = np.array(pts_per_stamp)

        # flag stamps with unusually high chisq values (use pseudo-reduced-chisq)
        reject_chisq = np.zeros_like(self.tts, dtype='bool')

        if self.stamp_chisq is not None:
            X2u = self.stamp_chisq[~np.isinf(self.stamp_chisq)] / (pts_per_stamp[~np.isinf(self.stamp_chisq)])
            mad = astropy.stats.mad_std(X2u)
            med = np.median(X2u)
            reject_chisq[~np.isinf(self.stamp_chisq)] = np.abs(X2u-med)/mad > chisq_sigma
            reject_chisq[np.isinf(self.stamp_chisq)]  = True

        # print out results
        if verbose:
            print('%d out of %d transits rejected with high chisq' \
                  %(np.sum(reject_chisq[enough_pts]), np.sum(nonempty)))
            print('%d out of %d transits rejected with insufficient coverage' \
                  %(np.sum(~enough_pts[nonempty]), np.sum(nonempty)))

        # save the results
        self.quality = enough_pts * ~reject_chisq

        return None


    def grab_stamps(self, stamptype, cadence='any'):
        '''
        stamptype: 'time', 'flux', 'error', 'mask', 'model', or 'cadno'
        cadence: 'short', 'long', or 'any'
        '''
        if stamptype == 'time':  stamps = self.time_stamps
        if stamptype == 'flux':  stamps = self.flux_stamps
        if stamptype == 'error': stamps = self.error_stamps
        if stamptype == 'mask':  stamps = self.mask_stamps
        if stamptype == 'model': stamps = self.model_stamps
        if stamptype == 'cadno': stamps = self.cadno_stamps
      
        if cadence == 'any':
            use = self.quality * ~(self.stamp_cadence=='none')
        elif cadence == 'short':
            use = self.quality * (self.stamp_cadence=='short')
        elif cadence == 'long':
            use = self.quality * (self.stamp_cadence=='long')
        else:
            raise ValueError('cadence must be "short", "long", or "any"')

        stamps_out = []
        for i, s in enumerate(stamps):
            if use[i]: stamps_out.append(s)

        return stamps_out
    
    
    
    def grab_icov(self, cadence='any'):
        '''
        cadence: 'short', 'long', or 'any'
        '''
        if cadence == 'any':
            use = self.quality * ~(self.stamp_cadence=='none')
        elif cadence == 'short':
            use = self.quality * (self.stamp_cadence=='short')
        elif cadence == 'long':
            use = self.quality * (self.stamp_cadence=='long')
        else:
            raise ValueError('cadence must be "short", "long", or "any"')

        icov_out = []
        for i, c in enumerate(self.icov):
            if use[i]: icov_out.append(c)

        return icov_out

    
    
###############################################################################################################################



def read_csv_file(filename, k_index=0, v_index=1):
    """
    Read a csv file and return keys and values for use in a dictionary
    
    Parameters
    ----------
    filename : string
        csv file
    k_index : int
        index where keys start
    v_index: int
        index where values start
        
        
    Returns
    -------
        keys : list of keys
        values : list of values
    """
    data = []
    with open(filename) as infile:
        reader = csv.reader(infile)

        for row in reader:
            data.append(row)

        keys   = data[k_index]
        values = data[v_index:]

        return keys, values


    
def get_csv_data(keyname, keys, values):
    """
    Put the keys and values outputs of read_csv_file() into a useable format
    
    Parameters
    ----------
    keyname : string
        column definition
    keys : list
        keys
    values : list
        values corresponding to each key
    """
    kid = keys.index(keyname)
    
    outdata = []
    for row in values:
        outdata.append(row[kid])
    
    return outdata



def download_lkf(target, qstart=1, qend=17, qlist=None, cadence='long', download_dir=None, verbose=False):
    """
    Download a series of KeplerLightCurveFiles from MAST using lightkurve
    Can either download a contiguous block over (qstart,qend) or a list of specified quarters 

    Parameters
    ----------
    target : string
        name of target lightkurve (see lightkurve documentation)
    qstart : int
        first quarter do download (default=1)
    qend : int
        last quarter to download (default=17)
    qlist : list
        specified quarters to download
    cadence : string
        'short' or 'long' (default='long')
    download_dir : string
        directory to download files to (default=None --> cache)
    verbose : bool
        boolean flag; 'True' to print quarter number

    Returns
    -------
    lkflist : list
        a list of lightkurve.KeplerLightCurveFile() objects
    """
    lkflist = []
    
    if qlist is not None:
        for q in qlist:
            lkflist.append(lk.search_lightcurvefile(target, mission='Kepler', quarter=q, \
                                                    cadence=cadence).download(download_dir=download_dir))
    else:
        for q in range(qstart,qend+1):
            if verbose: print(q)
            lkflist.append(lk.search_lightcurvefile(target, mission='Kepler', quarter=q, \
                                                    cadence=cadence).download(download_dir=download_dir))
            
    return lkflist



###############################################################################################################################



def make_transitmask(time, tts, duration, masksize=1.5):
    '''
    Make a transit mask for a Planet (1=near transit; 0=not)
    
    planet: wings.Planet() object
    time: array of time values
    masksize: number of transit durations from transit center to consider near transit (default=1.5)
    
    --returns
        transitmask: boolean array (1=near transit; 0=not)
    '''  
    transitmask = np.zeros_like(time, dtype='bool')
    
    for t0 in tts:
        neartransit = np.abs(time-t0)/duration < masksize
        transitmask += neartransit
    
    return transitmask



def detrend_and_join(lkflist, window_length=101, break_tolerance=25, polyorder=3, \
                     sigma_upper=5, sigma_lower=10, masklist=None):
    '''
    Perform basic flattening, outlier rejection, etc. and joining with PDCSAP lightcurves

    lkflist: list of lightkurve.KeplerLightCurveFile() objects
    window_length: length of window for Savitsky-Golay filter (default=101; 2 days for 29 min cadence)
    break_tolerance: tolerance for gaps in data (default=25)
    polyorder: polyomial detrending order for Savitsky-Golay filter (default=3)
    sigma_upper: sigma threshold for outlier rejection
    masklist: list of boolean arrays to mask data before flattening, one mask per lkf in lkflist; 'True' near transits

    Returns a single lightkurve.LightCurve() object with detrended Kepler PDCSAP flux
    Default values are optimized for long cadence data; use window_length=3001, break_tolerance=750 for short cadence
    
    See lightkurve documentation for further description of input parameters
    '''
    # do the first quarter
    pdcsap = lkflist[0].PDCSAP_FLUX
    if masklist is None: mask = None
    else:
        mask = np.copy(masklist[0])
        masklisthere = np.copy(masklist)
    
    pdcsap, dqmask = remove_flagged_cadences(pdcsap)
    if mask is not None: mask = mask[dqmask]

    pdcsap = pdcsap.flatten(break_tolerance=break_tolerance, window_length=window_length, polyorder=polyorder, mask=mask)

    pdcsap, outliers = pdcsap.remove_outliers(sigma_lower=sigma_lower, sigma_upper=sigma_upper, iters=None, return_mask=True)
    if mask is not None: masklisthere[0] = mask[~outliers]

    pdcsap = pdcsap.normalize()


    # do the remaining quarters
    for q, lkf in enumerate(lkflist[1:]):
        lkf = lkf.PDCSAP_FLUX
        if masklist is None: mask = None
        else: mask = np.copy(masklist[q+1])
        
        lkf, dqmask = remove_flagged_cadences(lkf)
        if mask is not None: mask = mask[dqmask]

        lkf = lkf.flatten(break_tolerance=break_tolerance, window_length=window_length, polyorder=polyorder, mask=mask)

        lkf, outliers = lkf.remove_outliers(sigma_lower=float('inf'), sigma_upper=sigma_upper, iters=None, return_mask=True)
        if mask is not None: masklisthere[q+1] = mask[~outliers]

        lkf = lkf.normalize()

        pdcsap = pdcsap.append(lkf)

        
    # final smoothing on joined lightcurve
    if masklist is None: mask = None
    else: mask = np.hstack(masklisthere)
    pdcsap = pdcsap.flatten(break_tolerance=break_tolerance, window_length=window_length, polyorder=polyorder, mask=mask).normalize()   

    return pdcsap



def remove_flagged_cadences(lkf, bitmask='default'):
    '''
    Remove cadences flagged by Kepler pipeline data quality flags
    See lightkurve documentation for description of input parameters

    lkf: lk.KeplerLightCurveFile() object
    bitmask: 'default', 'hard', or 'hardest' -- how aggressive to be on cutting flagged cadences

    --returns
        lkf: LightCurveFile with flagged cadences removed
        qmask: boolean mask indicating with cadences were removed (True=data is good quality)
    '''
    qmask = lk.KeplerQualityFlags.create_quality_mask(lkf.quality, bitmask)
    
    lkf.time         = lkf.time[qmask]
    lkf.flux         = lkf.flux[qmask]
    lkf.flux_err     = lkf.flux_err[qmask]
    lkf.centroid_col = lkf.centroid_col[qmask]
    lkf.centroid_row = lkf.centroid_row[qmask]
    lkf.cadenceno    = lkf.cadenceno[qmask]
    lkf.quality      = lkf.quality[qmask]
    
    return lkf, qmask


def cut_stamps(planet, time, data, dtype, stampsize=1.5):
    '''
    Cut out a stamp centered on each transit time from a full Kepler lightcurve

    planet: wings.Planet() object
    time: array of time values
    data: array of data values for stamp (same length as time)
    dtype: ndarray datatype to use for stamps
    stampsize: distance from each transit center to cut, in transit durations (default=1.5)
    '''
    stamps = []

    # cut out the stamps
    for t0 in planet.tts:
        neartransit = np.abs(time - t0)/planet.duration < stampsize
        stamps.append(np.array(data[neartransit], dtype=dtype))
        
    return stamps



def combine_stamps(sc_stamps, lc_stamps):
    '''
    Combine short and long cadence stamps, using SC wherever available

    sc_stamps: short cadence stamps
    lc_stamps: long cadence stamps

    --returns
        stamps_out: single list of stamps
        stamp_cadence: array of len(stamps_out) specifying cadence of each stamp as 'short', 'long', or 'none'
    '''
    # check lengths
    if len(sc_stamps) != len(lc_stamps):
        raise ValueError('inconsistent number of stamps')
        
    Nstamps = len(sc_stamps)
    
    # add stamps to list, prioritizing short cadence
    stamps_out = []
    stamp_cadence = []
    for i in range(Nstamps):
        if len(sc_stamps[i]) > 0:
            stamps_out.append(sc_stamps[i])
            stamp_cadence.append('short')
        elif len(lc_stamps[i]) > 0:
            stamps_out.append(lc_stamps[i])
            stamp_cadence.append('long')
        else:
            stamps_out.append([])
            stamp_cadence.append('none')
            
    stamp_cadence = np.array(stamp_cadence)

    return stamps_out, stamp_cadence



###############################################################################################################################

#############################################
# FUCTIONS FOR GENERATING COVARIANCE MATRIX #
#############################################


def generate_acor_fxn(time, flux, mask, Npts, verbose=True):
    '''
    Generate an autocorrelation function on out-of-transit flux data
    
    time: array of time values (typically short cadence)
    flux: array of flux values
    mask: boolean mask where any planet transits (1=transit; 0=out-of-transit)
    Npts: number of points to use in each generation of acor fxn (typically 3*max(transit_duration))
    
    --returns
        xcor: lag-time values used
        acor: autocorrelation function
        wcor: corresponding weights (for now, wcor=1.0)
    '''
    Nsamples = int((len(flux) - len(flux) % Npts)/Npts)

    # generate autocorrelation function
    acor = np.zeros((Nsamples,2*Npts+1))
    msum = np.zeros(Nsamples)
    gaps = np.zeros(Nsamples)

    for i in range(Nsamples):
        acor[i] = np.correlate(1-flux[Npts*i:Npts*(i+1)+1],1-flux[Npts*i:Npts*(i+1)+1],mode='full')
        msum[i] = np.sum(mask[Npts*i:Npts*(i+1)])
        gaps[i] = (time[Npts*i:Npts*(i+1)+1][1:] - time[Npts*i:Npts*(i+1)+1][:-1]).max()/(SCIT/3600/24)

    acor  = np.median(acor[(msum==0)*(gaps<2)], axis=0)
    acor  = acor[Npts:]/acor[Npts]

    xcor = (time[:Npts+1]-time.min())*24 + SCIT/3600
    wcor = 1.0

    if verbose:
        print('Number of samples = %d' %Nsamples)
        print('Number of points per sample = %d' %Npts)
        print('Number of transit-free and gap-free samples = %d' %np.sum((msum==0)*(gaps<2)))
        
    return xcor, acor, wcor



def decaying_sinusoid(x, theta):
    '''
    Model a linear combination of decaying sinusoids -- A*exp(-x/tau)*sin(k*x-phi)
    
    x: array of values at which to evaluate function
    theta: array of parameters [A1, tau1, k1, phi1, A2, tau2, k2, phi2...]
    
    --returns
        model: functional model evaluated for (x, theta)
    '''
    model  = np.zeros_like(x)
    Nterms = int(len(theta)/4)
    theta  = theta.reshape(Nterms,-1)
       
    for i in range(Nterms):
        A, tau, k, phi = theta[i]
        model += A*np.exp(-x/tau)*np.sin(k*x-phi)
    
    return model



def residuals_for_acor(theta, xdata, ydata, yerror):
    '''
    Residuals for modeling the autocorrelation function -- use with op.leastsq()
    
    theta: array of parameters [A1, tau1, k1, phi1, A2, tau2, k2, phi2...]
    xdata: array of lag times
    ydata: array of corresponding autocorrelation function
    yerror: array of corresponding errors
    '''
    ymodel = decaying_sinusoid(xdata, theta)
    res    = (ymodel-ydata)/yerror

    return res



def lnlike_acor(theta, xdata, ydata, yerror):
    '''
    Log-likelihood for modeling the autocorrelation function -- use with op.minimize()
    
    theta: array of parameters [A1, tau1, k1, phi1, A2, tau2, k2, phi2...]
    xdata: array of lag times
    ydata: array of corresponding autocorrelation function
    yerror: array of corresponding errors
    '''
    ymodel = decaying_sinusoid(xdata, theta)
    res    = (ymodel-ydata)/yerror
    
    return -0.5*np.sum(res**2)




def residuals_for_acor_inception(A, acor_model, res_acor, ydata, yerror):
    '''
    Residuals for modeling the high-frequency component of the ACF -- use with op.leastsq()
    
    A: (float) scaleable amplitdue of high-f component relative to low-f component
    acor_model: smooth model of low-f ACF
    res_acor: autocorrelation function of residuals of ACF (true ACF - model ACF)
    ydata: true ACF
    yerror: array of corresponding errors
    '''
    ymodel = acor_model + A*res_acor
    res    = (ymodel-ydata)/yerror

    return res



def model_acor_fxn(xcor, acor, wcor, Nterms=5, do_plots=True):
    '''
    Model the autocorrelation function as a sum of decaying sinusoids A*exp(x/tau)*sin(k*x-phi)
    
    xcor: lag-time values used
    acor: autocorrelation function
    wcor: corresponding weights
    Nterms: number of terms to use in sinusoid model (default=5)
    
    --returns
        acor_model: model of autocorrelation function
        acor_theta: fitteed parameter values
    '''
    # initial estimate for model parameters
    theta  = np.zeros((Nterms,4))

    # automatically probe a variety of frequencies (k) and decay speeds (taue)
    adj = np.e*(np.arange(Nterms)+1)

    for i in range(Nterms):
        theta[i] = [acor.max()/(3*(i+2)), xcor.max()/adj[i], xcor.max()/adj[i], -pi/2]

    theta = theta.reshape(-1)
    theta_guess = theta.copy()
    
     # do a minimization
    theta_out, success = op.leastsq(residuals_for_acor, theta, maxfev=300*(len(xcor)+1), args=(xcor, acor, wcor))

    fxn = lambda *args: -lnlike_acor(*args)
    result = op.minimize(fxn, theta_out, args=(xcor, acor, wcor))

    # pull the results and generate a model
    acor_theta = result['x']
    acor_model = decaying_sinusoid(xcor, acor_theta)
    acor_guess = decaying_sinusoid(xcor, theta_guess)
    
    
    # do acf inception (autocorrelation of residuals)
    rcor = acor - acor_model
    
    res_acor = np.correlate(rcor, rcor, 'full')
    res_acor = res_acor[int(len(res_acor)/2)+1:]
    res_acor = np.hstack([res_acor,0])

    # do a minimization on high-freq (res_acor) component amplitude
    A = np.array([np.std(rcor)/np.std(res_acor)/np.sqrt(2)])
    A_out, success = op.leastsq(residuals_for_acor_inception, A, args=(acor_model, res_acor, acor, 1.0))    
    
    # combine low-freq (acor_model) and high-freq(res_acor) components
    full_acor_model = acor_model + A_out*res_acor
    full_rcor = acor-full_acor_model
    
    
    # plot the results
    if do_plots:
        xcor_lc, acor_lc = bin_acor(xcor, acor)
        
        fig = plt.figure(figsize=(18,6))

        ax = plt.subplot2grid(shape=(4,1), loc=(0,0), rowspan=3, colspan=1)
        ax.plot(xcor, acor, c='orange', label='ACF from SC data')
        ax.plot(xcor, full_acor_model, c='cornflowerblue', label='Model ACF')
        ax.plot(xcor, acor_model, c='mediumblue', lw=2, label='Model low-freq component')
        ax.plot(xcor_lc, acor_lc, 'o', c='red', mec='darkred', ms=9, label='Binned to long cadence')
        ax.plot(xcor, acor_guess, color='k', ls=':', lw=2)
        ax.set_xticks([])
        ax.set_xlim(0, xcor.max())
        ax.set_ylabel('Normalized ACF', fontsize=24)
        ax.legend(loc='upper right', fontsize=16)

        ylim = 1.1*np.max(np.abs(rcor))

        ax = plt.subplot2grid(shape=(4,1), loc=(3,0), rowspan=1, colspan=1)
        ax.plot(xcor, full_rcor, c='cornflowerblue')
        ax.set_xlabel('lag time [hrs]', fontsize=24)
        ax.set_xlim(0, xcor.max())
        ax.set_ylim(-ylim, ylim)

        plt.show()
        
    return full_acor_model, acor_theta



def bin_acor(xcor, acor):
    '''
    Bin autocorrelation function from short cadence to long cadence
    
    xcor: lag-time values used
    acor: autocorrelation function
    --returns
        xcor_lc: lag-time values binned to long cadence
        acor_lc: autocorrelation function binned to long cadence
    '''
    Npts = len(xcor)
    
    # make sure xcor & acor have a length divisible by 30
    xcor_lc = xcor[:Npts-Npts%30]
    acor_lc = acor[:Npts-Npts%30]

    # bin over every 30 cadences
    xcor_lc = xcor_lc.reshape(int(len(xcor_lc)/30),30).sum(axis=1)/30
    acor_lc = acor_lc.reshape(int(len(acor_lc)/30),30).sum(axis=1)/30

    # pad the end to avoid off-by-one clipping problems
    xcor_lc = np.hstack([xcor_lc, xcor_lc[-1]+(xcor_lc[-1]-xcor_lc[-2])])
    acor_lc = np.hstack([acor_lc, 0])    
    
    return xcor_lc, acor_lc



def make_covariance_matrix(acor, Nsize, sigma_white, sigma_red, do_plots=False):
    '''
    Make a covariance matrix from the autocorrelation function
    
    acor: (modeled) 1D autocorrelation function
    Nsize: output size of matrix (should be equal to length of longest stamp)
    sigma_white: 1-sigma error due to gaussian noise
    sigma_red: 1-sigma error due to correlated noise
    
    --returns
        covmatrix: covariance matrix
    '''
    # make sure that acor is as big as needed
    if len(acor) < Nsize:
        acor = np.hstack([acor, np.zeros(Nsize-len(acor))])
    if len(acor) > Nsize:
        acor = acor[:Nsize]

    # broadcast autocorrelation function into a matrix and normalize 
    covmatrix = np.zeros((Nsize,Nsize))
    
    for i in range(Nsize):
        covmatrix[i] += np.roll(acor,i)

    covmatrix = np.triu(covmatrix)
    covmatrix += covmatrix.T
    covmatrix -= np.eye(Nsize)*covmatrix
    covmatrix /= covmatrix.max()
    
    # make diagonal matrix
    dij = np.eye(Nsize)

    # combine scaled diagonal and off-diagonal terms
    covmatrix = covmatrix*sigma_red**2 + dij*sigma_white**2
    
    # make inverse matrix
    icovmatrix = np.linalg.inv(covmatrix)

    # plot the results
    if do_plots:
        fig, axes = plt.subplots(1,2, figsize=(8,8))
        ax = axes[0]
        ax.imshow(covmatrix, vmin=0.1*covmatrix.min(), vmax=0.1*covmatrix.max())
        ax = axes[1]
        ax.imshow(icovmatrix, vmin=0.3*icovmatrix.min(), vmax=0.3*icovmatrix.max())
        plt.show()
    
    return covmatrix