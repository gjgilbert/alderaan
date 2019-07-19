# import relevant modules
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op
import scipy.signal as sig
import scipy.interpolate as interp
from   scipy import stats

import csv
import sys
import os
import imp
import timeit
import progressbar
import warnings

import astropy
from   astropy.io import fits as pyfits
import lightkurve as lk
import exoplanet as exo
import starry
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



#####

class Planet:
    def __init__(self, epoch=None, period=None, depth=None, duration=None, index=None, tts=None, tts_err=None, quality=None, \
                 pshape=None, ptime=None, time_stamps=None, flux_stamps=None, error_stamps=None, mask_stamps=None, model_stamps=None, \
                 stamp_cadence=None, stamp_coverage=None, stamp_chisq=None, icov=None):


        self.epoch            = epoch            # reference transit time in range (0, period)
        self.period           = period           # orbital period
        self.depth            = depth            # transit depth
        self.duration         = duration         # transit duration

        self.index            = index            # index of each transit in range (0,1600) -- Kepler baseline
        self.tts              = tts              # all midtransit times in range (0,1600) -- Kepler baseline
        self.tts_err          = tts_err          # corresponding 1-sigma error bars on transit times
        self.quality          = quality          # boolean flag per transit; True=good

        self.pshape           = pshape           # [T0, P, rp, zeta, b^2, e^1/2*sinw, e^1/2*cosw, u1, u2]
        self.ptime            = ptime            # parameters describing ttvs; specifics depend on ttv method

        self.time_stamps      = time_stamps      # list of time stamps (one per transit) centered on midtransit
        self.flux_stamps      = flux_stamps      # list of flux stamps
        self.error_stamps     = error_stamps     # list of error stamps
        self.mask_stamps      = mask_stamps      # list of mask stamps, (mask=1 where OTHER planets transit)
        self.model_stamps     = model_stamps     # list of model stamps

        self.stamp_cadence    = stamp_cadence    # 'short', 'long', or 'none'
        self.stamp_coverage   = stamp_coverage   # fraction of transit/baseline covered by useable cadences
        self.stamp_chisq      = stamp_chisq      # chi-sq per transit

        self.icov             = icov             # inverse covariance matrix
        
    

    def pshape_names(self):
        '''
        Convenience function to return names of transit shape parameters
        '''      
        return ['T0', 'P', 'rp', 'zeta', 'b^2', 'e^1/2*sinw', 'e^1/2*cosw', 'u1', 'u2']



    def pshape_values(self, Rstar=1.0):
        '''
        Convenience function to print transit shape parameters with names
        '''
        print('T0\t %.4f'        %self.pshape[0])
        print('P\t %.4f'         %self.pshape[1])
        print('rp (rE)\t %.4f'   %(self.pshape[2]*Rstar*RSRE))
        print('dur \t %.4f'      %(24/self.pshape[3]))
        print('b\t %.4f'         %(np.sqrt(self.pshape[4])))
        print('esinw\t %.4f'     %self.pshape[5])
        print('ecosw\t %.4f'     %self.pshape[6])
              
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
            print('%d out of %d transits rejected with high chisq' %(np.sum(reject_chisq[enough_pts]), np.sum(nonempty)))
            print('%d out of %d transits rejected with insufficient coverage' %(np.sum(~enough_pts[nonempty]), np.sum(nonempty)))

        # save the results
        self.quality = enough_pts * ~reject_chisq

        return None


    def grab_stamps(self, stamptype, cadence='any'):
        '''
        stamptype: 'time', 'flux', 'error', 'mask', or 'model'
        cadence: 'short', 'long', or 'any'
        '''
        if stamptype == 'time':  stamps = self.time_stamps
        if stamptype == 'flux':  stamps = self.flux_stamps
        if stamptype == 'error': stamps = self.error_stamps
        if stamptype == 'mask':  stamps = self.mask_stamps
        if stamptype == 'model': stamps = self.model_stamps
      
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


    
    def update_model_stamps(self, Rstar):
        '''
        Update model stamps by calculating based on pshape and tts
        '''
        tts         = self.tts[self.quality]
        time_stamps = self.grab_stamps('time')
        cadences    = self.stamp_cadence[self.quality]
        
        model_flux  = calculate_model_flux(self.pshape, Rstar, tts, time_stamps, cadences)

        new_model_stamps = []
        j = 0
        for i, good in enumerate(self.quality):
            if good:
                new_model_stamps.append(model_flux[j])
                j += 1
            else:
                new_model_stamps.append(np.array([]))

        self.model_stamps = new_model_stamps
                
        return None



    def update_ephemeris(self):
        '''
        Fit a linear ephemeris to transit times; update period and epoch
        '''
        ephemeris = np.polyfit(self.index[self.quality], self.tts[self.quality], 1)
        
        self.period = ephemeris[0]
        self.epoch  = ephemeris[1]%ephemeris[0]
        
        return None



    def plot_folded_lightcurve(self, undersample=10):
        '''
        Plot a phase-folded lightcurve

        undersample: factor to under sample short cadence data to speed up plotting (default=10)
        '''
        # grab long cadence data
        lcts = self.grab_stamps('time', 'long')
        lcfs = self.grab_stamps('flux', 'long')
        lces = self.grab_stamps('error', 'long')

        # fold on transit times
        lc_folded_time_stamps = []
        for i, stamp in enumerate(lcts):
            lc_folded_time_stamps.append(stamp-self.tts[self.quality*(self.stamp_cadence=='long')][i])

        # linearize and sort
        if len(lc_folded_time_stamps) > 0:
            order   = np.argsort(np.hstack(lc_folded_time_stamps))
            lctime  = np.hstack(lc_folded_time_stamps)[order]
            lcflux  = np.hstack(lcfs)[order]
            lcerror = np.hstack(lces)[order]
        else:
            lctime, lcflux, lcerror = None, None, None

        # grab long cadence data
        scts = self.grab_stamps('time', 'short')
        scfs = self.grab_stamps('flux', 'short')
        sces = self.grab_stamps('error', 'short')

        # fold on transit times
        sc_folded_time_stamps = []
        for i, stamp in enumerate(scts):
            sc_folded_time_stamps.append(stamp-self.tts[self.quality*(self.stamp_cadence=='short')][i])

        # linearize and sort
        if len(sc_folded_time_stamps) > 0:
            order   = np.argsort(np.hstack(sc_folded_time_stamps))
            sctime  = np.hstack(sc_folded_time_stamps)[order][::undersample]
            scflux  = np.hstack(scfs)[order][::undersample]
            scerror = np.hstack(sces)[order][::undersample]
        else:
            sctime, scflux, scerror = None, None, None

        # calculate model
        btp = pshape_to_btp(self.pshape)
        btp.t0 = 0.0
        
        if sctime is not None:
            modeltime = sctime*1.0
            btm = batman.TransitModel(btp, modeltime)
        elif lctime is not None:
            modeltime = lctime*1.0
            btm = batman.TransitModel(btp, modeltime, supersample_factor=30, exp_time=LCIT/60/24)
        else:
            raise ValueError('None values for both lctime and sctime')
        
        modelflux = btm.light_curve(btp)

        # plot the data
        if scflux is not None:
            ylower = np.percentile(scflux,0.3)
            yupper = np.percentile(scflux,99.7)
        elif lcflux is not None:
            ylower = np.percentile(lcflux,0.3)
            yupper = np.percentile(lcflux,99.7)
            
        fig = plt.figure(figsize=(12,6))
        if sctime is not None: plt.plot(sctime, scflux, '.', c='grey', alpha=0.2)
        if lctime is not None: plt.plot(lctime, lcflux, 'k.', ms=12, alpha=0.3)
        plt.plot(modeltime, modelflux, c='darkorange', lw=3)
        plt.xlim(modeltime.min(), modeltime.max())
        plt.ylim(ylower,yupper)
        plt.xlabel('Time from transit midpoint [days]', fontsize=20)
        plt.ylabel('Normalized flux', fontsize=20)
        plt.show()
        
        return fig



    def plot_ttvs(self):
        # grab transit times, indexes, and uncertainties
        tts   = self.tts[self.quality]
        index = self.index[self.quality]
        
        if self.tts_err is not None:
            tts_err = self.tts_err[self.quality]
        else:
            tts_err = np.zeros_like(tts)
            
        # make a linear ephemeris; calculate O-C
        ephem_fit = np.polyfit(index, tts, 1)
        ephemeris = np.polyval(ephem_fit, index)
        omc       = tts-ephemeris

        # calculate a periodogram (use angular frequencies)
        freqs = np.linspace(2*pi/1600, pi/self.period,len(tts))
        pgram = sig.lombscargle(tts,omc,freqs)
        harmonics = np.sort(np.hstack([1/np.arange(1,10), np.arange(2,10)]))
        
        # make the plot
        fig = plt.figure(figsize=(12,6))
        
        ax = plt.subplot2grid(shape=(4,3), loc=(0,0), colspan=2, rowspan=3)
        ax.errorbar(tts, omc*24*60, yerr=tts_err*24*60, fmt='o', color='grey')
        ax.set_xticklabels('')
        ax.set_ylabel('O-C [min]', fontsize=16)
        ax.set_ylim(np.percentile(omc*24*60,0.3)-np.median(tts_err*24*60),np.percentile(omc*24*60,99.7)+np.median(tts_err*24*60))

        ax = plt.subplot2grid(shape=(4,3), loc=(3,0), colspan=2, rowspan=1)
        ax.errorbar(tts, omc*24*60, yerr=tts_err*24*60, fmt='o', color='grey')
        ax.set_xlabel('time [BJKD]', fontsize=16)

        ax = plt.subplot2grid(shape=(4,3), loc=(0,2), colspan=1, rowspan=3)
        ax.plot(2*pi/freqs, pgram, color='grey', label=np.round(2*pi/freqs[np.argmax(pgram)],2))
        ax.legend(loc='upper right')
        ax.set_xscale('log')
        ax.set_xlim(2*pi/freqs.max(),2*pi/freqs.min())
        ax.yaxis.tick_right()
        ax.vlines(2*pi/freqs[np.argmax(pgram)], pgram.min(), pgram.max(), color='darkorange', linestyle='--')

        plt.show()

        return fig
    

#####



#################################################
# FUNCTIONS FOR READING IN AND DOWNLOADING DATA #
#################################################


# Read in a cks.csv files
def read_csv_file(filename, k_index=0, v_index=1):
    data = []
    with open(filename) as infile:
        reader = csv.reader(infile)

        for row in reader:
            data.append(row)

        keys   = data[k_index]
        values = data[v_index:]

        
        return keys, values
    


# Pull data from csv files read in with wings.read_csv_file()
def get_csv_data(keyname,keys,values):
    '''
    keyname = (string) of column definition, see CKS documentation
    '''
    kid = keys.index(keyname)
    
    outdata = []
    for row in values:
        outdata.append(row[kid])
    
    return outdata



# Download a KeplerLightCurveFile from MAST using lightkurve
def download_lkf(target, qstart=1, qend=17, qlist=None, cadence='long', download_dir=None, verbose=False):
    '''
    Download a series of KeplerLightCurveFiles from MAST using lightkurve
    Can either download a contiguous block over (qstart,qend) or a slist of specified quarters 

    target: name of target lightkurve (see lightkurve documentation)
    qstart: first quarter do download (default=1)
    qend: last quarter to download (default=17)
    qlist: list of specific quarters to download
    cadence: 'short' or 'long' (default='long')
    download_dir: directory to download files to (default=None --> cache)
    verbose: boolean flag; 'True' to print quarter number

    Returns a list of lightkurve.KeplerLightCurveFile() objects
    '''
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



# Make a transit mask for a Planet (1=near transit; 0=not)
def make_transitmask(planet, time, masksize=1.5):
    '''
    Make a transit mask for a Planet (1=near transit; 0=not)
    
    planet: wings.Planet() object
    time: array of time values
    masksize: number of transit durations from transit center to consider near transit (default=1.5)
    
    --returns
        transitmask: boolean array (1=near transit; 0=not)
    '''
    tts = planet.tts
    duration = planet.duration
    
    transitmask = np.zeros_like(time, dtype='bool')
    
    for t0 in tts:
        neartransit = np.abs(time-t0)/duration < masksize
        transitmask += neartransit
    
    return transitmask



# Perform basic flattening, outlier rejection, etc. and joining with PDCSAP lightcurves
def detrend_and_join(lkflist, window_length=101, break_tolerance=25, polyorder=3, sigma_upper=5, masklist=None):
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

    pdcsap, outliers = pdcsap.remove_outliers(sigma_lower=float('inf'), sigma_upper=sigma_upper, iters=None, return_mask=True)
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



# Remove cadences flagged by Kepler pipeline
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



# Put a list of Planet objects in order by period
def sort_by_period(planets):
    '''
    Put a list of wings.Planet() objects in order by period
    
    planets: list of wings.Planet() objects
    '''
    NPL = len(planets)
    
    periods = []
    for planet in planets:
        periods.append(planet.period)

    periods = np.asarray(periods)
    order = np.argsort(periods)

    sorted_planets = []
    for npl in range(NPL):
        sorted_planets.append(planets[order[npl]])

    planets = sorted_planets
        
    return planets




####################################
# FUCTIONS FOR WORKING WITH STAMPS #
####################################



# cut out a stamp centered on each transit time
def cut_stamps(planet, time, data, stampsize=1.5):
    '''
    Cut out a stamp centered on each transit time from a full Kepler lightcurve

    planet: wings.Planet() object
    time: array of time values
    data: array of data values for stamp (same length as time)
    stampsize: distance from each transit center to cut, in transit durations (default=1.5)
    '''
    stamps = []

    # cut out the stamps
    for t0 in planet.tts:
        neartransit = np.abs(time - t0)/planet.duration < stampsize
        stamps.append(data[neartransit])
        
    return stamps


# Combine short and long cadence stamps, using SC wherever available
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



# Remove cadences from stamps where other planets transit
def mask_overlapping_transits(planet):
    '''
    Remove cadences from stamps where other planets transit

    planet: wings.Planet() object

    -- automatically updates time_, flux_, error_, and mask_stamps on Planet object
    '''
    for i, m in enumerate(planet.mask_stamps):
        if len(m) > 0:
            planet.time_stamps[i]  = planet.time_stamps[i][~m]
            planet.flux_stamps[i]  = planet.flux_stamps[i][~m]
            planet.error_stamps[i] = planet.error_stamps[i][~m]
            planet.mask_stamps[i]  = planet.mask_stamps[i][~m]

    return None



def clip_outlier_cadences(planet, sigma=5.0, kernel_size=7):
    '''
    Do some iterative sigma rejection on each stamp

    planet: wings.Planet() object
    sigma: rejection threshold for clipping (default=5.0)
    kernel_size: size of window for median filter (default=7)

    -- automatically updates time_, flux_, and error_stamps on Planet object
    '''
    p = planet

    for i, f in enumerate(p.flux_stamps):
        if len(f) > 0:
            loop = True
            while loop:
                smoothed = sig.medfilt(p.flux_stamps[i], kernel_size=kernel_size)
                outliers = np.abs(p.flux_stamps[i]-smoothed)/p.error_stamps[i] > sigma

                if np.sum(outliers) > 0:
                    p.time_stamps[i]  = p.time_stamps[i][~outliers]
                    p.flux_stamps[i]  = p.flux_stamps[i][~outliers]
                    p.error_stamps[i] = p.error_stamps[i][~outliers]
                else:
                    loop = False

    return None



# Fit a linear polynomial out-of-transit flux to flatten data flux stamps
def flatten_stamps(planet, jitter=0.1):
    '''
    Fit a linear polynomial to out-of-transit flux to flatten data flux stamps

    planet: wings.Planet() object
    jitter: fudge factor to avoid fitting in-transit flux if there are unresolved TTVs (default=0.1)

    -- automatically updates flux_stamps on Planet object
    '''
    p = planet

    for i, flux in enumerate(p.flux_stamps):
        if len(flux) > 0:
            time = p.time_stamps[i]

            intransit = np.abs(time-p.tts[i])/p.duration < 0.5+jitter

            if np.sum(~intransit) > 0:
                coeffs = np.polyfit(time[~intransit],flux[~intransit],1)
                linfit = np.polyval(coeffs, time)
            else:
                linfit = 1.0

            p.flux_stamps[i] = flux/linfit

    return None





#############################################
# FUCTIONS FOR GENERATING COVARIANCE MATRIX #
#############################################


# Generate an autocorrelation function on out-of-transit flux data
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



# Model a linear combination of decaying sinusoids -- A*exp(-x/tau)*sin(k*x-phi)
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



# Residuals for modeling the autocorrelation function -- use with op.leastsq()
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



# Log-likelihood for modeling the autocorrelation function -- use with op.minimize()
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



# Model the autocorrelation function as a sum of decaying sinusoids
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
    
    # plot the results
    if do_plots:
        xcor_lc, acor_lc = bin_acor(xcor, acor)
        
        fig = plt.figure(figsize=(18,6))

        ax = plt.subplot2grid(shape=(4,1), loc=(0,0), rowspan=3, colspan=1)
        ax.plot(xcor, acor, c='orange', label='ACF from SC data')
        ax.plot(xcor, acor_model, c='mediumblue', lw=2, label='Model fit')
        ax.plot(xcor_lc, acor_lc, 'ro', label='Binned to LC')
        ax.plot(xcor, acor_guess, color='cornflowerblue', ls=':', lw=2)
        ax.set_xticks([])
        ax.set_xlim(0, xcor.max())
        ax.set_ylabel('Normalized ACF')
        ax.legend(loc='upper right')

        ylim = 1.1*np.max(np.abs(acor-acor_model))

        ax = plt.subplot2grid(shape=(4,1), loc=(3,0), rowspan=1, colspan=1)
        ax.plot(xcor, acor-acor_model, c='mediumblue')
        ax.set_xlabel('lag time [hrs]', fontsize=16)
        ax.set_xlim(0, xcor.max())
        ax.set_ylim(-ylim, ylim)

        plt.show()
        
    return acor_model, acor_theta



# Bin autocorrelation function from short cadence to long cadence
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



# Make a covariance matrix from the autocorrelation function
def make_covariance_matrix(xcor, theta, Nsize, sigma_white, sigma_red, do_plots=False):
    '''
    Make a covariance matrix from the autocorrelation function
    
    xcor: lag-time values used
    theta: fitted parameters from modeling autocorrelation function
    Nsize: output size of matrix (should be equal to length of longest stamp)
    sigma_white: 1-sigma error due to gaussian noise
    sigma_red: 1-sigma error due to correlated noise
    
    --returns
        covmatrix: covariance matrix
    '''
    # make diagonal matrix; per-point variance of photon noise
    dij = np.diag(np.ones(Nsize)*sigma_white**2)

    # make a lag matrix
    lagmatrix = np.zeros((Nsize,Nsize))
    xlag = np.arange(Nsize)

    for i in range(Nsize):
        lagmatrix[i] += np.roll(xlag,i)

    lagmatrix = np.triu(lagmatrix)
    lagmatrix += lagmatrix.T
    lagmatrix *= xcor[1]-xcor[0]
    lagmatrix += SCIT/3600

    # make covariance matrix
    covmatrix = decaying_sinusoid(lagmatrix, theta) 
    covmatrix /= covmatrix.max()
    covmatrix *= sigma_red**2
    covmatrix += dij
    
    icovmatrix = np.linalg.inv(covmatrix)

    # plot the results
    if do_plots:
        fig, axes = plt.subplots(1,2, figsize=(8,8))
        ax = axes[0]
        ax.imshow(covmatrix, vmin=2*covmatrix.min(), vmax=0.05*covmatrix.max())
        ax = axes[1]
        ax.imshow(icovmatrix, vmin=0.5*icovmatrix.min(), vmax=0.02*icovmatrix.max())
        plt.show()
    
    return covmatrix




##############################################
# FUNCTIONS FOR GENERATING LIGHTCURVE MODELS #
##############################################



# Convert pshape vector to a parameterization convenient for starry and exoplanet
def pshape_to_pstarry(pshape, Rstar):
    '''
    Convert pshape vector to a parameterization convenient for starry and exoplanet
        - pshape semimajor axis should be in units of STELLAR radii
        - pstarry semimajor axis should be in units of SOLAR radii
        - rp is really rp/Rstar
        - zeta is inverse transit duration (see Pal 2008 for motivation)
    
    pshape: [T0, P, rp, zeta, b^2, e^1/2*sinw, e^1/2*cosw, u1, u2]
    Rstar: stellar radius [solar radii]    

    -- returns pstarry: [T0, P, rp, a, inc, ecc, w, u1, u2]
    '''
    # pull quantities
    T0, P, rp, zeta, b2, esinw, ecosw, u1, u2 = pshape
    
    # calculate transit duration and impact parameter
    D = 1/zeta
    b = np.sqrt(b2)
    
    # calculate eccentricity and argument
    ecc = esinw**2 + ecosw**2
    w   = np.arctan2(esinw,ecosw)
    
    # calculate ecentricity corrections
    E1 = (1-ecc**2)/(1+ecc*np.sin(w))                                   # Winn 2010, Eq.7 (ecc portion)
    E2 = E1/np.sqrt(1-ecc**2)                                           # Winn 2010, Eq.16
    
    # calculate inclination
    k2 = (1+rp)**2
    acosi = b/E1
    asini = np.sqrt(k2-b2)/np.sin((pi*D)/(P*E2))
    inc   = np.arctan2(asini,acosi)

    # calculate semi-major axis
    a = b/E1/np.cos(inc)*Rstar

    return np.array([T0, P, rp, a, inc, ecc, w, u1, u2])



# Convert pstarry to pshape
def pstarry_to_pshape(pstarry, Rstar):
    '''
    Convert pstarry to pshape
        - pshape semimajor axis should be in units of STELLAR radii
        - pstarry semimajor axis should be in units of SOLAR radii
        - rp is really rp/Rstar
        - zeta is inverse transit duration (see Pal 2008 for motivation)

    pstarry: [T0, P, rp, a, inc, ecc, w, u1, u2]
    Rstar: stellar radius [solar radii]    

    -- returns pshape: [T0, P, rp, zeta, b^2, e^1/2*sinw, e^1/2*cosw, u1, u2]

    '''
    # pull quantities
    T0, P, rp, a, inc, ecc, w, u1, u2 = pstarry

    # calculate ecentricity corrections
    E1 = (1-ecc**2)/(1+ecc*np.sin(w))                                   # Winn 2010, Eq.7 (ecc portion)
    E2 = E1/np.sqrt(1-ecc**2)                                           # Winn 2010, Eq.16

    # calculate impact parameter
    b = a*E1*np.cos(inc)                                                # Winn 2010, Eq.7
    b2 = b**2
    
    # calculate transit duration
    k2 = (1+rp)**2                                                      # convenience variable
    D  = (P/pi)*E2*np.arcsin(np.sqrt(k2-b2)/(a*np.sin(inc)))            # Winn 2010, Eq.14

    # calculate eccentricity components
    esinw = np.sqrt(ecc)*np.sin(w)
    ecosw = np.sqrt(ecc)*np.cos(w)

    return np.array([T0, P, rp, 1/D, b2, esinw, ecosw, u1, u2])



def calculate_model_flux(pshape, Rstar, tts, time_stamps, cadences):
    '''
    Generate a list of lightcurve models a series of individual transits using starry & exoplanet

    pshape: array of transit parameters -- [T0, P, rp, zeta, b^2, esinw, ecosw, u1, u2]
    Rstar: stellar radius [solar radii]
    omc: list of observed-minus-calculated ttv offset from a linear ephemeris [days]
    time_stamps: list of time stamps (one per transit)
    cadences: cadence of each transit

    -- returns modellist: a list of model_stamps
    '''
    # check that vector lengths are consistent
    if len(time_stamps) != len(tts):
        raise ValueError('inconsistent input lengths')
    if len(cadences) != len(tts):
        raise ValueError('inconsistent input lengths')
   
    # convert pshape parameters to a form more easily used by starry
    T0, P, rp, a, inc, ecc, w, u1, u2 = pshape_to_pstarry(pshape, Rstar)
    b = np.sqrt(pshape[4])


    # split up short and long cadence; keep count of how many are in each list
    sc_time = []
    lc_time = []
    sc_pos = [0]
    lc_pos = [0]

    for i, cad in enumerate(cadences):
        if cad == 'short':
            sc_time.append(time_stamps[i])
            sc_pos.append(len(time_stamps[i]))
        elif cad == 'long':
            lc_time.append(time_stamps[i])
            lc_pos.append(len(time_stamps[i]))
        else:
            raise ValueError('all cadences must be "short" or "long"')

    sc_pos = np.cumsum(np.array(sc_pos))
    lc_pos = np.cumsum(np.array(lc_pos))

    # generate the light curve model
    exoSLC = exo.StarryLightCurve([u1, u2])
    orbit  = exo.orbits.KeplerianOrbit(t0=T0, period=P, a=a, b=b, ecc=ecc, omega=w, r_star=Rstar)

    sc_model = 1 + exoSLC.get_light_curve(orbit=orbit, r=rp, t=np.hstack(sc_time), texp=SCIT/3600/24, oversample=1).eval()
    lc_model = 1 + exoSLC.get_light_curve(orbit=orbit, r=rp, t=np.hstack(lc_time), texp=LCIT/60/24, oversample=30).eval()

    # turn it into a list
    j = 0
    k = 0
    N = len(tts)
    modellist = [None]*N
    for i, cad in enumerate(cadences):
        if cad == 'short':
            modellist[i] = sc_model[sc_pos[j]:sc_pos[j+1]]
            j += 1
        elif cad == 'long':
            modellist[i] = lc_model[lc_pos[k]:lc_pos[k+1]]
            k += 1
            
    return modellist


# Calculate the error-scaled residuals when fitting ttvs (for use in op.leastsq)
def residuals_for_shape(p0, p1, varyp, Rstar, tts, time_stamps, flux_stamps, cadences, icov):
    '''
    Calculate the error-scaled residuals when fitting ttvs (for use in op.leastsq)


    p0, p1, varyp are in parameters -- [T0, P, rp, zeta, b^2, esinw, ecosw, u1, u2]

    p0: list parameters to vary
    p1: list parameters to hold fixed
    varyp: boolean array (1 = vary; 0 = fix) setting which parameters to vary in LM fit
    Rstar: stellar radius [solar radii]
    tts: list of midtransit times [t0, t1, t2,...]
    time_stamps: list of time stamps
    flux_stamps: list of corresponding fluxes
    cadences: cadence of each transit
    icov: list of covariance matrices (one per transit)

    -- returns res: error-scalled residuals
    '''
    print('CALLING residuals_for_shape()')
    
    # make shape parameter vector
    pshape = np.zeros_like(varyp, dtype='float')
    pshape[varyp] = p0
    pshape[~varyp] = p1

    # get number of transits being fitted
    N = len(time_stamps)

    # rename data flux (it's clunky but makes the equations and I/O easier to follow)
    data_flux = flux_stamps    

    # calculate the model flux
    model_flux = calculate_model_flux(pshape, Rstar, tts, time_stamps, cadences)


    # calculate residuals - op.leastsq expects a vector same length as data
    res = [None]*N
    for i in range(N):
        y = data_flux[i] - model_flux[i]
        res[i] = np.dot(y.T,np.dot(icov[i],y))*np.ones_like(data_flux[i])/len(data_flux[i])

    res = np.hstack(res)
    
    # add Rayleigh prior on eccentricity; width taken from Mills+ 2019
    ecc = pshape[5]**2 + pshape[6]**2
    rayleigh_prior = stats.rayleigh.pdf(0.0355, scale=0.0355)/stats.rayleigh.pdf(ecc, scale=0.0355)
    
    return res*rayleigh_prior



# Calculate mid-transit times from a specified ttv model
def calculate_tts_from_model(transit_index, ptime, method):
    '''
    Calculate mid-transit times from a specified ttv model
    
    transit_index: list of indexes corresponding to tts
    ptime: list of parameters describing ttv model
            [L0, L1]                 if method == 'linear'
            [Q0, Q1, Q2]             if method == 'quadratic'
            [T0, P, freq, Amp, tphi] if method == 'sinusoidal'
    method: 'linear', 'quadratic', 'cubic', 'spectroscopic'            
    '''
    # get transit times
    if method == 'linear':
        L0, L1 = ptime
        tts_here = L0 + transit_index*L1
        
    elif method == 'quadratic':
        Q0, Q1, Q2 = ptime
        tts_here = Q0 + Q1*transit_index + Q2*transit_index**2
        
    elif method == 'sinusoidal':
        T0, P, freq, Amp, tphi = ptime
        tts_here = T0 + transit_index*P                         # linear ephemeris
        tts_here += Amp*np.sin(2*pi*freq*(tts_here-tphi))       # with sinusoidal perturbation
    
    else:
        raise ValueError('unsupported fitting method')
        
    return tts_here



# Slide transit model away from transit center and find minimum chi-sq
def slide_ttvs(planet, Rstar, slide_offset=0.5, delta_chisq=2.0, do_plots=False, verbose=False):
    '''
    Slide transit model away from transit center and find minimum chi-sq

    planet: wings.Planet() object
    Rstar: stellar radius [solar radii]
    slide_offset: number of transit durations away from center to move slide (default=0.5)
    delta_chisq: change in chisq value from minimum to use when fitting parabola for minimum (default=2.0) 
    do_plots: boolean flag, True to plot lightcurves and chisq-vs-offset

    -- returns
        tts_new: maximum likelihood time for each transit center
        tts_err: 1-sigma error bars on tts_new

    NOTE: works with only good transits throughout, then interpolates to empty/bad transits
    '''
    p = planet

    # get the tts and transit indexes
    tts_here = p.tts[p.quality]
    transit_index = p.index[p.quality]

    # grab stamps
    time_stamps  = p.grab_stamps('time')
    flux_stamps  = p.grab_stamps('flux')
    error_stamps = p.grab_stamps('error')

    cadence = p.stamp_cadence[p.quality]

    # pull pstarry values
    T0, P, rp, a, inc, ecc, w, u1, u2 = pshape_to_pstarry(p.pshape, Rstar)

    # grab covariance matrices
    icov = p.grab_icov()

    # make arrays for new transit times and uncertainties
    tts_new = np.zeros_like(tts_here)
    err_new = np.zeros_like(tts_here)

    # determine number of points to use
    Npts = np.round(p.duration*5*slide_offset*24*3600/SCIT*1.618)
    Npts += Npts % 2 + 1
    Npts = int(Npts)

    # estimate uncertainty for each transit time
    for i, t in enumerate(tts_here):
        # create vector of transit times and chisq offset from best-fit t0
        tc_vector = tts_here[i] + np.linspace(-p.duration*slide_offset, p.duration*slide_offset, Npts)
        chisq_vector = np.zeros_like(tc_vector)

        # grab stamps
        tstamp = time_stamps[i]
        fstamp = flux_stamps[i]
        estamp = error_stamps[i]

        # slide along transit time vector and calculate chisq
        for j, tc in enumerate(tc_vector):
            orbit = exo.orbits.KeplerianOrbit(t0=tc, period=P, a=a, incl=inc, ecc=ecc, omega=w, r_star=Rstar)
        
            if cadences[i] == 'short':
                mstamp = exo.StarryLightCurve([u1,u2]).get_light_curve(orbit=orbit, r=rp, t=time_stamps[i], \
                                                                      texp=SCIT/3600/24, oversample=1).eval()
            elif cadences[i] == 'long':
                mstamp = exo.StarryLightCurve([u1,u2]).get_light_curve(orbit=orbit, r=rp, t=time_stamps[i], \
                                                                      texp=LCIT/60/24, oversmample=30).eval()
            y = fstamp-mstamp
            chisq_vector[j] = np.dot(y.T,np.dot(icov[i],y))

        # grab points near minimum chisq
        min_chisq = chisq_vector.min()
        tcfit = tc_vector[chisq_vector < min_chisq+delta_chisq]
        x2fit = chisq_vector[chisq_vector < min_chisq+delta_chisq]

        # eliminate points far from the local minimum
        spacing = np.median(tcfit[1:]-tcfit[:-1])
        faraway = np.abs(tcfit-np.median(tcfit))/spacing > len(tcfit)/2

        tcfit = tcfit[~faraway]
        x2fit = x2fit[~faraway]

        # make sure there are at least 8 pts to fit a parabola
        if len(tcfit) < 8:
            tts_new[i] = np.nan
            err_new[i] = np.nan
            bad_initial_fit = False
        else:
            quad_coeffs = np.polyfit(tcfit, x2fit, 2)
            quadfit = np.polyval(quad_coeffs, tcfit)
            qtc_min = -quad_coeffs[1]/(2*quad_coeffs[0])
            qx2_min = np.polyval(quad_coeffs, qtc_min)
            qtc_err = np.sqrt(1/quad_coeffs[0])

            tts_new[i] = np.mean([qtc_min,np.median(tcfit)])
            err_new[i] = qtc_err

            # check that the fit is well-conditioned (ie. a negative t**2 coefficient)
            if quad_coeffs[0] < 0.0:
                bad_initial_fit = True
            else:
                bad_initial_fit = False
            
        # for poorly conditioned fits, refit a narrower range around the minimum
        tightfactor = 1
        loop = False
        
        if bad_initial_fit:
            if verbose: print('\t', int(tts_new[i]), 'bad initial fit')
            loop = True
            
            while loop:
                tightfactor += 1
                
                min_chisq = chisq_vector.min()
                tcfit = tc_vector[chisq_vector < min_chisq+delta_chisq/np.sqrt(tightfactor)]
                x2fit = chisq_vector[chisq_vector < min_chisq+delta_chisq/np.sqrt(tightfactor)]

                spacing = np.median(tcfit[1:]-tcfit[:-1])
                faraway = np.abs(tcfit-np.median(tcfit))/spacing > len(tcfit)/2
                
                tcfit = tcfit[~faraway]
                x2fit = x2fit[~faraway]

                # make sure there are at least 8 pts to fit a parabola
                if len(tcfit) < 8:
                    loop = False
                    tts_new[i] = np.nan
                    err_new[i] = np.nan
                else:
                    quad_coeffs = np.polyfit(tcfit, x2fit, 2)
                    quadfit = np.polyval(quad_coeffs, tcfit)
                    qtc_min = -quad_coeffs[1]/(2*quad_coeffs[0])
                    qx2_min = np.polyval(quad_coeffs, qtc_min)
                    qtc_err = np.sqrt(1/quad_coeffs[0])

                    # if a good fit was found save and inflate errors
                    if quad_coeffs[0] > 0.0:
                        loop = False
                        tts_new[i] = np.mean([qtc_min,np.median(tcfit)])
                        err_new[i] = qtc_err*np.sqrt(tightfactor)
                    

        # check that uncertainity is not overestimated (this can occur when the minimum in not parabola-like)
        if len(tcfit) > 0:
            high_too_high = tts_new[i] + err_new[i] > tcfit.max()
            low_too_low   = tts_new[i] - err_new[i] < tcfit.min()

            if high_too_high*low_too_low*(bad_initial_fit != True):
                if verbose: print('\t', int(tts_new[i]),'rescaling uncertainty')
                err_new[i] = (tcfit.max()-tcfit.min())/2

        # check that the recovered transit time is within the expected range
        out_of_bounds_low  = tts_new[i] < tc_vector.min()
        out_of_bounds_high = tts_new[i] > tc_vector.max()

        if out_of_bounds_low+out_of_bounds_high:
            if verbose: print('\t', int(tts_new[i]), 'out of bounds')
            tts_new[i] = np.nan
            err_new[i] = np.nan

        # plot the results
        if do_plots*(np.isnan(tts_new[i])==False):
            if verbose: print('transit time = %.3f +/- %.3f' %(tts_new[i], err_new[i]))
            
            btp    = pshape_to_btp(p.pshape)
            btp.t0 = tts_new[i]
            btm    = batman.TransitModel(btp, tstamp)
            mstamp = btm.light_curve(btp)

            spacing = np.median(tcfit[1:]-tcfit[:-1])
            faraway = np.abs(tc_vector-np.median(tcfit))/spacing > len(tcfit)/2

            use_wide = (chisq_vector < min_chisq+12)
            use_1    = (chisq_vector < min_chisq+1)*~faraway
            use_2    = (chisq_vector < min_chisq+2)*~faraway

            if cadence[i] == 'long':  undersample = 1
            if cadence[i] == 'short': undersample = 10

            fig, axes = plt.subplots(1,2, figsize=(12,3))

            ax = axes[0]
            ax.errorbar(tstamp[::undersample]-tts_new[i], fstamp[::undersample], yerr=estamp[::undersample], fmt='ko')
            ax.plot(tstamp-tts_new[i], mstamp, c='orange', lw=3)
            ax.set_title(int(tts_new[i]))
            
            ax = axes[1]
            ax.plot(tc_vector[use_wide], chisq_vector[use_wide], 'k.')
            ax.plot(tc_vector[use_2], chisq_vector[use_2], '.', c='orange')
            ax.plot(tc_vector[use_1], chisq_vector[use_1], '.', c='red')
            ax.plot(tcfit, quadfit, c='r', lw=1)
            ax.plot(tts_new[i], qx2_min, 'r*', ms=15, mec='k', mew=0.5)
            ax.plot(tts_new[i]-err_new[i], qx2_min+1, 'r|', ms=15, mew=3)
            ax.plot(tts_new[i]+err_new[i], qx2_min+1, 'r|', ms=15, mew=3)
            ax.plot(tts_new[i], qx2_min+1.5, 'kv', ms=10, fillstyle='none')
           
            plt.show() 

    # interpolate where fits where bad  
    nan_tts = np.isnan(tts_new) + np.isnan(err_new)
    tts_new[nan_tts] = np.interp(transit_index[nan_tts], transit_index[~nan_tts], tts_new[~nan_tts])
    err_new[nan_tts] = p.duration*2*slide_offset

    # do not update empty transits (these will be handled later); do update error bars on their tts
    tts_out = np.copy(p.tts)
    err_out = np.zeros_like(p.tts)

    empty = ~p.quality
    err_out[empty]  = p.duration*2*slide_offset
    tts_out[~empty] = tts_new
    err_out[~empty] = err_new

    return tts_out, err_out



# Run a Levenberg-Marquardt minimization, first for transit shape, then for ttvs
def do_LM_fit(planet, Rstar, varyp, ttv_method, do_plots=False, verbose=False):
    '''
    Run a Levenberg-Marquardt minimization, first for transit shape, then for ttvs

    planet: wings.Planet() object
    Rstar: stellar radius [solar radii]
    varyp: boolean array of which transit parameters to vary -- [T0, P, rp, zeta, b^2, esinw, ecosw, u1, u2]
    ttv_method: 'linear', 'quadratic', 'sinusoidal', or 'slide'
    do_plots: boolean flag; 'True' to display plots during fitting

    --returns
        planet: wings.Planet() object with values updated from results of LM fit
        X2_shape: chi-sq value from fitting transit shape
        X2_time: chi-sq value from fitting ttvs
    '''
    ### PART 1 -- PREPARATION ###
    p = planet

    # set up parameter vectors
    p0 = p.pshape[varyp]
    p1 = p.pshape[~varyp]   
    
    # grab stamps (time, flux, error)
    time_stamps  = p.grab_stamps('time')
    flux_stamps  = p.grab_stamps('flux')
    error_stamps = p.grab_stamps('error')
    
    # grab transit times and indexes
    tts = p.tts[p.quality]
    transit_index = p.index[p.quality]
    cadences = p.stamp_cadence[p.quality]

    # grab covariance matricies
    icov = p.grab_icov()


    ### PART 2 -- FITTING TRANSIT SHAPE ###
    
    # do the least squares fit for shape
    print('...fitting shape')
    pout_shape, success = op.leastsq(residuals_for_shape, p0, args=(p1, varyp, Rstar, tts, time_stamps, flux_stamps, cadences, icov))


    # update shape parameters
    p.pshape[varyp] = pout_shape*1.0
    p.depth         = p.pshape[2]**2
    p.duration      = 1./p.pshape[3]

    # update model stamps and calculate chi-sq
    p.update_model_stamps()    
    p.calculate_stamp_chisq()
    
    X2_shape = np.sum(p.stamp_chisq[p.quality])


    ### PART 3 -- FITTING TTVS ###
    
    # do the least squares fit for ttvs
    print('...fitting ttvs:', ttv_method)

    if ttv_method == 'linear':
        pout_time, success = op.leastsq(residuals_for_time, p.ptime, args=(p.pshape, ttv_method, btmlist, transit_index, \
                                                                           np.hstack(flux_stamps), np.hstack(error_stamps)))
    elif ttv_method == 'slide':
        tts_new, tts_err_new = slide_ttvs(planet, do_plots=do_plots, verbose=verbose)
        
    else:
        raise ValueError('unsupported ttv method')

    # update ptime and tts
    if ttv_method == 'linear':
        p.ptime = pout_time
        p.tts = calculate_tts_from_model(p.index, p.ptime, ttv_method)

    if ttv_method == 'slide':
        p.tts = tts_new
        p.tts_err = tts_err_new

    # update ephemeris (epoch & period)
    p.update_ephemeris()
    p.pshape[0] = p.epoch*1.0
    p.pshape[1] = p.period*1.0

    # update model stamps and calculate chi-sq
    p.update_model_stamps()
    p.calculate_stamps_chisq()
    
    X2_time = np.sum(p.stamp_chisq[p.quality])

    return p, X2_shape, X2_time



###################################
# FUNCTIONS FOR WORKING WITH TTVS #
###################################



def identify_uncertain_ttv_outliers(tts, omc, error, sigma=5.0, do_plots=False):
    '''
    Identify TTVs with unusually high uncertainties

    tts: array of transit times (days)
    omc: array of TTVs (O-C) [days]
    error: array of corresponding errors (days)
    index: array of corresponding transit indexes
    sigma: threshold to reject outliers; default=5.0
    do_plots: boolean flag, True to plot lightcurves and chisq-vs-offset

    -- returns
        outliers: boolean array of len(tts); True where TTVs have been rejected
    '''
    N = len(tts)
    
    # get scatter and uncertainty
    scatter = astropy.stats.mad_std(omc)
    uncertainty = np.median(error)

    # flag outliers
    outliers = np.abs(error-uncertainty)/np.std(error) > sigma

    # display the results
    if do_plots:      
        fig, ax = plt.subplots(1,1,figsize=(9,6))

        ylim = (np.maximum(np.abs(omc.min()),np.abs(omc.max())) + uncertainty)*24*60

        ax.errorbar(tts, omc*24*60, yerr=error*24*60, fmt='o', color='grey')
        ax.errorbar(tts[outliers], omc[outliers]*24*60, yerr=error[outliers]*24*60, fmt='o', color='firebrick')
        ax.set_ylabel('O-C [min]', fontsize=16)
        ax.set_ylim(-ylim,ylim)
        ax.set_xlabel('time [BJKD]', fontsize=16)

        plt.show()

    return outliers




def identify_global_ttv_outliers(tts, omc, error, sigma=5.0, do_plots=False):
    '''
    Identify TTV outliers via robust sigma clipping (based on MAD and median)

    tts: array of transit times (days)
    omc: array of TTVs (O-C) [days]
    error: array of corresponding errors (days)
    sigma: threshold to reject outliers; default=5.0
    do_plots: boolean flag, True to plot lightcurves and chisq-vs-offset

    -- returns
        outliers: boolean array of len(tts); True where TTVs have been rejected
    '''
    N = len(tts)
    
    # get scatter and uncertainty
    scatter = astropy.stats.mad_std(omc)
    uncertainty = np.median(error)

    # flag outliers
    outliers = np.abs(omc-np.median(omc))/scatter > sigma

    # display the results
    if do_plots:      
        fig, ax = plt.subplots(1,1,figsize=(9,6))

        ylim = (np.maximum(np.abs(omc.min()),np.abs(omc.max())) + uncertainty)*24*60

        ax.errorbar(tts, omc*24*60, yerr=error*24*60, fmt='o', color='grey')
        ax.errorbar(tts[outliers], omc[outliers]*24*60, yerr=error[outliers]*24*60, fmt='o', color='firebrick')
        ax.hlines(sigma*scatter*24*60,tts.min(),tts.max(),color='orange',linestyle='--')
        ax.hlines(-sigma*scatter*24*60,tts.min(),tts.max(),color='orange', linestyle='--')
        ax.set_ylabel('O-C [min]', fontsize=16)
        ax.set_ylim(-ylim,ylim)
        ax.set_xlabel('time [BJKD]', fontsize=16)

        plt.show()

    return outliers



def identify_local_ttv_outliers(tts, omc, error, window_length=9, known_outliers=None, do_plots=False, verbose=False):
    '''
    Identify TTV outliers based on a smoothed TTV model

    tts: array of transit times [days]
    omc: array of TTVs (O-C) [days]
    error: array of corresponding errors [days]
    window_length: number of points to fit a polynomial to surrounding each ttv (should be an odd integer >= 9)
    known_outliers: (optional) array of pre-identified outliers
    do_plots: boolean flag, True to plot lightcurves and chisq-vs-offset

    -- returns
        outliers: boolean array of len(tts); True where TTVs have been rejected
    '''
    N = len(tts)

    # if there are only a few TTVs, don't do local rejection
    if N < window_length*2:
        if verbose: print('    too few TTVs for local rejection')
        outliers = np.zeros_like(tts,dtype='bool')
        poly_trend = np.polyval(np.polyfit(tts, omc, 2, w=1/error), tts)
        return outliers, poly_trend

    # check that window length is odd; compute half window length
    if window_length % 2 != 1:
        raise ValueError('window_length must be an odd integer')
    else:
        wl2 = int(window_length/2)
    
    # set up outlier array
    if known_outliers is None:
        outliers = np.zeros_like(omc, dtype='bool')
    else:
        outliers = known_outliers.copy()

    # get scatter and uncertainty
    scatter = astropy.stats.mad_std(omc)
    uncertainty = np.median(error)

    # pad the arrays
    omc_padded = np.pad(omc, wl2, mode='constant', constant_values=np.nan)
    tts_padded = np.pad(tts, wl2, mode='constant', constant_values=np.nan)

    # reject 5-sigma outliers from a median filtered trend
    loop = True
    iterations = 0
    while loop:
        iterations += 1
        med_trend = np.zeros_like(omc)
        med_trend[~outliers] = sig.medfilt(omc[~outliers], kernel_size=window_length)
        med_trend[outliers]  = np.interp(tts[outliers], tts[~outliers], med_trend[~outliers])
        
        residuals = omc - med_trend
        residuals[outliers] = 0.0

        scale = np.sqrt(uncertainty**2+astropy.stats.mad_std(residuals[~outliers])**2)
        
        med_out = np.abs(residuals)/scale > 3.0

        if np.sum(med_out) > 0:        
            outliers += med_out
        else:
            loop = False

    # iteratively reject one outlier at a time compared to a local trend
    loop = True
    iterations = 0
    while loop:
        iterations += 1

        weight = error*~outliers
        weight_padded = np.pad(weight, wl2, mode='constant', constant_values=np.nan)
        
        poly_trend = np.zeros_like(omc)

        # make a trend by fitting a polynomial around each point
        for i in range(N):
            omc_here = np.array(omc_padded[i:i+window_length])
            omc_here[wl2] = np.nan
            omc_here = omc_here[~np.isnan(omc_here)]

            tts_here = np.array(tts_padded[i:i+window_length])
            tts_here[wl2] = np.nan
            tts_here = tts_here[~np.isnan(tts_here)]      

            weight_here = np.array(weight_padded[i:i+window_length])
            weight_here[wl2] = np.nan
            weight_here = weight_here[~np.isnan(weight_here)]

            omc_here    = omc_here[weight_here > 0]
            tts_here    = tts_here[weight_here > 0]
            weight_here = weight_here[weight_here > 0]

            if len(omc_here) < 3:
                poly_trend[i] = med_trend[i]
            elif len(omc_here) < 5:
                poly_trend[i] = np.interp(tts[i], tts_here, omc_here)
            else:
                polyorder = 2
                qfit = np.polyfit(tts_here, omc_here, polyorder, w=1/weight_here)
                poly_trend[i] = np.polyval(qfit, tts[i])

        # quickly smooth the polynomial trend with a narrow Hann window
        hann_window = sig.windows.hann(3)
        poly_trend  = sig.convolve(poly_trend, hann_window, mode='same') / np.sum(hann_window)

        # calculate residuals
        residuals = omc-poly_trend
        residuals[outliers] = 0.0

        # identify worst outlier and reject if above threshold
        worst = np.argmax(np.abs(residuals))

        alpha = 1-1/N
        scale = np.sqrt(uncertainty**2 + astropy.stats.mad_std(residuals[~outliers])**2)
        rlow,rhigh = stats.norm.interval(alpha,scale=scale)

        if np.abs(residuals[worst]) > rhigh:
            outliers[worst] = True
        else:
            loop = False
            
        # make some plots
        if do_plots:           
            fig, axes = plt.subplots(1,2, figsize=(18,6))

            ax = axes[0]
            ax.errorbar(tts, omc, yerr=error, fmt='o', color='grey', fillstyle='none')
            ax.errorbar(tts[outliers], omc[outliers], yerr=error[outliers], fmt='o', color='orange', fillstyle='none')
            ax.errorbar(tts[worst], omc[worst], yerr=error[worst], fmt='o', color='firebrick')
            ax.plot(tts, poly_trend, 'cornflowerblue', lw=2)
            ax.plot(tts, med_trend, 'mediumblue', lw=2)
            ax.set_ylim(omc.min(), omc.max())
            
            ax = axes[1]
            ax.errorbar(tts, residuals/scale, yerr=error/scale, fmt='o', color='grey', fillstyle='none')
            ax.errorbar(tts[worst], residuals[worst]/scale, yerr=error[worst]/scale, fmt='o', color='firebrick')
            ax.hlines(rlow/scale, tts.min(), tts.max(), color='firebrick', linestyle='--')
            ax.hlines(rhigh/scale, tts.min(), tts.max(), color='firebrick', linestyle='--')
            
            plt.show()

    # colate the results for output
    new_outliers = outliers*~known_outliers

    return new_outliers, poly_trend



def do_ttv_lombscargle_analysis(planet, verbose=False, do_plots=False):
    '''
    Find the best-fit sinusoidal ttv model using a Lomb-Scargle periodogram

    planet: wings.Planet() object

    -- returns
        tts_new: a smooth (interpolated) ttv model
        peak_freq: peak frequency from LS periodogram
        peak_fap:corresponding false-alarm probability
    '''
    p = planet
    
    # pull transit times, uncertainties, and indexes
    tts     = p.tts[p.quality]
    tts_err = p.tts_err[p.quality]
    index   = p.index[p.quality]
    
    # initialize outlier array
    outliers = np.zeros_like(tts, dtype='bool')
    
    loop = True
    iterations = 0
    while loop:
        iterations += 1
        
        # make a linear ephemeris; calculate O-C
        lin_coeffs = np.polyfit(index[~outliers], tts[~outliers], deg=1, w=1/tts_err[~outliers])
        ephemeris  = np.polyval(lin_coeffs, index)
        omc        = tts-ephemeris

        # get scatter and uncertainty
        scatter = astropy.stats.mad_std(omc[~outliers])
        uncertainty = np.median(tts_err[~outliers])

        # identify outliers based on TTV uncertainty
        out_uncertain = identify_uncertain_ttv_outliers(tts, omc, tts_err)

        # identify global TTV outliers
        out_global = identify_global_ttv_outliers(tts, omc, tts_err)

        # identify local TTV outliers
        known_outliers = out_uncertain + out_global
        out_local, local_trend = identify_local_ttv_outliers(tts, omc, tts_err, known_outliers=known_outliers, \
                                                                   window_length=9)
        
        # combine outlier arrays
        outliers = out_uncertain + out_global + out_local

        # compute a Lomb-Scargle periodogram
        lombscargle = astropy.stats.LombScargle(tts[~outliers], omc[~outliers], uncertainty)
        freq, power = lombscargle.autopower(minimum_frequency=1/(tts.max()-tts.min()), maximum_frequency=0.5/p.period, \
                                           samples_per_peak=10)

        peak_freq  = freq[np.argmax(power)]
        peak_fap   = lombscargle.false_alarm_probability(power.max(), method='bootstrap')

        # if no significant period is found, compute an LS periodogram based on the local trend
        if peak_fap >= 0.003:
            lombscargle = astropy.stats.LombScargle(tts[~outliers], local_trend[~outliers], uncertainty)
            freq, power = lombscargle.autopower(minimum_frequency=1/(tts.max()-tts.min()), \
                                                maximum_frequency=0.5/p.period, samples_per_peak=10)

            peak_freq  = freq[np.argmax(power)]
            peak_fap   = lombscargle.false_alarm_probability(power.max(), method='bootstrap')


        # if a significant period is found repeat search for local outliers based on phased TTVs
        if peak_fap < 0.003:
            tts_phased = tts % (1/peak_freq)
            mod_phased = lombscargle.model(tts_phased, peak_freq)
            res_phased = omc-mod_phased
            out_phased = np.abs(res_phased)/uncertainty > 3.0
            out_phased = out_phased*~out_uncertain*~out_global

            outliers    = out_uncertain + out_global + out_phased
            local_trend = mod_phased*1.0

            
        # finish the loop
        if iterations >= 3:
            loop = False
            
            # prepare interpolated tts for output
            full_ephemeris = p.epoch + p.period*p.index

            tts_new = np.zeros_like(p.tts)
            tts_new[p.quality] = ephemeris + local_trend
            tts_new[~p.quality] = full_ephemeris[~p.quality] + np.interp(p.tts[~p.quality], tts, local_trend)
            tts_new[p.tts < tts.min()] = full_ephemeris[p.tts < tts.min()]
            tts_new[p.tts > tts.max()] = full_ephemeris[p.tts > tts.max()]

            tts_old = p.tts.copy()
            #p.tts   = tts_new*1.0
            
            #display the results
            if verbose:
                print(' ', np.sum(out_uncertain), 'uncertainty outliers found')
                print(' ', np.sum(out_global), 'global outliers found')
                print(' ', np.sum(out_local), 'local outliers found')
                if peak_fap < 0.003: print(' ', np.sum(out_phased), 'phased outliers found')
                print('')
                print('  scatter: %.1f min' %(scatter*24*60))
                print('  uncertainty: %.1f min' %(uncertainty*24*60))
                print('')
                print('  peak period: %.2f days' %(1/peak_freq))
                print('  false alarm: %.3f' %peak_fap)    

            # make a quadratic polynomial trend
            qfit = np.polyfit(tts[~outliers], omc[~outliers], deg=2, w=1/tts_err[~outliers])
            qtrend = np.polyval(qfit, tts)
            
            # make some plots
            if do_plots:
                fig = plt.figure(figsize=(18,6))

                ylim = np.maximum(np.abs(omc[~known_outliers].min()),np.abs(omc[~known_outliers].max()))*24*60

                ax = plt.subplot2grid(shape=(4,4), loc=(0,0), colspan=2, rowspan=3)
                ax.errorbar(tts[~outliers], omc[~outliers]*24*60, yerr=tts_err[~outliers]*24*60, fmt='o', color='grey')
                ax.errorbar(tts[outliers], omc[outliers]*24*60, yerr=tts_err[outliers]*24*60, fmt='o', color='firebrick', fillstyle='none')
                ax.plot(tts, local_trend*24*60, color='cornflowerblue', lw=2)
                if peak_fap >= 0.003: ax.plot(tts, qtrend*24*60, color='mediumblue', lw=2)
                ax.set_ylabel('O-C [min]', fontsize=16)
                ax.set_ylim(-ylim,ylim)
                ax.set_xlabel('time [BJKD]', fontsize=16)

                tts_phased = np.linspace(0,1/peak_freq)
                omc_phased = lombscargle.model(tts_phased, peak_freq)

                ax = plt.subplot2grid(shape=(4,4), loc=(0,2), colspan=1, rowspan=3)
                ax.plot(tts_phased, omc_phased*24*60, 'darkorange', lw=3)
                ax.errorbar(tts[~outliers]%(1/peak_freq), omc[~outliers]*24*60, yerr=tts_err[~outliers]*24*60, \
                            fmt='o', color='grey', fillstyle='none', lw=1.0)
                ax.errorbar(tts[outliers]%(1/peak_freq), omc[outliers]*24*60, yerr=tts_err[outliers]*24*60, \
                            fmt='o', color='firebrick', fillstyle='none')
                ax.set_ylim(-ylim,ylim)
                ax.set_yticklabels('')

                ax = plt.subplot2grid(shape=(4,4), loc=(0,3), colspan=1, rowspan=3)
                ax.plot(freq, power, 'grey', lw=1.0)
                ax.vlines(peak_freq, 0, 1.1*power.max(), color='darkorange', linestyle='--', lw=2.0)
                ax.set_xlim(freq.min(),freq.max())
                ax.set_ylim(0.0,1.1*power.max())
                ax.set_yticklabels('')

                plt.show()
    
    return tts_new, peak_freq, peak_fap
