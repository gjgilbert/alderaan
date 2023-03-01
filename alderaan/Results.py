from   astropy.io import fits
import batman
import copy
import corner
import dynesty
import glob
import matplotlib.pyplot as plt
import numpy as np
import numpy.polynomial.polynomial as poly
import pandas as pd
import pickle
import warnings

from .io import load_detrended_lightcurve
from .utils import weighted_percentile
from .constants import lcit, scit

__all__ = ['Results']

class _Lightcurve:
    def __init__(self, key_values):
        
        self._keys = []
  
        for k, v in key_values.items():
            self._keys.append(k)
            setattr(self, k, copy.copy(v))
            
        required_keys = ['time', 'flux', 'error', 'cadno', 'quarter']
        
        for k in required_keys:
            if k not in self._keys:
                raise ValueError('Key {0} must be provided'.format(k))
                
    def keys(self):
        return self._keys
    
    def plot(self, quarter=None):
        if quarter is None:
            quarter = self.quarter

        plt.figure(figsize=(20,4))
        plt.plot(self.time[self.quarter == quarter], self.flux[self.quarter == quarter], 'k.', ms=1)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.xlabel("Time [BKJD]", fontsize=24)
        plt.ylabel("Flux", fontsize=24)
        if np.any(self.quarter==quarter):
            plt.xlim(self.time[self.quarter == quarter].min(), self.time[self.quarter == quarter].max())
        else:
            warnings.warn("Attempting to plot quarter with no data")
        plt.show()
        
        return None
    
    
class _TransitTimes:
    def __init__(self, key_values):
        self._keys = []
  
        for k, v in key_values.items():
            self._keys.append(k)
            setattr(self, k, copy.copy(v))
            
        required_keys = ['inds', 'tts', 'model', 'outlier_prob', 'outlier_flag']
        
        for k in required_keys:
            if k not in self._keys:
                raise ValueError('Key {0} must be provided'.format(k))
                
        # calculate least squares period, epoch, and omc
        self.t0, self.period = poly.polyfit(self.inds, self.model, 1)
        self.omc = self.tts - poly.polyval(self.inds, [self.t0, self.period])
        self.trend = self.model - poly.polyval(self.inds, [self.t0, self.period])
        
        self._keys += ['t0', 'period', 'omc', 'trend']

    def keys(self):
        return self._keys


class _MultiTransitTimes:
    def __init__(self, _TTlist):
        self._keys = ['npl'] + _TTlist[0].keys()
        self.npl = len(_TTlist)
        
        for k in self._keys[1:]:
            v = [None]*self.npl
            for n, _TT in enumerate(_TTlist):
                v[n] = getattr(_TT, k)
            setattr(self, k, copy.copy(v))
        
    def keys(self):
        return self._keys
    
    def linear_ephemeris(self, n):
        return self.t0[n] + self.period[n]*self.inds[n]
        
        
    
class _Posteriors():
    def __init__(self, dynestyResults):
        self.samples = copy.copy(dynestyResults.samples)
        self.logwt   = copy.copy(dynestyResults.logwt)
        self.logz    = copy.copy(dynestyResults.logz)
        self.logl    = copy.copy(dynestyResults.logl)
        self._keys   = ['samples', 'logwt', 'logz', 'logl']

    def keys(self):
        return self._keys    
    
    def nsamp(self):
        """
        Number of samples
        """
        return self.samples.shape[0]
    
    def neff(self):
        """
        Number of effective samples (Kish ESS); wrapper for dynesty.utils.get_neff_from_logwt
        """
        return dynesty.utils.get_neff_from_logwt(self.logwt)
    
    def weights(self):
        """
        Convenience function for getting normalized weights
        """
        wt = np.exp(self.logwt - self.logz[-1])
        
        return wt/np.sum(wt)
    
    def summary(self):
        if hasattr(self, '_summary') == False:
            samples = self.samples
            weights = self.weights()
            npl = (samples.shape[1]-2) // 5

            labels = []
            for n in range(npl):
                labels += 'C0_{0} C1_{0} r_{0} b_{0} T14_{0}'.format(n).split()
            labels += ['q1', 'q2']

            _summary = pd.DataFrame(columns=['mean', 'stdev', 'median', 
                                            'CI_2.5', 'CI_16', 'CI_84', 'CI_97.5'], index=labels)

            for i, lab in enumerate(labels):
                avg = np.sum(weights*samples[:,i])
                var = np.sum(weights*(samples[:,i] - avg)**2)
                
                _summary['mean'][lab]   = copy.copy(avg)
                _summary['stdev'][lab]  = np.sqrt(var)
                _summary['median'][lab] = weighted_percentile(samples[:,i], 50, w=weights)
                _summary['CI_2.5'][lab] = weighted_percentile(samples[:,i], 2.5, w=weights)
                _summary['CI_16'][lab] = weighted_percentile(samples[:,i], 15.9, w=weights)
                _summary['CI_84'][lab] = weighted_percentile(samples[:,i], 84.1, w=weights)
                _summary['CI_97.5'][lab] = weighted_percentile(samples[:,i], 97.5, w=weights)
                
            self._summary = copy.copy(_summary)
            
        return self._summary
    
    
    
class Results:
    def __init__(self, target=None, data_dir=None):
        self.target = target
        self.npl          = None
        self.lightcurve   = None
        self.transittimes = None
        self.posteriors   = None
        self._keys  = ['target', 'npl', 'lightcurve', 'transittimes', 'posteriors']
        
        if target is None:
            warnings.warn('Results object initiated without target')
        else:
            self.load(target, data_dir)
                   
    def keys(self):
        return self._keys

    
    def load(self, target, data_dir):
        # lightcurve
        lightcurve_files = glob.glob(data_dir + '{0}/*filtered.fits'.format(target))
        
        for lcf in lightcurve_files:
            self.load_lightcurve(lcf)
            
        # transit times
        transittime_files = glob.glob(data_dir + '{0}/*quick.ttvs'.format(target))
        transittime_files.sort()
        
        self.load_transittimes(transittime_files)
        self.npl = len(transittime_files)
        self._dataframe = [None]*self.npl
        
            
        # posteriors
        self.load_posteriors(glob.glob(data_dir + '{0}/*nested.pkl'.format(target))[0])
        
    
    def load_lightcurve(self, file):       
        _lightcurve = {}
        with fits.open(file) as hduL:
            _lightcurve['time'] = np.array(hduL['TIME'].data, dtype='float64')
            _lightcurve['flux'] = np.array(hduL['FLUX'].data, dtype='float64')
            _lightcurve['error'] = np.array(hduL['ERROR'].data, dtype='float64')
            _lightcurve['cadno'] = np.array(hduL['CADNO'].data, dtype='int')
            _lightcurve['quarter'] = np.array(hduL['QUARTER'].data, dtype='int')
            
        if self.lightcurve is not None:
            _lightcurve['time'] = np.hstack([self.lightcurve.time, _lightcurve['time']])
            _lightcurve['flux'] = np.hstack([self.lightcurve.flux, _lightcurve['flux']])
            _lightcurve['error'] = np.hstack([self.lightcurve.error, _lightcurve['error']])
            _lightcurve['cadno'] = np.hstack([self.lightcurve.cadno, _lightcurve['cadno']])
            _lightcurve['quarter'] = np.hstack([self.lightcurve.quarter, _lightcurve['quarter']])
            
        self.lightcurve = _Lightcurve(_lightcurve)
        
    
    def load_transittimes(self, files):
        _transittimes = [None]*len(files)
        
        for n, f in enumerate(files):
            darr = np.loadtxt(f).swapaxes(0,1)
            ddic = {}
            keys = 'inds tts model outlier_prob outlier_flag'.split()

            for i, k in enumerate(keys):
                ddic[k] = darr[i]

            ddic['inds'] = np.asarray(ddic['inds'], dtype='int')
            ddic['outlier_flag'] = np.asarray(ddic['outlier_flag'], dtype='bool')
        
            _transittimes[n] = _TransitTimes(ddic)
        
        self.transittimes = _MultiTransitTimes(_transittimes)

    
    def load_posteriors(self, file):
        with open(file, 'rb') as f:
            dynestyResults = dynesty.results.Results(pickle.load(f))
            self.posteriors = _Posteriors(dynestyResults)        

    
    def plot_lightcurve(self, quarter=None):
        self.lightcurve.plot(quarter)

    
    def plot_omc(self, show_outliers=False):
        fig, ax = plt.subplots(self.npl, figsize=(12,3*self.npl))
        if self.npl == 1: ax = [ax]
        for i in range(self.npl):
            tts = self.transittimes.tts[i]
            out = self.transittimes.outlier_flag[i]
            omc = self.transittimes.omc[i]
            trend = self.transittimes.trend[i]
            
            if show_outliers:
                ax[i].scatter(tts, omc*24*60, c=self.transittimes.outlier_prob[i], cmap='viridis')
                ax[i].plot(tts[out], omc[out]*24*60, 'rx')
                ax[i].plot(tts, trend*24*60, 'k')
            else:
                ax[i].plot(tts[~out], omc[~out]*24*60, 'o', c='lightgrey')
                ax[i].plot(tts[~out], trend[~out]*24*60, c='C{0}'.format(i), lw=3)
            ax[i].set_ylabel('O-C [min]', fontsize=20)
            
        ax[self.npl-1].set_xlabel('Time [BJKD]', fontsize=20)
        plt.show() 
        
    
    def plot_corner(self, n, limbdark=True, physical=True):
        C0, C1, r, b, T14 = self.posteriors.samples[:,5*n:5*(n+1)].swapaxes(0,1)
        q1, q2 = self.posteriors.samples[:,-2:].swapaxes(0,1)
        weights = self.posteriors.weights()
        
        if physical:
            # least squares period and epoch
            Leg0 = self._legendre(n,0)
            Leg1 = self._legendre(n,1)
            ephem = self.transittimes.model[n] + np.outer(C0,Leg0) + np.outer(C1,Leg1)            
            t0, P = poly.polyfit(self.transittimes.inds[n], ephem.T, 1)            
            
            # limb darkening (see Kipping 2013)
            u1 = 2*np.sqrt(q1)*q2
            u2 = np.sqrt(q1)*(1-2*q2)
            
            if limbdark:
                samples = np.array([P, t0, r, b, T14, u1, u2]).swapaxes(0,1)
                labels = 'P t0 r b T14 u1 u2'.split()
            else:
                samples = np.array([P, t0, r, b, T14]).swapaxes(0,1)
                labels = 'P t0 r b T14'.split()

        else:
            if limbdark:
                samples = np.array([C0, C1, r, b, T14, q1, q2]).swapaxes(0,1)
                labels = 'C0 C1 r b T14 q1 q2'.split()
            else:
                samples = np.array([C0, C1, r, b, T14]).swapaxes(0,1)
                labels = 'C0 C1 r b T14'.split()
                
        resamp = np.zeros((3*len(weights),len(labels)))
        domain = np.zeros((len(labels),2))
        
        for i in range(len(labels)):
            resamp[:,i] = np.random.choice(samples[:,i], p=weights, replace=True, size=3*len(weights))
            domain[i,0] = np.percentile(resamp[:,i], 0.1)
            domain[i,1] = np.percentile(resamp[:,i],99.9)
        
        # impact parameter
        domain[3,0] = 0.
        domain[3,1] = np.max([domain[3,1], 1.])
        
        cfig = corner.corner(resamp, labels=labels, hist_bin_factor=3, range=domain)
        
        
    def plot_transit(self, n, index):
        loc = self.transittimes.inds[n] == index
        tc  = float(self.transittimes.model[n][loc])
        T14 = self.posteriors.summary()['median']['T14_{0}'.format(n)]
        use = np.abs(self.lightcurve.time - tc)/T14 < 1.5
                
        theta = self._batman_theta(n)
        
        t_obs = self.lightcurve.time[use]
        f_obs = (self.lightcurve.flux[use] - 1.0)*1000
        t_mod = np.arange(t_obs.min(),t_obs.max(),scit)
        f_mod = (batman.TransitModel(theta, t_mod-tc).light_curve(theta) - 1.0)*1000
                
        plt.figure(figsize=(12,4))
        plt.plot(t_obs, f_obs, '.', color='lightgrey')
        plt.plot(t_mod, f_mod, color='C{0}'.format(n), lw=3)
        plt.xlim(tc-1.55*T14, tc+1.55*T14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.xlabel("Time [BKJD]", fontsize=24)
        plt.ylabel("Flux [ppt]", fontsize=24)
        plt.text(t_obs.max(), f_obs.max(), "PERIOD = {0}".format(np.round(theta.per,2)), \
                 fontsize=20, ha='right', va='top')
        plt.show()
        
        
    def plot_folded(self, n, max_pts=1000):
        tts = self.transittimes.inds[n]
        
        time = self.lightcurve.time
        flux = self.lightcurve.flux
        
    def summary(self):
        return self.posteriors.summary()
    
    
    def dataframe(self, n):
        if self._dataframe[n] is None:
            # raw posteriors
            C0, C1, r, b, T14 = self.posteriors.samples[:,5*n:5*(n+1)].swapaxes(0,1)
            q1, q2 = self.posteriors.samples[:,-2:].swapaxes(0,1)
            ln_wt = self.posteriors.logwt

            # least squares period and epoch
            Leg0 = self._legendre(n,0)
            Leg1 = self._legendre(n,1)
            ephem = self.transittimes.model[n] + np.outer(C0,Leg0) + np.outer(C1,Leg1)            
            t0, P = poly.polyfit(self.transittimes.inds[n], ephem.T, 1)

            # limb darkening (see Kipping 2013)
            u1 = 2*np.sqrt(q1)*q2
            u2 = np.sqrt(q1)*(1-2*q2)

            # build dataframe
            data = np.vstack([P, t0, r, b, T14, u1, u2, ln_wt]).T
            labels = 'PERIOD T0 ROR IMPACT DUR14 LD_U1 LD_U2 LN_WT'.split()
            self._dataframe[n] = pd.DataFrame(data, columns=labels)
            
        return self._dataframe[n]

    
    def _batman_theta(self, n):
        theta = batman.TransitParams()
        theta.per = self.transittimes.period[n]
        theta.t0  = 0.
        theta.rp  = self.posteriors.summary()['median']['r_{0}'.format(n)]
        theta.b   = self.posteriors.summary()['median']['b_{0}'.format(n)]
        theta.T14 = self.posteriors.summary()['median']['T14_{0}'.format(n)]
    
        q1 = self.posteriors.summary()['median']['q1']
        q2 = self.posteriors.summary()['median']['q2']
        theta.u = [2*np.sqrt(q1)*q2, np.sqrt(q1)*(1-2*q2)]
        theta.limb_dark = 'quadratic'
        
        return theta
    
    
    def _legendre(self, n, k):
        t = self.transittimes.linear_ephemeris(n)
        x = 2*(t-self.lightcurve.time.min())/(self.lightcurve.time.max()-self.lightcurve.time.min()) - 1
        
        if k==0:
            return np.ones_like(x)
        if k==1:
            return x
        else:
            return ValueError("only configured for 0th and 1st order Legendre polynomials")