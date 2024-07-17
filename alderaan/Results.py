from   astropy.io import fits
import batman
import copy
import corner
import dynesty
import glob
import matplotlib.pyplot as plt
import numpy as np
import numpy.polynomial.polynomial as poly
import os
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
            self._keys.append(k.lower())
            setattr(self, k.lower(), np.array(v))
            
        required_keys = ['index', 'ttime', 'model', 'out_prob', 'out_flag']
        
        self.index = self.index.astype(int)
        self.out_flag = self.out_flag.astype(bool)
            
        # calculate least squares period, epoch, and omc
        self.t0, self.period = poly.polyfit(self.index, self.model, 1)
        self.omc_ttime = self.ttime - poly.polyval(self.index, [self.t0, self.period])
        self.omc_model = self.model - poly.polyval(self.index, [self.t0, self.period])

        self._keys += ['t0', 'period', 'omc_ttime', 'omc_model']
            
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
        return self.t0[n] + self.period[n]*self.index[n]        
        
    
class _Posteriors:
    def __init__(self, key_values):
        self._keys = ['samples', 'ln_like', 'ln_wt', 'ln_z']
              
        df = pd.DataFrame(key_values)
        
        self.ln_like = np.array(df['LN_LIKE'])
        self.ln_wt = np.array(df['LN_WT'])
        self.ln_z = np.array(df['LN_WT'])
        
        self.samples = df.drop(columns=['LN_LIKE', 'LN_WT', 'LN_Z'])        
            

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
        return dynesty.utils.get_neff_from_logwt(self.ln_wt)
    
    
    def weights(self):
        """
        Convenience function for getting normalized weights
        """
        wt = np.exp(self.ln_wt - self.ln_z[-1])
        return wt/np.sum(wt)

    
    def summary(self):
        if hasattr(self, '_summary') == False:
            samples = self.samples
            weights = self.weights()
            npl = (samples.shape[1]-2) // 5

            labels = list(self.samples.keys())
            _summary = pd.DataFrame(columns=['mean', 'stdev', 'median', 
                                            'CI_2.5', 'CI_16', 'CI_84', 'CI_97.5'], index=labels)

            for i, lab in enumerate(labels):
                avg = np.sum(weights*samples[lab])
                var = np.sum(weights*(samples[lab] - avg)**2)
                
                _summary['mean'][lab]   = copy.copy(avg)
                _summary['stdev'][lab]  = np.sqrt(var)
                _summary['median'][lab] = weighted_percentile(samples[lab], 50, w=weights)
                _summary['CI_2.5'][lab] = weighted_percentile(samples[lab], 2.5, w=weights)
                _summary['CI_16'][lab] = weighted_percentile(samples[lab], 15.9, w=weights)
                _summary['CI_84'][lab] = weighted_percentile(samples[lab], 84.1, w=weights)
                _summary['CI_97.5'][lab] = weighted_percentile(samples[lab], 97.5, w=weights)
                
            self._summary = copy.copy(_summary)
            
        return self._summary
    
    
    
class Results:
    def __init__(self, target=None, data_dir=None):
        
        if target is not None:
            self.target = target      
            
            if data_dir is not None:
                results_file = os.path.join(data_dir, '{0}/{0}-results.fits'.format(target))
                
                with fits.open(results_file) as hduL:
                    if hduL[0].header['TARGET'] != self.target:
                        raise ValueError('specified target does not match FITS header')
                    
                    self.npl = hduL[0].header['NPL']
                    
            else:
                self.npl = 0
                warnings.warn('Results object initiated without data')
                   
        else:
            self.target = None
            warnings.warn('Results object initiated without target')
        
        
        # initialize extensions
        self.lightcurve = None
        self.transittimes = None
        self.posteriors = None
        
        self._keys  = ['target', 'npl', 'lightcurve', 'transittimes', 'posteriors']
        
        # read in lightcurves
        lightcurve_files = glob.glob(os.path.join(data_dir, '{0}/*filtered.fits'.format(target)))
        for lcf in lightcurve_files:
            self.load_lightcurve(lcf)
        
        # read in results
        results_file = os.path.join(data_dir, '{0}/{0}-results.fits'.format(target))
        self.load_posteriors(results_file)
        self.load_transittimes(results_file)
        
        # make samples container
        self._samples = [None]*self.npl
        
    
    def keys(self):
        return self._keys
    
    def methods(self):
        print(['load_lightcurve',
               'load_posteriors',
               'load_transittimes',
               'plot_folded',
               'plot_lightcurve',
               'plot_omc',
               'plot_transit',
               'summary'
              ]
             )
            

    def load_lightcurve(self, f):    
        _lightcurve = {}
        with fits.open(f) as hduL:
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
        
        
    def load_posteriors(self, f):
        with fits.open(f) as hduL:
            data = hduL['SAMPLES'].data
            keys = data.names
            
            _posteriors = []
            for k in keys:
                _posteriors.append(data[k])

            self.posteriors = _Posteriors(pd.DataFrame(np.array(_posteriors).T, columns=keys))

    
    def load_transittimes(self, f):
        _transittimes = [None]*self.npl
        
        with fits.open(f) as hduL:
            for n in range(self.npl):
                data = hduL['TTIMES_{0}'.format(str(n).zfill(2))].data
                keys = data.names
                
                _tts = []
                for k in keys:
                    _tts.append(data[k])
                
                _transittimes[n] = _TransitTimes(pd.DataFrame(np.array(_tts).T, columns=keys))

        self.transittimes = _MultiTransitTimes(_transittimes)
        
        
    def plot_lightcurve(self, quarter=None):
        self.lightcurve.plot(quarter)

    
    def plot_omc(self, show_outliers=False):
        fig, ax = plt.subplots(self.npl, figsize=(12,3*self.npl))
        if self.npl == 1: ax = [ax]
        for i in range(self.npl):
            tts = self.transittimes.ttime[i]
            out = self.transittimes.out_flag[i]
            omc_ttime = self.transittimes.omc_ttime[i]
            omc_model = self.transittimes.omc_model[i]
            
            if show_outliers:
                ax[i].scatter(tts, omc_ttime*24*60, c=self.transittimes.out_prob[i], cmap='viridis')
                ax[i].plot(tts[out], omc_ttime[out]*24*60, 'rx')
                ax[i].plot(tts, omc_model*24*60, 'k')
            else:
                ax[i].plot(tts[~out], omc_ttime[~out]*24*60, 'o', c='lightgrey')
                ax[i].plot(tts[~out], omc_model[~out]*24*60, c='C{0}'.format(i), lw=3)
            ax[i].set_ylabel('O-C [min]', fontsize=20)
            
        ax[self.npl-1].set_xlabel('Time [BJKD]', fontsize=20)
        plt.show() 
        
    
    def plot_transit(self, n, index):
        loc = self.transittimes.index[n] == index
        tc  = float(self.transittimes.model[n][loc])
        T14 = self.posteriors.summary()['median']['DUR14_{0}'.format(n)]
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
        tts = self.transittimes.ttime[n]
        time = self.lightcurve.time
        flux = self.lightcurve.flux
        
        
    def summary(self):
        return self.posteriors.summary()
    
    
    def samples(self, n):
        if self._samples[n] is None:
            # raw posteriors
            C0  = self.posteriors.samples['C0_{0}'.format(n)]
            C1  = self.posteriors.samples['C1_{0}'.format(n)]
            r   = self.posteriors.samples['ROR_{0}'.format(n)]
            b   = self.posteriors.samples['IMPACT_{0}'.format(n)]
            T14 = self.posteriors.samples['DUR14_{0}'.format(n)]
            q1  = self.posteriors.samples['LD_Q1']
            q2  = self.posteriors.samples['LD_Q2']
            ln_wt = self.posteriors.ln_wt

            # least squares period and epoch
            centered_index = self.transittimes.index[n] - self.transittimes.index[n][-1]//2
           
            LegX = centered_index / (self.transittimes.index[n][-1]/2)
            Leg0 = np.ones_like(LegX)
            ephem = self.transittimes.model[n] +  np.outer(C0,Leg0) + np.outer(C1,LegX)
            t0, P = poly.polyfit(self.transittimes.index[n], ephem.T, 1)

            # limb darkening (see Kipping 2013)
            u1 = 2*np.sqrt(q1)*q2
            u2 = np.sqrt(q1)*(1-2*q2)

            # build dataframe
            data = np.vstack([P, t0, r, b, T14, u1, u2, ln_wt]).T
            labels = 'PERIOD T0 ROR IMPACT DUR14 LD_U1 LD_U2 LN_WT'.split()
            self._samples[n] = pd.DataFrame(data, columns=labels)
            
        return self._samples[n]
    
    
    def _batman_theta(self, n):
        theta = batman.TransitParams()
        theta.per = self.transittimes.period[n]
        theta.t0  = 0.
        theta.rp  = self.posteriors.summary()['median']['ROR_{0}'.format(n)]
        theta.b   = self.posteriors.summary()['median']['IMPACT_{0}'.format(n)]
        theta.T14 = self.posteriors.summary()['median']['DUR14_{0}'.format(n)]
    
        q1 = self.posteriors.summary()['median']['LD_Q1']
        q2 = self.posteriors.summary()['median']['LD_Q2']
        theta.u = [2*np.sqrt(q1)*q2, np.sqrt(q1)*(1-2*q2)]
        theta.limb_dark = 'quadratic'
        
        return theta