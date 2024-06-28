import numpy as np
import numpy.polynomial.polynomial as poly


__all__ = ['Ephemeris']


class Ephemeris:
    def __init__(self, inds, tts):
        self.inds = inds
        self.tts  = tts
    
        # calculate least squares period and epoch
        self.t0, self.period = poly.polyfit(inds, tts, 1)
        
        # calculate ttvs
        self.ttvs = tts - poly.polyval(inds, [self.t0, self.period])
    
        # calculate full set of transit times
        self.full_transit_times = self.t0 + self.period*np.arange(self.inds.max()+1)
        self.full_transit_times[self.inds] = self.tts

        # set up histogram for identifying transit offsets
        ftts = self.full_transit_times

        self._bin_edges = np.concatenate([[ftts[0] - 0.5*self.period],
                                         0.5*(ftts[1:] + ftts[:-1]),
                                         [ftts[-1] + 0.5*self.period]
                                        ])

        self._bin_values = np.concatenate([[ftts[0]], ftts, [ftts[-1]]])
    
    
    def _get_model_dt(self, t):
        _inds = np.searchsorted(self._bin_edges, t)
        _vals = self._bin_values[_inds]
        return _vals
    
    def _warp_times(self, t):
        return t - self._get_model_dt(t)