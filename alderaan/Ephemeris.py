__all__ = ["Ephemeris"]


import numpy as np


class Ephemeris:
    """
    A class for storing the emphemeris information of a single planet

    Parameters
    ----------

    Attributes
    ----------


    """

    def __init__(self, inds, tts):
        self.inds = inds
        self.tts = tts

        # calculate least squares period and epoch
        self.period, self.t0 = self._fit_ephem()

        # calculate ttvs
        self.ttvs = self.tts - self.period * self.inds + self.t0

        # calculate full set of transit times
        self.full_transit_times = self.t0 + self.period * np.arange(self.inds.max() + 1)
        self.full_transit_times[self.inds] = self.tts

        # set up histogram for identifying transit offsets
        ftts = self.full_transit_times

        self._bin_edges = np.concatenate(
            [
                [ftts[0] - 0.5 * self.period],
                0.5 * (ftts[1:] + ftts[:-1]),
                [ftts[-1] + 0.5 * self.period],
            ]
        )

        self._bin_values = np.concatenate([[ftts[0]], ftts, [ftts[-1]]])

    def _fit_ephem(self):
        A = np.ones((len(self.inds), 2))
        A[:, 0] = self.inds

        return np.linalg.lstsq(A, self.tts)[0]

    def _get_model_dt(self, t, return_inds=False):
        _inds = np.searchsorted(self._bin_edges, t)
        _vals = self._bin_values[_inds]

        if return_inds:
            return _vals, _inds
        return _vals

    def _warp_times(self, t, return_inds=False):
        warps = self._get_model_dt(t, return_inds=return_inds)

        if return_inds:
            return t - warps[0], warps[1]
        else:
            return t - warps
