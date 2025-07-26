import numpy as np

__all__ = ['Ephemeris']

class Ephemeris:
    def __init__(self, transit_index, transit_times, transit_error=None):
        """
        Docstring
        """
        # populate attributes
        self.index = transit_index
        self.ttime = transit_times
        self.error = transit_error

        # calculate period and epoch from linear emphemeris fit
        self.period, self.epoch = self._fit_linear_ephemeris()


    def _fit_linear_ephemeris(self):
        A = np.ones((len(self.index), 2))
        A[:, 0] = self.index

        return np.linalg.lstsq(A, self.ttime, rcond=None)[0]
        

    # TODO: use linear interpolation for internal times
    def _full_ttime_vector(self, return_index=True):
        full_index_vector = np.arange(self.index.max()+1)
        full_ttime_vector = self.epoch + self.period*full_index_vector
        full_ttime_vector[self.index] = self.ttime

        if return_index:
            return full_index_vector, full_ttime_vector
        return full_ttime_vector
    
