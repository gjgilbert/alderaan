import numpy as np

__all__ = ['Ephemeris']

class Ephemeris:
    def __init__(self, 
                 index, 
                 ttime, 
                 error=None,
                 quality=None,
                 t_min=None,
                 t_max=None
                ):
        """
        Docstring
        """
        # populate attributes
        self.index = index
        self.ttime = ttime
        self.error = error
        self.quality = quality

        # calculate period and epoch from linear emphemeris fit
        self.period, self.epoch = self.fit_linear_ephemeris()

        # set (t_min, t_max)
        if t_min is not None:
            self.t_min = t_min
        else:
            self.t_min = self.ttime.min() - self.period / 2
        
        if t_max is not None:
            self.t_max = t_max
        else:
            self.t_max = self.ttime.max() + self.period / 2

        # put epoch in range (t_min, t_min + period)
        self.epoch = self._adjust_epoch(self.t_min)

    
    def _adjust_epoch(self, t_min):
        if self.epoch < t_min:
            adj = 1 + (t_min - self.epoch) // self.period
            self.epoch += adj * self.period
        if self.epoch > (t_min + self.period):
            adj = (self.epoch - t_min) // self.period
            self.epoch -= adj * self.period

        return self.epoch
    
    
    def fit_linear_ephemeris(self):
        A = np.ones((len(self.index), 2))
        A[:, 0] = self.index

        return np.linalg.lstsq(A, self.ttime, rcond=None)[0]
    

    def eval_linear_ephemeris(self, index=None):
        if index is None:
            index = self.index

        return self.epoch + self.period*index
        

    # TODO: use linear interpolation for internal times
    def full_ephemeris(self, return_index=True):
        full_index_vector = np.arange(self.index.max()+1)
        full_ttime_vector = self.eval_linear_ephemeris(full_index_vector)
        full_ttime_vector[self.index] = self.ttime

        if return_index:
            return full_index_vector, full_ttime_vector
        return full_ttime_vector