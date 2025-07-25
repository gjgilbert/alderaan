import numpy as np

__all__ = ['Planet']

class Planet:
    def __init__(self, df, planet_no, t_min, t_max):
        """
        Docstring
        """
        # read transit parameters from pandas dataframe
        self = self._from_dataframe(df, planet_no)

        # put epoch in range (t_min, t_min + period)
        self = self._adjust_epoch(t_min)

        # initialize with linear ephemeris
        self.tts = np.arange(self.epoch, t_max, self.period)
        self.index = np.array(np.round((self.tts - self.epoch) / self.period), dtype=int)
        
        # set quality flag vector
        self.quality = np.ones(len(self.tts), dtype='bool')


    def _from_dataframe(self, df, index):
        self.period = float(df.at[index, 'period'])
        self.epoch = float(df.at[index, 'epoch'])
        self.depth = float(df.at[index, 'depth'])
        self.duration = float(df.at[index, 'duration'])
        self.impact = float(df.at[index, 'impact'])

        return self
    

    def _adjust_epoch(self, t_min):
        if self.epoch < t_min:
            adj = 1 + (t_min - self.epoch) // self.period
            self.epoch += adj * self.period
        if self.epoch > (t_min + self.period):
            adj = (self.epoch - t_min) // self.period
            self.epoch -= adj * self.period

        return self