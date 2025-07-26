import numpy as np
import pymc3 as pm
import pymc3_ext as pmx

__all__ = ['OMC']

class OMC:
    def __init__(self, ephemeris):
        # set initial O-C estimates
        self.index = ephemeris.index
        self.xtime = ephemeris.ttime
        self.yomc = ephemeris.ttime - ephemeris.eval_linear_ephemeris()
        self.yerr = ephemeris.error

        # set static reference period, epoch, and linear ephemeris
        self = self._set_static_references(ephemeris)


    def _set_static_references(self, ephemeris):
        self._static_period = ephemeris.period.copy()
        self._static_epoch = ephemeris.epoch.copy()
        self._static_ephemeris = ephemeris.eval_linear_ephemeris()

        ephem_1 = self._static_epoch + self.index*self._static_period
        ephem_2 = self._static_ephemeris

        assert np.allclose(ephem_1, ephem_2, rtol=1e-10, atol=1e-10), "static reference ephemeris is not self-consistent"

        return self
    

    def polymodel(self, polyorder, xt_predict=None):
        """
        Build a PyMC3 model to fit TTV observed-minus-calculated data
        Fits data with a polynomial (up to cubic)

        Arguments
        ----------
        polyorder : int
            polynomial order
        xt_predict : ndarray
            time values to predict OMC model; if not provided xtime will be used

        Returns
        -------
        model : pm.Model()
        """
        