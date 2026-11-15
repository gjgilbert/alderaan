__all__ = ["TransitTimes"]

import numpy as np
from astropy.io import fits

from alderaan.utils.stats import robust_polyfit


class TransitTimes:
    """
    Regularized transit times for one planet (output of RegularizeOMC)

    Parameters
    ----------
    mission, target : str
    index : ndarray
        transit index
    ttime : ndarray
        observed transit times
    model : ndarray
        regularized (OMC model) transit times
    out_prob : ndarray
        outlier probability; NaN if not computed
    out_flag : ndarray
        True for transit times flagged as outliers

    Attributes
    ----------
    period, epoch : float
        least squares linear ephemeris fit to the model transit times
    omc_ttime, omc_model : ndarray
        observed and model transit times minus the linear ephemeris
    """

    def __init__(self, mission, target, index, ttime, model, out_prob, out_flag):
        self.mission = mission
        self.target = target
        self.index = np.asarray(index, dtype=int)
        self.ttime = np.asarray(ttime, dtype=float)
        self.model = np.asarray(model, dtype=float)
        self.out_prob = np.asarray(out_prob, dtype=float)
        self.out_flag = np.asarray(out_flag, dtype=bool)

        lengths = {
            len(x)
            for x in (self.index, self.ttime, self.model, self.out_prob, self.out_flag)
        }
        if len(lengths) != 1:
            raise ValueError(
                "index, ttime, model, out_prob, out_flag must have equal lengths"
            )

        self.period, self.epoch = robust_polyfit(self.index, self.model, 1)
        self.omc_ttime = self.ttime - self.linear_ephemeris()
        self.omc_model = self.model - self.linear_ephemeris()

    def linear_ephemeris(self):
        return self.epoch + self.period * self.index

    def to_hdu(self, n):
        """
        BinTableHDU named TTIMES_nn for planet n
        """
        columns = [
            fits.Column(name="INDEX", format="K", array=self.index),
            fits.Column(name="TTIME", format="D", array=self.ttime),
            fits.Column(name="MODEL", format="D", array=self.model),
            fits.Column(name="OUT_PROB", format="D", array=self.out_prob),
            fits.Column(name="OUT_FLAG", format="K", array=self.out_flag.astype(int)),
        ]
        hdu = fits.BinTableHDU.from_columns(columns, name=f"TTIMES_{n:02d}")
        hdu.header["MISSION"] = self.mission
        hdu.header["TARGET"] = self.target
        return hdu

    @classmethod
    def from_hdu(cls, hdu):
        data = hdu.data
        return cls(
            hdu.header["MISSION"],
            hdu.header["TARGET"],
            data["INDEX"],
            data["TTIME"],
            data["MODEL"],
            data["OUT_PROB"],
            data["OUT_FLAG"],
        )
