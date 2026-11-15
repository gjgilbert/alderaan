__all__ = ["TransitTimes"]

import numpy as np
from astropy.io import fits

from alderaan.utils.stats import robust_polyfit

# name -> (dtype, FITS column format); index, ttime_obs, ttime_mod are always set,
# the rest are optional (None)
FIELDS = {
    "index": (int, "K"),
    "ttime_obs": (float, "D"),
    "ttime_mod": (float, "D"),
    "error_obs": (float, "D"),
    "out_prob": (float, "D"),
    "out_flag": (bool, "L"),
}


class TransitTimes:
    """
    Observed and modeled transit times for one planet (immutable; transforms
    return new objects)

    Parameters
    ----------
    mission, target : str
    index : ndarray
        transit index
    ttime_obs : ndarray
        observed (measured) transit times; NaN where no measurement exists
    error_obs : ndarray, optional
        measured transit time uncertainties; NaN where no measurement exists
    ttime_mod : ndarray, optional
        modeled transit times; set equal to ttime_obs if not given
    out_prob : ndarray, optional
        outlier probability (RegularizeOMC); NaN if not computed
    out_flag : ndarray, optional
        True for observed transit times flagged as outliers (RegularizeOMC)

    Attributes
    ----------
    period, epoch : float
        least squares linear ephemeris fit to ttime_mod
    """

    def __init__(
        self,
        mission,
        target,
        index,
        ttime_obs,
        error_obs=None,
        ttime_mod=None,
        out_prob=None,
        out_flag=None,
    ):
        self.mission = mission
        self.target = target
        arrays = {
            "index": index,
            "ttime_obs": ttime_obs,
            "ttime_mod": ttime_obs if ttime_mod is None else ttime_mod,
            "error_obs": error_obs,
            "out_prob": out_prob,
            "out_flag": out_flag,
        }
        for k, (dtype, _) in FIELDS.items():
            x = arrays[k]
            setattr(self, k, None if x is None else np.asarray(x, dtype=dtype))

        lengths = {len(getattr(self, k)) for k in self._fields()}
        if len(lengths) != 1:
            raise ValueError(f"{', '.join(self._fields())} must have equal lengths")

        self.period, self.epoch = robust_polyfit(self.index, self.ttime_mod, 1)

    def _fields(self):
        """Names of array fields that are set"""
        return [k for k in FIELDS if getattr(self, k) is not None]

    @classmethod
    def from_linear_ephemeris(cls, mission, target, period, epoch, t_min, t_max):
        """
        Transit times predicted by a linear ephemeris, clipped to (t_min, t_max);
        no measurements exist, so ttime_obs = ttime_mod = prediction
        """
        ttime = np.arange(epoch, t_max, period)
        index = np.array(np.round((ttime - epoch) / period), dtype=int)
        return cls(mission, target, index, ttime).clip(t_min, t_max)

    def linear_ephemeris(self):
        return self.epoch + self.period * self.index

    def subset(self, mask):
        """New TransitTimes with only transits where mask is True"""
        return TransitTimes(
            self.mission,
            self.target,
            **{k: getattr(self, k)[mask] for k in self._fields()},
        )

    def clip(self, t_min, t_max):
        """
        New TransitTimes with ttime_mod within (t_min, t_max), transit index re-zeroed
        """
        use = (self.ttime_mod >= t_min) & (self.ttime_mod <= t_max)
        arrays = {k: getattr(self, k)[use] for k in self._fields()}
        arrays["index"] = arrays["index"] - arrays["index"][0]
        return TransitTimes(self.mission, self.target, **arrays)

    def _to_hdu(self, n):
        """
        BinTableHDU named TTIMES_nn for planet n; unset optional fields are omitted
        (used by Results.to_fits)
        """
        columns = [
            fits.Column(name=k.upper(), format=FIELDS[k][1], array=getattr(self, k))
            for k in self._fields()
        ]
        hdu = fits.BinTableHDU.from_columns(columns, name=f"TTIMES_{n:02d}")
        hdu.header["MISSION"] = self.mission
        hdu.header["TARGET"] = self.target
        return hdu

    @classmethod
    def _from_hdu(cls, hdu):
        """Build from a TTIMES_nn table (used by Results.from_fits)"""
        data = hdu.data
        names = data.columns.names
        return cls(
            hdu.header["MISSION"],
            hdu.header["TARGET"],
            **{k: data[k.upper()] for k in FIELDS if k.upper() in names},
        )
