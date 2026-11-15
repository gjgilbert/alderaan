__all__ = ["TransitTimes"]

import numpy as np
from astropy.io import fits

from alderaan.utils.stats import robust_polyfit

OPTIONAL_FIELDS = ("error_obs", "out_prob", "out_flag")


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
        self.index = np.asarray(index, dtype=int)
        self.ttime_obs = np.asarray(ttime_obs, dtype=float)
        self.error_obs = (
            None if error_obs is None else np.asarray(error_obs, dtype=float)
        )
        self.ttime_mod = (
            self.ttime_obs if ttime_mod is None else np.asarray(ttime_mod, dtype=float)
        )
        self.out_prob = None if out_prob is None else np.asarray(out_prob, dtype=float)
        self.out_flag = None if out_flag is None else np.asarray(out_flag, dtype=bool)

        lengths = {len(getattr(self, k)) for k in self._fields()}
        if len(lengths) != 1:
            raise ValueError(f"{', '.join(self._fields())} must have equal lengths")

        self.period, self.epoch = robust_polyfit(self.index, self.ttime_mod, 1)

    def _fields(self):
        """Names of array fields that are set"""
        return ["index", "ttime_obs", "ttime_mod"] + [
            k for k in OPTIONAL_FIELDS if getattr(self, k) is not None
        ]

    @classmethod
    def from_linear_ephemeris(cls, mission, target, period, epoch, t_min, t_max):
        """
        Transit times predicted by a linear ephemeris, clipped to (t_min, t_max);
        no measurements exist, so ttime_obs = ttime_mod = prediction
        """
        ttime = np.arange(epoch, t_max, period)
        index = np.array(np.round((ttime - epoch) / period), dtype=int)
        return cls(mission, target, index, ttime, ttime_mod=ttime).clip(t_min, t_max)

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
        formats = {
            "index": "K",
            "ttime_obs": "D",
            "ttime_mod": "D",
            "error_obs": "D",
            "out_prob": "D",
            "out_flag": "L",
        }
        columns = [
            fits.Column(name=k.upper(), format=formats[k], array=getattr(self, k))
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
            data["INDEX"],
            data["TTIME_OBS"],
            ttime_mod=data["TTIME_MOD"],
            **{k: data[k.upper()] for k in OPTIONAL_FIELDS if k.upper() in names},
        )
