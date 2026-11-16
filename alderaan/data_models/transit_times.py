__all__ = ["TransitTimes"]

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.stats import mad_std

from alderaan.utils.plotting import annotate_plot, label_axes, save_and_close
from alderaan.utils.stats import robust_polyfit, sigma_cutoff

# name -> (dtype, FITS column format)
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
        (default all NaN)
    ttime_mod : ndarray, optional
        modeled transit times; set equal to ttime_obs if not given
    out_prob : ndarray, optional
        outlier probability (RegularizeOMC); NaN if not computed (default all NaN)
    out_flag : ndarray, optional
        True for observed transit times flagged as outliers (RegularizeOMC);
        default all False

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

        npts = len(index)
        if npts < 2:
            raise ValueError(f"need at least 2 transits for an ephemeris, got {npts}")

        arrays = {
            "index": index,
            "ttime_obs": ttime_obs,
            # copy so ttime_mod never shares memory with ttime_obs
            "ttime_mod": np.copy(ttime_obs) if ttime_mod is None else ttime_mod,
            "error_obs": np.full(npts, np.nan) if error_obs is None else error_obs,
            "out_prob": np.full(npts, np.nan) if out_prob is None else out_prob,
            "out_flag": np.zeros(npts, dtype=bool) if out_flag is None else out_flag,
        }
        for k, (dtype, _) in FIELDS.items():
            setattr(self, k, np.asarray(arrays[k], dtype=dtype))

        if any(len(getattr(self, k)) != npts for k in FIELDS):
            raise ValueError(f"{', '.join(FIELDS)} must have equal lengths")

        self.period, self.epoch = robust_polyfit(self.index, self.ttime_mod, 1)

    @classmethod
    def from_linear_ephemeris(cls, mission, target, period, epoch, t_min, t_max):
        """
        Transit times predicted by a linear ephemeris within [t_min, t_max),
        indexed from 0; no measurements exist, so ttime_obs = ttime_mod = prediction
        """
        # start at the first transit at or after t_min, even if epoch is later
        t_first = epoch - period * np.floor((epoch - t_min) / period)
        ttime = np.arange(t_first, t_max, period)
        return cls(mission, target, np.arange(len(ttime)), ttime)

    def linear_ephemeris(self):
        return self.epoch + self.period * self.index

    def ttv_scatter(self):
        """Robust scatter (mad_std) of observed about modeled transit times"""
        obs = np.isfinite(self.ttime_obs)
        return mad_std(self.ttime_obs[obs] - self.ttime_mod[obs])

    def ttv_buffer(self):
        """ttv_scatter scaled so not even one normal outlier is expected"""
        npts = np.sum(np.isfinite(self.ttime_obs))
        return sigma_cutoff(npts) * self.ttv_scatter()

    def subset(self, mask):
        """New TransitTimes with only transits where mask is True"""
        return TransitTimes(
            self.mission,
            self.target,
            **{k: getattr(self, k)[mask] for k in FIELDS},
        )

    def clip(self, t_min, t_max):
        """
        New TransitTimes with ttime_mod within [t_min, t_max] (inclusive), transit
        index re-zeroed
        """
        use = (self.ttime_mod >= t_min) & (self.ttime_mod <= t_max)
        if np.sum(use) < 2:
            raise ValueError(
                f"need at least 2 transits for an ephemeris, got {np.sum(use)} "
                f"within ({t_min:.2f}, {t_max:.2f})"
            )
        arrays = {k: getattr(self, k)[use] for k in FIELDS}
        arrays["index"] = arrays["index"] - arrays["index"][0]
        return TransitTimes(self.mission, self.target, **arrays)

    def plot_omc(self, show_outliers=False, color="C0", ax=None, fig_path=None):
        """
        Plot observed and modeled O-C (minutes) relative to linear_ephemeris();
        returns the Figure

        Parameters
        ----------
        show_outliers : bool
            color observed times by outlier probability and mark flagged times
            with a red x; default plots only unflagged observed times
        color : str
            model line color (black when outlier probabilities are shown)
        ax : matplotlib Axes, optional
            draw here; default makes a new figure
        fig_path : str, optional
            save the figure here, closing it unless ax was given; default leaves
            it open
        """
        new_figure = ax is None
        if new_figure:
            _, ax = plt.subplots(1, 1, figsize=(8, 3))

        omc_obs = (self.ttime_obs - self.linear_ephemeris()) * 24 * 60
        omc_mod = (self.ttime_mod - self.linear_ephemeris()) * 24 * 60
        obs = np.isfinite(self.ttime_obs)
        flag = self.out_flag
        good = obs & ~flag

        line_color = color
        if show_outliers:
            prob = self.out_prob[obs]
            if np.all(np.isnan(prob)):
                ax.scatter(self.ttime_obs[obs], omc_obs[obs], c="lightgrey")
            else:
                ax.scatter(
                    self.ttime_obs[obs],
                    omc_obs[obs],
                    c=prob,
                    cmap="viridis",
                    vmin=0,
                    vmax=1,
                )
                line_color = "k"
            ax.plot(self.ttime_obs[obs & flag], omc_obs[obs & flag], "rx")
        else:
            ax.plot(self.ttime_obs[good], omc_obs[good], "o", c="lightgrey")
        ax.plot(self.ttime_mod, omc_mod, c=line_color, lw=2)

        rms = mad_std(omc_obs[good] - omc_mod[good])
        annotate_plot(ax, f"P = {self.period:.2f} d", "upper right")
        annotate_plot(ax, f"residual RMS = {rms:.1f} min", "lower right")

        shown = obs if show_outliers else good
        yrange = max(5, 1.1 * np.max(np.abs(omc_obs[shown]), initial=0))
        ax.set_ylim(-yrange, yrange)
        label_axes(ax, "Time [BKJD]", "O-C [min]")
        if new_figure:
            ax.figure.tight_layout()
        save_and_close(ax.figure, fig_path, close=new_figure)

        return ax.figure

    def _to_hdu(self, n):
        """
        BinTableHDU named TTIMES_nn for planet n (used by Results.to_fits)
        """
        columns = [
            fits.Column(name=k.upper(), format=FIELDS[k][1], array=getattr(self, k))
            for k in FIELDS
        ]
        hdu = fits.BinTableHDU.from_columns(columns, name=f"TTIMES_{n:02d}")
        hdu.header["MISSION"] = self.mission
        hdu.header["TARGET"] = self.target
        return hdu

    @classmethod
    def _from_hdu(cls, hdu):
        """
        Build from a TTIMES_nn table (used by Results.from_fits); columns missing
        from older files take their defaults
        """
        data = hdu.data
        names = data.columns.names
        return cls(
            hdu.header["MISSION"],
            hdu.header["TARGET"],
            **{k: data[k.upper()] for k in FIELDS if k.upper() in names},
        )
