"""TransitTimes: observed and modeled transit times for one planet"""

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
    "ttime_err": (float, "D"),
    "phot_coverage": (bool, "L"),
    "phot_noise": (bool, "L"),
    "kept_ttv": (bool, "L"),
}


def linear_ephemeris_times(period, epoch, t_min, t_max):
    """Transit times of a linear ephemeris within [t_min, t_max)"""
    # start at the first transit at or after t_min, even if epoch is later
    t_first = epoch - period * np.floor((epoch - t_min) / period)
    return np.arange(t_first, t_max, period)


class TransitTimes:
    """
    Observed and modeled transit times (days) for one planet

    Treated as immutable; transforms (clip) return new objects.

    Parameters
    ----------
    mission, target : str
    index : ndarray
        transit index; starts at 0 and strictly increases
    ttime_obs : ndarray
        observed (measured) transit times; NaN where no measurement exists
    ttime_err : ndarray, optional
        measured transit time uncertainties; NaN where no measurement exists
        (default all NaN)
    ttime_mod : ndarray, optional
        modeled transit times, must be finite; default a copy of ttime_obs
    phot_coverage, phot_noise : ndarray, optional
        False for transits with poor photometric coverage or high noise
        (FlagBadTransits); flagged transits stay in masks and models but their
        photometry is not fit or measured; default all True
    kept_ttv : ndarray, optional
        False for observed transit times rejected as outliers (RegularizeOMC);
        affects only the OMC fit; default all True
    model_name : str, optional
        OMC model that set ttime_mod (RegularizeOMC: 'smooth', 'poly_<k>',
        'lasso_<k>', 'sinusoid', 'matern32' or 'independent'); None if not
        regularized
    model_dof : float, optional
        degrees of freedom of that model (effective for matern32; nonzero
        coefficients for lasso_<k>; the number of kept times for 'independent');
        None if not regularized or 'smooth'
    ttv_scatter : float, optional
        robust scatter (mad_std, days) of kept observed times about the OMC
        model (for 'independent', the matern32 GP that fills unmeasured epochs);
        set by RegularizeOMC; None if not regularized

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
        ttime_err=None,
        ttime_mod=None,
        phot_coverage=None,
        phot_noise=None,
        kept_ttv=None,
        model_name=None,
        model_dof=None,
        ttv_scatter=None,
    ):
        self.mission = mission
        self.target = target
        self.model_name = model_name
        self.model_dof = model_dof
        self.ttv_scatter = ttv_scatter

        npts = len(index)
        if npts < 2:
            raise ValueError(f"need at least 2 transits for an ephemeris, got {npts}")

        arrays = {
            "index": index,
            "ttime_obs": ttime_obs,
            # copy so ttime_mod never shares memory with ttime_obs
            "ttime_mod": np.copy(ttime_obs) if ttime_mod is None else ttime_mod,
            "ttime_err": np.full(npts, np.nan) if ttime_err is None else ttime_err,
            "phot_coverage": (
                np.ones(npts, dtype=bool) if phot_coverage is None else phot_coverage
            ),
            "phot_noise": np.ones(npts, dtype=bool)
            if phot_noise is None
            else phot_noise,
            "kept_ttv": np.ones(npts, dtype=bool) if kept_ttv is None else kept_ttv,
        }
        for k, (dtype, _) in FIELDS.items():
            setattr(self, k, np.asarray(arrays[k], dtype=dtype))

        if any(len(getattr(self, k)) != npts for k in FIELDS):
            raise ValueError(f"{', '.join(FIELDS)} must have equal lengths")

        # transit model warps assume index counts epochs from zero
        if self.index[0] != 0 or np.any(np.diff(self.index) <= 0):
            raise ValueError("index must start at 0 and be strictly increasing")

        if not np.all(np.isfinite(self.ttime_mod)):
            raise ValueError("ttime_mod must be finite")

        self.period, self.epoch = robust_polyfit(self.index, self.ttime_mod, 1)

    @classmethod
    def from_linear_ephemeris(cls, mission, target, period, epoch, t_min, t_max):
        """
        Linear ephemeris transit times within [t_min, t_max), indexed from 0

        ttime_obs and ttime_mod are both the prediction.
        """
        ttime = linear_ephemeris_times(period, epoch, t_min, t_max)
        return cls(mission, target, np.arange(len(ttime)), ttime)

    def linear_ephemeris(self):
        """Transit times of the (period, epoch) ephemeris at each index"""
        return self.epoch + self.period * self.index

    def ttv_buffer(self):
        """
        Timing tolerance (days): ttv_scatter times sigma_cutoff(n kept times)

        The cutoff is set so not even one normal outlier is expected among the
        kept observed times; the scatter already includes the timing errors.
        Raises ValueError if ttv_scatter is not set.
        """
        if self.ttv_scatter is None:
            raise ValueError("ttv_scatter not set; regularize the transit times first")
        kept = self.kept_ttv & np.isfinite(self.ttime_obs)
        return sigma_cutoff(np.sum(kept)) * self.ttv_scatter

    def clip(self, t_min, t_max):
        """New TransitTimes with ttime_mod in [t_min, t_max], index re-zeroed"""
        use = (self.ttime_mod >= t_min) & (self.ttime_mod <= t_max)
        if np.sum(use) < 2:
            raise ValueError(
                f"need at least 2 transits for an ephemeris, got {np.sum(use)} "
                f"within ({t_min:.2f}, {t_max:.2f})"
            )
        arrays = {k: getattr(self, k)[use] for k in FIELDS}
        arrays["index"] = arrays["index"] - arrays["index"][0]
        return TransitTimes(
            self.mission,
            self.target,
            **arrays,
            model_name=self.model_name,
            model_dof=self.model_dof,
            ttv_scatter=self.ttv_scatter,
        )

    def plot_omc(
        self, show_outliers=False, color="C0", title=None, ax=None, fig_path=None
    ):
        """
        Plot observed and modeled O-C (minutes) about linear_ephemeris()

        Returns the Figure.

        Parameters
        ----------
        show_outliers : bool
            also plot observed times rejected by kept_ttv, as red x (off-scale
            ones as red triangles at the edge); default plots only kept times
        color : str
            model line color
        title : str, optional
            axes title
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
        flag = ~self.kept_ttv
        good = obs & ~flag

        # y scale set by the kept times and the model, not by outliers
        yrange = max(
            5,
            1.1 * np.max(np.abs(omc_obs[good]), initial=0),
            1.1 * np.max(np.abs(omc_mod)),
        )
        ax.set_ylim(-yrange, yrange)

        ax.plot(self.ttime_obs[good], omc_obs[good], "o", c="lightgrey")
        if show_outliers:
            on = obs & flag & (np.abs(omc_obs) <= yrange)
            ax.plot(self.ttime_obs[on], omc_obs[on], "rx")
            for sign, marker in [(1, "^"), (-1, "v")]:
                off = obs & flag & (sign * omc_obs > yrange)
                # on the axis edge, above the corner annotations
                ax.plot(
                    self.ttime_obs[off],
                    np.full(np.sum(off), sign * yrange),
                    marker,
                    ls="none",
                    ms=5,
                    mfc="none",
                    mec="r",
                    mew=0.8,
                    clip_on=False,
                    zorder=4,
                )
        ax.plot(self.ttime_mod, omc_mod, c=color, lw=2)

        # typical (median) uncertainty, drawn once right of the data
        t0, t1 = self.ttime_mod[0], self.ttime_mod[-1]
        span = t1 - t0
        ax.set_xlim(t0 - 0.02 * span, t1 + 0.08 * span)
        err = np.median(self.ttime_err[good]) * 24 * 60 if np.any(good) else np.nan
        if np.isfinite(err):
            ax.errorbar(t1 + 0.05 * span, 0, yerr=err, fmt="o", c="grey", capsize=3)

        resid = omc_obs[good] - omc_mod[good]
        rms = mad_std(resid)
        annotate_plot(ax, f"P = {self.period:.2f} d", "upper right")
        annotate_plot(ax, f"residual RMS = {rms:.1f} min", "lower right")
        if self.model_name is not None:
            annotate_plot(ax, f"model = {self.model_name}", "upper left")
        if self.model_dof is not None and np.sum(good) > self.model_dof:
            chi2 = np.sum((resid / (self.ttime_err[good] * 24 * 60)) ** 2)
            chi2_red = chi2 / (np.sum(good) - self.model_dof)
            annotate_plot(ax, f"reduced χ² = {chi2_red:.2f}", "lower left")
        if title is not None:
            ax.set_title(title)
        label_axes(ax, "Time [BKJD]", "O-C [min]")
        if new_figure:
            ax.figure.tight_layout()
        save_and_close(ax.figure, fig_path, close=new_figure)

        return ax.figure

    def _to_hdu(self, n):
        """BinTableHDU named TTIMES_nn for planet n (used by Results.to_fits)"""
        columns = [
            fits.Column(name=k.upper(), format=FIELDS[k][1], array=getattr(self, k))
            for k in FIELDS
        ]
        hdu = fits.BinTableHDU.from_columns(columns, name=f"TTIMES_{n:02d}")
        hdu.header["MISSION"] = self.mission
        hdu.header["TARGET"] = self.target
        if self.model_name is not None:
            hdu.header["OMCMODEL"] = self.model_name
        if self.model_dof is not None:
            hdu.header["OMCDOF"] = self.model_dof
        if self.ttv_scatter is not None:
            hdu.header["TTVSCAT"] = self.ttv_scatter
        return hdu

    @classmethod
    def _from_hdu(cls, hdu):
        """Build from a TTIMES_nn table (used by Results.from_fits)"""
        data = hdu.data
        return cls(
            hdu.header["MISSION"],
            hdu.header["TARGET"],
            **{k: data[k.upper()] for k in FIELDS},
            model_name=hdu.header.get("OMCMODEL"),
            model_dof=hdu.header.get("OMCDOF"),
            ttv_scatter=hdu.header.get("TTVSCAT"),
        )
