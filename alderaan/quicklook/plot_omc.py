__all__ = ["PlotOMC"]

import matplotlib.pyplot as plt
import numpy as np
from astropy.stats import mad_std

from alderaan.quicklook.base import finish


class PlotOMC:
    """
    Quicklook plots of observed-minus-calculated (O-C) transit times

    Parameters
    ----------
    transit_times : list of TransitTimes
        one per planet, with OMC diagnostics (e.g. Results.transit_times);
        out_prob may be all NaN
    """

    def __init__(self, transit_times):
        self.transit_times = transit_times
        self.target = transit_times[0].target

    def plot_omc(self, fig_path=None, show_plot=False):
        npl = len(self.transit_times)

        fig, ax = plt.subplots(npl, figsize=(8, 3 * npl))
        if npl == 1:
            ax = [ax]

        for n, tt in enumerate(self.transit_times):
            obs = np.isfinite(tt.ttime_obs)
            xtime = tt.ttime_obs[obs]
            out_prob = tt.out_prob[obs]
            out_flag = tt.out_flag[obs]
            period = tt.period

            # O-C relative to a linear fit to the regularized transit times
            yobs = (tt.ttime_obs - tt.linear_ephemeris())[obs]
            ymod = (tt.ttime_mod - tt.linear_ephemeris())[obs]

            if np.all(np.isnan(out_prob)):
                marker_color, line_color, cmap = "lightgrey", f"C{n}", None
            else:
                marker_color, line_color, cmap = 1 - out_prob, "k", "viridis"

            ax[n].scatter(
                xtime, yobs * 24 * 60, c=marker_color, cmap=cmap, label="Observed TTVs"
            )
            ax[n].plot(
                tt.ttime_mod,
                (tt.ttime_mod - tt.linear_ephemeris()) * 24 * 60,
                lw=2,
                c=line_color,
                label="Regularized Model",
            )
            ax[n].plot(xtime[out_flag], yobs[out_flag] * 24 * 60, "rx")

            rms = mad_std(yobs - ymod, ignore_nan=True) * 24 * 60
            ax[n].text(
                0.95,
                0.10,
                f"residual RMS = {rms:.1f} min",
                transform=ax[n].transAxes,
                fontsize=12,
                ha="right",
                backgroundcolor="w",
            )
            ax[n].text(
                0.95,
                0.85,
                f"P = {period:.1f}",
                transform=ax[n].transAxes,
                fontsize=14,
                ha="right",
                backgroundcolor="w",
            )

            ax[n].tick_params(labelsize=12)
            ax[n].set_ylabel("O-C [min]", fontsize=20)

            yrange = 24 * 60 * np.max(np.abs(yobs))
            yrange = np.max([5, 1.1 * yrange])
            ax[n].set_ylim(-yrange, +yrange)

        ax[0].set_title(f"{self.target}", fontsize=20)
        ax[-1].set_xlabel("Time [BJKD]", fontsize=20)

        plt.tight_layout()

        finish(fig, fig_path, show_plot)
