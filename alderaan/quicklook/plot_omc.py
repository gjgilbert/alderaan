__all__ = ["PlotOMC"]

import matplotlib.pyplot as plt
import numpy as np
import numpy.polynomial.polynomial as poly
from astropy.stats import mad_std

from alderaan.quicklook.base import finish


class PlotOMC:
    """
    Quicklook plots of observed-minus-calculated (O-C) transit times

    Parameters
    ----------
    transit_times : list of pd.DataFrame
        one per planet; columns [index, ttime, model, out_prob, out_flag]
        (e.g. Results.transit_times); out_prob may be all NaN
    target : str
    """

    def __init__(self, transit_times, target):
        self.transit_times = transit_times
        self.target = target

    def plot_omc(self, fig_path=None, show_plot=False):
        npl = len(self.transit_times)

        fig, ax = plt.subplots(npl, figsize=(8, 3 * npl))
        if npl == 1:
            ax = [ax]

        for n, tt in enumerate(self.transit_times):
            index = tt["index"].to_numpy()
            xtime = tt["ttime"].to_numpy()
            out_prob = tt["out_prob"].to_numpy()
            out_flag = tt["out_flag"].to_numpy()

            # O-C relative to a linear fit to the model transit times
            epoch, period = poly.polyfit(index, tt["model"].to_numpy(), 1)
            linear = epoch + period * index
            yobs = xtime - linear
            ymod = tt["model"].to_numpy() - linear

            if np.all(np.isnan(out_prob)):
                marker_color, line_color, cmap = "lightgrey", f"C{n}", None
            else:
                marker_color, line_color, cmap = 1 - out_prob, "k", "viridis"

            ax[n].scatter(
                xtime, yobs * 24 * 60, c=marker_color, cmap=cmap, label="Observed TTVs"
            )
            ax[n].plot(
                xtime, ymod * 24 * 60, lw=2, c=line_color, label="Regularized Model"
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
