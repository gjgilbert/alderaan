__all__ = ["PlotPhotometry"]

import matplotlib.pyplot as plt
import numpy as np

from alderaan.constants import KEPLER_LCIT, KEPLER_SCIT
from alderaan.models.transit_model import TransitModel
from alderaan.quicklook.base import finish
from alderaan.utils.astro import bin_data


class PlotPhotometry:
    """
    Quicklook plots of photometry and transit models

    Parameters
    ----------
    litecurve : LiteCurve
        plotted by plot_litecurve
    planets : list of Planet, optional
        marks transit times on plot_litecurve
    results : Results, optional
        required by plot_transit, which plots results.litecurve
    """

    def __init__(self, litecurve, planets=None, results=None):
        self.litecurve = litecurve
        self.target = litecurve.target
        self.planets = planets
        self.results = results

    def plot_litecurve(self, fig_path=None, show_plot=False):
        lc = self.litecurve

        fig, ax = plt.subplots(1, 1, figsize=(20, 4))
        ax.plot(lc.time, lc.flux, "k.", ms=0.5)
        ax.tick_params(labelsize=12)
        ax.set_xlabel("Time [BJKD]", fontsize=24)
        ax.set_ylabel("Flux", fontsize=24)
        ax.set_xlim(lc.time.min(), lc.time.max())

        yrange = 1.3 * np.max(
            [
                np.abs(1 - np.percentile(lc.flux, 0.1)),
                np.abs(1 - np.percentile(lc.flux, 99.9)),
            ]
        )
        ax.set_ylim(1 - yrange, 1 + yrange)

        if self.planets is not None:
            ymin = 1 - yrange * (1.3 + 0.1 * len(self.planets))
            ymax = 1 + yrange
            ax.set_ylim(ymin, ymax)
            for n, p in enumerate(self.planets):
                ax.plot(
                    p.transit_times,
                    np.ones_like(p.transit_times) - (0.9 - 0.1 * n) * (1 - ymin),
                    "^",
                    c=f"C{n}",
                )

        plt.tight_layout()

        finish(fig, fig_path, show_plot)

    def plot_transit(self, n, index=None, max_pts=1000, fig_path=None, show_plot=False):
        """
        Plot long cadence photometry of planet n folded on the model transit times,
        with the median posterior model and residuals

        index : int, optional
            plot a single transit; default folds all transits
        """
        results = self.results
        if results is None or results.litecurve is None:
            raise ValueError("PlotPhotometry.results with a litecurve is required")

        time = results.litecurve.time
        flux = results.litecurve.flux

        params = results.median_transit_params(n)
        T14 = params[4]

        # fold photometry
        tt = results.transit_times[n]
        tcs = tt.model if index is None else tt.model[tt.index == index]

        t_folded, f_folded = [], []
        for tc in tcs:
            use = np.abs(time - tc) / T14 < 1.5
            t_folded.append(time[use] - tc)
            f_folded.append(flux[use])

        t_folded = np.hstack(t_folded)
        f_folded = np.hstack(f_folded)

        # median models: unbinned and integrated over a long cadence exposure
        lc_offsets = np.linspace(-KEPLER_LCIT / 2, KEPLER_LCIT / 2, 29)
        t_mod = np.arange(t_folded.min(), t_folded.max(), KEPLER_SCIT)

        transit_flux = TransitModel.transit_flux
        f_mod = transit_flux(t_mod, *params)
        f_mod_lc = transit_flux(t_mod, *params, exptime_offsets=lc_offsets)
        residuals = f_folded - transit_flux(
            t_folded, *params, exptime_offsets=lc_offsets
        )

        t_binned, f_binned = bin_data(t_folded, f_folded, T14 / 11)
        _, r_binned = bin_data(t_folded, residuals, T14 / 11)

        # deterministic subset of points to plot
        rng = np.random.default_rng(0)
        inds = rng.choice(
            len(t_folded), size=min(max_pts, len(t_folded)), replace=False
        )

        fig, ax = plt.subplots(
            2, 1, figsize=(12, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
        )

        ax[0].plot(t_folded[inds] * 24, f_folded[inds], "o", color="lightgrey")
        ax[0].plot(t_binned * 24, f_binned, "s", color="w", mec="k", ms=10)
        ax[0].plot(t_mod * 24, f_mod, color=f"C{n}", lw=3)
        ax[0].plot(t_mod * 24, f_mod_lc, "k--", lw=2)
        ax[0].set_ylabel("Flux", fontsize=24)
        ax[0].set_title(f"{self.target} - Planet {n}", fontsize=18)

        ax[1].plot(t_folded[inds] * 24, residuals[inds], ".", color="lightgrey")
        ax[1].plot(t_binned * 24, r_binned, "s", color="w", mec="k", ms=10)
        ax[1].axhline(0, color="k", ls="--", lw=2)
        ax[1].set_xlim(t_folded.min() * 24, t_folded.max() * 24)
        ax[1].set_xlabel("Time from mid-transit [hrs]", fontsize=24)
        ax[1].text(
            0.02,
            0.8,
            f"RMS = {int(np.std(residuals) * 1e6)} ppm",
            transform=ax[1].transAxes,
            fontsize=11,
        )

        finish(fig, fig_path, show_plot)
