__all__ = ["Results"]

import logging

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

from alderaan.constants import KEPLER_LCIT, KEPLER_SCIT
from alderaan.data_models.litecurve import LiteCurve
from alderaan.data_models.posteriors import Posteriors
from alderaan.data_models.transit_times import TransitTimes
from alderaan.transit_model import transit_flux
from alderaan.utils.astro import ld_q_to_u
from alderaan.utils.plotting import annotate_plot, label_axes, save_and_close
from alderaan.utils.stats import bin_data, weighted_percentile

logger = logging.getLogger(__name__)


class Results:
    """
    Packaged pipeline outputs for one target

    Parameters
    ----------
    posteriors : Posteriors, optional
        nested sampling posteriors
    transit_times : list of TransitTimes, optional
        one per planet, in planet order
    litecurve : LiteCurve, optional
        final detrended photometry (the data fit by the sampler)

    mission and target are taken from the supplied objects, which must agree
    """

    def __init__(self, posteriors=None, transit_times=None, litecurve=None):
        self.posteriors = posteriors
        self.transit_times = transit_times
        self.litecurve = litecurve

        components = []
        if posteriors is not None:
            components.append(posteriors)
        if transit_times is not None:
            components += transit_times
        if litecurve is not None:
            components.append(litecurve)

        missions = {c.mission for c in components}
        targets = {c.target for c in components}
        if len(missions) > 1 or len(targets) > 1:
            raise ValueError(
                f"inconsistent metadata: missions {missions}, targets {targets}"
            )
        self.mission = missions.pop() if missions else None
        self.target = targets.pop() if targets else None

        if posteriors is not None and transit_times is not None:
            if posteriors.npl != len(transit_times):
                raise ValueError(
                    f"posteriors have {posteriors.npl} planets; "
                    f"{len(transit_times)} transit time tables"
                )

    @property
    def npl(self):
        if self.transit_times is not None:
            return len(self.transit_times)
        if self.posteriors is not None:
            return self.posteriors.npl
        return None

    def to_fits(self, filepath):
        """
        Write PRIMARY, SAMPLES + EPHEMERIS (if posteriors), and TTIMES_nn (if
        transit_times); the detrended litecurve is written by LiteCurve.to_fits
        """
        primary_hdu = fits.PrimaryHDU()
        primary_hdu.header["MISSION"] = self.mission
        primary_hdu.header["TARGET"] = self.target
        primary_hdu.header["NPL"] = self.npl

        # component HDU builders are private to data_models; Results owns the file
        hdul = fits.HDUList([primary_hdu])
        if self.posteriors is not None:
            hdul += list(self.posteriors._to_hdus())
        if self.transit_times is not None:
            for n, tt in enumerate(self.transit_times):
                hdul.append(tt._to_hdu(n))

        hdul.writeto(filepath, overwrite=True)
        logger.info("wrote %s", filepath)

    @classmethod
    def from_fits(cls, results_filepath, detrended_filepath=None):
        """
        Load a results file written by Results.to_fits and, optionally, a detrended
        litecurve written by LiteCurve.to_fits

        Parameters
        ----------
        results_filepath : str
            path to {target}_results.fits
        detrended_filepath : str, optional
            path to {target}_*_detrended.fits; litecurve is None if omitted
        """
        logger.info("reading %s", results_filepath)
        with fits.open(results_filepath) as hdul:
            npl = hdul[0].header["NPL"]

            if "SAMPLES" in hdul:
                posteriors = Posteriors._from_hdus(hdul["SAMPLES"], hdul["EPHEMERIS"])
            else:
                posteriors = None

            if "TTIMES_00" in hdul:
                transit_times = [
                    TransitTimes._from_hdu(hdul[f"TTIMES_{n:02d}"]) for n in range(npl)
                ]
            else:
                transit_times = None

        if detrended_filepath is None:
            litecurve = None
        else:
            litecurve = LiteCurve.from_fits(detrended_filepath)

        return cls(
            posteriors=posteriors, transit_times=transit_times, litecurve=litecurve
        )

    def summary(self):
        return self.posteriors.summary()

    def planet_samples(self, n):
        return self.posteriors.planet_samples(n)

    # plot methods: fig_path saves the figure there and closes it; default leaves
    # it open

    def plot_lightcurve(self, visit=None, ax=None, fig_path=None):
        """
        Plot detrended photometry with transit times marked; returns the Figure
        (see LiteCurve.plot)
        """
        mark_times = [tt.ttime_mod for tt in self.transit_times]
        return self.litecurve.plot(
            visit=visit, mark_times=mark_times, ax=ax, fig_path=fig_path
        )

    def plot_omc(self, n, show_outliers=False, fig_path=None):
        """
        Plot O-C for planet n; returns the Figure (see TransitTimes.plot_omc)
        """
        return self.transit_times[n].plot_omc(
            show_outliers=show_outliers, color=f"C{n}", fig_path=fig_path
        )

    def median_transit_params(self, n):
        """
        (t0, period, ror, b, T14, u1, u2) at the posterior median, with t0 = 0;
        argument order of transit_flux
        """
        median = self.summary()["median"]
        period = weighted_percentile(
            self.planet_samples(n)["PERIOD"].to_numpy(),
            50,
            w=self.posteriors.weights(),
        )

        return (
            0.0,
            period,
            median[f"ROR_{n}"],
            median[f"IMPACT_{n}"],
            median[f"DUR14_{n}"],
            *ld_q_to_u(median["LD_Q1"], median["LD_Q2"]),
        )

    def plot_transit(self, n, index, ax=None, fig_path=None):
        """
        Plot photometry and median posterior model around one transit of planet n
        (transit index `index`); returns the Figure
        """
        tt = self.transit_times[n]
        if not np.any(tt.index == index):
            raise ValueError(f"planet {n} has no transit with index {index}")

        tc = tt.ttime_mod[tt.index == index][0]
        params = self.median_transit_params(n)
        T14 = params[4]

        lc = self.litecurve
        use = np.abs(lc.time - tc) / T14 < 1.5
        if not np.any(use):
            raise ValueError(f"no photometry near planet {n} transit {index}")

        t_obs = lc.time[use]
        f_obs = (lc.flux[use] - 1.0) * 1000
        t_mod = np.arange(t_obs.min(), t_obs.max(), KEPLER_SCIT)
        f_mod = (transit_flux(t_mod - tc, *params) - 1.0) * 1000

        new_figure = ax is None
        if new_figure:
            _, ax = plt.subplots(1, 1, figsize=(12, 4))
        ax.plot(t_obs, f_obs, "o", color="lightgrey")
        ax.plot(t_mod, f_mod, color=f"C{n}", lw=2)
        ax.set_xlim(tc - 1.55 * T14, tc + 1.55 * T14)
        annotate_plot(ax, f"P = {params[1]:.2f} d", "lower right")
        label_axes(
            ax,
            "Time [BKJD]",
            "Flux [ppt]",
            f"{self.target} - Planet {n}, transit {index}",
        )
        if new_figure:
            ax.figure.tight_layout()
        save_and_close(ax.figure, fig_path)

        return ax.figure

    def plot_folded_transit(self, n, max_pts=1000, fig_path=None):
        """
        Plot photometry of planet n folded on the model transit times, with the
        median posterior model and residuals; returns the Figure

        max_pts : int
            maximum number of unbinned points drawn (deterministic subset)
        """
        params = self.median_transit_params(n)
        T14 = params[4]

        time = self.litecurve.time
        flux = self.litecurve.flux

        # fold photometry on the model transit times
        t_folded, f_folded = [], []
        for tc in self.transit_times[n].ttime_mod:
            use = np.abs(time - tc) / T14 < 1.5
            t_folded.append(time[use] - tc)
            f_folded.append(flux[use])

        t_folded = np.hstack(t_folded)
        f_folded = np.hstack(f_folded)

        # median models: unbinned and integrated over a long cadence exposure
        lc_offsets = np.linspace(-KEPLER_LCIT / 2, KEPLER_LCIT / 2, 29)
        t_mod = np.arange(t_folded.min(), t_folded.max(), KEPLER_SCIT)

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
        label_axes(ax[0], ylabel="Flux", title=f"{self.target} - Planet {n}")

        ax[1].plot(t_folded[inds] * 24, residuals[inds], ".", color="lightgrey")
        ax[1].plot(t_binned * 24, r_binned, "s", color="w", mec="k", ms=10)
        ax[1].axhline(0, color="k", ls="--", lw=2)
        ax[1].set_xlim(t_folded.min() * 24, t_folded.max() * 24)
        annotate_plot(ax[1], f"RMS = {int(np.std(residuals) * 1e6)} ppm", "lower left")
        label_axes(ax[1], xlabel="Time from mid-transit [hrs]")
        fig.tight_layout()
        save_and_close(fig, fig_path)

        return fig
