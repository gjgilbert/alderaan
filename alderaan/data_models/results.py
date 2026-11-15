__all__ = ["Results"]

import logging

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.stats import mad_std

from alderaan.constants import KEPLER_LCIT, KEPLER_SCIT
from alderaan.data_models.litecurve import LiteCurve
from alderaan.data_models.posteriors import Posteriors
from alderaan.data_models.transit_times import TransitTimes
from alderaan.models.transit_model import TransitModel
from alderaan.utils.astro import bin_data, ld_q_to_u
from alderaan.utils.stats import weighted_percentile

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

        hdul = fits.HDUList([primary_hdu])
        if self.posteriors is not None:
            hdul += list(self.posteriors.to_hdus())
        if self.transit_times is not None:
            for n, tt in enumerate(self.transit_times):
                hdul.append(tt.to_hdu(n))

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
            path to {target}-results.fits
        detrended_filepath : str, optional
            path to {target}_*_detrended.fits; litecurve is None if omitted
        """
        logger.info("reading %s", results_filepath)
        with fits.open(results_filepath) as hdul:
            npl = hdul[0].header["NPL"]

            if "SAMPLES" in hdul:
                posteriors = Posteriors.from_hdus(hdul["SAMPLES"], hdul["EPHEMERIS"])
            else:
                posteriors = None

            if "TTIMES_00" in hdul:
                transit_times = [
                    TransitTimes.from_hdu(hdul[f"TTIMES_{n:02d}"]) for n in range(npl)
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

    def plot_lightcurve(self, visit=None):
        """
        Plot detrended photometry; all visits if visit is None
        """
        lc = self.litecurve
        use = np.ones(len(lc.time), dtype=bool) if visit is None else lc.visit == visit
        if not np.any(use):
            raise ValueError(f"no data for visit {visit}")

        fig, ax = plt.subplots(1, 1, figsize=(20, 4))
        ax.plot(lc.time[use], lc.flux[use], "k.", ms=1)
        ax.tick_params(labelsize=12)
        ax.set_xlabel("Time [BKJD]", fontsize=24)
        ax.set_ylabel("Flux", fontsize=24)
        ax.set_xlim(lc.time[use].min(), lc.time[use].max())

        return fig

    def plot_omc(self, show_outliers=False):
        """
        Plot O-C transit times and regularized model for each planet
        """
        fig, ax = plt.subplots(self.npl, figsize=(12, 3 * self.npl))
        if self.npl == 1:
            ax = [ax]

        for n, tt in enumerate(self.transit_times):
            obs = np.isfinite(tt.ttime_obs)
            good = obs & ~tt.out_flag
            bad = obs & tt.out_flag
            omc_obs = (tt.ttime_obs - tt.linear_ephemeris()) * 24 * 60
            omc = (tt.ttime_mod - tt.linear_ephemeris()) * 24 * 60

            if show_outliers:
                ax[n].scatter(
                    tt.ttime_obs[obs], omc_obs[obs], c=tt.out_prob[obs], cmap="viridis"
                )
                ax[n].plot(tt.ttime_obs[bad], omc_obs[bad], "rx")
                ax[n].plot(tt.ttime_mod, omc, "k")
            else:
                ax[n].plot(tt.ttime_obs[good], omc_obs[good], "o", c="lightgrey")
                ax[n].plot(tt.ttime_mod, omc, c=f"C{n}", lw=3)

            rms = mad_std(omc_obs[good] - omc[good])
            ax[n].text(
                0.95,
                0.10,
                f"residual RMS = {rms:.1f} min",
                transform=ax[n].transAxes,
                ha="right",
            )
            ax[n].set_ylabel("O-C [min]", fontsize=20)

        ax[0].set_title(self.target, fontsize=20)
        ax[-1].set_xlabel("Time [BKJD]", fontsize=20)

        return fig

    def plot_transit(self, n, index=None, folded=True, max_pts=1000):
        """
        Plot photometry and median posterior model for planet n

        folded=True : phase-folded on all model transit times
        folded=False : single transit at transit index `index`
        """
        if folded:
            return self._plot_phase_folded_transit(n, max_pts)

        if index is None:
            raise ValueError("index is required when folded=False")

        return self._plot_single_transit(n, index)

    def median_transit_params(self, n):
        """
        (t0, period, ror, b, T14, u1, u2) at the posterior median, with t0 = 0;
        argument order of TransitModel.transit_flux
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

    def _plot_single_transit(self, n, index):
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
        f_mod = (TransitModel.transit_flux(t_mod - tc, *params) - 1.0) * 1000

        fig, ax = plt.subplots(1, 1, figsize=(12, 4))
        ax.plot(t_obs, f_obs, "o", color="lightgrey")
        ax.plot(t_mod, f_mod, color=f"C{n}", lw=2)
        ax.set_xlim(tc - 1.55 * T14, tc + 1.55 * T14)
        ax.tick_params(labelsize=12)
        ax.set_xlabel("Time [BKJD]", fontsize=24)
        ax.set_ylabel("Flux [ppt]", fontsize=24)
        ax.set_title(f"{self.target} - Planet {n}, transit {index}", fontsize=18)

        return fig

    def _plot_phase_folded_transit(self, n, max_pts=1000):
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

        return fig
