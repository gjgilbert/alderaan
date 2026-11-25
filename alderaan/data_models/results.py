"""Results: posteriors, transit times, and litecurve for one target"""

__all__ = ["Results"]

import logging
import os

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

from alderaan.constants import EXPTIME, KEPLER_SCIT, SEED
from alderaan.data_models.litecurve import LiteCurve
from alderaan.data_models.posteriors import Posteriors
from alderaan.data_models.transit_times import TransitTimes
from alderaan.transit_model import exposure_offsets, transit_flux
from alderaan.utils.astro import ld_q_to_u
from alderaan.utils.plotting import annotate_plot, label_axes, save_and_close
from alderaan.utils.stats import bin_data, legendre_p1, weighted_percentile

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

    Notes
    -----
    mission and target are taken from the supplied objects, which must agree.
    Plot methods return the Figure.
    """

    def __init__(self, posteriors=None, transit_times=None, litecurve=None):
        self.posteriors = posteriors
        self.transit_times = transit_times
        self.litecurve = litecurve
        self.meta = {}  # FITS header cards, written to PRIMARY
        self._summary = None  # set on the first summary() call

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
        """Number of planets, from transit_times or posteriors; None if neither"""
        if self.transit_times is not None:
            return len(self.transit_times)
        if self.posteriors is not None:
            return self.posteriors.npl
        return None

    def to_fits(self, filepath):
        """
        Write PRIMARY, SAMPLES, EPHEMERIS and BATCHES, and TTIMES_nn HDUs to filepath

        PRIMARY holds MISSION, TARGET, NPL and meta; SAMPLES, EPHEMERIS, BATCHES are
        written if posteriors is set, one TTIMES_nn per planet if transit_times is
        set. The litecurve is not written; use LiteCurve.to_fits.
        """
        primary_hdu = fits.PrimaryHDU()
        primary_hdu.header["MISSION"] = self.mission
        primary_hdu.header["TARGET"] = self.target
        primary_hdu.header["NPL"] = self.npl
        primary_hdu.header.update(self.meta)

        # component HDU builders are private to data_models; Results owns the file
        hdul = fits.HDUList([primary_hdu])
        if self.posteriors is not None:
            hdul += list(self.posteriors._to_hdus())
        if self.transit_times is not None:
            for n, tt in enumerate(self.transit_times):
                hdul.append(tt._to_hdu(n))

        # write then rename, so a crash never leaves a partial file at filepath
        hdul.writeto(filepath + ".tmp", overwrite=True)
        os.replace(filepath + ".tmp", filepath)
        logger.info("wrote %s", filepath)

    @classmethod
    def from_fits(cls, results_filepath, detrended_filepath=None):
        """
        Load a Results.to_fits file and, optionally, a LiteCurve.to_fits file

        Parameters
        ----------
        results_filepath : str
            path to {target}_results.fits
        detrended_filepath : str, optional
            path to {target}_detrended.fits; litecurve is None if omitted
        """
        logger.info("reading %s", results_filepath)
        with fits.open(results_filepath) as hdul:
            npl = hdul[0].header["NPL"]
            skip = ("SIMPLE", "BITPIX", "NAXIS", "EXTEND", "MISSION", "TARGET", "NPL")
            meta = {k: v for k, v in hdul[0].header.items() if k not in skip}

            if "SAMPLES" in hdul:
                posteriors = Posteriors._from_hdus(hdul)
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

        results = cls(
            posteriors=posteriors, transit_times=transit_times, litecurve=litecurve
        )
        results.meta = meta
        return results

    def summary(self):
        """
        Posterior summary statistics (see Posteriors.summary)

        Computed on the first call and cached; posteriors must not be replaced
        or modified after that.
        """
        if self._summary is None:
            self._summary = self.posteriors.summary()
        return self._summary

    def planet_samples(self, n):
        """Physical posterior samples for planet n (see Posteriors.planet_samples)"""
        return self.posteriors.planet_samples(n)

    # plot methods: fig_path saves the figure there and closes it (unless ax was
    # given); default leaves it open

    def plot_lightcurve(self, visit=None, ax=None, fig_path=None):
        """Plot detrended photometry with ttime_mod marked (see LiteCurve.plot)"""
        mark_times = [tt.ttime_mod for tt in self.transit_times]
        return self.litecurve.plot(
            visit=visit, mark_times=mark_times, ax=ax, fig_path=fig_path
        )

    def plot_omc(self, n, show_outliers=False, fig_path=None):
        """Plot O-C for planet n (see TransitTimes.plot_omc)"""
        tt = self.transit_times[n]
        return tt.plot_omc(
            show_outliers=show_outliers,
            color=f"C{n}",
            title=f"{tt.target}-{n}",
            fig_path=fig_path,
        )

    def median_transit_params(self, n):
        """
        Posterior median (t0, period, ror, b, T14, u1, u2) of planet n, with t0 = 0

        In transit_flux argument order. period is the weighted median of
        planet_samples PERIOD; u1, u2 are converted from the median q1, q2.
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

    def median_transit_times(self, n):
        """
        Transit times of planet n at the posterior median of C0 and C1

        ttime_mod + C0 + C1 * legx (see Posteriors).
        """
        tt = self.transit_times[n]
        post = self.posteriors
        median = self.summary()["median"]
        legx = legendre_p1(
            post.ref_epoch[n] + post.ref_period[n] * tt.index,
            (post.t_start, post.t_end),
        )
        return tt.ttime_mod + median[f"C0_{n}"] + median[f"C1_{n}"] * legx

    def plot_transit(self, n, index, ax=None, fig_path=None):
        """
        Plot photometry and median posterior model around one transit of planet n

        index is the transit index (TransitTimes.index). Flux is in ppt; the model
        is integrated over the exposure as sampled.
        """
        tt = self.transit_times[n]
        if not np.any(tt.index == index):
            raise ValueError(f"planet {n} has no transit with index {index}")

        tc = self.median_transit_times(n)[tt.index == index][0]
        params = self.median_transit_params(n)
        T14 = params[4]

        lc = self.litecurve
        use = np.abs(lc.time - tc) / T14 < 1.5
        if not np.any(use):
            raise ValueError(f"no photometry near planet {n} transit {index}")

        t_obs = lc.time[use]
        f_obs = (lc.flux[use] - 1.0) * 1000
        t_mod = np.arange(t_obs.min(), t_obs.max(), KEPLER_SCIT)
        om = lc.obsmode[use][0]
        offsets = exposure_offsets(EXPTIME[om], self.posteriors.supersample[om])
        f_mod = (
            transit_flux(t_mod - tc, *params, exptime_offsets=offsets) - 1.0
        ) * 1000

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
        save_and_close(ax.figure, fig_path, close=new_figure)

        return ax.figure

    def _planet_model(self, n):
        """
        Median model flux of planet n at every litecurve cadence, and its window

        Returns
        -------
        model : ndarray
            median model flux, integrated over each obsmode's exposure as sampled;
            1 away from transits
        window : ndarray
            True within 1.5 T14 of a median transit time (the folded window)
        """
        lc = self.litecurve
        params = self.median_transit_params(n)
        T14 = params[4]
        offsets = {
            om: exposure_offsets(EXPTIME[om], self.posteriors.supersample[om])
            for om in np.unique(lc.obsmode)
        }
        in_obsmode = {om: lc.obsmode == om for om in offsets}
        if np.any(np.diff(lc.time) < 0):
            raise ValueError("litecurve.time must be sorted")
        # cadences near each transit: a superset of every mask below
        half = 1.5 * T14 + max(EXPTIME[om] for om in offsets)
        ttimes = self.median_transit_times(n)
        lo = np.searchsorted(lc.time, ttimes - 2 * half)
        hi = np.searchsorted(lc.time, ttimes + 2 * half)

        model = np.ones_like(lc.flux)
        window = np.zeros(len(lc.time), dtype=bool)
        for tc, i, j in zip(ttimes, lo, hi):
            dt = lc.time[i:j] - tc
            window[i:j] |= np.abs(dt) < 1.5 * T14
            for om, offs in offsets.items():
                # beyond the window too, so the model returns to 1 for short T14
                q = in_obsmode[om][i:j] & (np.abs(dt) < 1.5 * T14 + EXPTIME[om])
                if q.any():
                    model[i:j][q] = transit_flux(dt[q], *params, exptime_offsets=offs)

        return model, window

    def reduced_chi2(self):
        """
        Quicklook reduced chi2: {"CHI2RED": star, "CHI2RED{n}": planet n}

        Residuals of the median model, with litecurve.error, in the 1.5 T14
        windows of _planet_model. Planet models add as in TransitModel. The star
        value uses the union of windows, dof = N - (5 npl + 2); planet n's value
        divides the sibling models out of its window, dof = N - 7. Keys are FITS
        header cards for meta.
        """
        lc = self.litecurve
        models, windows = zip(*[self._planet_model(n) for n in range(self.npl)])
        total = 1 + np.sum([m - 1 for m in models], axis=0)

        union = np.any(windows, axis=0)
        chi2 = np.sum(((lc.flux - total) / lc.error)[union] ** 2)
        out = {"CHI2RED": float(chi2 / (union.sum() - (5 * self.npl + 2)))}

        for n, (model, window) in enumerate(zip(models, windows)):
            sibling = total - (model - 1)
            # (flux / sibling - model) / (error / sibling)
            resid = (lc.flux - sibling * model) / lc.error
            chi2 = np.sum(resid[window] ** 2)
            out[f"CHI2RED{n}"] = float(chi2 / (window.sum() - 7))

        return out

    def plot_folded_transit(self, n, max_pts=1000, fig_path=None):
        """
        Plot folded photometry of planet n with the median model and residuals

        Folded on median_transit_times. The model is drawn unbinned (solid) and
        integrated over each supersampled obsmode's exposure (dashed).

        Parameters
        ----------
        max_pts : int
            maximum number of unbinned points drawn (deterministic subset)
        """
        params = self.median_transit_params(n)
        T14 = params[4]

        # sibling planet models divided out, as in reduced_chi2
        models = [self._planet_model(m)[0] for m in range(self.npl)]
        sibling = 1 + np.sum([models[m] - 1 for m in range(self.npl) if m != n], axis=0)

        time = self.litecurve.time
        flux = self.litecurve.flux / sibling
        obsmode = self.litecurve.obsmode

        t_folded, f_folded, m_folded, om_folded = [], [], [], []
        n_transit = 0
        for tc in self.median_transit_times(n):
            use = np.abs(time - tc) / T14 < 1.5
            n_transit += np.any(use)
            t_folded.append(time[use] - tc)
            f_folded.append(flux[use])
            m_folded.append(models[n][use])
            om_folded.append(obsmode[use])

        t_folded = np.hstack(t_folded)
        f_folded = np.hstack(f_folded)
        m_folded = np.hstack(m_folded)
        om_folded = np.hstack(om_folded)

        # exposure integration as sampled (TransitModel.supersample)
        supersample = self.posteriors.supersample
        exptime_offsets = {
            om: exposure_offsets(EXPTIME[om], supersample[om])
            for om in np.unique(om_folded)
        }

        # median models: unbinned, and integrated over each obsmode's exposure
        t_mod = np.arange(t_folded.min(), t_folded.max(), KEPLER_SCIT)
        f_mod = transit_flux(t_mod, *params)
        f_mod_integrated = [
            transit_flux(t_mod, *params, exptime_offsets=offsets)
            for om, offsets in exptime_offsets.items()
            if supersample[om] > 1
        ]

        residuals = f_folded - m_folded

        t_binned, f_binned, f_binned_err = bin_data(t_folded, f_folded, T14 / 11)
        _, r_binned, r_binned_err = bin_data(t_folded, residuals, T14 / 11)

        rng = np.random.default_rng(SEED)
        inds = rng.choice(
            len(t_folded), size=min(max_pts, len(t_folded)), replace=False
        )

        fig, ax = plt.subplots(
            2, 1, figsize=(8, 4), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
        )

        # empty bins: open circles on the unbinned model (top) and on zero (bottom)
        empty = np.isnan(f_binned)
        f_empty = transit_flux(t_binned[empty], *params)

        # flux plotted as ppm offset from 1, residuals in ppm
        ax[0].plot(
            t_folded[inds] * 24,
            (f_folded[inds] - 1) * 1e6,
            "o",
            color="lightgrey",
            ms=3,
        )
        ax[0].errorbar(
            t_binned * 24,
            (f_binned - 1) * 1e6,
            f_binned_err * 1e6,
            fmt="o",
            color="k",
            ms=5,
        )
        ax[0].plot(
            t_binned[empty] * 24, (f_empty - 1) * 1e6, "o", mfc="none", mec="k", ms=5
        )
        ax[0].plot(t_mod * 24, (f_mod - 1) * 1e6, color=f"C{n}", lw=3)
        for f in f_mod_integrated:
            ax[0].plot(t_mod * 24, (f - 1) * 1e6, "k--", lw=2)
        label_axes(
            ax[0],
            ylabel="Flux [ppm]",
            title=f"{self.target} - Planet {n}",
            label_size=15,
            title_size=16,
        )

        ax[1].plot(
            t_folded[inds] * 24, residuals[inds] * 1e6, ".", color="lightgrey", ms=3
        )
        ax[1].errorbar(
            t_binned * 24, r_binned * 1e6, r_binned_err * 1e6, fmt="o", color="k", ms=5
        )
        ax[1].plot(
            t_binned[empty] * 24, np.zeros(empty.sum()), "o", mfc="none", mec="k", ms=5
        )
        ax[1].axhline(0, color="k", ls="--", lw=2)
        ax[1].set_xlim(t_folded.min() * 24, t_folded.max() * 24)

        # headroom above the data for the top annotations; residuals symmetric
        lo, hi = ax[0].get_ylim()
        ax[0].set_ylim(lo, hi + 0.25 * (hi - lo))
        ymax = np.max(np.abs(ax[1].get_ylim()))
        ax[1].set_ylim(-ymax, ymax)

        annotate_plot(ax[0], f"N_transit = {n_transit}", "upper left")
        annotate_plot(ax[0], f"P = {params[1]:.2f} d", "upper right")
        annotate_plot(
            ax[0], f"residual RMS = {int(np.std(residuals) * 1e6)} ppm", "lower right"
        )
        chi2_red = self.meta[f"CHI2RED{n}"]
        annotate_plot(
            ax[0],
            f"reduced χ² = {chi2_red:.2f}",
            "lower left",
            edgecolor="r" if chi2_red > 2 else "none",
        )
        label_axes(ax[1], xlabel="Time from mid-transit [hrs]", label_size=15)
        ax[1].set_ylabel("Residuals", fontsize=12)
        fig.tight_layout()
        save_and_close(fig, fig_path)

        return fig
