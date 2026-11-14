__all__ = [
    "plot_omc",
    "plot_litecurve",
    "dynesty_runplot",
    "dynesty_traceplot",
    "dynesty_cornerplot",
    "plot_results_omc",
    "plot_results_transit",
]

import matplotlib.pyplot as plt
import numpy as np
from astropy.stats import mad_std
from dynesty import plotting as dyplot
from matplotlib.ticker import FormatStrFormatter

from alderaan.constants import KEPLER_LCIT, KEPLER_SCIT
from alderaan.data_models.planet import Planet
from alderaan.models.transit_model import TransitModel
from alderaan.utils.astro import bin_data
from alderaan.utils.stats import robust_polyfit


def plot_omc(
    xtime,
    yobs,
    yerr,
    ymod,
    quality,
    out_prob,
    period,
    target,
    filepath=None,
    interactive=False,
):
    """
    Plot observed-minus-calculated
    Each data argument is a list with one entry per planet
    """
    npl = len(xtime)

    # set colors
    if np.any([op is None for op in out_prob]):
        marker_color = ["lightgrey"] * npl
        line_color = [f"C{n}" for n in range(npl)]
        cmap = None
    else:
        marker_color = [1 - op for op in out_prob]
        line_color = ["k"] * npl
        cmap = "viridis"

    # make plot
    fig, ax = plt.subplots(npl, figsize=(8, 3 * npl))
    if npl == 1:
        ax = [ax]

    for n in range(npl):
        ax[n].scatter(
            xtime[n],
            yobs[n] * 24 * 60,
            c=marker_color[n],
            cmap=cmap,
            label="Observed TTVs",
        )
        if ymod[n] is not None:
            ax[n].plot(
                xtime[n],
                ymod[n] * 24 * 60,
                lw=2,
                c=line_color[n],
                label="Regularized Model",
            )
        if len(quality[n]) > 0:
            ax[n].plot(xtime[n][~quality[n]], yobs[n][~quality[n]] * 24 * 60, "rx")

        if yerr[n] is not None:
            err = np.nanmedian(yerr[n]) * 24 * 60
            ax[n].text(
                0.05,
                0.10,
                f"measured error = {err:.1f} min",
                transform=ax[n].transAxes,
                fontsize=12,
                ha="left",
                backgroundcolor="w",
            )

        if ymod[n] is not None:
            rms = mad_std(yobs[n] - ymod[n], ignore_nan=True) * 24 * 60
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
            f"P = {period[n]:.1f}",
            transform=ax[n].transAxes,
            fontsize=14,
            ha="right",
            backgroundcolor="w",
        )

        ax[n].tick_params(labelsize=12)
        ax[n].set_ylabel("O-C [min]", fontsize=20)

        yrange = 24 * 60 * np.max([np.abs(yobs[n].min()), np.abs(yobs[n].max())])
        yrange = np.max([5, 1.1 * yrange])
        ax[n].set_ylim(-yrange, +yrange)

    ax[0].set_title(f"{target}", fontsize=20)
    ax[n].set_xlabel("Time [BJKD]", fontsize=20)

    plt.tight_layout()

    _save_or_close(fig, filepath, interactive)

    return fig


def plot_litecurve(litecurve, target, planets=None, filepath=None, interactive=False):
    # shorthand
    lc = litecurve

    if planets is not None:
        if isinstance(planets, Planet):
            planets = [planets]

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

    if planets is not None:
        ymin = 1 - yrange * (1.3 + 0.1 * len(planets))
        ymax = 1 + yrange
        ax.set_ylim(ymin, ymax)
        for n, p in enumerate(planets):
            ax.plot(
                p.transit_times,
                np.ones_like(p.transit_times) - (0.9 - 0.1 * n) * (1 - ymin),
                "^",
                c=f"C{n}",
            )

    plt.tight_layout()

    _save_or_close(fig, filepath, interactive)

    return fig, ax


def dynesty_runplot(results, target, filepath=None, interactive=False):
    fig, ax = dyplot.runplot(
        results, logplot=True, label_kwargs={"fontsize": 16}, color="#0d0887"
    )

    ax[0].set_title(target, fontsize=24)
    fig.tight_layout()

    _save_or_close(fig, filepath, interactive, bbox_inches="tight", dpi=96)

    return fig, ax


def dynesty_traceplot(results, target, planet_no, filepath=None, interactive=False):
    fig, ax = dyplot.traceplot(
        results,
        labels="C0 C1 r b T14 q1 q2".split(),
        dims=np.arange(5 * planet_no, 5 * (planet_no + 1)),
        label_kwargs={"fontsize": 14},
    )

    fig.tight_layout()
    fig.suptitle(
        f"{target} - Planet {planet_no}", fontsize=18, y=fig.subplotpars.top + 0.02
    )

    _save_or_close(fig, filepath, interactive, bbox_inches="tight", dpi=96)

    return fig, ax


def dynesty_cornerplot(results, target, planet_no, filepath=None, interactive=False):
    fig, ax = dyplot.cornerplot(
        results,
        labels="C0 C1 r b T14 q1 q2".split(),
        dims=np.arange(5 * planet_no, 5 * (planet_no + 1)),
        label_kwargs={"fontsize": 14},
        color=f"C{planet_no}",
    )

    fig.tight_layout()
    fig.suptitle(
        f"{target} - Planet {planet_no}", fontsize=18, y=fig.subplotpars.top + 0.02
    )

    _save_or_close(fig, filepath, interactive, bbox_inches="tight", dpi=96)

    return fig, ax


def plot_quick_fit_ttvs(
    target,
    planet_no,
    _t_obs,
    _t_mod,
    _f_obs,
    _f_mod,
    tc,
    tc_offset,
    tc_fit,
    x2_fit,
    chisq,
    transit_window_size,
    filepath=None,
    interactive=False,
):
    # recompute a fdew quantities
    quad_coeffs = robust_polyfit(tc_fit, x2_fit, 2)
    quad_model = np.polyval(quad_coeffs, tc_fit)
    qtc_min = -quad_coeffs[1] / (2 * quad_coeffs[0])
    qtc_err = np.sqrt(1 / quad_coeffs[0])
    qx2_min = np.polyval(quad_coeffs, qtc_min)
    _ttj = np.nanmean([qtc_min, np.mean(tc_fit)])
    _errj = qtc_err * (1 + np.std(x2_fit - quad_model))

    # make some plots
    fig, ax = plt.subplots(1, 2, figsize=(8, 3))

    ax[0].plot(_t_obs, _f_obs, "ko")
    ax[0].plot(_t_mod, _f_mod, c=f"C{planet_no}", lw=3)

    xticks = np.array(
        [tc - transit_window_size / 2, tc, tc + transit_window_size / 2]
    ).round(2)
    ax[0].set_xticks(xticks)
    ax[0].yaxis.set_major_formatter(FormatStrFormatter("%.4f"))
    ax[0].set_xlabel("Time [BJKD]", fontsize=14)
    ax[0].set_ylabel("Flux", fontsize=14)

    display = np.abs(chisq - qx2_min) < 2.5

    _x = tc_offset[display]
    _y_obs = (chisq - qx2_min)[display]
    _y_mod = np.polyval(quad_coeffs, _x) - qx2_min

    ax[1].plot(_x, _y_obs, "o", mec="k", mfc="w")
    ax[1].plot(_x, _y_mod, c=f"C{planet_no}", lw=3)
    ax[1].axvline(qtc_min, color="k", ls=":")
    ax[1].axvline(tc_fit[np.argmin(x2_fit)], color="k", ls=":")
    ax[1].axvline(np.mean(tc_fit), color="k", ls=":")
    ax[1].axvline(_ttj, color="k", lw=2)

    xticks = np.array([_ttj - 1.5 * _errj, _ttj, _ttj + 1.5 * _errj])
    ax[1].set_xticks(xticks, np.round(xticks - _ttj, 4))
    ax[1].set_ylim(-0.5, 2.5)
    ax[1].set_xlabel(r"$\Delta t_c$", fontsize=14)
    ax[1].set_ylabel(r"$\Delta \chi^2$", fontsize=14)

    plt.suptitle(f"{target} - Planet {planet_no}", fontsize=18)
    plt.tight_layout()

    _save_or_close(fig, filepath, interactive)

    return fig, ax


def _save_or_close(fig, filepath, interactive, **savefig_kwargs):
    if filepath is not None:
        fig.savefig(filepath, **savefig_kwargs)
    if not interactive:
        plt.close(fig)


def plot_results_omc(results, filepath=None, interactive=False):
    """
    Plot observed-minus-calculated transit times from a Results (wraps plot_omc)
    """
    xtime, yobs, ymod, quality, out_prob, period = [], [], [], [], [], []
    for n, tt in enumerate(results.transit_times):
        epoch, P = results.linear_ephemeris(n)
        linear = epoch + P * tt["index"].to_numpy()

        xtime.append(tt["ttime"].to_numpy())
        yobs.append(xtime[-1] - linear)
        ymod.append(tt["model"].to_numpy() - linear)
        quality.append(~tt["out_flag"].to_numpy())
        out_prob.append(tt["out_prob"].to_numpy())
        period.append(P)

    yerr = [None] * results.npl
    return plot_omc(
        xtime,
        yobs,
        yerr,
        ymod,
        quality,
        out_prob,
        period,
        results.target,
        filepath=filepath,
        interactive=interactive,
    )


def plot_results_transit(
    results, n, index=None, max_pts=1000, filepath=None, interactive=False
):
    """
    Plot long cadence photometry of planet n folded on the model transit times,
    with the median posterior model and residuals

    index : int, optional
        plot a single transit; default folds all transits
    """
    if results.litecurve is None:
        raise ValueError("Results.litecurve is required to plot transits")

    time = results.litecurve.time
    flux = results.litecurve.flux

    # median posterior transit parameters
    median = results.summary()["median"]
    q1, q2 = median["LD_Q1"], median["LD_Q2"]
    T14 = median[f"DUR14_{n}"]
    params = (
        0.0,
        results.linear_ephemeris(n)[1],
        median[f"ROR_{n}"],
        median[f"IMPACT_{n}"],
        T14,
        2 * np.sqrt(q1) * q2,
        np.sqrt(q1) * (1 - 2 * q2),
    )

    # fold photometry
    tt = results.transit_times[n]
    tcs = tt["model"] if index is None else tt["model"][tt["index"] == index]

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

    f_mod = TransitModel.transit_flux(t_mod, *params)
    f_mod_lc = TransitModel.transit_flux(t_mod, *params, exptime_offsets=lc_offsets)
    residuals = f_folded - TransitModel.transit_flux(
        t_folded, *params, exptime_offsets=lc_offsets
    )

    t_binned, f_binned = bin_data(t_folded, f_folded, T14 / 11)
    _, r_binned = bin_data(t_folded, residuals, T14 / 11)

    # deterministic subset of points to plot
    rng = np.random.default_rng(0)
    inds = rng.choice(len(t_folded), size=min(max_pts, len(t_folded)), replace=False)

    fig, ax = plt.subplots(
        2, 1, figsize=(12, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )

    ax[0].plot(t_folded[inds] * 24, f_folded[inds], "o", color="lightgrey")
    ax[0].plot(t_binned * 24, f_binned, "s", color="w", mec="k", ms=10)
    ax[0].plot(t_mod * 24, f_mod, color=f"C{n}", lw=3)
    ax[0].plot(t_mod * 24, f_mod_lc, "k--", lw=2)
    ax[0].set_ylabel("Flux", fontsize=24)
    ax[0].set_title(f"{results.target} - Planet {n}", fontsize=18)

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

    _save_or_close(fig, filepath, interactive)

    return fig, ax
