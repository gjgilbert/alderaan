__all__ = [
    "plot_omc",
    "plot_holczer",
    "plot_omc_model_selection",
    "plot_folded_transit",
    "plot_tc_vs_chisq",
    "plot_acf",
    "plot_synthetic_noise",
]


from astropy.stats import mad_std
import numpy as np
import matplotlib.pyplot as plt
from .utils import bin_data


def plot_omc(linear_ephemeris, tts_obs, tts_mod, out=None, colors="auto"):
    """
    Docstring
    """
    npl = len(linear_ephemeris)

    if out is None:
        out = []
        for n in range(npl):
            out.append(np.zeros(len(linear_ephemeris[n]), dtype="bool"))

    if colors == "auto":
        colors = []
        for n in range(npl):
            colors.append(f"C{n}")

    fig, ax = plt.subplots(npl, figsize=(8, 2 * npl))
    if npl == 1:
        ax = [ax]

    for n in range(npl):
        xtime = linear_ephemeris[n]
        yomc_obs = (tts_obs[n] - linear_ephemeris[n]) * 24 * 60
        yomc_mod = (tts_mod[n] - linear_ephemeris[n]) * 24 * 60

        ax[n].plot(xtime, yomc_obs, "o", c="lightgrey")
        ax[n].plot(xtime, yomc_mod, lw=2, c=colors[n])
        ax[n].plot(xtime[out[n]], yomc_obs[out[n]], "rx")
        ax[n].set_ylabel("O-C [min]", fontsize=14)
    ax[npl - 1].set_xlabel("Time [BJKD]", fontsize=14)

    return fig, ax


def plot_holczer(data, n):
    xtime = data["tts"][n]
    yomc = data["tts"][n] - data["epoch"][n] - data["period"][n] * data["inds"][n]
    out = data["out"][n]

    xtime_interp = data["full_tts"][n]
    yomc_interp = (
        data["full_tts"][n]
        - data["epoch"][n]
        - data["period"][n] * data["full_inds"][n]
    )

    fig = plt.figure(figsize=(8, 3))

    plt.plot(xtime[~out], yomc[~out] * 24 * 60, "o", c="grey", label="Holczer+2016")
    plt.plot(xtime[out], yomc[out] * 24 * 60, "rx")
    plt.plot(xtime_interp, yomc_interp * 24 * 60, "k+", label="Interpolation")
    plt.xlabel("Time [BJKD]", fontsize=16)
    plt.ylabel("O-C [min]", fontsize=16)
    plt.legend(fontsize=12)

    return fig


def plot_tc_vs_chisq(t_obs, f_obs, t_mod, f_mod, tc_fit, x2_fit, x2_mod, tc, color):
    """
    Docstring
    """
    fig, ax = plt.subplots(1, 2, figsize=(8, 3))

    ax[0].plot((t_obs - tc) * 24, (f_obs - 1) * 1e6, "ko")
    ax[0].plot(t_mod * 24, (f_mod - 1) * 1e6, c=color, lw=2)
    ax[0].set_xlabel(r"$\Delta t$ (hours)", fontsize=14)
    ax[0].set_ylabel(r"$\Delta F$ (ppm)", fontsize=14)

    ax[1].plot((tc_fit - tc) * 24, x2_fit - x2_fit.min(), "ko")
    ax[1].plot((tc_fit - tc) * 24, x2_mod - x2_fit.min(), c=color, lw=3)
    ax[1].axvline(0, color="k", ls=":", lw=2)
    ax[1].set_xlabel(r"$\Delta t$ (hours)", fontsize=14)
    ax[1].set_ylabel(r"$\Delta \chi^2$", fontsize=14)

    plt.tight_layout()

    return fig, ax


def plot_omc_model_selection(
    xtime, yomc, fg_prob, bad, full_linear_ephemeris, full_omc_trend, err, rms, target
):
    """
    Docstring
    """
    fig = plt.figure(figsize=(12, 4))

    plt.scatter(xtime, yomc * 24 * 60, c=1 - fg_prob, cmap="viridis", label="MAP TTVs")
    plt.plot(xtime[bad], yomc[bad] * 24 * 60, "rx")
    plt.plot(
        full_linear_ephemeris, full_omc_trend * 24 * 60, "k", label="Regularized model"
    )
    plt.xlabel("Time [BJKD]", fontsize=20)
    plt.ylabel("O-C [min]", fontsize=20)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14, loc="upper right")
    plt.title(target, fontsize=20)
    plt.text(
        xtime.min(),
        yomc.min() * 24 * 60,
        f"measured error = {err:.1f} min",
        fontsize=16,
        ha="left",
        backgroundcolor="w",
    )
    plt.text(
        xtime.max(),
        yomc.min() * 24 * 60,
        f"residual RMS = {rms:.1f} min",
        fontsize=16,
        ha="right",
        backgroundcolor="w",
    )

    return fig


def plot_folded_transit(lc, sc, tts, depth, duration, target, n):
    """
    Docstring
    """
    t_folded = []
    f_folded = []

    # grab the data
    for t0 in tts:
        if sc is not None:
            use = np.abs(sc.time - t0) / duration < 1.5

            if np.sum(use) > 0:
                t_folded.append(sc.time[use] - t0)
                f_folded.append(sc.flux[use])

        if lc is not None:
            use = np.abs(lc.time - t0) / duration < 1.5

            if np.sum(use) > 0:
                t_folded.append(lc.time[use] - t0)
                f_folded.append(lc.flux[use])

    # sort the data
    t_folded = np.hstack(t_folded)
    f_folded = np.hstack(f_folded)

    order = np.argsort(t_folded)
    t_folded = t_folded[order]
    f_folded = f_folded[order]

    # bin the data
    t_binned, f_binned = bin_data(t_folded, f_folded, duration / 11)

    # set undersampling factor and plotting limits
    inds = np.arange(len(t_folded), dtype="int")
    inds = np.random.choice(inds, size=np.min([3000, len(inds)]), replace=False)

    ymin = -3 * mad_std(f_folded) - depth
    ymax = +3 * mad_std(f_folded)

    # plot the data
    fig = plt.figure(figsize=(8, 3))
    plt.plot(t_folded[inds] * 24, (f_folded[inds] - 1) * 1e6, ".", c="lightgrey")
    plt.plot(
        t_binned * 24,
        (f_binned - 1) * 1e6,
        "o",
        ms=8,
        color=f"C{n}",
        label=f"{target}-{n}",
    )
    plt.xlim(t_folded.min() * 24, t_folded.max() * 24)
    plt.ylim(ymin * 1e6, ymax * 1e6)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel("Time from mid-transit [hrs]", fontsize=16)
    plt.ylabel("Flux (ppm)", fontsize=16)
    plt.legend(fontsize=12, loc="lower right", framealpha=1)

    return fig


def plot_acf(xcor, acf_emp, acf_mod, xf, yf, freqs, target_name, season):
    """
    Docstring
    """
    fig = plt.figure(figsize=(20, 5))

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.8)

    ax = plt.subplot2grid(shape=(5, 10), loc=(0, 0), rowspan=3, colspan=7)
    ax.plot(xcor * 24, acf_emp, color="lightgrey")
    ax.plot(xcor * 24, acf_mod, c="red")
    ax.set_xlim(xcor.min() * 24, xcor.max() * 24)
    ax.set_xticks(np.arange(0, xcor.max() * 24, 2))
    ax.set_xticklabels([])
    ax.set_ylim(acf_emp.min() * 1.1, acf_emp.max() * 1.1)
    ax.set_ylabel("ACF", fontsize=20)
    ax.text(
        xcor.max() * 24 - 0.15,
        acf_emp.max(),
        f"{target_name}, {season}",
        va="top",
        ha="right",
        fontsize=20,
    )

    ax = plt.subplot2grid(shape=(5, 10), loc=(0, 7), rowspan=5, colspan=3)
    ax.plot(xf / 24 / 3600 * 1e3, yf, color="k", lw=0.5)
    for f in freqs:
        ax.axvline(f / 24 / 3600 * 1e3, color="red", zorder=0, lw=3, alpha=0.3)
    ax.set_xlim(xf.min() / 24 / 3600 * 1e3, xf.max() / 24 / 3600 * 1e3)
    ax.set_ylim(yf.min(), 1.2 * yf.max())
    ax.set_ylabel("Power", fontsize=20)
    ax.set_yticks([])
    ax.set_xlabel("Frequency [mHz]", fontsize=20)

    for i, sf in enumerate(np.sort(freqs)[::-1]):
        ax.text(
            xf.min() / 24 / 3600 * 1e3 + 0.1,
            yf.max() * (1.1 - 0.1 * i),
            f"{0:.2f} min".format(24 * 60 / sf),
            fontsize=16,
        )

    ax = plt.subplot2grid(shape=(5, 10), loc=(3, 0), rowspan=2, colspan=7)
    ax.plot(xcor * 24, acf_emp - acf_mod, c="lightgrey")
    ax.set_xlim(xcor.min() * 24, xcor.max() * 24)
    ax.set_xticks(np.arange(0, xcor.max() * 24, 2))
    ax.set_xlabel("Lag time [hours]", fontsize=20)
    ax.set_ylabel("Residuals", fontsize=20)

    return fig


def plot_synthetic_noise(t, white_noise, red_noise, target, season, depths):
    """
    Docstring
    """
    fig = plt.figure(figsize=(20, 5))

    plt.plot(t, white_noise + red_noise, ".", c="lightgrey")
    plt.plot(t, red_noise, c="r", lw=4, label=f"{target}, SEASON {season}")
    plt.axhline(depths.max(), c="k", ls=":", lw=2)
    plt.axhline(depths.min(), c="k", ls="--", lw=2)
    plt.axhline(-depths.min(), c="k", ls="--", lw=2)
    plt.axhline(-depths.max(), c="k", ls=":", lw=2)
    plt.xlim(t.min(), t.max())
    plt.ylim(np.percentile(white_noise, 1), np.percentile(white_noise, 99))
    plt.xlabel("Time [days]", fontsize=24)
    plt.ylabel("Flux", fontsize=24)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=20, loc="upper right", framealpha=1)
    # plt.savefig(os.path.join(FIGURE_DIR, TARGET + f'_synthetic_noise_season_{z}.png'), bbox_inches='tight')

    return fig
