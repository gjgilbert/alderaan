__all__ = ['plot_omc',
           'plot_litecurve',
           'plot_folded_transit',
           'dynesty_runplot',
           'dynesty_traceplot',
           'dynesty_cornerplot',
          ]

from astropy.stats import mad_std
from dynesty import plotting as dyplot
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np

from alderaan.planet import Planet
from alderaan.ephemeris import Ephemeris
from alderaan.modules.omc import OMC
from alderaan.utils.astro import bin_data


def plot_omc(data, target, filepath=None, interactive=False, time_label="Time [BJKD]"):
    """
    Plot observed-minus-calculated
    Input can be an alderaan Ephmeris or OMC object or a list of these
    """
    # check inputs
    if isinstance(data, OMC):
        omc_list = [data]
    elif isinstance(data, Ephemeris):
        omc_list = [OMC(data)]
    elif isinstance(data, list) & all([isinstance(d, OMC) for d in data]):
        omc_list = [d for d in data]
    elif isinstance(data, list) &  all([isinstance(d, Ephemeris) for d in data]):
        omc_list = [OMC(d) for d in data]
    else:
        raise ValueError("expected input: Ephemeris or OMC, or a list of these")
        
    npl = len(omc_list)

    # set colors
    if np.any([omc.out_prob is None for omc in omc_list]):
        marker_color = ['lightgrey']*npl
        line_color = [f'C{n}' for n in range(npl)]
        cmap = None
    else:
        marker_color = [1 - omc.out_prob for omc in omc_list]
        line_color = ['k' for n in range(npl)]
        cmap = 'viridis'
    
    # make plot
    fig, ax = plt.subplots(npl, figsize=(8, 3*npl))
    if npl == 1:
        ax = [ax]

    for n, omc in enumerate(omc_list):
        ax[n].scatter(omc.xtime, omc.yobs*24*60, c=marker_color[n], cmap=cmap, label='Observed TTVs')
        if omc.ymod is not None:
            ax[n].plot(omc.xtime, omc.ymod*24*60, lw=2, c=line_color[n], label='Regularized Model')
        if len(omc.quality) > 0:
            ax[n].plot(omc.xtime[~omc.quality], omc.yobs[~omc.quality]*24*60, 'rx')

        if omc.yerr is not None:
            err = np.nanmedian(omc.yerr) * 24 * 60
            ax[n].text(
                0.05,
                0.10,
                f"measured error = {err:.1f} min",
                transform=ax[n].transAxes,
                fontsize=12,
                ha='left',
                backgroundcolor='w',
            )
            
        if omc.ymod is not None:
            rms = mad_std(omc.yobs - omc.ymod, ignore_nan=True) * 24 * 60
            ax[n].text(
                0.95,
                0.10,
                f"residual RMS = {rms:.1f} min",
                transform=ax[n].transAxes,
                fontsize=12,
                ha='right',
                backgroundcolor='w',
            )

        ax[n].text(
            0.95,
            0.85,
            f"P = {omc._static_period:.1f}",
            transform=ax[n].transAxes,
            fontsize=14,
            ha='right',
            backgroundcolor='w'
        )

        ax[n].tick_params(labelsize=12)
        ax[n].set_ylabel("O-C [min]", fontsize=20)
        
        yrange = 24*60*np.max([np.abs(omc.yobs.min()),np.abs(omc.yobs.max())])
        yrange = np.max([5, 1.1*yrange])
        ax[n].set_ylim(-yrange, +yrange)

    ax[0].set_title(f"{target}", fontsize=20)
    ax[n].set_xlabel(time_label, fontsize=20)

    plt.tight_layout()

    if filepath is not None:
        plt.savefig(filepath)
    if not interactive:
        plt.close(fig)
    
    return fig


def plot_litecurve(litecurve, target, planets=None, filepath=None, interactive=False, time_label="Time [BJKD]"):
    # shorthand
    lc = litecurve

    if planets is not None:
        if isinstance(planets, Planet):
            planets = [planets]
    
    fig, ax = plt.subplots(1,1, figsize=(20,4))
    ax.plot(lc.time, lc.flux, 'k.', ms=0.5)
    ax.tick_params(labelsize=12)
    ax.set_xlabel(time_label, fontsize=24)
    ax.set_ylabel("Flux", fontsize=24)
    ax.set_xlim(lc.time.min(), lc.time.max())

    yrange = 1.3*np.max([
        np.abs(1-np.percentile(lc.flux, 0.1)),
        np.abs(1-np.percentile(lc.flux, 99.9))
    ])
    ax.set_ylim(1-yrange, 1+yrange)

    if planets is not None:
        ymin = 1 - yrange*(1.3 + 0.1*len(planets))
        ymax = 1 + yrange 
        ax.set_ylim(ymin, ymax)
        for n, p in enumerate(planets):
            ax.plot(
                p.ephemeris.ttime, 
                np.ones_like(p.ephemeris.ttime) - (0.9-0.1*n)*(1-ymin), 
                '^',
                c=f'C{n}'
            )

    plt.tight_layout()

    if filepath is not None:
        plt.savefig(filepath)
    if not interactive:
        plt.close(fig)
    
    return fig, ax


def plot_folded_transit(litecurve, planet, planet_no, target, filepath=None, interactive=False, max_pts=3000):
    """
    Plot phase-folded transit light curve for a single planet.

    Data within ±1.5 transit durations of each transit time are folded 
    to a common mid-transit at t=0, then binned and plotted.

    Args:
        litecurve (LiteCurve): detrended light curve
        planet (Planet): planet object with ephemeris and duration attributes
        planet_no (int): planet index (used for color and labeling)
        target (str): target name for title/label
        filepath (str, optional): if provided, save figure to this path
        interactive (bool): if False (default), close figure after saving
        max_pts (int): maximum number of individual points to plot (default 3000)

    Returns:
        tuple : (fig, ax) matplotlib figure and axes
    """
    time = litecurve.time
    flux = litecurve.flux
    tts = planet.ephemeris.ttime
    duration = planet.duration

    # fold photometry around each transit time
    t_folded = []
    f_folded = []

    for t0 in tts:
        use = np.abs(time - t0) / duration < 1.5

        if np.sum(use) > 0:
            t_folded.append(time[use] - t0)
            f_folded.append(flux[use])

    if len(t_folded) == 0:
        fig, ax = plt.subplots(1, 1, figsize=(8, 3))
        ax.text(0.5, 0.5, "No in-transit data", transform=ax.transAxes,
                ha='center', va='center', fontsize=14)
        ax.set_title(f"{target} - Planet {planet_no}", fontsize=16)
        if filepath is not None:
            plt.savefig(filepath, bbox_inches='tight')
        if not interactive:
            plt.close(fig)
        return fig, ax

    t_folded = np.hstack(t_folded)
    f_folded = np.hstack(f_folded)

    # sort by time
    order = np.argsort(t_folded)
    t_folded = t_folded[order]
    f_folded = f_folded[order]

    # bin the data
    t_binned, f_binned = bin_data(t_folded, f_folded, duration / 11)

    # subsample individual points for plotting
    inds = np.arange(len(t_folded), dtype=int)
    if len(inds) > max_pts:
        inds = np.random.choice(inds, size=max_pts, replace=False)

    # convert to hours and ppm
    t_hrs = t_folded[inds] * 24
    f_ppm = (f_folded[inds] - 1) * 1e6
    t_bin_hrs = t_binned * 24
    f_bin_ppm = (f_binned - 1) * 1e6

    # y-axis limits
    scatter = mad_std(f_folded)
    depth = planet.depth  # already in fractional units
    ymin = (-3 * scatter - depth) * 1e6
    ymax = (+3 * scatter) * 1e6

    # plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 3))
    ax.plot(t_hrs, f_ppm, '.', c='lightgrey', zorder=1)
    ax.plot(t_bin_hrs, f_bin_ppm, 'o', ms=8, color=f'C{planet_no}',
            label=f'{target} - {planet_no}', zorder=2)

    ax.set_xlim(t_folded.min() * 24, t_folded.max() * 24)
    ax.set_ylim(ymin, ymax)
    ax.tick_params(labelsize=12)
    ax.set_xlabel("Time from mid-transit [hrs]", fontsize=16)
    ax.set_ylabel("Flux (ppm)", fontsize=16)
    ax.legend(fontsize=12, loc='lower right', framealpha=1)

    # annotate with number of transits
    n_transits = len(planet.ephemeris.ttime)
    ax.text(0.05, 0.10, f"N transits = {n_transits}",
            transform=ax.transAxes, fontsize=12, ha='left', backgroundcolor='w')

    plt.tight_layout()

    if filepath is not None:
        plt.savefig(filepath, bbox_inches='tight')
    if not interactive:
        plt.close(fig)

    return fig, ax


def dynesty_runplot(results, target, filepath=None, interactive=False):
    fig, ax = dyplot.runplot(results, logplot=True, label_kwargs={'fontsize':16}, color='#0d0887')
    
    ax[0].set_title(target, fontsize=24)
    fig.tight_layout()
    
    if filepath is not None:
        plt.savefig(filepath, bbox_inches='tight', dpi=96)
    if not interactive:
        plt.close(fig)

    return fig, ax


def dynesty_traceplot(results, target, planet_no, filepath=None, interactive=False):
    fig, ax = dyplot.traceplot(
        results,
        labels=_parameter_labels(1, subscripts=False),
        dims=np.arange(5 * planet_no, 5 * (planet_no + 1)),
        label_kwargs={'fontsize':14},
    )
    
    fig.tight_layout()
    fig.suptitle(f"{target} - Planet {planet_no}", fontsize=18, y=fig.subplotpars.top + 0.02)
    
    if filepath is not None:
        plt.savefig(filepath, bbox_inches='tight', dpi=96)
    if not interactive:
        plt.close(fig)
        
    return fig, ax


def dynesty_cornerplot(results, target, planet_no, filepath=None, interactive=False):
    fig, ax = dyplot.cornerplot(
        results,
        labels=_parameter_labels(1, subscripts=False),
        dims=np.arange(5 * planet_no, 5 * (planet_no + 1)),
        label_kwargs={'fontsize':14},
        color=f'C{planet_no}',
    )
    
    fig.tight_layout()
    fig.suptitle(f"{target} - Planet {planet_no}", fontsize=18, y=fig.subplotpars.top + 0.02)
    
    if filepath is not None:
        plt.savefig(filepath, bbox_inches='tight', dpi=96)
    if not interactive:
        plt.close(fig)
        
    return fig, ax


def _parameter_labels(npl, subscripts=True):
    labels = []
    
    if subscripts:
        for n in range(npl):
            labels = labels + f'$C0_{n}$ $C1_{n}$ $r_{n}$ $b_{n}$ $T14_{n}$'.split()
    else:
        for n in range(npl):
            labels = labels + f'C0 C1 r b T14'.split()

    labels += 'q1 q2'.split()
    
    return labels


def plot_quick_fit_ttvs(target, planet_no, _t_obs, _t_mod, _f_obs, _f_mod, tc, tc_offset, tc_fit, x2_fit, chisq, transit_window_size, filepath=None, interactive=False, time_label="Time [BJKD]"):
    # recompute a fdew quantities
    quad_coeffs = np.polyfit(tc_fit, x2_fit, 2)
    quad_model = np.polyval(quad_coeffs, tc_fit)
    qtc_min = -quad_coeffs[1] / (2 * quad_coeffs[0])
    qtc_err = np.sqrt(1 / quad_coeffs[0])
    qx2_min = np.polyval(quad_coeffs, qtc_min)
    _ttj = np.nanmean([qtc_min, np.mean(tc_fit)])
    _errj = qtc_err * (1 + np.std(x2_fit - quad_model))

    # make some plots
    fig, ax = plt.subplots(1,2, figsize=(8,3))
    
    ax[0].plot(_t_obs, _f_obs, 'ko')
    ax[0].plot(_t_mod, _f_mod, c=f'C{planet_no}', lw=3)

    xticks = np.array([tc-transit_window_size/2, tc, tc+transit_window_size/2]).round(2)
    ax[0].set_xticks(xticks)
    ax[0].yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
    ax[0].set_xlabel(time_label, fontsize=14)
    ax[0].set_ylabel("Flux", fontsize=14)

    display = np.abs(chisq - qx2_min) < 2.5

    _x = tc_offset[display]
    _y_obs = (chisq-qx2_min)[display]
    _y_mod = np.polyval(quad_coeffs, _x) - qx2_min

    ax[1].plot(_x, _y_obs, 'o', mec='k', mfc='w')
    ax[1].plot(_x, _y_mod, c=f'C{planet_no}', lw=3)
    ax[1].axvline(qtc_min, color='k', ls=':')
    ax[1].axvline(tc_fit[np.argmin(x2_fit)], color='k', ls=':')
    ax[1].axvline(np.mean(tc_fit), color='k', ls=':')
    ax[1].axvline(_ttj, color='k', lw=2)

    xticks = np.array([_ttj - 1.5 * _errj, _ttj, _ttj + 1.5 * _errj])
    ax[1].set_xticks(xticks, np.round(xticks - _ttj, 4))
    ax[1].set_ylim(-0.5, 2.5)
    ax[1].set_xlabel("$\Delta t_c$", fontsize=14)
    ax[1].set_ylabel("$\Delta \chi^2$", fontsize=14)
    
    plt.suptitle(f"{target} - Planet {planet_no}", fontsize=18)
    plt.tight_layout()

    if filepath is not None:
        plt.savefig(filepath)
    if not interactive:
        plt.close(fig)
    
    return fig, ax
