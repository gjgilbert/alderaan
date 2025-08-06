__all__ = ['plot_omc',
           'plot_litecurve',
           'dynesty_runplot',
           'dynesty_traceplot',
           'dynesty_cornerplot',
          ]

from astropy.stats import mad_std
from dynesty import plotting as dyplot
import matplotlib.pyplot as plt
import numpy as np
from alderaan.schema.planet import Planet
from alderaan.schema.ephemeris import Ephemeris
from alderaan.modules.omc import OMC


def plot_omc(data, target, filepath=None, interactive=False):
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
            rms = mad_std(omc.yobs - omc.ymod[n], ignore_nan=True) * 24 * 60
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
    ax[n].set_xlabel("Time [BJKD]", fontsize=20)

    plt.tight_layout()

    if filepath is not None:
        plt.savefig(filepath)
    if not interactive:
        plt.close(fig)
    
    return fig


def plot_litecurve(litecurve, target, planets=None, filepath=None, interactive=False):
    # shorthand
    lc = litecurve

    if planets is not None:
        if isinstance(planets, Planet):
            planets = [planets]
    
    fig, ax = plt.subplots(1,1, figsize=(20,4))
    ax.plot(lc.time, lc.flux, 'k.', ms=0.5)
    ax.tick_params(labelsize=12)
    ax.set_xlabel("Time [BJKD]", fontsize=24)
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