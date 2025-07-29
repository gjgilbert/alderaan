__all__ = ['plot_omc']

from astropy.stats import mad_std
import matplotlib.pyplot as plt
import numpy as np
from src.schema.ephemeris import Ephemeris
from src.modules.omc import OMC


def plot_omc(data, target, filepath):
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

    ax[0].set_title(f"{target}", fontsize=20)
    ax[n].set_xlabel("Time [BJKD]", fontsize=20)

    plt.tight_layout()
    plt.savefig(filepath)

    return fig, ax
