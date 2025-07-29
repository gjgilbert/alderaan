__all__ = ['plot_omc']

from astropy.stats import mad_std
import matplotlib.pyplot as plt
import numpy as np
from src.schema.ephemeris import Ephemeris
from src.modules.omc import OMC


def plot_omc(data, target, filepath, ymod=None, fg_prob=None, outliers=None):
    """
    Plot observed-minus-calculated
    Input can be an alderaan Ephmeris or OMC object or a list of these
    """
    # check inputs
    if isinstance(data, OMC):
        omc_list = list(data)
    elif isinstance(data, Ephemeris):
        omc_list = list(OMC(data))
    elif isinstance(data, list):
        if all([isinstance(d, OMC) for d in data]):
            omc_list = [d for d in data]
        elif all([isinstance(d, OMC) for d in data]):
            omc_list = [OMC(d) for d in data]
    else:
        raise ValueError("expected input: Ephemeris or OMC, or a list of these")
    
    # set colors
    if fg_prob is None:
        marker_color = 1 - fg_prob
        line_color = [f'C{n}' for n in range(npl)]
        cmap = 'Greys'
    else:
        marker_color = 0.5*np.ones(len(omc.xtime))
        line_color = ['k' for n in range(npl)]
        cmap = 'viridis'
    
    # make plot
    npl = len(omc_list)

    fig, ax = plt.subplots(npl, figsize=(8, 2*npl))
    if npl == 1:
        ax = [ax]

    for n, omc in enumerate(omc_list):
        ax[n].scatter(omc.xtime, omc.yomc*24*60, c=marker_color, cmap=cmap, label='Observed TTVs')
        if ymod is not None:
            ax[n].plot(omc.xtime, ymod*24*60, lw=2, c=line_color[n], label='Regularized Model')
        if outliers is not None:
            ax[n].plot(omc.xtime[outliers], omc.yomc[outliers]*24*60, 'rx')

        if omc.yerr is not None:
            err = np.nanmedian(omc.yerr)
            ax[n].text(
                omc.xtime.min(),
                omc.yomc.min() * 24 * 60,
                f"measured error = {err:.1f} min",
                fontsize=16,
                ha='left',
                backgroundcolor='w',
            )
            
        if ymod is not None:
            rms = mad_std(omc.yomc - ymod, ignore_nan=True)
            ax[n].text(
                omc.xtime.min(),
                omc.yomc.min() * 24 * 60,
                f"residual RMS = {rms:.1f} min",
                fontsize=16,
                ha='left',
                backgroundcolor='w',
            )

        ax[n].set_xlabel("Time [BJKD]", fontsize=20)
        ax[n].set_ylabel("O-C [min]", fontsize=20)
        ax[n].set_xticks(fontsize=14)
        ax[n].set_yticks(fontsize=14)
        ax[n].legend(fontsize=14, loc="upper right")

    ax[0].set_title(f"{target}", fontsize=20)

    return fig, ax
