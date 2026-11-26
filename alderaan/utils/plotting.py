"""
House style for data-model plot methods

Convention shared by every plot method: it returns the Figure; ``fig_path``
saves the figure there as a PNG and closes it (unless the caller passed ``ax``,
which keeps the caller's figure open); ``fig_path=None`` leaves it open.
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

DPI = 100

# corner name -> (x, y, ha, va) in axes coordinates
_CORNERS = {
    "upper left": (0.03, 0.95, "left", "top"),
    "upper right": (0.97, 0.95, "right", "top"),
    "lower left": (0.03, 0.05, "left", "bottom"),
    "lower right": (0.97, 0.05, "right", "bottom"),
}


def label_axes(ax, xlabel=None, ylabel=None, title=None, label_size=20, title_size=18):
    """Set axis labels (label_size), title (title_size), and tick labels (12)"""
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=label_size)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=label_size)
    if title is not None:
        ax.set_title(title, fontsize=title_size)
    ax.tick_params(labelsize=12)


def annotate_plot(ax, text, loc="upper right", edgecolor="none"):
    """Write text flush in an axes corner, fontsize 12 on a white background"""
    x, y, ha, va = _CORNERS[loc]
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=12,
        bbox={"facecolor": "w", "edgecolor": edgecolor, "pad": 2},
    )


def save_and_close(fig, fig_path, compress_level=6, close=True):
    """
    Write fig to fig_path as a 256-color PNG at DPI, then close it if close is True

    No-op if fig_path is None. Writes the Agg buffer's RGB channels with Pillow,
    skipping the opaque alpha channel savefig would encode (faster, smaller file).
    The 256-color palette halves the file size with no visible loss for plots.
    dpi is set before drawing because it fixes the buffer's pixel dimensions.
    """
    if fig_path is None:
        return
    fig.set_dpi(DPI)
    fig.patch.set_facecolor("w")
    fig.canvas.draw()
    rgb = np.asarray(fig.canvas.buffer_rgba())[..., :3]
    Image.fromarray(rgb, "RGB").quantize(256).save(
        fig_path, format="png", dpi=(DPI, DPI), compress_level=compress_level
    )
    if close:
        plt.close(fig)
    logger.debug("wrote %s", fig_path)
