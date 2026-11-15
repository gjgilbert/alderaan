"""House style for data-model plot methods"""

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


def label_axes(ax, xlabel=None, ylabel=None, title=None):
    """Set axis labels (fontsize 20), title (18), and tick labels (12)"""
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=20)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=20)
    if title is not None:
        ax.set_title(title, fontsize=18)
    ax.tick_params(labelsize=12)


def corner_text(ax, text, loc="upper right"):
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
        bbox={"facecolor": "w", "edgecolor": "none", "pad": 2},
    )


def save_and_close(fig, fig_path, compress_level=6):
    """
    Write fig to fig_path as an RGB PNG at DPI and close it; no-op if fig_path is
    None (figure stays open for interactive use)

    Writes the Agg buffer's RGB channels directly with Pillow, skipping the opaque
    alpha channel that Figure.savefig would encode (faster, smaller file); dpi is
    set before drawing because it fixes the buffer's pixel dimensions
    """
    if fig_path is None:
        return
    fig.set_dpi(DPI)
    fig.patch.set_facecolor("w")
    fig.canvas.draw()
    rgb = np.asarray(fig.canvas.buffer_rgba())[..., :3]
    Image.fromarray(rgb, "RGB").save(
        fig_path, format="png", dpi=(DPI, DPI), compress_level=compress_level
    )
    plt.close(fig)
    logger.debug("wrote %s", fig_path)
