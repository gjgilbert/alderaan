import logging

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def finish(fig, fig_path=None, show_plot=False, **savefig_kwargs):
    """
    Save the figure if fig_path is given, show it if show_plot, then close it
    """
    if fig_path is not None:
        fig.savefig(fig_path, **savefig_kwargs)
        logger.debug("wrote %s", fig_path)
    if show_plot:
        plt.show()
    plt.close(fig)
