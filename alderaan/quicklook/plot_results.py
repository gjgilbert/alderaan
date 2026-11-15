__all__ = ["PlotResults"]

import numpy as np
from dynesty import plotting as dyplot
from dynesty.utils import Results as DynestyResults

from alderaan.quicklook.base import finish

PARAMETER_LABELS = "C0 C1 r b T14 q1 q2".split()


class PlotResults:
    """
    Quicklook plots of nested sampling results

    Parameters
    ----------
    results : Results
        results.posteriors must include ln_vol, ln_zerr, n_live
    """

    def __init__(self, results):
        post = results.posteriors
        if any(x is None for x in (post.ln_vol, post.ln_zerr, post.n_live)):
            raise ValueError("Posteriors is missing ln_vol, ln_zerr, or n_live")

        self.results = results

        # dynesty plotting functions take a dynesty Results; samples_u and
        # samples_id are required by its constructor but unused by these plots
        samples = post.samples.to_numpy()
        self._dynesty_results = DynestyResults(
            {
                "samples": samples,
                "samples_u": samples,
                "samples_id": np.zeros(len(samples), dtype=int),
                "samples_n": post.n_live,
                "logl": post.ln_like,
                "logwt": post.ln_wt,
                "logz": post.ln_z,
                "logzerr": post.ln_zerr,
                "logvol": post.ln_vol,
                "niter": post.sampler_info["NITER"],
                "eff": post.sampler_info["EFF"],
            }
        )

    def plot_runplot(self, fig_path=None, show_plot=False):
        fig, ax = dyplot.runplot(
            self._dynesty_results,
            logplot=True,
            label_kwargs={"fontsize": 16},
            color="#0d0887",
        )

        ax[0].set_title(self.results.target, fontsize=24)
        fig.tight_layout()

        finish(fig, fig_path, show_plot, bbox_inches="tight", dpi=96)

    def plot_traceplot(self, n, fig_path=None, show_plot=False):
        fig, _ = dyplot.traceplot(
            self._dynesty_results,
            labels=PARAMETER_LABELS,
            dims=np.arange(5 * n, 5 * (n + 1)),
            label_kwargs={"fontsize": 14},
        )

        fig.tight_layout()
        fig.suptitle(
            f"{self.results.target} - Planet {n}",
            fontsize=18,
            y=fig.subplotpars.top + 0.02,
        )

        finish(fig, fig_path, show_plot, bbox_inches="tight", dpi=96)

    def plot_cornerplot(self, n, fig_path=None, show_plot=False):
        fig, _ = dyplot.cornerplot(
            self._dynesty_results,
            labels=PARAMETER_LABELS,
            dims=np.arange(5 * n, 5 * (n + 1)),
            label_kwargs={"fontsize": 14},
            color=f"C{n}",
        )

        fig.tight_layout()
        fig.suptitle(
            f"{self.results.target} - Planet {n}",
            fontsize=18,
            y=fig.subplotpars.top + 0.02,
        )

        finish(fig, fig_path, show_plot, bbox_inches="tight", dpi=96)
