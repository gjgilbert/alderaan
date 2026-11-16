__all__ = ["Posteriors"]

import dynesty
import numpy as np
import pandas as pd
from astropy.io import fits
from dynesty import plotting as dyplot

from alderaan.transit_model import PARAMETER_LABELS
from alderaan.utils.astro import ld_q_to_u
from alderaan.utils.plotting import save_and_close
from alderaan.utils.stats import legendre_p1, weighted_percentile

# dynesty run outputs kept by Posteriors: per-sample arrays, mapped to their FITS
# SAMPLES columns (names as in legacy); niter, eff, batch_nlive go in the header
SAMPLE_COLUMNS = {
    "logwt": "LN_WT",
    "logl": "LN_LIKE",
    "logz": "LN_Z",
    "logzerr": "LN_ZERR",
    "logvol": "LN_VOL",
    "samples_n": "N_LIVE",
}
SAMPLER_KEYS = (*SAMPLE_COLUMNS, "niter", "eff", "batch_nlive")


class Posteriors:
    """
    Nested sampling posteriors for one target (thin wrapper around dynesty results)

    Parameters
    ----------
    mission, target : str
    samples : pd.DataFrame
        nested samples; columns npl * [C0_n, C1_n, ROR_n, IMPACT_n, DUR14_n]
        + [LD_Q1, LD_Q2]
    sampler_info : dict
        dynesty run outputs, keyed by their dynesty.results.Results names:
        logwt, logl, logz, logzerr, logvol, samples_n : ndarray, one entry per
            sample (log weight, log likelihood, cumulative log evidence and its
            error, log prior volume, number of live points)
        niter, eff : number of iterations, sampling efficiency (%)
        batch_nlive : list of live points per batch
    ref_period, ref_epoch : ndarray
        per planet; least squares linear ephemeris of the transit times the sampler
        perturbed with (C0, C1)
    t_start, t_end : float
        photometry span of the Legendre mapping used by the sampler,
        legx = legendre_p1(ref_epoch + ref_period * index, (t_start, t_end))
        (see TransitModel._build_warps)
    """

    def __init__(
        self,
        mission,
        target,
        samples,
        sampler_info,
        ref_period,
        ref_epoch,
        t_start,
        t_end,
    ):
        self.mission = mission
        self.target = target
        self.samples = samples
        self.sampler_info = sampler_info
        self.ref_period = np.asarray(ref_period, dtype=float)
        self.ref_epoch = np.asarray(ref_epoch, dtype=float)
        self.t_start = float(t_start)
        self.t_end = float(t_end)

        if len(self.ref_epoch) != self.npl:
            raise ValueError("ref_epoch must have one entry per planet")

        if samples.shape[1] != 5 * self.npl + 2:
            raise ValueError(
                f"{samples.shape[1]} sample columns inconsistent with npl = {self.npl}"
            )

        if set(sampler_info) != set(SAMPLER_KEYS):
            raise ValueError(f"sampler_info keys must be {SAMPLER_KEYS}")
        for k in SAMPLE_COLUMNS:
            if len(sampler_info[k]) != len(samples):
                raise ValueError(f"sampler_info[{k!r}] must have one entry per sample")

    @property
    def npl(self):
        return len(self.ref_period)

    @classmethod
    def from_dynesty(cls, mission, target, results, tm):
        """
        Package dynesty results with the sampling ephemeris of each planet

        Parameters
        ----------
        mission, target : str
        results : dynesty.results.Results
        tm : TransitModel
            transit model that was sampled
        """
        npl = (results.samples.shape[1] - 2) // 5

        if tm.npl != npl:
            raise ValueError(f"{tm.npl} planets for {npl} sampled planets")

        labels = []
        for n in range(npl):
            labels += f"C0_{n} C1_{n} ROR_{n} IMPACT_{n} DUR14_{n}".split()
        labels += ["LD_Q1", "LD_Q2"]

        samples = pd.DataFrame(results.samples, columns=labels)
        sampler_info = {k: getattr(results, k) for k in SAMPLER_KEYS}

        return cls(
            mission,
            target,
            samples,
            sampler_info,
            tm.ref_period,
            tm.ref_epoch,
            tm.t_start,
            tm.t_end,
        )

    def _to_hdus(self):
        """
        SAMPLES and EPHEMERIS BinTableHDUs (used by Results.to_fits)
        """
        info = self.sampler_info
        samples_df = self.samples.assign(
            **{col: info[k] for k, col in SAMPLE_COLUMNS.items()}
        )
        samples_hdu = fits.BinTableHDU(
            data=samples_df.to_records(index=False), name="SAMPLES"
        )
        samples_hdu.header["MISSION"] = self.mission
        samples_hdu.header["TARGET"] = self.target
        samples_hdu.header["NITER"] = info["niter"]
        samples_hdu.header["EFF"] = info["eff"]
        samples_hdu.header["NBATCH"] = len(info["batch_nlive"])
        for i, nlive in enumerate(info["batch_nlive"]):
            samples_hdu.header[f"NLIVE{i}"] = nlive

        ephem_hdu = fits.BinTableHDU.from_columns(
            [
                fits.Column(name="PLANET", format="K", array=np.arange(self.npl)),
                fits.Column(name="REF_PERIOD", format="D", array=self.ref_period),
                fits.Column(name="REF_EPOCH", format="D", array=self.ref_epoch),
            ],
            name="EPHEMERIS",
        )
        ephem_hdu.header["MISSION"] = self.mission
        ephem_hdu.header["TARGET"] = self.target
        ephem_hdu.header["T_START"] = self.t_start
        ephem_hdu.header["T_END"] = self.t_end

        return samples_hdu, ephem_hdu

    @classmethod
    def _from_hdus(cls, samples_hdu, ephem_hdu):
        """Build from SAMPLES and EPHEMERIS tables (used by Results.from_fits)"""
        header = samples_hdu.header
        samples = pd.DataFrame(
            {
                k: np.asarray(samples_hdu.data[k], dtype=float)
                for k in samples_hdu.columns.names
            }
        )

        sampler_info = {
            k: samples.pop(col).to_numpy() for k, col in SAMPLE_COLUMNS.items()
        }
        sampler_info["samples_n"] = sampler_info["samples_n"].astype(int)
        sampler_info["niter"] = header["NITER"]
        sampler_info["eff"] = header["EFF"]
        sampler_info["batch_nlive"] = [
            header[f"NLIVE{i}"] for i in range(header["NBATCH"])
        ]

        ephem = ephem_hdu.data

        return cls(
            header["MISSION"],
            header["TARGET"],
            samples,
            sampler_info,
            ephem["REF_PERIOD"],
            ephem["REF_EPOCH"],
            ephem_hdu.header["T_START"],
            ephem_hdu.header["T_END"],
        )

    @property
    def nsamp(self):
        return len(self.samples)

    def neff(self):
        """
        Number of effective samples (Kish ESS)
        """
        return dynesty.utils.get_neff_from_logwt(self.sampler_info["logwt"])

    def weights(self):
        """
        Normalized posterior weights
        """
        wt = np.exp(self.sampler_info["logwt"] - self.sampler_info["logz"][-1])
        return wt / np.sum(wt)

    def summary(self):
        """
        Weighted posterior summary statistics for each sampled parameter
        """
        weights = self.weights()

        rows = {}
        for label in self.samples.columns:
            x = self.samples[label].to_numpy()
            avg = np.sum(weights * x)
            rows[label] = {
                "mean": avg,
                "stdev": np.sqrt(np.sum(weights * (x - avg) ** 2)),
                "median": weighted_percentile(x, 50, w=weights),
                "CI_2.5": weighted_percentile(x, 2.5, w=weights),
                "CI_16": weighted_percentile(x, 15.9, w=weights),
                "CI_84": weighted_percentile(x, 84.1, w=weights),
                "CI_97.5": weighted_percentile(x, 97.5, w=weights),
            }

        return pd.DataFrame.from_dict(rows, orient="index")

    def planet_samples(self, n):
        """
        Physical posterior samples for planet n

        The sampled transit times are ttime + C0 + C1 * legx with
        legx = legendre_p1(ref_epoch + ref_period * index, (t_start, t_end)), which is
        linear in index; adding that line to the reference ephemeris gives the
        sampled period and epoch

        Returns
        -------
        pd.DataFrame
            columns [PERIOD, EPOCH, ROR, IMPACT, DUR14, LD_U1, LD_U2, LN_WT]
        """
        s = self.samples
        C0 = s[f"C0_{n}"].to_numpy()
        C1 = s[f"C1_{n}"].to_numpy()

        # legx at index 0, and its change per transit
        span = (self.t_start, self.t_end)
        x0 = legendre_p1(self.ref_epoch[n], span)
        dx = legendre_p1(self.ref_epoch[n] + self.ref_period[n], span) - x0

        period = self.ref_period[n] + C1 * dx
        epoch = self.ref_epoch[n] + C0 + C1 * x0
        u1, u2 = ld_q_to_u(s["LD_Q1"].to_numpy(), s["LD_Q2"].to_numpy())

        return pd.DataFrame(
            {
                "PERIOD": period,
                "EPOCH": epoch,
                "ROR": s[f"ROR_{n}"].to_numpy(),
                "IMPACT": s[f"IMPACT_{n}"].to_numpy(),
                "DUR14": s[f"DUR14_{n}"].to_numpy(),
                "LD_U1": u1,
                "LD_U2": u2,
                "LN_WT": self.sampler_info["logwt"],
            }
        )

    def _dynesty_results(self):
        """
        dynesty Results for dynesty.plotting; samples_u and samples_id are required
        by its constructor but unused by these plots
        """
        samples = self.samples.to_numpy()
        return dynesty.utils.Results(
            {
                "samples": samples,
                "samples_u": samples,
                "samples_id": np.zeros(len(samples), dtype=int),
                **self.sampler_info,
            }
        )

    # dynesty plots below return the Figure; fig_path saves it there and closes it

    def dynesty_runplot(self, fig_path=None):
        """dynesty run plot"""
        fig, ax = dyplot.runplot(
            self._dynesty_results(),
            logplot=True,
            label_kwargs={"fontsize": 16},
            color="#0d0887",
        )
        ax[0].set_title(self.target, fontsize=18)
        fig.tight_layout()
        save_and_close(fig, fig_path)

        return fig

    def dynesty_traceplot(self, n, fig_path=None):
        """dynesty trace plot of planet n parameters"""
        fig, _ = dyplot.traceplot(
            self._dynesty_results(),
            labels=PARAMETER_LABELS,
            dims=np.arange(5 * n, 5 * (n + 1)),
            label_kwargs={"fontsize": 14},
        )
        self._finish_planet_plot(fig, n, fig_path)

        return fig

    def dynesty_cornerplot(self, n, fig_path=None):
        """dynesty corner plot of planet n parameters"""
        fig, _ = dyplot.cornerplot(
            self._dynesty_results(),
            labels=PARAMETER_LABELS,
            dims=np.arange(5 * n, 5 * (n + 1)),
            label_kwargs={"fontsize": 14},
            color=f"C{n}",
        )
        self._finish_planet_plot(fig, n, fig_path)

        return fig

    def _finish_planet_plot(self, fig, n, fig_path):
        # leave the top 4% of the figure for the title
        fig.suptitle(f"{self.target} - Planet {n}", fontsize=18, y=0.99, va="top")
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        save_and_close(fig, fig_path)
