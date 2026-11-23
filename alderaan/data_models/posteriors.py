"""Posteriors: transit-model posterior samples and the sampling ephemeris"""

__all__ = ["Posteriors"]

import dynesty
import numpy as np
import pandas as pd
from astropy.io import fits
from dynesty import plotting as dyplot

from alderaan.transit_model import TRANSIT_PARAMETER_LABELS
from alderaan.utils.astro import ld_q_to_u
from alderaan.utils.plotting import save_and_close
from alderaan.utils.stats import legendre_p1, weighted_percentile

# dynesty run outputs kept by Posteriors: per-sample arrays written as FITS
# SAMPLES columns (with their formats); per-sample integrals rebuilt from logl
# and logvol on load (Posteriors._integrals); batch_nlive and batch_logl_bounds
# go in the BATCHES table; niter and eff in the SAMPLES header
SAMPLE_COLUMNS = {
    "logl": ("LN_LIKE", np.float64),
    "logvol": ("LN_VOL", np.float64),
    "samples_n": ("N_LIVE", np.int32),
    "ncall": ("N_CALL", np.int32),
    "samples_id": ("SAMP_ID", np.int32),
    "samples_batch": ("SAMP_BATCH", np.int16),
}
INTEGRAL_KEYS = ("logwt", "logz", "logzerr", "information")
SAMPLER_KEYS = (
    *SAMPLE_COLUMNS,
    *INTEGRAL_KEYS,
    "niter",
    "eff",
    "batch_nlive",
    "batch_logl_bounds",
)


class Posteriors:
    """
    Nested sampling posteriors for one target (thin wrapper around dynesty results)

    Parameters
    ----------
    mission, target : str
    samples : pd.DataFrame
        nested samples of TransitModel theta; columns npl * [C0_n, C1_n, ROR_n,
        IMPACT_n, DUR14_n] + [LD_Q1, LD_Q2]
    sampler_info : dict
        dynesty run outputs, keyed by their dynesty.results.Results names:
        logwt, logl, logz, logzerr, logvol, samples_n, ncall, information :
            ndarray, one entry per sample (log weight, log likelihood, cumulative
            log evidence and its error, log prior volume, number of live points,
            likelihood calls, cumulative information in nats)
        samples_id, samples_batch : ndarray, one entry per sample (live point
            strand and batch; used by dynesty.utils.resample_run)
        niter, eff : number of iterations, sampling efficiency (%)
        batch_nlive : list of live points per batch
        batch_logl_bounds : ndarray, shape (nbatch, 2); logl bounds per batch
    ref_period, ref_epoch : ndarray
        per planet; least squares linear ephemeris of the transit times the sampler
        perturbed with (C0, C1)
    t_start, t_end : float
        photometry span of the Legendre mapping used by the sampler,
        legx = legendre_p1(ref_epoch + ref_period * index, (t_start, t_end))
        (see TransitModel._build_warps)
    supersample : dict
        exposure supersample factor per obsmode used by the sampler
        (TransitModel.supersample; 1 = not integrated)
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
        supersample,
    ):
        self.mission = mission
        self.target = target
        self.samples = samples
        self.sampler_info = sampler_info
        self.ref_period = np.asarray(ref_period, dtype=float)
        self.ref_epoch = np.asarray(ref_epoch, dtype=float)
        self.t_start = float(t_start)
        self.t_end = float(t_end)
        self.supersample = {str(om): int(ss) for om, ss in supersample.items()}
        self.meta = {}  # FITS header cards, written to SAMPLES; not read back

        if len(self.ref_epoch) != self.npl:
            raise ValueError("ref_epoch must have one entry per planet")

        if samples.shape[1] != 5 * self.npl + 2:
            raise ValueError(
                f"{samples.shape[1]} sample columns inconsistent with npl = {self.npl}"
            )

        if set(sampler_info) != set(SAMPLER_KEYS):
            raise ValueError(f"sampler_info keys must be {SAMPLER_KEYS}")
        for k in (*SAMPLE_COLUMNS, *INTEGRAL_KEYS):
            if len(sampler_info[k]) != len(samples):
                raise ValueError(f"sampler_info[{k!r}] must have one entry per sample")

    @property
    def npl(self):
        """Number of planets"""
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
            tm.supersample,
        )

    def _to_hdus(self):
        """SAMPLES, EPHEMERIS and BATCHES BinTableHDUs (used by Results.to_fits)"""
        info = self.sampler_info
        # float32 parameters err by < 0.005 posterior sd; LN_LIKE and LN_VOL keep
        # float64 (LN_LIKE can reach ~1e12)
        samples_df = self.samples.astype(np.float32).assign(
            **{
                col: np.asarray(info[k], fmt)
                for k, (col, fmt) in SAMPLE_COLUMNS.items()
            }
        )
        samples_hdu = fits.BinTableHDU(
            data=samples_df.to_records(index=False), name="SAMPLES"
        )
        samples_hdu.header["MISSION"] = self.mission
        samples_hdu.header["TARGET"] = self.target
        samples_hdu.header["NITER"] = info["niter"]
        samples_hdu.header["EFF"] = info["eff"]
        # convergence summary (derived from the columns; not read back)
        samples_hdu.header["NCALL"] = int(np.sum(info["ncall"]))
        samples_hdu.header["NEFF"] = self.neff()
        samples_hdu.header["LN_Z"] = info["logz"][-1]
        samples_hdu.header["LN_ZERR"] = info["logzerr"][-1]
        samples_hdu.header["INFO"] = info["information"][-1]
        samples_hdu.header.update(self.meta)

        # a table, not header cards: the baseline batch bounds are -inf, inf
        bounds = np.asarray(info["batch_logl_bounds"], dtype=float)
        batches_hdu = fits.BinTableHDU.from_columns(
            [
                fits.Column(name="NLIVE", format="J", array=info["batch_nlive"]),
                fits.Column(name="LOGL_MIN", format="D", array=bounds[:, 0]),
                fits.Column(name="LOGL_MAX", format="D", array=bounds[:, 1]),
            ],
            name="BATCHES",
        )

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
        ephem_hdu.header["NOBSMODE"] = len(self.supersample)
        for i, (om, ss) in enumerate(self.supersample.items()):
            ephem_hdu.header[f"OBSMODE{i}"] = om
            ephem_hdu.header[f"SUPSAMP{i}"] = ss

        return samples_hdu, ephem_hdu, batches_hdu

    @classmethod
    def _from_hdus(cls, hdul):
        """Build from the SAMPLES, EPHEMERIS, BATCHES tables of a Results file"""
        samples_hdu, ephem_hdu = hdul["SAMPLES"], hdul["EPHEMERIS"]
        header = samples_hdu.header
        samples = pd.DataFrame(
            {
                k: np.asarray(samples_hdu.data[k], dtype=float)
                for k in samples_hdu.columns.names
            }
        )
        # TEMPORARY: the 20261123 batch stores the integrals, no strands and no
        # BATCHES table; strand -1 and NaN bounds make resample_run raise.
        # Remove after the next batch.
        if "BATCHES" not in hdul:
            samples = samples.drop(columns=["LN_WT", "LN_Z", "LN_ZERR", "INFO"])
            samples["SAMP_ID"] = -1
            samples["SAMP_BATCH"] = -1
            nbatch = header["NBATCH"]
            batch_nlive = [header[f"NLIVE{i}"] for i in range(nbatch)]
            batch_logl_bounds = np.full((nbatch, 2), np.nan)
        else:
            batches = hdul["BATCHES"].data
            batch_nlive = [int(n) for n in batches["NLIVE"]]
            batch_logl_bounds = np.column_stack(
                [batches["LOGL_MIN"], batches["LOGL_MAX"]]
            ).astype(float)

        sampler_info = {
            k: samples.pop(col).to_numpy() for k, (col, _) in SAMPLE_COLUMNS.items()
        }
        for k in ("samples_n", "ncall", "samples_id", "samples_batch"):
            sampler_info[k] = sampler_info[k].astype(int)
        sampler_info |= cls._integrals(sampler_info["logl"], sampler_info["logvol"])
        sampler_info["niter"] = header["NITER"]
        sampler_info["eff"] = header["EFF"]
        sampler_info["batch_nlive"] = batch_nlive
        sampler_info["batch_logl_bounds"] = batch_logl_bounds

        ephem = ephem_hdu.data
        eh = ephem_hdu.header
        supersample = {
            eh[f"OBSMODE{i}"]: eh[f"SUPSAMP{i}"] for i in range(eh["NOBSMODE"])
        }

        return cls(
            header["MISSION"],
            header["TARGET"],
            samples,
            sampler_info,
            ephem["REF_PERIOD"],
            ephem["REF_EPOCH"],
            eh["T_START"],
            eh["T_END"],
            supersample,
        )

    @staticmethod
    def _integrals(logl, logvol):
        """Per-sample logwt, logz, logzerr, information, as dynesty computes them"""
        # start volume 0 (dynesty's logvol_init): ln_likelihood is never -inf
        logwt, logz, logzvar, information = dynesty.utils.compute_integrals(
            logl=logl, logvol=logvol
        )
        return {
            "logwt": logwt,
            "logz": logz,
            "logzerr": np.sqrt(np.maximum(logzvar, 0)),
            "information": information,
        }

    def neff(self):
        """Number of effective samples (Kish ESS)"""
        return dynesty.utils.get_neff_from_logwt(self.sampler_info["logwt"])

    def weights(self):
        """Normalized posterior weights"""
        wt = np.exp(self.sampler_info["logwt"] - self.sampler_info["logz"][-1])
        return wt / np.sum(wt)

    def summary(self):
        """
        Weighted posterior summary statistics for each sampled parameter

        Returns
        -------
        pd.DataFrame
            indexed by sample column; columns [mean, stdev, median, CI_2.5, CI_16,
            CI_84, CI_97.5] (CI_16 and CI_84 are the 15.9 and 84.1 percentiles)
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

        Sampled transit times are ttime_mod + C0 + C1 * legx, with legx linear in
        transit index (see class docstring); adding that line to the reference
        ephemeris gives the sampled period and epoch.

        Returns
        -------
        pd.DataFrame
            one row per sample; columns [PERIOD, EPOCH, ROR, IMPACT, DUR14, LD_U1,
            LD_U2, LN_WT]: the transit basis (see alderaan.transit_model) and
            the log weight
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
        dynesty Results for dynesty.plotting and dynesty.utils (e.g. resample_run)

        samples_u, samples_it and blob are not stored: samples_u holds the
        samples (the constructor requires it) and the others zeros (resample_run
        only copies them).
        """
        samples = self.samples.to_numpy()
        zeros = np.zeros(len(samples), dtype=int)
        return dynesty.utils.Results(
            {
                "samples": samples,
                "samples_u": samples,
                "samples_it": zeros,
                "blob": zeros,
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
        fig.set_size_inches(7.2, 9)  # same as the traceplot; smaller PNG
        fig.tight_layout()
        save_and_close(fig, fig_path)

        return fig

    def dynesty_traceplot(self, n, fig_path=None):
        """dynesty trace plot of planet n parameters"""
        fig, _ = dyplot.traceplot(
            self._dynesty_results(),
            labels=TRANSIT_PARAMETER_LABELS,
            dims=np.arange(5 * n, 5 * (n + 1)),
            label_kwargs={"fontsize": 14},
        )
        fig.set_size_inches(7.2, 9)  # 0.6 x the dynesty default; smaller PNG
        self._finish_planet_plot(fig, n, fig_path)

        return fig

    def dynesty_cornerplot(self, n, fig_path=None):
        """dynesty corner plot of planet n parameters"""
        fig, _ = dyplot.cornerplot(
            self._dynesty_results(),
            labels=TRANSIT_PARAMETER_LABELS,
            dims=np.arange(5 * n, 5 * (n + 1)),
            label_kwargs={"fontsize": 14},
            color=f"C{n}",
        )
        self._finish_planet_plot(fig, n, fig_path)

        return fig

    def _finish_planet_plot(self, fig, n, fig_path):
        """Title, lay out and save a planet n figure"""
        # leave the top 4% of the figure for the title
        fig.suptitle(f"{self.target} - Planet {n}", fontsize=18, y=0.99, va="top")
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        save_and_close(fig, fig_path)
