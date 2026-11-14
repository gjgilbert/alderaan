__all__ = ["Results"]

import logging

import dynesty
import numpy as np
import numpy.polynomial.polynomial as poly
import pandas as pd
from astropy.io import fits

from alderaan.utils.stats import weighted_percentile

logger = logging.getLogger(__name__)

TTIMES_COLUMNS = ["index", "ttime", "model", "out_prob", "out_flag"]


class Results:
    """
    Packaged pipeline outputs for one target

    Parameters
    ----------
    mission, target, run_id : str
    samples : pd.DataFrame
        nested samples; columns num_planets * [C0_n, C1_n, ROR_n, IMPACT_n, DUR14_n]
        + [LD_Q1, LD_Q2]
    ln_wt, ln_like, ln_z : ndarray
        log weight, log likelihood, and cumulative log evidence per sample
    ln_vol, ln_zerr, n_live : ndarray, optional
        log prior volume, log evidence error, and number of live points per sample;
        used for sampler diagnostic plots (absent in older results files)
    transit_times : list of pd.DataFrame
        one per planet; columns [index, ttime, model, out_prob, out_flag]
    sampler_info : dict, optional
        sampler diagnostics written to the SAMPLES header (NITER, NBATCH, NLIVEi, EFF)
    """

    def __init__(
        self,
        mission,
        target,
        run_id,
        samples,
        ln_wt,
        ln_like,
        ln_z,
        transit_times,
        sampler_info=None,
        ln_vol=None,
        ln_zerr=None,
        n_live=None,
    ):
        self.mission = mission
        self.target = target
        self.run_id = run_id
        self.samples = samples
        self.ln_wt = np.asarray(ln_wt)
        self.ln_like = np.asarray(ln_like)
        self.ln_z = np.asarray(ln_z)
        self.transit_times = transit_times
        self.sampler_info = {} if sampler_info is None else sampler_info
        self.ln_vol = None if ln_vol is None else np.asarray(ln_vol)
        self.ln_zerr = None if ln_zerr is None else np.asarray(ln_zerr)
        self.n_live = None if n_live is None else np.asarray(n_live)

        if samples.shape[1] != 5 * self.npl + 2:
            raise ValueError(
                f"{samples.shape[1]} sample columns inconsistent with npl = {self.npl}"
            )

        nsamp = len(samples)
        if not (len(self.ln_wt) == len(self.ln_like) == len(self.ln_z) == nsamp):
            raise ValueError("ln_wt, ln_like, ln_z must have one entry per sample")

        for name in ("ln_vol", "ln_zerr", "n_live"):
            x = getattr(self, name)
            if x is not None and len(x) != nsamp:
                raise ValueError(f"{name} must have one entry per sample")

    @property
    def npl(self):
        return len(self.transit_times)

    @classmethod
    def from_dynesty(cls, results, mission, target, run_id, transit_times):
        """
        Package dynesty results with per-planet transit time tables

        Parameters
        ----------
        results : dynesty.results.Results
        mission, target, run_id : str
        transit_times : list of pd.DataFrame
            one per planet, in planet order; columns as in TTIMES_COLUMNS
        """
        npl = (results.samples.shape[1] - 2) // 5

        if len(transit_times) != npl:
            raise ValueError(
                f"{len(transit_times)} transit time tables for {npl} planets"
            )

        labels = []
        for n in range(npl):
            labels += f"C0_{n} C1_{n} ROR_{n} IMPACT_{n} DUR14_{n}".split()
        labels += ["LD_Q1", "LD_Q2"]

        samples = pd.DataFrame(results.samples, columns=labels)

        sampler_info = {"NITER": results.niter, "NBATCH": len(results.batch_nlive)}
        for i, nlive in enumerate(results.batch_nlive):
            sampler_info[f"NLIVE{i}"] = nlive
        sampler_info["EFF"] = results.eff

        return cls(
            mission,
            target,
            run_id,
            samples,
            results.logwt,
            results.logl,
            results.logz,
            transit_times,
            sampler_info=sampler_info,
            ln_vol=results.logvol,
            ln_zerr=results.logzerr,
            n_live=results.samples_n,
        )

    @classmethod
    def from_fits(cls, filepath):
        """
        Load a results file written by Results.to_fits

        Parameters
        ----------
        filepath : str
            path to {target}-results.fits
        """
        logger.info("reading %s", filepath)
        with fits.open(filepath) as hdul:
            header = hdul[0].header
            npl = header["NPL"]

            samples_hdu = hdul["SAMPLES"]
            samples = pd.DataFrame(
                {
                    k: np.asarray(samples_hdu.data[k], dtype=float)
                    for k in samples_hdu.columns.names
                }
            )
            ln_wt = samples.pop("LN_WT").to_numpy()
            ln_like = samples.pop("LN_LIKE").to_numpy()
            ln_z = samples.pop("LN_Z").to_numpy()

            # sampler diagnostics; absent in older results files
            names = samples_hdu.columns.names
            ln_vol = samples.pop("LN_VOL").to_numpy() if "LN_VOL" in names else None
            ln_zerr = samples.pop("LN_ZERR").to_numpy() if "LN_ZERR" in names else None
            n_live = (
                samples.pop("N_LIVE").to_numpy(dtype=int) if "N_LIVE" in names else None
            )

            sampler_info = {
                k: samples_hdu.header[k]
                for k in samples_hdu.header
                if k in ("NITER", "NBATCH", "EFF") or k.startswith("NLIVE")
            }

            transit_times = []
            for n in range(npl):
                data = hdul[f"TTIMES_{n:02d}"].data
                transit_times.append(
                    _ttimes_dataframe(
                        np.column_stack([data[k.upper()] for k in TTIMES_COLUMNS])
                    )
                )

            mission, target, run_id = (
                header["MISSION"],
                header["TARGET"],
                header["RUN_ID"],
            )

        return cls(
            mission,
            target,
            run_id,
            samples,
            ln_wt,
            ln_like,
            ln_z,
            transit_times,
            sampler_info=sampler_info,
            ln_vol=ln_vol,
            ln_zerr=ln_zerr,
            n_live=n_live,
        )

    def to_hdulist(self):
        """
        Build the standard results HDUList: PRIMARY, SAMPLES, TTIMES_nn
        """
        primary_hdu = fits.PrimaryHDU()
        primary_hdu.header["MISSION"] = self.mission
        primary_hdu.header["TARGET"] = self.target
        primary_hdu.header["RUN_ID"] = self.run_id
        primary_hdu.header["NPL"] = self.npl

        samples_df = self.samples.assign(
            LN_WT=self.ln_wt, LN_LIKE=self.ln_like, LN_Z=self.ln_z
        )
        if self.ln_vol is not None:
            samples_df = samples_df.assign(
                LN_VOL=self.ln_vol, LN_ZERR=self.ln_zerr, N_LIVE=self.n_live
            )
        samples_hdu = fits.BinTableHDU(
            data=samples_df.to_records(index=False), name="SAMPLES"
        )
        for k, v in self.sampler_info.items():
            samples_hdu.header[k] = v

        hdul = fits.HDUList([primary_hdu, samples_hdu])

        for n, tt in enumerate(self.transit_times):
            ttimes_df = tt.astype({"out_flag": int})
            ttimes_df.columns = [k.upper() for k in ttimes_df.columns]
            hdul.append(
                fits.BinTableHDU(
                    data=ttimes_df.to_records(index=False), name=f"TTIMES_{n:02d}"
                )
            )

        return hdul

    def to_fits(self, filepath):
        self.to_hdulist().writeto(filepath, overwrite=True)
        logger.info("wrote %s", filepath)

    def weights(self):
        """
        Normalized posterior weights
        """
        wt = np.exp(self.ln_wt - self.ln_z[-1])
        return wt / np.sum(wt)

    def neff(self):
        """
        Number of effective samples (Kish ESS)
        """
        return dynesty.utils.get_neff_from_logwt(self.ln_wt)

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

    def linear_ephemeris(self, n):
        """
        Least squares (epoch, period) fit to the model transit times of planet n
        """
        tt = self.transit_times[n]
        epoch, period = poly.polyfit(tt["index"], tt["model"], 1)
        return epoch, period

    def planet_samples(self, n):
        """
        Physical posterior samples for planet n

        Returns
        -------
        pd.DataFrame
            columns [PERIOD, T0, ROR, IMPACT, DUR14, LD_U1, LD_U2, LN_WT]
        """
        s = self.samples
        index = self.transit_times[n]["index"].to_numpy()
        model = self.transit_times[n]["model"].to_numpy()

        # least squares period and epoch of the perturbed ephemeris
        legx = (index - index[-1] // 2) / (index[-1] / 2)
        ephem = (
            model
            + np.outer(s[f"C0_{n}"], np.ones_like(legx))
            + np.outer(s[f"C1_{n}"], legx)
        )
        t0, period = poly.polyfit(index, ephem.T, 1)

        # limb darkening (see Kipping 2013)
        q1, q2 = s["LD_Q1"], s["LD_Q2"]
        u1 = 2 * np.sqrt(q1) * q2
        u2 = np.sqrt(q1) * (1 - 2 * q2)

        data = np.vstack(
            [
                period,
                t0,
                s[f"ROR_{n}"],
                s[f"IMPACT_{n}"],
                s[f"DUR14_{n}"],
                u1,
                u2,
                self.ln_wt,
            ]
        ).T
        labels = "PERIOD T0 ROR IMPACT DUR14 LD_U1 LD_U2 LN_WT".split()

        return pd.DataFrame(data, columns=labels)


def _ttimes_dataframe(data):
    df = pd.DataFrame(data, columns=TTIMES_COLUMNS)
    return df.astype({"index": int, "out_flag": bool})
