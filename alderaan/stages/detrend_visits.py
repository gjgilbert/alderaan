__all__ = ["DetrendVisits"]

import os
import warnings
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from astropy.stats import mad_std
from celerite2.backprop import LinAlgError

from alderaan.constants import kepler_lcit, kepler_scit
from alderaan.data_models.litecurve import LiteCurve
from alderaan.modules.detrend import GaussianProcessDetrender
from alderaan.modules.quicklook import plot_litecurve


class DetrendVisits:
    def __init__(
        self,
        shire,
        target,
        results_dir,
        quicklook_dir,
        tight_mask,
        clip_outliers,
        ttv_buffer=None,
    ):
        self.shire = deepcopy(shire)
        self.target = target
        self.results_dir = results_dir
        self.quicklook_dir = quicklook_dir
        self.tight_mask = tight_mask
        self.clip_outliers = clip_outliers
        self.ttv_buffer = ttv_buffer

        self.durations = np.array([p.duration for p in self.shire.planets])

    def _transit_mask(self, detrender):
        obsmode = detrender.litecurve.obsmode[0]
        exptime = detrender.exptime[obsmode]

        if self.tight_mask:
            abs_size = 2 * (exptime + np.max(self.ttv_buffer))
            return detrender.make_transit_mask(
                rel_size=1.0, abs_size=abs_size, mask_type="condensed"
            )

        abs_size = 2 * (exptime + 1 / 24)
        return detrender.make_transit_mask(
            rel_size=3.0, abs_size=abs_size, mask_type="condensed"
        )

    def _min_period(self, obsmode):
        if obsmode == "short cadence":
            return np.max([5 * np.max(self.durations), 91 * kepler_scit])
        if obsmode == "long cadence":
            return np.max([5 * np.max(self.durations), 13 * kepler_lcit])
        raise ValueError(f"unsuported obsmode: {obsmode}")

    def _clip_outliers(self, detrenders):
        for detrender in detrenders:
            mask = self._transit_mask(detrender)

            npts_initial = len(detrender.litecurve.time)

            detrender.clip_outliers(
                kernel_size=13, sigma_upper=5, sigma_lower=5, mask=mask
            )
            detrender.clip_outliers(
                kernel_size=13, sigma_upper=5, sigma_lower=1000, mask=None
            )

            npts_final = len(detrender.litecurve.time)

            print(
                f"  Quarter {detrender.litecurve.visit[0]} : "
                f"{npts_initial - npts_final} outliers rejected"
            )

    def _estimate_oscillation_periods(self, detrenders):
        oscillation_periods = np.zeros(len(detrenders))
        for j, detrender in enumerate(detrenders):
            min_period = self._min_period(detrender.litecurve.obsmode[0])
            oscillation_periods[j] = detrender.estimate_oscillation_period(
                min_period=min_period
            )

        osc_per_mu = np.nanmedian(oscillation_periods)
        osc_per_sd = np.max(
            [mad_std(oscillation_periods, ignore_nan=True), 0.1 * osc_per_mu]
        )

        print(
            f"\nNominal stellar oscillation period = "
            f"{osc_per_mu:.1f} +/- {osc_per_sd:.1f} days"
        )

        print("\nSaving stellar oscillation periods...\n")
        filename = os.path.join(
            self.results_dir, f"{self.target}_stellar_oscillations.txt"
        )
        oscillation_period_by_quarter = np.ones(18) * np.nan
        quarter_indexes = [detrender.litecurve.visit[0] - 1 for detrender in detrenders]
        oscillation_period_by_quarter[quarter_indexes] = oscillation_periods
        np.savetxt(filename, np.stack([np.arange(18), oscillation_period_by_quarter]).T)

        return oscillation_periods

    def _detrend(self, detrender, oscillation_periods):
        obsmode = detrender.litecurve.obsmode[0]

        if obsmode == "short cadence":
            gap_tolerance = np.max(
                [int(np.min(self.durations) / kepler_scit * 5 / 2), 91]
            )
        elif obsmode == "long cadence":
            gap_tolerance = np.max(
                [int(np.min(self.durations) / kepler_lcit * 5 / 2), 13]
            )
        else:
            raise ValueError(f"unsuported obsmode: {obsmode}")

        min_period = self._min_period(obsmode)
        mask = self._transit_mask(detrender)

        # use successively simpler models as fallbacks
        try:
            return detrender.detrend(
                "RotationTerm",
                np.nanmedian(oscillation_periods),
                min_period,
                transit_mask=mask,
                gap_tolerance=gap_tolerance,
                jump_tolerance=5.0,
                correct_ramp=True,
                return_trend=False,
                progressbar=False,
            )
        except LinAlgError:
            warnings.warn(
                "Initial detrending failed..."
                "attempting to refit without exponential ramp component",
                stacklevel=1,
            )

        try:
            return detrender.detrend(
                "RotationTerm",
                np.nanmedian(oscillation_periods),
                min_period,
                transit_mask=mask,
                gap_tolerance=gap_tolerance,
                jump_tolerance=5.0,
                correct_ramp=False,
                return_trend=False,
                progressbar=False,
            )
        except LinAlgError:
            warnings.warn(
                "Detrending with RotationTerm failed..."
                "attempting to detrend with SHOTerm",
                stacklevel=1,
            )

        return detrender.detrend(
            "SHOTerm",
            np.nanmedian(oscillation_periods),
            min_period,
            transit_mask=mask,
            gap_tolerance=gap_tolerance,
            jump_tolerance=5.0,
            correct_ramp=False,
            return_trend=False,
            progressbar=False,
        )

    def perform(self):
        print("\n\nSTAGE: DETREND VISITS\n")

        planets = self.shire.planets
        litecurves = self.shire.litecurve.split_visits()

        detrenders = [GaussianProcessDetrender(lc, planets) for lc in litecurves]

        if self.clip_outliers:
            self._clip_outliers(detrenders)

        oscillation_periods = self._estimate_oscillation_periods(detrenders)

        for j, detrender in enumerate(detrenders):
            print(f"\nDetrending {j + 1} of {len(detrenders)} litecurves", flush=True)
            litecurves[j] = self._detrend(detrender, oscillation_periods)

        litecurve = LiteCurve().from_list(litecurves)

        filepath = os.path.join(
            self.quicklook_dir, f"{self.target}_litecurve_detrended.png"
        )
        plot_litecurve(litecurve, self.target, planets, filepath)
        plt.close("all")

        # output cadence follows the last visit
        obsmode = detrenders[-1].litecurve.obsmode[0]
        if obsmode == "long cadence":
            filename = os.path.join(
                self.results_dir, f"{self.target}_lc_detrended.fits"
            )
            litecurve.to_fits(self.target, filename, cadence="LONG")
        elif obsmode == "short cadence":
            filename = os.path.join(
                self.results_dir, f"{self.target}_sc_detrended.fits"
            )
            litecurve.to_fits(self.target, filename, cadence="SHORT")

        self.shire.litecurve = litecurve

        return self.shire
