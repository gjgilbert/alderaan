import copy
import logging
import os
import subprocess
import sys
import time
import warnings

from aesara_theano_fallback import aesara as theano
from astropy.units import UnitsWarning

import alderaan
from alderaan.algorithms.detrend_visits import DetrendVisits
from alderaan.algorithms.flag_bad_transits import FlagBadTransits
from alderaan.algorithms.regularize_omc import RegularizeOMC
from alderaan.algorithms.transit_fitting import (
    QuickFitTransitShape,
    QuickFitTransitTimes,
    SampleTransitModel,
)
from alderaan.data_models.results import Results
from alderaan.quicklook import PlotOMC, PlotPhotometry, PlotResults
from alderaan.utils.kepler import load_kepler_data
from alderaan.utils.logger import setup_logging

logger = logging.getLogger(__name__)


def _git_hash():
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(alderaan.__file__),
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as e:
        logger.debug("git hash unavailable: %s", e)
        return "unknown"


def main(config):
    """
    Run the ALDERAAN pipeline on one target

    Parameters
    ----------
    config : dict
        pipeline configuration built by `alderaan run` (alderaan/cli/run.py)
    """
    pipeline_start_time = time.perf_counter()

    mission = config["RUN"]["mission"]
    target = config["RUN"]["target"]
    run_id = config["RUN"]["run_id"]
    output_dir = config["PATHS"]["output_dir"]

    log_level = config.get("LOGGING", {}).get("level", "INFO")
    log_path = setup_logging(
        os.path.join(output_dir, "logs"), target, run_id, log_level
    )

    warnings.simplefilter("always", UserWarning)
    warnings.filterwarnings(action="ignore", category=UnitsWarning, module="astropy")

    logger.info("alderaan %s (git %s)", alderaan.__version__, _git_hash())
    logger.info("argv: %s", " ".join(sys.argv))
    for section, values in config.items():
        logger.info("config [%s]: %s", section, values)
    logger.info("theano cache: %s", theano.config.compiledir)
    logger.info("log file: %s (console level %s)", log_path, log_level)

    try:
        results_path = _run(config, mission, target, run_id, output_dir)
    except KeyboardInterrupt:
        logger.error(
            "interrupted after %.1f min",
            (time.perf_counter() - pipeline_start_time) / 60,
        )
        raise
    except Exception:
        # the one catch-log-reraise point: the traceback reaches the log file
        logger.critical("uncaught exception; pipeline aborted", exc_info=True)
        raise

    logger.info(
        "pipeline completed in %.1f min; results: %s",
        (time.perf_counter() - pipeline_start_time) / 60,
        results_path,
    )


def _run(config, mission, target, run_id, output_dir):
    results_dir = os.path.join(output_dir, "results", run_id, target)
    os.makedirs(results_dir, exist_ok=True)

    quicklook_dir = os.path.join(output_dir, "quicklook", run_id, target)
    os.makedirs(quicklook_dir, exist_ok=True)

    config["PATHS"]["results_dir"] = results_dir

    # data i/o
    if mission == "Kepler":
        star, planets, litecurve = load_kepler_data(config)
    elif mission == "K2":
        raise NotImplementedError("K2 mission not yet implemented")
    elif mission == "TESS":
        raise NotImplementedError("TESS mission not yet implemented")
    else:
        raise ValueError(f"Unsupported mission : {mission}")

    raw_litecurve = copy.deepcopy(litecurve)

    PlotPhotometry(litecurve, planets).plot_litecurve(
        fig_path=os.path.join(quicklook_dir, f"{target}_litecurve_raw.png")
    )

    star, planets, litecurve, _, omc = RegularizeOMC(
        star, planets, litecurve, config
    ).perform(method="smooth")
    PlotOMC(omc).plot_omc(
        fig_path=os.path.join(quicklook_dir, f"{target}_omc_initial.png")
    )

    # 1st pass
    star, planets, litecurve = DetrendVisits(
        star, planets, litecurve, raw_litecurve, config
    ).perform(tight_mask=False, clip_outliers="local_trend")
    PlotPhotometry(litecurve, planets).plot_litecurve(
        fig_path=os.path.join(quicklook_dir, f"{target}_litecurve_detrended_pass1.png")
    )
    star, planets, litecurve = FlagBadTransits(
        star, planets, litecurve, config
    ).perform()
    star, planets, litecurve = QuickFitTransitShape(
        star, planets, litecurve, config
    ).perform()
    star, planets, litecurve = QuickFitTransitTimes(
        star, planets, litecurve, config
    ).perform()
    star, planets, litecurve, ttv_buffer, transit_times = RegularizeOMC(
        star, planets, litecurve, config
    ).perform(method="select")
    PlotOMC(transit_times).plot_omc(
        fig_path=os.path.join(quicklook_dir, f"{target}_omc.png")
    )

    # 2nd pass
    star, planets, litecurve = DetrendVisits(
        star, planets, litecurve, raw_litecurve, config
    ).perform(tight_mask=True, clip_outliers="transit_model", ttv_buffer=ttv_buffer)
    PlotPhotometry(litecurve, planets).plot_litecurve(
        fig_path=os.path.join(quicklook_dir, f"{target}_litecurve_detrended_pass2.png")
    )
    star, planets, litecurve = FlagBadTransits(
        star, planets, litecurve, config
    ).perform()
    posteriors = SampleTransitModel(star, planets, litecurve, config).perform()

    # package and save results
    results = Results(
        posteriors=posteriors, transit_times=transit_times, litecurve=litecurve
    )
    results_path = os.path.join(results_dir, f"{target}-results.fits")
    results.to_fits(results_path)

    plot_results = PlotResults(results)
    plot_results.plot_runplot(
        fig_path=os.path.join(quicklook_dir, f"{target}_dynesty_runplot.png")
    )
    for n in range(results.npl):
        plot_results.plot_traceplot(
            n,
            fig_path=os.path.join(
                quicklook_dir, f"{target}_dynesty_traceplot_{n:02d}.png"
            ),
        )
        plot_results.plot_cornerplot(
            n,
            fig_path=os.path.join(
                quicklook_dir, f"{target}_dynesty_cornerplot_{n:02d}.png"
            ),
        )
    logger.info("wrote quicklook plots to %s", quicklook_dir)

    return results_path
