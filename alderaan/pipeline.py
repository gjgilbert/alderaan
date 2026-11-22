"""Run every ALDERAAN stage on one target (`alderaan run`)"""

import copy
import logging
import os
import sys
import time

import alderaan
from alderaan.algorithms import (
    ClipOutliers,
    DetrendVisits,
    FlagBadTransits,
    QuickFitTransitShape,
    QuickFitTransitTimes,
    RegularizeOMC,
    SampleTransitModel,
)
from alderaan.data_models import Results
from alderaan.utils.kepler import load_kepler_data
from alderaan.utils.logger import setup_logging, teardown_logging

logger = logging.getLogger(__name__)


def main(
    target,
    mission,
    run_id,
    input_dir,
    output_dir,
    catalog,
    visits=None,
    log_level="INFO",
):
    """
    Run the ALDERAAN pipeline on one target (see `alderaan run --help`)
    """
    pipeline_start_time = time.perf_counter()

    # fail before creating any output directories
    if mission != "Kepler":
        raise NotImplementedError(
            f"mission {mission!r} not yet implemented; only Kepler is supported"
        )

    log_path = setup_logging(
        os.path.join(output_dir, run_id, "logs"), target, log_level
    )

    logger.info("alderaan %s (git %s)", alderaan.__version__, alderaan.git_hash())
    logger.info("argv: %s", " ".join(sys.argv))
    logger.info(
        "target %s, mission %s, run_id %s, visits %s", target, mission, run_id, visits
    )
    logger.info(
        "input_dir %s, output_dir %s, catalog %s",
        input_dir,
        output_dir,
        catalog,
    )
    logger.info("log file: %s (console level %s)", log_path, log_level)
    logger.info(
        "threads: %s",
        ", ".join(
            f"{v}={os.environ.get(v, 'unset')}"
            for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        ),
    )

    try:
        results_path = _run(
            target, mission, run_id, input_dir, output_dir, catalog, visits
        )
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
    else:
        logger.info(
            "pipeline completed in %.1f min; results: %s",
            (time.perf_counter() - pipeline_start_time) / 60,
            results_path,
        )
    finally:
        # close the log file and stop capturing warnings, even if the run fails
        teardown_logging()


def _run(target, mission, run_id, input_dir, output_dir, catalog, visits):
    results_dir = os.path.join(output_dir, run_id, "results", target)
    plots_dir = os.path.join(output_dir, run_id, "plots", target)

    # never mix outputs of two runs: a rerun needs a new run_id or an empty directory
    for d in (results_dir, plots_dir):
        if os.path.isdir(d) and os.listdir(d):
            raise FileExistsError(f"{d} is not empty; use a new run_id")
        os.makedirs(d, exist_ok=True)

    # data i/o (Kepler only; main() rejects other missions)
    star, planets, litecurve = load_kepler_data(target, input_dir, catalog, visits)

    raw_litecurve = copy.deepcopy(litecurve)

    litecurve.plot(
        mark_times=[p.transit_times.ttime_mod for p in planets],
        fig_path=os.path.join(plots_dir, f"{target}_litecurve_raw.png"),
    )

    # starting transit times
    planets, litecurve = RegularizeOMC(star, planets, litecurve).perform(
        method="smooth"
    )

    # no fitted transit model yet: clip on a local trend and detrend with a
    # wide mask
    planets, litecurve = ClipOutliers(star, planets, litecurve).perform(
        method="local_trend"
    )
    litecurve.reset(raw_litecurve)  # detrend raw photometry for surviving cadences
    planets, litecurve = DetrendVisits(star, planets, litecurve).perform(
        method="wide_mask"
    )
    # photometry flags hold until the next detrend
    planets, litecurve = FlagBadTransits(star, planets, litecurve).perform()
    planets, litecurve = QuickFitTransitShape(star, planets, litecurve).perform()
    planets, litecurve = QuickFitTransitTimes(star, planets, litecurve).perform()
    planets, litecurve = RegularizeOMC(star, planets, litecurve).perform(
        method="select"
    )
    for n, p in enumerate(planets):
        p.transit_times.plot_omc(
            show_outliers=True,
            title=f"{target}-{n}",
            fig_path=os.path.join(plots_dir, f"{target}_omc_{n:02d}.png"),
        )

    # final fit: clip and detrend mask from the fitted model and final times;
    # re-flag so the sampler's flags are judged on the photometry it fits
    planets, litecurve = ClipOutliers(star, planets, litecurve).perform(
        method="transit_model"
    )
    litecurve.reset(raw_litecurve)
    planets, litecurve = DetrendVisits(star, planets, litecurve).perform(
        method="tight_mask"
    )
    planets, litecurve = FlagBadTransits(star, planets, litecurve).perform()
    litecurve.to_fits(os.path.join(results_dir, f"{target}_detrended.fits"))
    litecurve.plot(
        mark_times=[p.transit_times.ttime_mod for p in planets],
        fig_path=os.path.join(plots_dir, f"{target}_litecurve_detrended.png"),
    )
    posteriors = SampleTransitModel(star, planets, litecurve).perform()

    # package and save results
    results = Results(
        posteriors=posteriors,
        transit_times=[p.transit_times for p in planets],
        litecurve=litecurve,
    )
    results_path = os.path.join(results_dir, f"{target}_results.fits")
    results.to_fits(results_path, run_id)

    posteriors.dynesty_runplot(
        fig_path=os.path.join(plots_dir, f"{target}_dynesty_runplot.png")
    )
    for n in range(results.npl):
        posteriors.dynesty_traceplot(
            n,
            fig_path=os.path.join(plots_dir, f"{target}_dynesty_traceplot_{n:02d}.png"),
        )
        posteriors.dynesty_cornerplot(
            n,
            fig_path=os.path.join(
                plots_dir, f"{target}_dynesty_cornerplot_{n:02d}.png"
            ),
        )
    logger.info("wrote plots to %s", plots_dir)

    return results_path
