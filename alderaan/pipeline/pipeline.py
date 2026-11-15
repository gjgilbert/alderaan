import copy
import logging
import os
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
from alderaan.utils.kepler import load_kepler_data
from alderaan.utils.logger import setup_logging

logger = logging.getLogger(__name__)


def main(
    target,
    mission,
    run_id,
    input_dir,
    output_dir,
    catalog_dir,
    visits=None,
    log_level="INFO",
):
    """
    Run the ALDERAAN pipeline on one target (see `alderaan run --help`)
    """
    pipeline_start_time = time.perf_counter()

    log_path = setup_logging(
        os.path.join(output_dir, run_id, "logs"), target, log_level
    )

    warnings.simplefilter("always", UserWarning)
    warnings.filterwarnings(action="ignore", category=UnitsWarning, module="astropy")

    logger.info("alderaan %s (git %s)", alderaan.__version__, alderaan._git_hash())
    logger.info("argv: %s", " ".join(sys.argv))
    logger.info(
        "target %s, mission %s, run_id %s, visits %s", target, mission, run_id, visits
    )
    logger.info(
        "input_dir %s, output_dir %s, catalog_dir %s",
        input_dir,
        output_dir,
        catalog_dir,
    )
    logger.info("theano cache: %s", theano.config.compiledir)
    logger.info("log file: %s (console level %s)", log_path, log_level)

    try:
        results_path = _run(
            target, mission, run_id, input_dir, output_dir, catalog_dir, visits
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

    logger.info(
        "pipeline completed in %.1f min; results: %s",
        (time.perf_counter() - pipeline_start_time) / 60,
        results_path,
    )


def _run(target, mission, run_id, input_dir, output_dir, catalog_dir, visits):
    results_dir = os.path.join(output_dir, run_id, "results", target)
    os.makedirs(results_dir, exist_ok=True)

    quicklook_dir = os.path.join(output_dir, run_id, "plots", target)
    os.makedirs(quicklook_dir, exist_ok=True)

    # data i/o
    if mission == "Kepler":
        star, planets, litecurve = load_kepler_data(
            target, input_dir, catalog_dir, visits
        )
    elif mission == "K2":
        raise NotImplementedError("K2 mission not yet implemented")
    elif mission == "TESS":
        raise NotImplementedError("TESS mission not yet implemented")
    else:
        raise ValueError(f"Unsupported mission : {mission}")

    raw_litecurve = copy.deepcopy(litecurve)

    litecurve.plot(
        mark_times=[p.transit_times.ttime_mod for p in planets],
        fig_path=os.path.join(quicklook_dir, f"{target}_litecurve_raw.png"),
    )

    planets, litecurve = RegularizeOMC(star, planets, litecurve).perform(
        method="smooth"
    )
    for n, p in enumerate(planets):
        p.transit_times.plot_omc(
            show_outliers=True,
            fig_path=os.path.join(quicklook_dir, f"{target}_omc_initial_{n:02d}.png"),
        )

    # 1st pass
    planets, litecurve = DetrendVisits(star, planets, litecurve, raw_litecurve).perform(
        tight_mask=False, clip_outliers="local_trend"
    )
    litecurve.plot(
        mark_times=[p.transit_times.ttime_mod for p in planets],
        fig_path=os.path.join(quicklook_dir, f"{target}_litecurve_detrended_pass1.png"),
    )
    planets, litecurve = FlagBadTransits(star, planets, litecurve).perform()
    planets, litecurve = QuickFitTransitShape(star, planets, litecurve).perform()
    planets, litecurve = QuickFitTransitTimes(star, planets, litecurve).perform()
    planets, litecurve = RegularizeOMC(star, planets, litecurve).perform(
        method="select"
    )
    for n, p in enumerate(planets):
        p.transit_times.plot_omc(
            show_outliers=True,
            fig_path=os.path.join(quicklook_dir, f"{target}_omc_{n:02d}.png"),
        )

    # 2nd pass
    planets, litecurve = DetrendVisits(star, planets, litecurve, raw_litecurve).perform(
        tight_mask=True, clip_outliers="transit_model"
    )
    (obsmode,) = set(litecurve.obsmode)  # fails loudly on mixed cadence
    cadence = {"long cadence": "lc", "short cadence": "sc"}[obsmode]
    litecurve.to_fits(os.path.join(results_dir, f"{target}_{cadence}_detrended.fits"))
    litecurve.plot(
        mark_times=[p.transit_times.ttime_mod for p in planets],
        fig_path=os.path.join(quicklook_dir, f"{target}_litecurve_detrended_pass2.png"),
    )
    planets, litecurve = FlagBadTransits(star, planets, litecurve).perform()
    posteriors = SampleTransitModel(star, planets, litecurve).perform()

    # package and save results
    results = Results(
        posteriors=posteriors,
        transit_times=[p.transit_times for p in planets],
        litecurve=litecurve,
    )
    results_path = os.path.join(results_dir, f"{target}-results.fits")
    results.to_fits(results_path)

    posteriors.plot_runplot(
        fig_path=os.path.join(quicklook_dir, f"{target}_dynesty_runplot.png")
    )
    for n in range(results.npl):
        posteriors.plot_traceplot(
            n,
            fig_path=os.path.join(
                quicklook_dir, f"{target}_dynesty_traceplot_{n:02d}.png"
            ),
        )
        posteriors.plot_cornerplot(
            n,
            fig_path=os.path.join(
                quicklook_dir, f"{target}_dynesty_cornerplot_{n:02d}.png"
            ),
        )
    logger.info("wrote quicklook plots to %s", quicklook_dir)

    return results_path
