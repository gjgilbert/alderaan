"""Run every ALDERAAN stage on one target (`alderaan run`)"""

import copy
import logging
import os
import sys
import time

import numpy as np

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
from alderaan.errors import UnsupportedTargetError
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

    Always logs a run summary. Every exception is re-raised, including
    UnsupportedTargetError for a known limit (logged as one line, no traceback).
    """
    pipeline_start_time = time.perf_counter()

    # fail before creating any output directories
    if mission != "Kepler":
        raise NotImplementedError(
            f"mission {mission!r} not yet implemented; only Kepler is supported"
        )

    log_path = setup_logging(
        os.path.join(output_dir, run_id, "logs"), target, run_id, log_level
    )

    logger.info("alderaan %s (git %s)", alderaan.__version__, alderaan.git_hash())
    logger.info("argv: %s", " ".join(sys.argv))
    logger.info(
        "target %s, mission %s, run_id %s, visits %s",
        target,
        mission,
        run_id,
        "all" if visits is None else visits,
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

    results = results_path = None
    status = "FAILED (uncaught exception; see CRITICAL traceback above)"
    try:
        results, results_path = _run(
            target, run_id, input_dir, output_dir, catalog, visits
        )
        status = "ok"
    except KeyboardInterrupt:
        logger.error("interrupted; pipeline aborted")
        status = "interrupted"
        raise
    except UnsupportedTargetError as e:
        # a known limit, not a bug: one line, no traceback
        logger.error("target not processed (unsupported): %s", e)
        status = f"unsupported (known limit): {e}"
        raise
    except Exception:
        # the one catch-log-reraise point: the traceback reaches the log file
        logger.critical("uncaught exception; pipeline aborted", exc_info=True)
        raise
    finally:
        logger.info(
            "%s",
            run_summary(
                target,
                run_id,
                status,
                time.perf_counter() - pipeline_start_time,
                results,
                results_path,
            ),
        )
        teardown_logging()


def run_summary(target, run_id, status, elapsed_s, results=None, results_path=None):
    """
    End-of-run verdict block for the log

    Lists status, data, sampler, posterior medians per planet, results path, and
    elapsed time. Data lines appear only if results is given (status ok).
    Surrounding blank lines make the block stand out in the log.
    """
    lines = [
        f"===== run summary: {target} (run_id {run_id}) =====",
        f"  status:   {status}",
    ]
    if results is not None:
        lc = results.litecurve
        post = results.posteriors
        info = post.sampler_info
        visits = " ".join(str(v) for v in np.unique(lc.visit))
        lines += [
            f"  data:     {len(lc.time)} cadences, visits {visits}, "
            f"t = {lc.t_start:.1f}-{lc.t_end:.1f} d",
            f"  sampler:  ln Z = {info['logz'][-1]:.2f} +/- {info['logzerr'][-1]:.2f}, "
            f"{post.neff():.0f} effective samples, {np.sum(info['ncall'])} calls",
            "  posterior medians:",
            f"    {'planet':<12s}{'P [d]':>12s}{'ror':>9s}{'b':>7s}{'T14 [hr]':>10s}"
            f"  {'OMC model':<14s}{'good/all transits':>17s}",
        ]
        for n, tt in enumerate(results.transit_times):
            _, period, ror, b, t14, _, _ = results.median_transit_params(n)
            ngood = np.sum(tt.phot_coverage & tt.phot_noise)
            name = results.meta.get(f"PLNAME{n}", str(n))
            lines.append(
                f"    {name:<12s}{period:>12.6f}{ror:>9.4f}{b:>7.3f}{t14 * 24:>10.3f}"
                f"  {str(tt.model_name):<14s}{f'{ngood}/{len(tt.ttime_mod)}':>17s}"
            )
        lines.append(f"  results:  {results_path}")
    lines.append(f"  elapsed:  {elapsed_s / 60:.1f} min")
    return "\n\n" + "\n".join(lines) + "\n"


def _run(target, run_id, input_dir, output_dir, catalog, visits):
    """Run every stage, write results and plots; return (results, results_path)"""
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

    # no fitted transit model yet: clip on a local trend, detrend with a wide mask
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
            snr=p.meta["snr_per_transit"],
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
    litecurve.meta.update(
        RUN_ID=run_id, VERSION=alderaan.__version__, GIT_HASH=alderaan.git_hash()
    )
    litecurve.to_fits(os.path.join(results_dir, f"{target}_detrended.fits"))
    litecurve.plot(
        mark_times=[p.transit_times.ttime_mod for p in planets],
        fig_path=os.path.join(plots_dir, f"{target}_litecurve_detrended.png"),
    )
    posteriors = SampleTransitModel(star, planets, litecurve).perform()

    results = Results(
        posteriors=posteriors,
        transit_times=[p.transit_times for p in planets],
        litecurve=litecurve,
    )
    results.meta.update(litecurve.meta)
    for n, p in enumerate(planets):
        results.meta[f"PLNAME{n}"] = p.meta["PLNAME"]
    results.meta.update(results.reduced_chi2())
    results_path = os.path.join(results_dir, f"{target}_results.fits")
    results.to_fits(results_path)

    posteriors.dynesty_runplot(
        fig_path=os.path.join(plots_dir, f"{target}_dynesty_runplot.png")
    )
    for n in range(results.npl):
        results.plot_folded_transit(
            n, fig_path=os.path.join(plots_dir, f"{target}_folded_transit_{n:02d}.png")
        )
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

    return results, results_path
