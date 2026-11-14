import os
import shutil
import sys
import warnings
from datetime import datetime
from timeit import default_timer as timer

from aesara_theano_fallback import aesara as theano
from astropy.units import UnitsWarning

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


def main(config):
    """
    Run the ALDERAAN pipeline on one target

    Parameters
    ----------
    config : dict
        pipeline configuration built by `alderaan run` (alderaan/cli/run.py)
    """
    pipeline_start_time = timer()

    sys.stdout.flush()
    sys.stderr.flush()

    warnings.simplefilter("always", UserWarning)
    warnings.filterwarnings(action="ignore", category=UnitsWarning, module="astropy")

    print("")
    print("+" * shutil.get_terminal_size().columns)
    print("ALDERAAN Pipeline")
    print(f"Initialized {datetime.now().strftime('%d-%b-%Y at %H:%M:%S')}")
    print("+" * shutil.get_terminal_size().columns)
    print("")

    mission = config["RUN"]["mission"]
    target = config["RUN"]["target"]
    run_id = config["RUN"]["run_id"]

    input_dir = config["PATHS"]["input_dir"]
    output_dir = config["PATHS"]["output_dir"]
    catalog_name = config["CATALOGS"]["koi_catalog"]

    print("")
    print(f"   MISSION : {mission}")
    print(f"   TARGET  : {target}")
    print(f"   RUN ID  : {run_id}")
    print("")
    print(f"   Input directory   : {input_dir}")
    print(f"   Output directory  : {output_dir}")
    print(f"   Input catalog     : {catalog_name}")
    print("")
    print(f"   theano cache : {theano.config.compiledir}")
    print("")

    os.makedirs(output_dir, exist_ok=True)

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

    PlotPhotometry(litecurve, target, planets).plot_litecurve(
        fig_path=os.path.join(quicklook_dir, f"{target}_litecurve_raw.png")
    )

    star, planets, litecurve, _, omc = RegularizeOMC(
        star, planets, litecurve, config
    ).perform(method="smooth")
    PlotOMC(omc, target).plot_omc(
        fig_path=os.path.join(quicklook_dir, f"{target}_omc_initial.png")
    )

    # 1st pass
    star, planets, litecurve = DetrendVisits(star, planets, litecurve, config).perform(
        tight_mask=False, clip_outliers="local_trend"
    )
    PlotPhotometry(litecurve, target, planets).plot_litecurve(
        fig_path=os.path.join(quicklook_dir, f"{target}_litecurve_detrended.png")
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
    PlotOMC(transit_times, target).plot_omc(
        fig_path=os.path.join(quicklook_dir, f"{target}_omc.png")
    )

    # 2nd pass
    star, planets, litecurve = DetrendVisits(star, planets, litecurve, config).perform(
        tight_mask=True, clip_outliers="transit_model", ttv_buffer=ttv_buffer
    )
    PlotPhotometry(litecurve, target, planets).plot_litecurve(
        fig_path=os.path.join(quicklook_dir, f"{target}_litecurve_detrended.png")
    )
    star, planets, litecurve = FlagBadTransits(
        star, planets, litecurve, config
    ).perform()
    dynesty_results = SampleTransitModel(star, planets, litecurve, config).perform()

    # package and save results
    results = Results.from_dynesty(
        dynesty_results, mission, target, run_id, transit_times
    )
    results.to_fits(os.path.join(results_dir, f"{target}-results.fits"))

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

    print(f"\ncumulative runtime = {((timer() - pipeline_start_time) / 60):.1f} min")
    print("\n\n=== exiting main pipeline ===\n\n")
