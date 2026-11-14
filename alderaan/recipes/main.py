#!/usr/bin/env python3

import argparse
import os
import shutil
import sys
import warnings
from configparser import ConfigParser
from datetime import datetime
from pathlib import Path
from timeit import default_timer as timer

from aesara_theano_fallback import aesara as theano
from astropy.units import UnitsWarning

from alderaan.data_models.shire import Shire
from alderaan.stages.cleanup_omc import CleanupOMC
from alderaan.stages.detrend_visits import DetrendVisits
from alderaan.stages.flag_bad_transits import FlagBadTransits
from alderaan.stages.load_kepler_data import LoadKeplerData
from alderaan.stages.quick_fit_transit_shape import QuickFitTransitShape
from alderaan.stages.quick_fit_transit_times import QuickFitTransitTimes
from alderaan.stages.regularize_omc import RegularizeOMC
from alderaan.stages.reset_litecurves import ResetLitecurves
from alderaan.stages.sample_transit_model import SampleTransitModel
from alderaan.utils.io import copy_input_target_catalog, resolve_config_path

BASE_PATH = Path(__file__).resolve().parents[2]


def main():
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

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--mission", required=True, type=str, help="Observing mission"
    )
    parser.add_argument(
        "-t", "--target", required=True, type=str, help="Target ID for star"
    )
    parser.add_argument(
        "-c", "--config", required=True, type=str, help="Path to config file"
    )
    args = parser.parse_args()

    config = ConfigParser()
    config.read(args.config)

    for k in config["PATHS"]:
        config["PATHS"][k] = resolve_config_path(config["PATHS"][k], BASE_PATH)

    mission = args.mission
    target = args.target
    run_id = config["RUN"]["run_id"]

    data_dir = config["PATHS"]["data_dir"]
    outputs_dir = config["PATHS"]["outputs_dir"]
    catalog_dir = config["PATHS"]["catalog_dir"]
    catalog_name = str(config["ARGS"]["catalog_csv"])
    catalog_filepath = os.path.join(catalog_dir, catalog_name)

    print("")
    print(f"   MISSION : {mission}")
    print(f"   TARGET  : {target}")
    print(f"   RUN ID  : {run_id}")
    print("")
    print(f"   Data directory    : {data_dir}")
    print(f"   Config file       : {args.config}")
    print(f"   Input catalog     : {catalog_name}")
    print("")
    print(f"   theano cache : {theano.config.compiledir}")
    print("")

    os.makedirs(outputs_dir, exist_ok=True)

    results_dir = os.path.join(outputs_dir, "results", run_id, target)
    os.makedirs(results_dir, exist_ok=True)

    quicklook_dir = os.path.join(outputs_dir, "quicklook", run_id, target)
    os.makedirs(quicklook_dir, exist_ok=True)

    copy_input_target_catalog(
        catalog_filepath, os.path.join(outputs_dir, "results", run_id, f"{run_id}.csv")
    )

    # data i/o
    if mission == "Kepler":
        star, planets, litecurve_raw = LoadKeplerData(
            target, catalog_filepath, catalog_dir, data_dir, quicklook_dir
        ).perform()
    elif mission == "K2":
        raise NotImplementedError("K2 mission not yet implemented")
    elif mission == "TESS":
        raise NotImplementedError("TESS mission not yet implemented")
    else:
        raise ValueError(f"Unsupported mission : {mission}")

    shire = Shire(star, planets, litecurve_raw)
    shire = CleanupOMC(shire, target, quicklook_dir, full=True).perform()

    # 1st pass
    shire = DetrendVisits(
        shire,
        target,
        results_dir,
        quicklook_dir,
        tight_mask=False,
        clip_outliers=True,
    ).perform()
    shire = FlagBadTransits(shire).perform()
    shire = QuickFitTransitShape(shire).perform()
    shire = QuickFitTransitTimes(shire, target, quicklook_dir).perform()
    shire, ttv_buffer = RegularizeOMC(
        shire, target, results_dir, quicklook_dir
    ).perform()
    shire = ResetLitecurves(shire, litecurve_raw).perform()

    # 2nd pass
    shire = DetrendVisits(
        shire,
        target,
        results_dir,
        quicklook_dir,
        tight_mask=True,
        clip_outliers=False,
        ttv_buffer=ttv_buffer,
    ).perform()
    shire = FlagBadTransits(shire).perform()
    SampleTransitModel(
        shire, mission, target, run_id, results_dir, quicklook_dir
    ).perform()

    print(f"\ncumulative runtime = {((timer() - pipeline_start_time) / 60):.1f} min")
    print("\n\n=== exiting main pipeline ===\n\n")


if __name__ == "__main__":
    main()
