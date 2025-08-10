import os
import sys

from aesara_theano_fallback import aesara as theano
import argparse
from astropy.units import UnitsWarning
from astropy.stats import mad_std
from celerite2.backprop import LinAlgError
from configparser import ConfigParser
from datetime import datetime
import gc
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import shutil
from timeit import default_timer as timer
import warnings

from alderaan.constants import *
from alderaan.ephemeris import Ephemeris
from alderaan.litecurve import LiteCurve
from alderaan.planet import Planet
from alderaan.modules.detrend import GaussianProcessDetrender
from alderaan.modules.omc import OMC
from alderaan.modules.transit_model import ShapeTransitModel, TTimeTransitModel
from alderaan.modules.quality_control import QualityControl
from alderaan.modules.quicklook import plot_litecurve, plot_omc, dynesty_cornerplot, dynesty_runplot, dynesty_traceplot
from alderaan.utils.io import resolve_config_path, parse_koi_catalog, parse_holczer16_catalog, copy_input_target_catalog


def initialize_pipeline():
    # flush buffer
    sys.stdout.flush()
    sys.stderr.flush()

    # filter warnings
    warnings.simplefilter('always', UserWarning)
    warnings.filterwarnings(
        action='ignore', category=UnitsWarning, module='astropy'
    )

    # start timer
    global_start_time = timer()

    return global_start_time


def cleanup():
    sys.stdout.flush()
    sys.stderr.flush()
    plt.close('all')
    gc.collect()


def main():
    # initialize program
    global_start_time = initialize_pipeline()

    print("")
    print("+" * shutil.get_terminal_size().columns)
    print("ALDERAAN Pipeline")
    print(f"Initialized {datetime.now().strftime('%d-%b-%Y at %H:%M:%S')}")
    print("+" * shutil.get_terminal_size().columns)
    print("")

    # read inputs from config
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mission', required=True, type=str, help='Observing mission')
    parser.add_argument('-t', '--target', required=True, type=str, help='Target ID for star')
    parser.add_argument('-c', '--config', required=True, type=str, help='Path to config file')
    args = parser.parse_args()

    config = ConfigParser()
    config.read(args.config)

    base_path = Path(__file__).resolve().parents[2]
    for key, value in config["PATHS"].items():
        config['PATHS'][key] = resolve_config_path(config['PATHS'][key], base_path)


    mission = args.mission
    target = args.target
    run_id = config['RUN']['run_id']

    data_dir =  config['PATHS']['data_dir']
    outputs_dir = config['PATHS']['outputs_dir']
    catalog_dir = config['PATHS']['catalog_dir']

    catalog_csv = os.path.join(catalog_dir, str(config['ARGS']['catalog_csv']))

    print("")
    print(f"   MISSION : {mission}")
    print(f"   TARGET  : {target}")
    print(f"   RUN ID  : {run_id}")
    print("")
    print(f"   Data directory    : {data_dir}")
    print(f"   Config file       : {args.config}")
    print(f"   Input catalog     : {os.path.basename(catalog_csv)}")
    print("")
    print(f"   theano cache : {theano.config.compiledir}")
    print("")

    # build directory structure
    print(f"outputs : {outputs_dir})")   
    
    results_dir = os.path.join(outputs_dir, 'results', run_id, target)
    print(f"results : {results_dir})")   
    
    quicklook_dir = os.path.join(outputs_dir, 'quicklook', run_id, target)
    print(f"quicklook : {quicklook_dir})")   
    
    catalog_csv_copy = os.path.join(outputs_dir, 'results', run_id, f'{run_id}.csv')
    print(f"catalog_csv_copy : {catalog_csv_copy})")
