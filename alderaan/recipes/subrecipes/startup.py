import os
import sys

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if base_path not in sys.path:
    sys.path.insert(0, base_path)

from aesara_theano_fallback import aesara as theano
import argparse
from astropy.units import UnitsWarning
from configparser import ConfigParser
from datetime import datetime
import shutil
from alderaan.utils.io import expand_config_path, copy_input_target_catalog
from timeit import default_timer as timer
import warnings


def load_context():
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
    config.read(os.path.join(base_path, args.config))

    mission = args.mission
    target = args.target
    run_id = config['RUN']['run_id']

    data_dir =  expand_config_path(config['PATHS']['data_dir'])
    outputs_dir = expand_config_path(config['PATHS']['outputs_dir'])
    catalog_dir = expand_config_path(config['PATHS']['catalog_dir'])

    catalog_csv = os.path.join(catalog_dir, str(config['ARGS']['catalog_csv']))

    print("")
    print(f"   MISSION : {mission}")
    print(f"   TARGET  : {target}")
    print(f"   RUN ID  : {run_id}")
    print("")
    print(f"   Base path         : {base_path}")
    print(f"   Data directory    : {data_dir}")
    print(f"   Config file       : {args.config}")
    print(f"   Input catalog     : {os.path.basename(catalog_csv)}")
    print("")
    print(f"   theano cache : {theano.config.compiledir}")
    print("")

    # build directory structure
    os.makedirs(outputs_dir, exist_ok=True)

    results_dir = os.path.join(outputs_dir, 'results', run_id, target)
    os.makedirs(results_dir, exist_ok=True)

    quicklook_dir = os.path.join(outputs_dir, 'quicklook', run_id, target)
    os.makedirs(quicklook_dir, exist_ok=True)

    # copy input catalog into results directory
    catalog_csv_copy = os.path.join(outputs_dir, 'results', run_id, f'{run_id}.csv')
    copy_input_target_catalog(catalog_csv, catalog_csv_copy)