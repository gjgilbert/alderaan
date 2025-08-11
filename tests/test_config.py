import os
import sys

import argparse
from configparser import ConfigParser
from pathlib import Path
import warnings
from alderaan.utils.io import resolve_config_path

warnings.simplefilter('always', UserWarning)

base_path = Path(__file__).resolve().parents[1]

# read inputs from config
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--mission', required=True, type=str, help='Observing mission')
parser.add_argument('-t', '--target', required=True, type=str, help='Target ID for star')
parser.add_argument('-c', '--config', required=True, type=str, help='Path to config file')
args = parser.parse_args()

config = ConfigParser()
config.read(args.config)

alderaan_base_path = Path(__file__).resolve().parents[2]
for key, value in config["PATHS"].items():
    config['PATHS'][key] = resolve_config_path(config['PATHS'][key], alderaan_base_path)

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

print("\npassing")