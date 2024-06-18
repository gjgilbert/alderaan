import argparse

parser = argparse.ArgumentParser(description="Inputs for ALDERAAN transit fiting pipeline")
parser.add_argument("--mission", default=None, type=str, required=True, \
                    help="Mission name; can be 'Kepler' or 'Simulated'")
parser.add_argument("--target", default=None, type=str, required=True, \
                    help="Target name; format should be K00000 or S00000")
parser.add_argument("--project_dir", default=None, type=str, required=True, \
                    help="Project directory for loading/saving ALDERAAN inputs/outputs")
parser.add_argument("--data_dir", default=None, type=str, required=True, \
                    help="Data directory for accessing MAST lightcurves")
parser.add_argument("--catalog", default=None, type=str, required=True, \
                    help="CSV file containing true input planetary parameters (e.g. DR25 catalog)")
parser.add_argument("--run_id", default=None, type=str, required=True, \
                    help="run identifier; when performing injection-and-recovery this will be used to locate the simulation directory")

args = parser.parse_args()
MISSION      = args.mission
TARGET       = args.target
PROJECT_DIR  = args.project_dir
DATA_DIR     = args.data_dir
CATALOG      = args.catalog
RUN_ID       = args.run_id