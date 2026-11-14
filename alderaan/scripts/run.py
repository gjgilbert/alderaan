"""`alderaan run`: parse arguments, build the config, and run the main recipe"""

import argparse
from pathlib import Path

from alderaan.utils.config import load_config
from alderaan.utils.io import resolve_config_path

BASE_PATH = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = str(BASE_PATH / "alderaan" / "configs" / "default_config.toml")
PATH_KEYS = ["input_dir", "output_dir", "catalog_dir"]


def build_parser():
    parser = argparse.ArgumentParser(
        prog="alderaan run", description="Run the ALDERAAN pipeline on one target"
    )
    parser.add_argument("-t", "--target", required=True, help="target ID, e.g. K00137")
    parser.add_argument(
        "-m",
        "--mission",
        required=True,
        choices=["Kepler", "K2", "TESS"],
        help="observing mission",
    )
    parser.add_argument("--run_id", required=True, help="run identifier")
    parser.add_argument(
        "-c",
        "--config",
        default=DEFAULT_CONFIG,
        help="TOML config file (default: packaged default_config.toml)",
    )
    parser.add_argument("--input_dir", help="photometry directory; overrides config")
    parser.add_argument("--output_dir", help="outputs directory; overrides config")
    parser.add_argument("--catalog_dir", help="catalog directory; overrides config")
    parser.add_argument(
        "--visits",
        nargs="+",
        type=int,
        help="visits (Kepler quarters) to load (default: all)",
    )
    return parser


def build_config(args, parser):
    run = {"run_id": args.run_id, "mission": args.mission, "target": args.target}
    if args.visits is not None:
        run["visits"] = args.visits

    paths = {k: getattr(args, k) for k in PATH_KEYS if getattr(args, k) is not None}

    config = load_config(args.config, overrides={"RUN": run, "PATHS": paths})

    for k in PATH_KEYS:
        if k not in config["PATHS"]:
            parser.error(f"{k} must be set via --{k} or [PATHS] in {args.config}")
        config["PATHS"][k] = resolve_config_path(config["PATHS"][k], BASE_PATH)

    return config


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    config = build_config(args, parser)

    # deferred so --help and argument errors don't pay for the theano/pymc import
    from alderaan.recipes.main import main as run_recipe

    run_recipe(config)

    return 0
