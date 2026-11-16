"""`alderaan run`: parse arguments and run the pipeline"""

import argparse
from pathlib import Path


def build_parser():
    parser = argparse.ArgumentParser(
        prog="alderaan run", description="Run the ALDERAAN pipeline on one target"
    )
    parser.add_argument("-t", "--target", required=True, help="target ID, e.g. K00137")
    parser.add_argument(
        "-m",
        "--mission",
        required=True,
        choices=["Kepler"],
        help="observing mission",
    )
    parser.add_argument("--run_id", required=True, help="run identifier")
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="photometry directory",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="outputs directory",
    )
    parser.add_argument(
        "--catalog_dir",
        type=Path,
        required=True,
        help="catalog directory",
    )
    parser.add_argument(
        "--visits",
        nargs="+",
        type=int,
        help="visits (Kepler quarters) to load (default: all)",
    )
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="console log level (file always logs DEBUG)",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    # deferred so --help and argument errors don't pay for the pipeline imports
    from alderaan.pipeline import main as run_pipeline

    run_pipeline(
        target=args.target,
        mission=args.mission,
        run_id=args.run_id,
        input_dir=str(args.input_dir.resolve()),
        output_dir=str(args.output_dir.resolve()),
        catalog_dir=str(args.catalog_dir.resolve()),
        visits=args.visits,
        log_level=args.log_level,
    )

    return 0
