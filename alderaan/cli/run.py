"""`alderaan run`: parse arguments and run the pipeline on one target

Example::

    alderaan run -t K00137 -m Kepler \\
        --input_dir /path/to/kepler/lightcurves \\
        --output_dir /path/to/outputs \\
        --catalog reference/alderaan/kepler_merged_alderaan.csv \\
        --run_id my_run  # optional; default run_{UT launch stamp}

Add ``--visits 4 5 6`` to load only some Kepler quarters. A rerun into a
non-empty results or plots directory fails; use a new run_id.

Outputs, under ``{output_dir}/{run_id}/`` (NN is the planet index)::

    logs/
        alderaan_K00137_{UTstamp}.log        # DEBUG-level run log
    results/K00137/
        K00137_detrended.fits                # detrended photometry
        K00137_results.fits                  # posteriors, transit times, photometry
    plots/K00137/
        K00137_litecurve_raw.png
        K00137_litecurve_detrended.png
        K00137_omc_NN.png
        K00137_folded_transit_NN.png
        K00137_dynesty_runplot.png
        K00137_dynesty_traceplot_NN.png
        K00137_dynesty_cornerplot_NN.png
"""

import argparse
import time
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
    parser.add_argument(
        "--run_id",
        help="run identifier (default: run_{UT launch stamp})",
    )
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
        "--catalog",
        type=Path,
        required=True,
        help="planet catalog in the format of "
        "reference/alderaan/kepler_merged_alderaan.csv",
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
    args.run_id = args.run_id or time.strftime("run_%Y%m%dT%H%M%S", time.gmtime())

    # deferred so --help and argument errors don't pay for the pipeline imports
    from alderaan.pipeline import main as run_pipeline

    run_pipeline(
        target=args.target,
        mission=args.mission,
        run_id=args.run_id,
        input_dir=str(args.input_dir.resolve()),
        output_dir=str(args.output_dir.resolve()),
        catalog=str(args.catalog.resolve()),
        visits=args.visits,
        log_level=args.log_level,
    )

    return 0
