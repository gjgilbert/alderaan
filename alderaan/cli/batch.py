"""`alderaan batch`: run the pipeline on every target in a catalog, in parallel

Each target runs as a separate `python -m alderaan.cli run` subprocess, so every
target gets its own log file, clean process state, and exit code. A bounded pool
runs up to --jobs targets at once; launches are spaced --launch_interval apart so
the pool does not start in lockstep. BLAS/OpenMP pools are pinned to one thread
per process, so jobs do not oversubscribe the CPUs. The run is fail-soft: every
target is attempted, failures are reported, and the exit code is 1 if any failed.
Adapted from the KPF pipeline (scripts/_dispatch.py).
"""

import argparse
import concurrent.futures
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# Live child processes, each its own session leader (start_new_session=True). On
# interrupt every in-flight child's group is killed; `_interrupted` stops new launches.
_live_procs = set()
_live_lock = threading.Lock()
_interrupted = threading.Event()

# Launch throttle: pool workers launch at least `launch_interval` apart.
_launch_lock = threading.Lock()
_last_launch = [0.0]


def _default_jobs():
    """At most 25% of the CPUs, but up to 16 (never more than the CPU count)"""
    n = os.cpu_count() or 1
    return min(n, max(16, n // 4))


def build_parser():
    parser = argparse.ArgumentParser(
        prog="alderaan batch",
        description="Run the ALDERAAN pipeline on every target in a catalog",
    )
    parser.add_argument(
        "-m", "--mission", required=True, choices=["Kepler"], help="observing mission"
    )
    parser.add_argument(
        "--run_id",
        help="run identifier (default: batch_{UT launch stamp})",
    )
    parser.add_argument(
        "--input_dir", type=Path, required=True, help="photometry directory"
    )
    parser.add_argument(
        "--output_dir", type=Path, required=True, help="outputs directory"
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        required=True,
        help="planet catalog in the format of "
        "reference/alderaan/kepler_merged_alderaan.csv; every koi_id is run",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=_default_jobs(),
        help="max concurrent targets (default: ~25%% of CPUs, but up to 16)",
    )
    parser.add_argument(
        "--job_timeout",
        type=float,
        default=None,
        help="per-target wall-clock limit in seconds (default: none)",
    )
    parser.add_argument(
        "--launch_interval",
        type=float,
        default=1.0,
        help="min seconds between target launches (default: 1.0)",
    )
    parser.add_argument(
        "--log_level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="console log level of each target (its file always logs DEBUG)",
    )
    return parser


def _handle_termination_signal(signum, frame):
    """Turn SIGTERM/SIGHUP into KeyboardInterrupt so `kill` or a hangup cleans up"""
    raise KeyboardInterrupt


def configure_runtime():
    """Route SIGTERM/SIGHUP to KeyboardInterrupt; pin BLAS/OpenMP to one thread

    One process per job already, so an inner thread pool per job would give
    jobs x cores threads. setdefault, so an explicit caller value wins. Also
    selects the Agg backend and disables numpy MADV_HUGEPAGE (forces synchronous
    memory compaction on Linux, stalling concurrent jobs). Children inherit the
    environment. No numerical effect.
    """
    signal.signal(signal.SIGTERM, _handle_termination_signal)
    signal.signal(signal.SIGHUP, _handle_termination_signal)
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("NUMPY_MADVISE_HUGEPAGE", "0")
    for var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ.setdefault(var, "1")


def _kill_group(proc, sig=signal.SIGKILL):
    """Signal a child's whole process group, ignoring an already-dead child"""
    try:
        os.killpg(proc.pid, sig)
    except (ProcessLookupError, PermissionError):
        pass


def _terminate_all_children(grace=5.0):
    """SIGTERM every live child's process group, then SIGKILL any survivor"""
    with _live_lock:
        procs = list(_live_procs)
    for p in procs:
        _kill_group(p, signal.SIGTERM)
    deadline = time.monotonic() + grace
    for p in procs:
        try:
            p.wait(timeout=max(0.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            pass
    for p in procs:
        if p.poll() is None:
            _kill_group(p, signal.SIGKILL)


def _run_one(argv, timeout=None, launch_interval=0.0):
    """Run one subprocess; return (returncode, stderr)

    Returns (124, note) after killing the process group on timeout, and (130, "")
    without launching once teardown has begun.
    """
    if _interrupted.is_set():
        return 130, ""
    if launch_interval > 0:
        # hold the lock across the wait so workers launch one interval apart;
        # waiting on `_interrupted` lets teardown wake every worker at once
        with _launch_lock:
            wait = launch_interval - (time.monotonic() - _last_launch[0])
            if wait > 0:
                _interrupted.wait(wait)
            _last_launch[0] = time.monotonic()
        if _interrupted.is_set():
            return 130, ""
    proc = subprocess.Popen(
        argv,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    with _live_lock:
        _live_procs.add(proc)
    try:
        # teardown may have snapshotted _live_procs between Popen and add
        if _interrupted.is_set():
            _kill_group(proc)
            proc.communicate()
            return 130, ""
        try:
            _out, stderr = proc.communicate(timeout=timeout)
            return proc.returncode, stderr
        except subprocess.TimeoutExpired:
            _kill_group(proc)
            proc.communicate()
            return 124, f"timed out after {timeout}s; child process group killed"
    finally:
        with _live_lock:
            _live_procs.discard(proc)


def run_targets(tasks, jobs, log_dir, job_timeout=None, launch_interval=0.0):
    """Run `tasks` (a list of (target, argv)) in a pool of `jobs`; return failed targets

    On SIGINT/SIGTERM/SIGHUP, cancel the queue, terminate every running child, and
    exit 130.
    """
    _interrupted.clear()
    with _live_lock:
        _live_procs.clear()
    _last_launch[0] = 0.0

    failures = []
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=jobs)
    try:
        futures = {
            pool.submit(_run_one, argv, job_timeout, launch_interval): target
            for target, argv in tasks
        }
        for future in concurrent.futures.as_completed(futures):
            target = futures[future]
            rc, stderr = future.result()
            if rc == 0:
                logger.info("ok: %s", target)
            else:
                logger.warning("FAILED: %s (exit %d)", target, rc)
                failures.append((target, rc, stderr))
    except KeyboardInterrupt:
        _interrupted.set()
        # ignore repeat signals so the teardown completes
        sigs = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
        prev = {s: signal.signal(s, signal.SIG_IGN) for s in sigs}
        try:
            logger.warning("interrupted; terminating running targets")
            pool.shutdown(wait=False, cancel_futures=True)
            _terminate_all_children()
            logger.warning("all child processes stopped")
        finally:
            for s, handler in prev.items():
                signal.signal(s, handler)
        sys.exit(130)
    finally:
        pool.shutdown(wait=True)

    for target, rc, stderr in failures:
        lines = [
            f"FAILED {target} (exit {rc})",
            f"  inspect log: {os.path.join(log_dir, f'alderaan_{target}_*.log')}",
        ]
        tail = (stderr or "").strip().splitlines()[-20:]
        if tail:
            lines.append("  --- last stderr lines ---")
            lines.extend(f"  {line}" for line in tail)
        logger.error("\n".join(lines))
    return {target for target, _, _ in failures}


def read_targets(catalog):
    """Unique koi_id values of the catalog, in catalog order"""
    import pandas as pd

    return pd.read_csv(catalog, usecols=["koi_id"]).koi_id.drop_duplicates().tolist()


def main(argv=None):
    configure_runtime()
    args = build_parser().parse_args(argv)
    args.run_id = args.run_id or time.strftime("batch_%Y%m%dT%H%M%S", time.gmtime())

    import alderaan
    from alderaan.utils.logger import setup_logging, teardown_logging

    input_dir = str(args.input_dir.resolve())
    output_dir = str(args.output_dir.resolve())
    catalog = str(args.catalog.resolve())
    log_dir = os.path.join(output_dir, args.run_id, "logs")
    log_path = setup_logging(log_dir, "batch", "INFO")

    try:
        targets = read_targets(catalog)
        logger.info("alderaan %s (git %s)", alderaan.__version__, alderaan.git_hash())
        logger.info("argv: %s", " ".join(sys.argv))
        logger.info(
            "%d target(s) from %s; jobs %d, job_timeout %s, launch_interval %s",
            len(targets),
            catalog,
            args.jobs,
            args.job_timeout,
            args.launch_interval,
        )
        logger.info("batch log: %s", log_path)

        tasks = [
            (
                t,
                [
                    sys.executable,
                    "-m",
                    "alderaan.cli",
                    "run",
                    "-t",
                    t,
                    "-m",
                    args.mission,
                    "--run_id",
                    args.run_id,
                    "--input_dir",
                    input_dir,
                    "--output_dir",
                    output_dir,
                    "--catalog",
                    catalog,
                    "--log_level",
                    args.log_level,
                ],
            )
            for t in targets
        ]
        failed = run_targets(
            tasks, args.jobs, log_dir, args.job_timeout, args.launch_interval
        )
        logger.info(
            "done: %d/%d target(s) ok", len(targets) - len(failed), len(targets)
        )
    finally:
        teardown_logging()

    return 1 if failed else 0
