"""
`alderaan batch`: run the pipeline on every target in a catalog, in parallel

Each target runs as a separate `python -m alderaan.cli run` subprocess, so every
target gets its own log file, clean process state, and exit code. A bounded pool
runs up to --jobs targets at once; launches are spaced --launch_interval apart so
the pool does not start in lockstep. BLAS/OpenMP pools are pinned to one thread
per process, so jobs do not oversubscribe the CPUs. The run is fail-soft: every
target is attempted and failures are reported. A target that exits 3 is
unsupported by design (a known limit, e.g. < 3 transits); it is counted apart and
does not fail the batch. A target over --job_timeout has its process group killed
and is counted as failed (reported as exit 124).
Adapted from the KPF pipeline (scripts/_dispatch.py).

Exit codes: 0 no target failed; 1 any target failed; 130 interrupted
(SIGINT/SIGTERM/SIGHUP; running targets are terminated).

Example::

    alderaan batch -m Kepler \\
        --input_dir /path/to/kepler/lightcurves \\
        --output_dir /path/to/outputs \\
        --catalog reference/alderaan/kepler_merged_alderaan.csv \\
        --jobs 64 \\
        --run_id my_run  # optional; default UT launch stamp

Every target writes the same files as `alderaan run`, all under one run
directory ``{output_dir}/{run_id}/``::

    alderaan_batch_{run_id}.csv              # copy of the input catalog (same suffix)
    logs/
        alderaan_batch_{run_id}.log          # batch progress and failure report
        alderaan_K00001_{run_id}.log         # one run log per target
        alderaan_K00002_{run_id}.log
        ...
    results/K00001/                          # see `alderaan run` for file names
    results/K00002/
    ...
    plots/K00001/
    plots/K00002/
    ...
"""

import argparse
import concurrent.futures
import logging
import os
import shutil
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
    """25% of the CPUs, but at least 16 (never more than the CPU count)"""
    n = os.cpu_count() or 1
    return min(n, max(16, n // 4))


def build_parser():
    """Argument parser for `alderaan batch`"""
    parser = argparse.ArgumentParser(
        prog="alderaan batch",
        description="Run the ALDERAAN pipeline on every target in a catalog",
    )
    parser.add_argument(
        "-m", "--mission", required=True, choices=["Kepler"], help="observing mission"
    )
    parser.add_argument(
        "--run_id",
        help="run identifier (default: UT launch stamp, %%Y%%m%%dT%%H%%M%%S)",
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
        help="max concurrent targets (default: ~25%% of CPUs, at least 16, "
        "at most the CPU count)",
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
    """
    Route SIGTERM/SIGHUP to KeyboardInterrupt; pin BLAS/OpenMP to one thread

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
    """
    Run one subprocess; return (returncode, stderr)

    Returns (124, stderr) after killing the process group on timeout, and
    (130, "") without launching once teardown has begun.
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
            _out, stderr = proc.communicate()
            return 124, stderr
    finally:
        with _live_lock:
            _live_procs.discard(proc)


def run_targets(tasks, jobs, job_timeout=None, launch_interval=0.0):
    """
    Run `tasks` (a list of (target, argv)) in a pool of `jobs`

    Returns (failed, unsupported), two dicts of target -> reason; unsupported
    targets exit 3, with the reason from their stderr. Logs the last 20 stderr
    lines of each failed target.

    On SIGINT/SIGTERM/SIGHUP, cancel the queue, terminate every running child, and
    exit 130.
    """
    _interrupted.clear()
    with _live_lock:
        _live_procs.clear()
    _last_launch[0] = 0.0

    failures = []
    unsupported = {}
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
            elif rc == 3:
                # pipeline.main writes "... (unsupported): <reason>" to stderr
                marker = "(unsupported): "
                lines = [s for s in (stderr or "").splitlines() if marker in s]
                reason = lines[-1].split(marker, 1)[1] if lines else "see log"
                logger.warning("UNSUPPORTED: %s (known limit: %s)", target, reason)
                unsupported[target] = reason
            else:
                if rc == 124:
                    reason = f"timed out after {job_timeout}s; process group killed"
                else:
                    reason = f"exit {rc}"
                logger.error("FAILED: %s (%s)", target, reason)
                failures.append((target, reason, stderr))
    except KeyboardInterrupt:
        _interrupted.set()
        # ignore repeat signals so the teardown completes
        sigs = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
        prev = {s: signal.signal(s, signal.SIG_IGN) for s in sigs}
        try:
            logger.warning("interrupted; terminating running targets")
            pool.shutdown(wait=False, cancel_futures=True)
            _terminate_all_children()
            logger.info("all child processes stopped")
        finally:
            for s, handler in prev.items():
                signal.signal(s, handler)
        sys.exit(130)
    finally:
        pool.shutdown(wait=True)

    for target, reason, stderr in failures:
        tail = (stderr or "").strip().splitlines()[-20:]
        if tail:
            lines = [f"{target} ({reason}): last stderr lines"]
            lines.extend(f"  {line}" for line in tail)
            logger.error("\n".join(lines))
    return {target: reason for target, reason, _ in failures}, unsupported


def read_targets(catalog):
    """Unique koi_id values of the catalog, in catalog order"""
    import pandas as pd

    return pd.read_csv(catalog, usecols=["koi_id"]).koi_id.drop_duplicates().tolist()


def main(argv=None):
    """Run every catalog target; return exit code 1 if any failed, else 0"""
    start = time.monotonic()
    configure_runtime()
    args = build_parser().parse_args(argv)
    args.run_id = args.run_id or time.strftime("%Y%m%dT%H%M%S", time.gmtime())

    import alderaan
    from alderaan.utils.logger import setup_logging, teardown_logging

    input_dir = str(args.input_dir.resolve())
    output_dir = str(args.output_dir.resolve())
    catalog = str(args.catalog.resolve())
    log_dir = os.path.join(output_dir, args.run_id, "logs")
    log_path = setup_logging(log_dir, "batch", args.run_id, "INFO")

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
        catalog_copy = os.path.join(
            output_dir,
            args.run_id,
            f"alderaan_batch_{args.run_id}{args.catalog.suffix}",
        )
        shutil.copyfile(catalog, catalog_copy)
        logger.info("catalog copy: %s", catalog_copy)

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
        failed, unsupported = run_targets(
            tasks, args.jobs, args.job_timeout, args.launch_interval
        )
        summary = batch_summary(
            args.run_id,
            len(targets),
            failed,
            unsupported,
            log_dir,
            time.monotonic() - start,
        )
        logger.info("%s", summary)
    finally:
        teardown_logging()

    return 1 if failed else 0


def batch_summary(run_id, ntargets, failed, unsupported, log_dir, elapsed_s):
    """
    End-of-batch verdict block for the log

    Lists counts, each unsupported and failed target with its reason, the log
    path pattern, and elapsed time. Surrounding blank lines make the block stand
    out in the log.
    """
    nok = ntargets - len(failed) - len(unsupported)
    lines = [
        f"===== batch summary: run_id {run_id} =====",
        f"  targets:  {ntargets} ({nok} ok, {len(unsupported)} unsupported, "
        f"{len(failed)} failed)",
    ]
    for label, group in (
        ("unsupported (known limits)", unsupported),
        ("!!!!! FAILED !!!!!", failed),
    ):
        if group:
            lines.append(f"  {label}:")
            lines.extend(f"    {t:<10s}{reason}" for t, reason in sorted(group.items()))
    lines += [
        f"  logs:     {os.path.join(log_dir, f'alderaan_<target>_{run_id}.log')}",
        f"  elapsed:  {elapsed_s / 3600:.2f} hr",
    ]
    return "\n\n" + "\n".join(lines) + "\n"
