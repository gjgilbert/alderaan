import logging
import os
import subprocess

__version__ = "0.2.2"

# random seed for every stochastic step, so runs are reproducible (Kepler's birth date)
SEED = 15711227

logger = logging.getLogger(__name__)


def git_hash():
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(__file__),
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as e:
        logger.debug("git hash unavailable: %s", e)
        return "unknown"
