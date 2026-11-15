import logging
import os
import subprocess

__version__ = "0.2.2"

logger = logging.getLogger(__name__)


def _git_hash():
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
