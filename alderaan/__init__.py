"""ALDERAAN: transit photometry pipeline for TTVs and transit shapes"""

import logging
import os
import subprocess

__version__ = "0.3.0"

logger = logging.getLogger(__name__)


def git_hash():
    """Short HEAD hash; '-dirty' if the package source has uncommitted changes"""

    def git(*args):
        return subprocess.run(
            ["git", *args],
            cwd=os.path.dirname(__file__),
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

    try:
        dirty = git("status", "--porcelain", "--", ".") != ""
        return git("rev-parse", "--short", "HEAD") + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError) as e:
        logger.debug("git hash unavailable: %s", e)
        return "unknown"
