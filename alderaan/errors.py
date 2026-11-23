"""Exceptions for known limits of the pipeline"""


class UnsupportedTargetError(Exception):
    """The pipeline cannot process this target by design (a known limit, not a bug)"""
