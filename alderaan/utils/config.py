"""Load TOML pipeline configs with optional in-memory section overrides"""

import os
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


def load_config(path, overrides=None):
    """
    Load a TOML config file as a nested dict

    Parameters
    ----------
    path : str
        path to the TOML config file
    overrides : dict, optional
        {section: {key: value}}; merged into each section, adding new sections

    Returns
    -------
    config : dict
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"config file not found: {path}")

    with open(path, "rb") as f:
        config = tomllib.load(f)

    for section, values in (overrides or {}).items():
        config.setdefault(section, {}).update(values)

    return config


def resolve_config_path(path_str, base_path):
    """Substitute {base_path} and return an absolute directory path ending in '/'"""
    return os.path.join(str(Path(path_str.format(base_path=base_path)).resolve()), "")
