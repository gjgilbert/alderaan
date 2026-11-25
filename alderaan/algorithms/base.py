"""Base class for algorithms that operate on a star, planets, and litecurve"""

import functools
import logging
import time
from copy import deepcopy

import numpy as np

from alderaan.data_models import LiteCurve, Planet, Star
from alderaan.errors import UnsupportedTargetError


def _log_perform(perform):
    """Wrap perform() to log its start, arguments and runtime"""
    # perform is defined in the class body: its module and qualname give the
    # subclass module's logger and the class name
    logger = logging.getLogger(perform.__module__)
    name = perform.__qualname__.rsplit(".", 1)[0]

    @functools.wraps(perform)
    def wrapper(self, *args, **kwargs):
        """Logged call of perform (functools.wraps replaces this docstring)"""
        params = [repr(a) for a in args] + [f"{k}={v!r}" for k, v in kwargs.items()]
        logger.info("start %s(%s)", name, ", ".join(params))
        t0 = time.perf_counter()
        out = perform(self, *args, **kwargs)
        logger.info("done %s in %.1f s", name, time.perf_counter() - t0)
        return out

    return wrapper


class BaseAlg:
    """
    Type-check and deepcopy inputs; provide shared transit bookkeeping

    Parameters
    ----------
    star : Star
    planets : list of Planet
    litecurve : LiteCurve

    Notes
    -----
    Subclasses take these same parameters (not repeated in their docstrings).
    perform() returns the updated (planets, litecurve) unless its docstring says
    otherwise. Each subclass's perform() is wrapped to log its start (with
    arguments) and runtime to the subclass module's logger; failures propagate to
    pipeline.py, which logs them.
    """

    def __init_subclass__(cls, **kwargs):
        """Wrap the subclass's own perform() with _log_perform"""
        super().__init_subclass__(**kwargs)
        if "perform" in cls.__dict__:
            cls.perform = _log_perform(cls.perform)

    def __init__(self, star, planets, litecurve):
        if not isinstance(litecurve, LiteCurve):
            raise TypeError("litecurve must be a LiteCurve")
        if not isinstance(planets, list):
            raise TypeError("planets must be a list of Planets")
        for p in planets:
            if not isinstance(p, Planet):
                raise TypeError("planets must be a list of Planets")
        if not isinstance(star, Star):
            raise TypeError("star must be a Star")

        self.litecurve = deepcopy(litecurve)
        self.npl = len(planets)
        self.planets = deepcopy(planets)
        self.star = deepcopy(star)

    def make_transit_mask(
        self,
        rel_size=0.0,
        abs_size=0.0,
        buffer=0.0,
        time=None,
        phot_only=False,
        max_period_fraction=None,
    ):
        """
        Mask cadences near a transit of any planet

        Each masked window has full width max(rel_size * T14, abs_size) + buffer

        Parameters
        ----------
        rel_size : float or array-like
            full width of masked region in units of transit duration (T14);
            one value for all planets or one per planet
        abs_size : float or array-like
            floor on rel_size * T14 in days, so short transits are not under-masked;
            one value for all planets or one per planet
        buffer : float or array-like
            extra full width in days added after the floor (e.g. cadence + TTVs);
            one value for all planets or one per planet
        time : ndarray
            cadence times to mask (default = self.litecurve.time)
        phot_only : bool
            mask only transits with good photometry (TransitTimes.phot_coverage
            and phot_noise), e.g. to select the photometry a fit uses
            (default False: every transit)
        max_period_fraction : float
            raise UnsupportedTargetError if a window is wider than this fraction
            of the planet's period (default None: no limit)

        Returns
        -------
        mask : ndarray, bool
            True for cadences near a transit of any planet
        """
        rel_size = np.broadcast_to(rel_size, self.npl)
        abs_size = np.broadcast_to(abs_size, self.npl)
        buffer = np.broadcast_to(buffer, self.npl)
        if np.any((rel_size == 0) & (abs_size == 0)):
            raise ValueError("either rel_size or abs_size must be nonzero")

        if time is None:
            time = self.litecurve.time

        mask = np.zeros((self.npl, len(time)), dtype=int)

        for n, p in enumerate(self.planets):
            half_width = 0.5 * (
                np.max([rel_size[n] * p.duration, abs_size[n]]) + buffer[n]
            )
            if max_period_fraction and 2 * half_width > max_period_fraction * p.period:
                raise UnsupportedTargetError(
                    f"planet {n}: transit mask width {2 * half_width:.3f} d exceeds "
                    f"{max_period_fraction:.3g} of the period ({p.period:.3f} d)"
                )
            tt = p.transit_times
            phot = tt.phot_coverage & tt.phot_noise
            ttime = tt.ttime_mod[phot] if phot_only else tt.ttime_mod
            for tc in ttime:
                mask[n] += np.abs(time - tc) < half_width

        return np.sum(mask, axis=0).astype(bool)

    def identify_overlapping_transits(self, rel_size=None, abs_size=None):
        """
        Flag transits whose midpoint is near a transit midpoint of another planet

        A transit overlaps if either tolerance (when given) is met.

        Parameters
        ----------
        rel_size : float
            tolerance on midpoint separation in units of the mean T14 of the two
            planets
        abs_size : float
            tolerance on midpoint separation in days

        Returns
        -------
        overlap : list of ndarray, bool
            one array per planet; True for transits that overlap another planet
        """
        if (rel_size is None) and (abs_size is None):
            raise ValueError("at least one of rel_size or abs_size must be provided")

        overlap = [None] * self.npl

        for i in range(self.npl):
            overlap[i] = np.zeros(
                len(self.planets[i].transit_times.ttime_mod), dtype=bool
            )
            ttime_i = self.planets[i].transit_times.ttime_mod

            for j in range(self.npl):
                if j == i:
                    continue
                for tc_j in self.planets[j].transit_times.ttime_mod:
                    if rel_size is not None:
                        overlap[i] |= np.abs(ttime_i - tc_j) < rel_size * 0.5 * (
                            self.planets[i].duration + self.planets[j].duration
                        )
                    if abs_size is not None:
                        overlap[i] |= np.abs(ttime_i - tc_j) < abs_size

        return overlap
