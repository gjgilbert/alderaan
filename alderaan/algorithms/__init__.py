"""Pipeline stages, each a BaseAlg subclass with a perform() method"""

from alderaan.algorithms.clip_outliers import ClipOutliers
from alderaan.algorithms.detrend_visits import DetrendVisits
from alderaan.algorithms.flag_bad_transits import FlagBadTransits
from alderaan.algorithms.quick_fit_transit_shape import QuickFitTransitShape
from alderaan.algorithms.quick_fit_transit_times import QuickFitTransitTimes
from alderaan.algorithms.regularize_omc import RegularizeOMC
from alderaan.algorithms.sample_transit_model import SampleTransitModel

__all__ = [
    "ClipOutliers",
    "DetrendVisits",
    "FlagBadTransits",
    "QuickFitTransitShape",
    "QuickFitTransitTimes",
    "RegularizeOMC",
    "SampleTransitModel",
]
