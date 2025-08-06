import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import astropy
from astropy.stats import mad_std
from celerite2.backprop import LinAlgError
import numpy as np
from alderaan.constants import *
from alderaan.schema.ephemeris import Ephemeris
from alderaan.schema.litecurve import LiteCurve
from alderaan.schema.planet import Planet
from alderaan.modules.detrend import GaussianProcessDetrender
from alderaan.utils.io import parse_koi_catalog
import warnings

warnings.simplefilter('always', UserWarning)
warnings.filterwarnings(
    action='ignore', category=astropy.units.UnitsWarning, module='astropy'
)

koi_id = 'K00148'

# load KOI catalog
filepath = '/data/user/gjgilbert/projects/alderaan/Catalogs/kepler_dr25_gaia_dr2_crossmatch.csv'
catalog = parse_koi_catalog(filepath, koi_id)

assert np.all(np.diff(catalog.period) > 0), "Planets should be ordered by ascending period"

NPL = int(catalog.npl[0])
kic_id = int(catalog.kic_id[0])
planets = [None]*NPL

print(f"{NPL} planets loaded for {koi_id}")

for i in range(NPL):
    planets[i] = Planet(catalog, koi_id, i)
    print(i, planets[i].period)


# load lightcurves
data_dir = '/data/user/gjgilbert/data/MAST_downloads/'
kic_id = catalog.kic_id[0]

litecurve_master_raw = LiteCurve().load_kplr_pdcsap(data_dir, kic_id, 'long cadence')
litecurve_list_raw = litecurve_master_raw.split_quarters()

for i, lc in enumerate(litecurve_list_raw):
    lc = lc.remove_flagged_cadences(bitmask='default')

litecurve_master = LiteCurve().from_list(litecurve_list_raw)

# check for negative timestamps
t_min = litecurve_master.time.min()
t_max = litecurve_master.time.max()

if t_min < 0:
    raise ValueError("Lightcurve has negative timestamps...this will cause problems")

# update planet ephemerides
for n, p in enumerate(planets):
    if p.ephemeris is None:
        _ephemeris = Ephemeris(period=p.period, epoch=p.epoch, t_min=t_min, t_max=t_max)
        planets[n] = p.update_ephemeris(_ephemeris)

# split litecurves by quarter
litecurves = litecurve_master.split_quarters()

for j, litecurve in enumerate(litecurves):
    assert len(np.unique(litecurve.quarter)) == 1, "expected one quarter per litecurve"
    assert len(np.unique(litecurve.obsmode)) == 1, "expected one obsmode per litecurve"

print(f"{len(litecurves)} litecurves loaded")


# ################ #
# DETRENDING BLOCK #
# ################ #

# initialize detrenders
detrenders = []
for j, litecurve in enumerate(litecurves):
    detrenders.append(GaussianProcessDetrender(litecurve, planets))

# clip outliers
for j, detrender in enumerate(detrenders):    
    mask = detrender.make_transit_mask(rel_size=3.0, abs_size=2/24, mask_type='condensed')
    
    npts_initial = len(detrender.litecurve.time)

    detrender.clip_outliers(kernel_size=13, sigma_upper=5, sigma_lower=5, mask=mask)
    detrender.clip_outliers(kernel_size=13, sigma_upper=5, sigma_lower=1000, mask=None)
    
    npts_final = len(detrender.litecurve.time)

    print(f"Quarter {litecurve.quarter[0]} : {npts_initial-npts_final} outliers rejected")

# estimate oscillation periods
oscillation_periods = np.zeros(len(detrenders))
for j, detrender in enumerate(detrenders):
    obsmode = detrender.litecurve.obsmode[0]

    if obsmode == 'short cadence':
        min_period = np.max([5 * np.max(detrender.durs), 91 * kepler_scit])
    elif obsmode == 'long cadence':
        min_period = np.max([5 * np.max(detrender.durs), 13 * kepler_lcit])
    else:
        raise ValueError(f"unsuported obsmode: {obsmode}")

    oscillation_periods[j] = detrender.estimate_oscillation_period(min_period=min_period)

print(np.nanmedian(oscillation_periods))
print(mad_std(oscillation_periods, ignore_nan=True))

# detrend the litecurves
for j, detrender in enumerate(detrenders):
    print(f"Detrending {j+1} of {len(detrenders)} litecurves", flush=True)
    
    # set detrender arguments based on observing mode
    obsmode = detrender.litecurve.obsmode[0]

    if obsmode == 'short cadence':
        min_period = np.max([5 * np.max(detrender.durs), 91 * kepler_scit])
        gap_tolerance = np.max([int(np.min(detrender.durs) / kepler_scit * 5 / 2), 91])
        jump_tolerance = 5.0
    elif obsmode == 'long cadence':
        min_period = np.max([5 * np.max(detrender.durs), 13 * kepler_lcit])
        gap_tolerance = np.max([int(np.min(detrender.durs) / kepler_lcit * 5 / 2), 13])
        jump_tolerance = 5.0
    else:
        raise ValueError(f"unsuported obsmode: {obsmode}")
    
    # make transit mask
    mask = detrender.make_transit_mask(rel_size=3.0, abs_size=2/24, mask_type='condensed')
    
    # call detrender.detrend(), using successively simpler models as fallbacks

    try:
        litecurves[j] = detrender.detrend(
            'RotationTerm',
            np.nanmedian(oscillation_periods),
            min_period,
            transit_mask=mask,
            gap_tolerance=gap_tolerance,
            jump_tolerance=jump_tolerance,
            correct_ramp=True,
            return_trend=False, 
            progressbar=False
        )
    except LinAlgError:
        warnings.warn(
            "Initial detrending failed...attempting to refit without exponential ramp component"
        )

        try:
            litecurves[j] = detrender.detrend(
                'RotationTerm',
                np.nanmedian(oscillation_periods),
                min_period,
                transit_mask=mask,
                gap_tolerance=gap_tolerance,
                jump_tolerance=jump_tolerance,
                correct_ramp=False,
                return_trend=False, 
                progressbar=False
            )
        except LinAlgError:
            warnings.warn(
                "Detrending with RotationTerm failed...attempting to detrend with SHOTerm"
            )
            litecurves[j] = detrender.detrend(
                'SHOTerm',
                np.nanmedian(oscillation_periods),
                min_period,
                transit_mask=mask,
                gap_tolerance=gap_tolerance,
                jump_tolerance=jump_tolerance,
                correct_ramp=False,
                return_trend=False, 
                progressbar=False
            )

print("passing")