import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from aesara_theano_fallback import aesara as theano
import astropy
from astropy.stats import mad_std
from datetime import datetime
import gc
import matplotlib.pyplot as plt
import numpy as np
from alderaan.utils.io import parse_koi_catalog, parse_holczer16_catalog
from alderaan.schema.planet import Planet
from alderaan.schema.ephemeris import Ephemeris
from alderaan.schema.litecurve import LiteCurve
from alderaan.modules.omc import OMC
from alderaan.modules.quicklook import plot_litecurve, plot_omc
import shutil
from timeit import default_timer as timer
import warnings

# #################### #
# Initialization Block #
########################

# flush buffer
sys.stdout.flush()
sys.stderr.flush()

# filter warnings
warnings.simplefilter('always', UserWarning)
warnings.filterwarnings(
    action='ignore', category=astropy.units.UnitsWarning, module='astropy'
)

# start program timer
global_start_time = timer()

print("")
print("+" * shutil.get_terminal_size().columns)
print("ALDERAAN Pipeline")
print(f"Initialized {datetime.now().strftime('%d-%b-%Y at %H:%M:%S')}")
print("+" * shutil.get_terminal_size().columns)
print("")

# hard-code inputs
mission = 'Kepler'
target = 'K00148'
run_id = 'develop'

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
data_dir = os.path.join(project_dir, 'tests/testdata/')
catalog_csv = os.path.join(project_dir, 'tests/catalogs/kepler_dr25_gaia_dr2_crossmatch.csv')

print("")
print(f"   MISSION : {mission}")
print(f"   TARGET  : {target}")
print(f"   RUN ID  : {run_id}")
print("")
print(f"   Project directory : {project_dir}")
print(f"   Data directory    : {data_dir}")
print(f"   Input catalog     : {catalog_csv}")
print("")
print(f"   theano cache : {theano.config.compiledir}")
print("")

# build directory structure
outputs_dir = os.path.join(project_dir, 'outputs')
os.makedirs(outputs_dir, exist_ok=True)

results_dir = os.path.join(outputs_dir, 'results', run_id, target)
os.makedirs(results_dir, exist_ok=True)

quicklook_dir = os.path.join(outputs_dir, 'quicklook', run_id, target)
os.makedirs(quicklook_dir, exist_ok=True)

# ######### #
# I/O Block #
# ######### #

print('\n\nI/O BLOCK\n')

# load KOI catalog
catalog = parse_koi_catalog(catalog_csv, target)

assert np.all(np.diff(catalog.period) > 0), "Planets should be ordered by ascending period"

NPL = int(catalog.npl[0])
koi_id = catalog.koi_id[0]
kic_id = int(catalog.kic_id[0])

# load lightcurves
litecurve_master = LiteCurve(data_dir, kic_id, 'long cadence', data_source='Kepler PDCSAP')

t_min = litecurve_master.time.min()
t_max = litecurve_master.time.max()
if t_min < 0:
    raise ValueError("Lightcurve has negative timestamps...this will cause problems")

# split litecurves by quarter
litecurves = litecurve_master.split_quarters()

for j, litecurve in enumerate(litecurves):
    assert len(np.unique(litecurve.quarter)) == 1, "expected one quarter per litecurve"
    assert len(np.unique(litecurve.obsmode)) == 1, "expected one obsmode per litecurve"

print(f"{len(litecurves)} litecurves loaded for {target}")

# initialize planets (catch no ephemeris warning)
with warnings.catch_warnings(record=True) as catch:
    warnings.simplefilter('always', category=UserWarning)
    planets = [None]*NPL
    for n in range(NPL):
        planets[n] = Planet(catalog, target, n)

print(f"\n{NPL} planets loaded for {target}")
print([np.round(p.period,6) for p in planets])

# update planet ephemerides
for n, p in enumerate(planets):
    if p.ephemeris is None:
        _ephemeris = Ephemeris(period=p.period, epoch=p.epoch, t_min=t_min, t_max=t_max)
        planets[n] = p.update_ephemeris(_ephemeris)

# load Holczer+2016 catalog
filepath = os.path.join(project_dir, 'Catalogs/holczer_2016_kepler_ttvs.txt')
holczer_ephemerides = parse_holczer16_catalog(filepath, koi_id, NPL)

print(f"\n{len(holczer_ephemerides)} ephemerides found in Holczer+2016")

# match Holczer ephemerides to Planets
count = 0

for n, p in enumerate(planets):
    for ephem in holczer_ephemerides:
        match = np.isclose(ephem.period, p.period, rtol=0.01, atol=p.duration)

        if match:
            print(f"  Planet {n} : {p.period:.6f} --> {ephem.period:.6f}")
            planets[n] = p.update_ephemeris(ephem)
            count += 1

print(f"{count} matching ephemerides found ({len(holczer_ephemerides)} expected)")

# quicklook litecurve
filepath = os.path.join(quicklook_dir, f"{target}_litecurve_raw.png")
_ = plot_litecurve(litecurve_master, target, planets, filepath)

# end-of-block cleanup
sys.stdout.flush()
sys.stderr.flush()
plt.close('all')
gc.collect()

print(f"\ncumulative runtime = {((timer()-global_start_time)/60):.1f} min")


# ######### #
# OMC Block #
# ######### #

print('\n\nOMC BLOCK (initialization)\n')
print("regularizing ephemerides")

# initialize OMC object for each planet
omc_list = []
for n, p in enumerate(planets):
    omc_list.append(OMC(p.ephemeris))

# fit a regularized model
for n, p in enumerate(planets):
    omc = omc_list[n]
    npts = np.sum(omc.quality)

    _period = np.copy(p.period)

    # Matern-3/2 model | don't use GP on very noisy data
    if (npts >= 8) & (np.median(omc.yerr) <= 0.5 * mad_std(omc.yobs)):
        with warnings.catch_warnings(record=True) as catch:
            warnings.simplefilter('always', category=RuntimeWarning)
            trace = omc.sample(omc.matern32_model())

    # Polynomial model | require 2^N transits
    else:
        polyorder = np.max([1, np.min([3, int(np.log2(npts))-1])])
        with warnings.catch_warnings(record=True) as catch:
            warnings.simplefilter('always', category=RuntimeWarning)
            trace = omc.sample(omc.poly_model(polyorder))

    if len(catch) > 0:
        print(f"{len(catch)} RuntimeWarnings caught during sampling")

    # update ephemeris
    omc.ymod = np.nanmedian(trace['pred'], 0)
    omc_list[n] = omc

    p.ephemeris = p.ephemeris.update_from_omc(omc)
    p.ephemeris = p.ephemeris.interpolate('spline', full=True)
    planets[n] = p.update_ephemeris(p.ephemeris)

    # make quicklook plot
    filepath = os.path.join(quicklook_dir, f"{target}_omc_initial.png")
    _ = plot_omc(omc_list, target, filepath)

    print(f"Planet {n} : {_period:.6f} --> {planets[n].period:.6f}")

# end-of-block cleanup
sys.stdout.flush()
sys.stderr.flush()
plt.close('all')
gc.collect()

print(f"\ncumulative runtime = {((timer()-global_start_time)/60):.1f} min")

print("passing")