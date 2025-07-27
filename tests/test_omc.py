import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from astropy.stats import mad_std
import numpy as np
from src.utils.io import parse_koi_catalog, parse_holczer16_catalog
from src.schema.planet import Planet
from src.schema.ephemeris import Ephemeris
from src.modules.omc.omc import OMC
import warnings

import pymc3 as pm
import pymc3_ext as pmx

warnings.simplefilter('always', UserWarning)

koi_id = 'K00148'

# load KOI catalog
filepath = '/data/user/gjgilbert/projects/alderaan/Catalogs/kepler_dr25_gaia_dr2_crossmatch.csv'
catalog = parse_koi_catalog(filepath, koi_id)

assert np.all(np.diff(catalog.period) > 0), "Planets should be ordered by ascending period"

NPL = int(catalog.npl[0])
planets = [None]*int(catalog.npl[0])

print(f"{NPL} planets loaded for {koi_id}")

for i in range(NPL):
    planets[i] = Planet(catalog, koi_id, i)
    print(i, planets[i].period)

# load Holczer+2016 catalog
filepath = '/data/user/gjgilbert/projects/alderaan/Catalogs/holczer_2016_kepler_ttvs.txt'
holczer_ephemerides = parse_holczer16_catalog(filepath, koi_id, NPL)

print(f"{len(holczer_ephemerides)} ephemerides found in Holczer+2016")

for i, ephem in enumerate(holczer_ephemerides):
    print(i, ephem.period)

# match Holczer ephemerides to Planets
count = 0

for n, p in enumerate(planets):
    for ephem in holczer_ephemerides:
        match = np.isclose(ephem.period, p.period, rtol=0.1, atol=p.duration)

        if match:
            planets[n] = p.update_ephemeris(ephem)
            count += 1

print(f"{count} matching ephemerides found for {NPL} planets")

for n, p in enumerate(planets):
    print(n, p.period)

# initialize OMC object for each planet
omc_list = []
for n, p in enumerate(planets):
    omc_list.append(OMC(p.ephemeris))

# identify significant frequencies
if NPL == 1:
    critical_fap = 0.1
elif NPL > 1:
    critical_fap = 0.99

for n, p in enumerate(planets):
    omc = omc_list[n]
    omc.peakfreq, omc.peakfap = omc.identify_significant_frequencies(critical_fap)

    print(n, p.period, omc.peakfreq, omc.peakfap)

# in multi-planet systems, check for closely matching, low significance frequencies
if NPL > 1:
    freqs = [None]*NPL
    faps = [None]*NPL
    close = [False]*NPL

    for i in range(NPL):
        if omc_list[i].peakfreq is not None:
            freqs[i] = omc_list[i].peakfreq
            faps[i] = omc_list[i].peakfap
        else:
            freqs[i] = np.nan
            faps[i] = np.nan

    for i in range(NPL):
        if faps[i] > 0.1:
            df_min = 1 / (omc_list[i].xtime.max() - omc_list[i].xtime.min())
            for j in range(i+1, NPL):
                df_ij = np.abs(freqs[i]-freqs[j])
                if df_ij < df_min:
                    close[i] = True
                    close[j] = True

    for i in range(NPL):
        if not close[i]:
            omc_list[i].peakfreq = None
            omc_list[i].peakfap = None

for n, p in enumerate(planets):
    if omc_list[n].peakfreq is not None:
        print(f"Planet {n}: periodic signal found at Pttv = {1/omc_list[n].peakfreq:.1f} d")
    else:
        print(f"Planet {n}: no significant periodic component found")

# perform OMC model selection
for n, p in enumerate(planets):
    print(f"\nPlanet {n}: P = {p.period:.1f}")

    omc = OMC(p.ephemeris)
    npts = np.sum(omc.quality)

    trace = {}
    dof = {}

    # polynomial model | require 2^N transits
    max_polyorder = np.max([1,np.min([3, int(np.log2(npts))-1])])
    for polyorder in range(1, max_polyorder+1):
        trace[f'poly{polyorder}'] = omc.sample(omc.poly_model(polyorder))
        dof[f'poly{polyorder}'] = polyorder + 1
    
    # sinusoidal model | require at least 8 transits
    if npts >= 8:
        trace['sinusoid'] = omc.sample(omc.sin_model())
        dof['sinusoid'] = 3

    # Matern-3/2 model | don't use GP on very noisy data
    if (npts >= 8) & (np.median(omc.yerr) <= 0.5 * mad_std(omc.yomc)):
        trace['matern32'] = omc.sample(omc.matern32_model())
        dof['matern32'] = np.nanmedian(trace['matern32']["dof"])

    # model selection
    best_model = omc.select_best_model(trace, dof)

    print(f"Adopting {best_model} ephemeris")

    # calculate outlier probability using a gaussian mixture model
    ymod = np.nanmedian(trace[best_model]['pred'], 0)
    fg_prob, out = omc.calculate_outlier_probability(ymod)

    print(f"{np.sum(out):.0f} outliers found out of {len(out):.0f} transit times ({np.sum(out)/len(out)*100:.1f}%)")
    print(f"measured error: {np.median(omc.yerr[~out])*24*60:1.f} min")
    print(f"residual RMS: {mad_std(omc.yerr[~out] - ymod[~out])*24*60:.1f}")

    # update Ephemeris
    p.ephemeris.model = ymod
    p.ephemeris.quality = ~out
    p.ephemeris.fg_prob = fg_prob

    planets[n] = p.update_ephemeris(p.ephemeris)

print("passing")