import os
import sys

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if base_path not in sys.path:
    sys.path.insert(0, base_path)

from aesara_theano_fallback import aesara as theano
import argparse
import astropy
from astropy.stats import mad_std
from celerite2.backprop import LinAlgError
from configparser import ConfigParser
from datetime import datetime
import gc
import matplotlib.pyplot as plt
import numpy as np
import shutil
from alderaan.constants import *
from alderaan.schema.ephemeris import Ephemeris
from alderaan.schema.litecurve import LiteCurve
from alderaan.schema.planet import Planet
from alderaan.modules.detrend import GaussianProcessDetrender
from alderaan.modules.omc import OMC
from alderaan.modules.transit_model.transit_model import ShapeTransitModel, TTimeTransitModel
from alderaan.modules.quality_control import QualityControl
from alderaan.modules.quicklook import plot_litecurve, plot_omc, dynesty_cornerplot, dynesty_runplot, dynesty_traceplot
from alderaan.utils.io import expand_config_path, parse_koi_catalog, parse_holczer16_catalog, copy_input_target_catalog
from timeit import default_timer as timer
import warnings


def initialize_pipeline():
    # flush buffer
    sys.stdout.flush()
    sys.stderr.flush()

    # filter warnings
    warnings.simplefilter('always', UserWarning)
    warnings.filterwarnings(
        action='ignore', category=astropy.units.UnitsWarning, module='astropy'
    )

    # start timer
    global_start_time = timer()

    return global_start_time


def cleanup():
    sys.stdout.flush()
    sys.stderr.flush()
    plt.close('all')
    gc.collect()


def main():
    # initialize program
    global_start_time = initialize_pipeline()

    print("")
    print("+" * shutil.get_terminal_size().columns)
    print("ALDERAAN Pipeline")
    print(f"Initialized {datetime.now().strftime('%d-%b-%Y at %H:%M:%S')}")
    print("+" * shutil.get_terminal_size().columns)
    print("")

    # read inputs from config
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mission', required=True, type=str, help='Observing mission')
    parser.add_argument('-t', '--target', required=True, type=str, help='Target ID for star')
    parser.add_argument('-c', '--config', required=True, type=str, help='Path to config file')
    args = parser.parse_args()

    config = ConfigParser()
    config.read(os.path.join(base_path, args.config))

    mission = args.mission
    target = args.target
    run_id = config['RUN']['run_id']

    data_dir =  expand_config_path(config['PATHS']['data_dir'])
    outputs_dir = expand_config_path(config['PATHS']['outputs_dir'])
    catalog_dir = expand_config_path(config['PATHS']['catalog_dir'])

    catalog_csv = os.path.join(catalog_dir, str(config['ARGS']['catalog_csv']))

    print("")
    print(f"   MISSION : {mission}")
    print(f"   TARGET  : {target}")
    print(f"   RUN ID  : {run_id}")
    print("")
    print(f"   Base path         : {base_path}")
    print(f"   Data directory    : {data_dir}")
    print(f"   Config file       : {args.config}")
    print(f"   Input catalog     : {os.path.basename(catalog_csv)}")
    print("")
    print(f"   theano cache : {theano.config.compiledir}")
    print("")

    # build directory structure
    os.makedirs(outputs_dir, exist_ok=True)

    results_dir = os.path.join(outputs_dir, 'results', run_id, target)
    os.makedirs(results_dir, exist_ok=True)

    quicklook_dir = os.path.join(outputs_dir, 'quicklook', run_id, target)
    os.makedirs(quicklook_dir, exist_ok=True)

    # copy input catalog into results directory
    catalog_csv_copy = os.path.join(outputs_dir, 'results', run_id, f'{run_id}.csv')
    copy_input_target_catalog(catalog_csv, catalog_csv_copy)


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
    filepath = os.path.join(catalog_dir, 'holczer_2016_kepler_ttvs.txt')
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
    cleanup()

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
    cleanup()

    print(f"\ncumulative runtime = {((timer()-global_start_time)/60):.1f} min")


    # ################ #
    # DETRENDING BLOCK #
    # ################ #

    print('\n\nDETRENDING BLOCK (1st pass)\n')

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

        print(f"  Quarter {detrender.litecurve.quarter[0]} : {npts_initial-npts_final} outliers rejected")

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

    osc_per_mu = np.nanmedian(oscillation_periods)
    osc_per_sd = np.max([mad_std(oscillation_periods, ignore_nan=True), 0.1*osc_per_mu])

    print(f"\nNominal stellar oscillation period = {osc_per_mu:.1f} +/- {osc_per_sd:.1f} days")

    # detrend the litecurves
    for j, detrender in enumerate(detrenders):
        print(f"\nDetrending {j+1} of {len(detrenders)} litecurves", flush=True)
        
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
        transit_mask = detrender.make_transit_mask(rel_size=3.0, abs_size=2/24, mask_type='condensed')
        
        # call detrender.detrend(), using successively simpler models as fallbacks
        try:
            litecurves[j] = detrender.detrend(
                'RotationTerm',
                np.nanmedian(oscillation_periods),
                min_period,
                transit_mask=transit_mask,
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
                    transit_mask=transit_mask,
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
                    transit_mask=transit_mask,
                    gap_tolerance=gap_tolerance,
                    jump_tolerance=jump_tolerance,
                    correct_ramp=False,
                    return_trend=False, 
                    progressbar=False
                )

    # recombine litecurves
    litecurve = LiteCurve(litecurves)

    # quicklook litecurve
    filepath = os.path.join(quicklook_dir, f"{target}_litecurve_detrended.png")
    _ = plot_litecurve(litecurve, target, planets, filepath)

    # end-of-block cleanup
    cleanup()

    print(f"\ncumulative runtime = {((timer()-global_start_time)/60):.1f} min")


    # ##################### #
    # QUALITY CONTROL BLOCK #
    # ##################### #

    print("\n\nQUALITY CONTROL BLOCK")

    qc = QualityControl(litecurve, planets)

    # check for transits with poor photometric coverage
    with warnings.catch_warnings(record=True) as catch:
        warnings.simplefilter('always', category=RuntimeWarning)
        coverage = qc.check_coverage()

    # check for transits with unusually high noise
    with warnings.catch_warnings(record=True) as catch:
        warnings.simplefilter('always', category=RuntimeWarning)
        good_rms = qc.check_rms(rel_size=3.0, abs_size=2/24, sigma_cut=5.0)

    for n, p in enumerate(planets):
        print(f"\n  Planet {n}:")

        assert len(coverage[n]) == len(p.ephemeris.ttime)
        assert len(good_rms[n]) == len(p.ephemeris.ttime)

        _nbad = np.sum(~coverage[n])
        _ntot = len(coverage[n])
        print(f"    {np.sum(_nbad)} of {_ntot} transits ({int(100*_nbad/_ntot)}%) rejected for insufficent photometric coverage")

        _nbad = np.sum(~good_rms[n] & coverage[n])
        _ntot = len(good_rms[n])
        print(f"    {np.sum(_nbad)} of {_ntot} transits ({int(100*_nbad/_ntot)}%) rejected for high photometric noise")
        
        planets[n].ephemeris.quality = coverage[n] & good_rms[n]

    # end-of-block cleanup
    cleanup()

    print(f"\ncumulative runtime = {((timer()-global_start_time)/60):.1f} min")


    # ################### #
    # TRANSIT MODEL BLOCK #
    # ################### #

    print('\n\nTRANSIT MODEL BLOCK\n')

    limbdark = [catalog.limbdark_1[0], catalog.limbdark_2[0]]
    transitmodel = ShapeTransitModel(litecurve, planets, limbdark)

    print("Supersample factor")
    for obsmode in transitmodel.unique_obsmodes:
        print(f"  {obsmode} : {transitmodel._obsmode_to_supersample(obsmode)}")

    print("\nFitting initial transit model")
    theta = transitmodel.optimize()
    planets = transitmodel.update_planet_parameters(theta)
    limbdark = transitmodel.update_limbdark_parameters(theta)

    print("\nFitting independent transit times")
    ttvmodel = TTimeTransitModel(litecurve, planets, limbdark)

    for n, p in enumerate(planets):
        print(f"\nPlanet {n} : fitting {np.sum(p.ephemeris.quality)} transit times")

        ttime_new, ttime_err_new = ttvmodel.mazeh13_holczer16_method(n, quicklook_dir=quicklook_dir)

        assert len(ttime_new) == len(ttime_err_new)
        assert len(ttime_new) == len(p.ephemeris.ttime)

        _nfit = np.sum(~np.isnan(ttime_new))
        _ntot = np.sum(p.ephemeris.quality)

        print(f"Planet {n} : {_nfit} of {_ntot} transit times ({_nfit / _ntot * 100:.1f}%) fit successfully")

    print("\nSampling with DynamicNestedSampler")
    results = transitmodel.sample(progress_every=10)

    filepath = os.path.join(quicklook_dir, f'{koi_id}_dynesty_runplot.png')
    fig, ax = dynesty_runplot(results, koi_id, filepath=filepath)

    for n, p in enumerate(transitmodel.planets):
        filepath = os.path.join(quicklook_dir, f'{koi_id}_dynesty_traceplot_{n:02d}.png')
        fig, ax = dynesty_traceplot(results, koi_id, n, filepath=filepath)

        filepath=os.path.join(quicklook_dir, f'{koi_id}_dynesty_cornerplot_{n:02d}.png')
        fig, ax = dynesty_cornerplot(results, koi_id, n, filepath=filepath, interactive=True)


if __name__ == '__main__':
    main()