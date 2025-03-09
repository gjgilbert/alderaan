#!/usr/bin/env python
# coding: utf-8

# # Analyze Autocorrelated Noise


import os
import sys
import json
import shutil
import warnings
from   datetime import datetime
from   timeit import default_timer as timer

import argparse
import astropy.stats
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import numpy.polynomial.polynomial as poly
import pandas as pd
from   scipy import stats

import pymc3_ext as pmx
from   aesara_theano_fallback import aesara as theano


# #### Flush buffer and silence extraneous warnings


# flush buffer to avoid mixed outputs from progressbar
sys.stdout.flush()

# turn off FutureWarnings
warnings.filterwarnings('ignore', category=FutureWarning)

# supress UnitsWarnings (this code doesn't use astropy units)
warnings.filterwarnings(action='ignore', category=astropy.units.UnitsWarning, module='astropy')


# #### Initialize timer


print("")
print("+"*shutil.get_terminal_size().columns)
print("ALDERAAN Detrending and TTV Estimation")
print(f"Initialized {datetime.now().strftime('%d-%b-%Y at %H:%M:%S')}")
print("+"*shutil.get_terminal_size().columns)
print("")

# start program timer
global_start_time = timer()


# #### Parse inputs


try:
    parser = argparse.ArgumentParser(description="Inputs for ALDERAAN transit fiting pipeline")
    parser.add_argument("--mission", default=None, type=str, required=True, \
                        help="Mission name; can be 'Kepler' or 'Simulated'")
    parser.add_argument("--target", default=None, type=str, required=True, \
                        help="Target name; format should be K00000 or S00000")
    parser.add_argument("--project_dir", default=None, type=str, required=True, \
                        help="Project directory for accessing lightcurve data and saving outputs")
    parser.add_argument("--data_dir", default=None, type=str, required=True, \
                        help="Data directory for accessing MAST lightcurves")
    parser.add_argument("--catalog", default=None, type=str, required=True, \
                        help="CSV file containing input planetary parameters")
    parser.add_argument("--run_id", default=None, type=str, required=True, \
                        help="run identifier")
    parser.add_argument("--verbose", default=False, type=bool, required=False, \
                        help="'True' to enable verbose logging")
    parser.add_argument("--iplot", default=False, type=bool, required=False, \
                        help="'True' to enable interactive matplotlib backend; default 'agg'")

    args = parser.parse_args()
    MISSION      = args.mission
    TARGET       = args.target
    PROJECT_DIR  = args.project_dir
    DATA_DIR     = args.data_dir
    CATALOG      = args.catalog
    RUN_ID       = args.run_id
    VERBOSE      = args.verbose
    IPLOT        = args.iplot

except SystemExit:
    warnings.warn("No arguments were parsed from the command line")


print("")
print(f"   MISSION : {MISSION}")
print(f"   TARGET  : {TARGET}")
print(f"   RUN ID  : {RUN_ID}")
print("")
print(f"   Project directory : {PROJECT_DIR}")
print(f"   Data directory    : {DATA_DIR}")
print(f"   Input catalog     : {CATALOG}")
print("")
print(f"   theano cache : {theano.config.compiledir}")
print("")


# #### Build directory structure


# directories in which to place pipeline outputs for this run
RESULTS_DIR = os.path.join(PROJECT_DIR, 'Results', RUN_ID, TARGET)
FIGURE_DIR  = os.path.join(PROJECT_DIR, 'Figures', RUN_ID, TARGET)

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

sys.path.append(PROJECT_DIR)


# #### Import ALDERAAN routines


from alderaan.constants import scit, lcit
from alderaan.utils import boxcar_smooth
import alderaan.io as io
import alderaan.noise as noise
import alderaan.detrend as detrend


# #### Set matplotlib backends


if not IPLOT:
    mpl.use('agg')

if np.any(np.array(['agg', 'png', 'svg', 'pdf', 'ps']) == mpl.get_backend()):
    warnings.warn("Selected matplotlib backend does not support interactive plotting")
    IPLOT = False


# MAIN SCRIPT BEGINS HERE
def main():

    # # ################
    # # ----- DATA I/O -----
    # # ################


    print("\nLoading data...\n")


    # ## Read in planet and stellar properties
    #
    # ##### WARNING!!! Reference epochs are not always consistent between catalogs. If using DR25, you will need to correct from BJD to BJKD with an offset of 2454833.0 days - the cumulative exoplanet archive catalog has already converted epochs to BJKD


    # Read in the data from csv file
    if MISSION == 'Kepler':
        target_dict = pd.read_csv(os.path.join(PROJECT_DIR, 'Catalogs', CATALOG))
    elif MISSION == 'Simulated':
        target_dict = pd.read_csv(os.path.join(PROJECT_DIR, f'Simulations/{RUN_ID}/{RUN_ID}.csv'))
    else:
        raise ValueError("MISSION must be 'Kepler' or 'Simulated'")

    # set KOI_ID global variable
    if MISSION == 'Kepler':
        KOI_ID = TARGET
    elif MISSION == 'Simulated':
        KOI_ID = "K" + TARGET[1:]
    else:
        raise ValueError("MISSION must be 'Kepler' or 'Simulated'")

    # pull relevant quantities and establish GLOBAL variables
    use = np.array(target_dict['koi_id']) == KOI_ID

    KIC = np.array(target_dict['kic_id'], dtype='int')[use]
    NPL = np.array(target_dict['npl'], dtype='int')[use]
    PERIODS = np.array(target_dict['period'], dtype='float')[use]
    DEPTHS  = np.array(target_dict['depth'], dtype='float')[use]
    DURS = np.array(target_dict['duration'], dtype='float')[use]

    if MISSION == 'Kepler':
        DURS /= 24.  # [hrs] --> [days]

    # sort planet parameters by period
    order = np.argsort(PERIODS)

    PERIODS = PERIODS[order]
    DEPTHS  = DEPTHS[order]
    DURS    = DURS[order]

    # do some consistency checks
    if all(k == KIC[0] for k in KIC): KIC = KIC[0]
    else: raise ValueError("There are inconsistencies with KIC in the csv input file")

    if all(n == NPL[0] for n in NPL): NPL = NPL[0]
    else: raise ValueError("There are inconsistencies with NPL in the csv input file")


    # ## Read in detrended lightcurves and quick transit times
    # #### These can be generated by running the script "detrend_and_estimate_ttvs.py"


    if os.path.exists(os.path.join(RESULTS_DIR , f'{TARGET}_lc_detrended.fits')):
        lc = io.load_detrended_lightcurve(os.path.join(RESULTS_DIR, f'{TARGET}_lc_detrended.fits'))
        lc.season = lc.quarter % 4
    else:
        lc = None

    if os.path.exists(os.path.join(RESULTS_DIR , f'{TARGET}_sc_detrended.fits')):
        sc = io.load_detrended_lightcurve(os.path.join(RESULTS_DIR, f'{TARGET}_sc_detrended.fits'))
        sc.season = sc.quarter % 4
    else:
        sc = None


    # transit times
    epochs = np.zeros(NPL)
    periods = np.zeros(NPL)
    linear_ephemeris = [None]*NPL

    transit_inds = []
    indep_transit_times = []
    quick_transit_times = []

    for npl in range(NPL):
        fname_in = os.path.join(RESULTS_DIR, f'{TARGET}_{npl:02d}_quick.ttvs')
        data_in  = np.genfromtxt(fname_in)

        transit_inds.append(np.array(data_in[:,0], dtype='int'))
        indep_transit_times.append(np.array(data_in[:,1], dtype='float'))
        quick_transit_times.append(np.array(data_in[:,2], dtype='float'))

        # do a quick fit to get a linear ephemeris
        pfit = poly.polyfit(transit_inds[npl], quick_transit_times[npl], 1)

        epochs[npl] = pfit[1]
        periods[npl] = pfit[0]
        linear_ephemeris[npl] = poly.polyval(transit_inds[npl], pfit)

    if IPLOT:
        fig, axes = plt.subplots(NPL, figsize=(12,3*NPL))
        if NPL == 1: axes = [axes]

        for npl in range(NPL):
            xtime = linear_ephemeris[npl]
            yomc_i = (indep_transit_times[npl] - linear_ephemeris[npl])*24*60
            yomc_q = (quick_transit_times[npl] - linear_ephemeris[npl])*24*60

            axes[npl].plot(xtime, yomc_i, 'o', c='lightgrey')
            axes[npl].plot(xtime, yomc_q, lw=2, c=f'C{npl}')
            axes[npl].set_ylabel('O-C [min]', fontsize=20)
        axes[NPL-1].set_xlabel('Time [BJKD]', fontsize=20)
        plt.show()


    # # ####################
    # # --- PRELIMINARIES ---
    # # ####################


    print("\nRunning preliminaries...\n")


    # ## Establish time baseline


    time_min = []
    time_max = []

    if sc is not None:
        time_min.append(sc.time.min())
        time_max.append(sc.time.max())

    if lc is not None:
        time_min.append(lc.time.min())
        time_max.append(lc.time.max())

    TIME_START = np.min(time_min)
    TIME_END   = np.max(time_max)

    # put epochs in range (TIME_START, TIME_START + PERIOD)
    for npl in range(NPL):
        if epochs[npl] < TIME_START:
            adj = 1 + (TIME_START - epochs[npl])//periods[npl]
            epochs[npl] += adj*periods[npl]

        if epochs[npl] > (TIME_START + periods[npl]):
            adj = (epochs[npl] - TIME_START)//periods[npl]
            epochs[npl] -= adj*periods[npl]


    # ## Estimate TTV scatter w/ uncertainty buffer


    ttv_scatter = np.zeros(NPL)
    ttv_buffer  = np.zeros(NPL)

    for npl in range(NPL):
        # estimate TTV scatter
        ttv_scatter[npl] = astropy.stats.mad_std(indep_transit_times[npl]-quick_transit_times[npl])

        # based on scatter in independent times, set threshold so not even one outlier is expected
        N   = len(transit_inds[npl])
        eta = np.max([3., stats.norm.interval((N-1)/N)[1]])

        ttv_buffer[npl] = eta*ttv_scatter[npl] + lcit


    # ## Make masks of various widths


    if sc is not None:
        sc_wide_mask = np.zeros((NPL,len(sc.time)),dtype='bool')
        sc_narrow_mask = np.zeros((NPL,len(sc.time)),dtype='bool')
        sc_regular_mask = np.zeros((NPL,len(sc.time)),dtype='bool')

        for npl in range(NPL):
            tts = quick_transit_times[npl]
            wide_size = np.max([3/24,3.5*DURS[npl]]) + 2*ttv_buffer[npl]
            narrow_size = 0.5*DURS[npl] + ttv_buffer[npl]
            regular_size = np.max([3/24,1.5*DURS[npl]]) + 2*ttv_buffer[npl]

            sc_wide_mask[npl] = detrend.make_transitmask(sc.time, tts, wide_size)
            sc_narrow_mask[npl] = detrend.make_transitmask(sc.time, tts, narrow_size)
            sc_regular_mask[npl] = detrend.make_transitmask(sc.time, tts, regular_size)

        sc.mask = sc_regular_mask.sum(0) > 0

    else:
        sc_wide_mask = None
        sc_narrow_mask = None
        sc_regular_mask = None


    if lc is not None:
        lc_wide_mask = np.zeros((NPL,len(lc.time)),dtype='bool')
        lc_narrow_mask = np.zeros((NPL,len(lc.time)),dtype='bool')
        lc_regular_mask = np.zeros((NPL,len(lc.time)),dtype='bool')

        for npl in range(NPL):
            tts = quick_transit_times[npl]
            wide_size = np.max([3/24,5.5*DURS[npl]]) + 2*ttv_buffer[npl]
            narrow_size = 0.5*DURS[npl] + ttv_buffer[npl]
            regular_size = np.max([3/24,1.5*DURS[npl]]) + 2*ttv_buffer[npl]

            lc_wide_mask[npl] = detrend.make_transitmask(lc.time, tts, wide_size)
            lc_narrow_mask[npl] = detrend.make_transitmask(lc.time, tts, narrow_size)
            lc_regular_mask[npl] = detrend.make_transitmask(lc.time, tts, regular_size)

        lc.mask = lc_regular_mask.sum(0) > 0

    else:
        lc_wide_mask = None
        lc_narrow_mask = None


    # ## Determine what data type each season has


    season_dtype = []

    if sc is not None:
        sc_seasons = np.unique(sc.quarter%4)
    else:
        sc_seasons = np.array([])

    if lc is not None:
        lc_seasons = np.unique(lc.quarter%4)
    else:
        lc_seasons = np.array([])

    for z in range(4):
        if np.isin(z, sc_seasons):
            season_dtype.append('short')
        elif np.isin(z, lc_seasons):
            season_dtype.append('long')
        else:
            season_dtype.append('none')


    # # ####################################
    # # ----- AUTOCORRELATION ANALYSIS -----
    # # ####################################

    # ## ACF plotting function


    # generating figures inside imported modules creates issues with UChicago Midway RCC cluster
    # it's easier to just define the function here in the main script

    def plot_acf(xcor, acf_emp, acf_mod, xf, yf, freqs, target_name, season):
        fig = plt.figure(figsize=(20,5))

        plt.tight_layout()
        plt.subplots_adjust(wspace=0.8)

        ax = plt.subplot2grid(shape=(5,10), loc=(0,0), rowspan=3, colspan=7)
        ax.plot(xcor*24, acf_emp, color='lightgrey')
        ax.plot(xcor*24, acf_mod, c='red')
        ax.set_xlim(xcor.min()*24,xcor.max()*24)
        ax.set_xticks(np.arange(0,xcor.max()*24,2))
        ax.set_xticklabels([])
        ax.set_ylim(acf_emp.min()*1.1, acf_emp.max()*1.1)
        ax.set_ylabel('ACF', fontsize=20)
        ax.text(xcor.max()*24-0.15, acf_emp.max(), f'{target_name}, {season}', va='top', ha='right', fontsize=20)


        ax = plt.subplot2grid(shape=(5,10), loc=(0,7), rowspan=5, colspan=3)
        ax.plot(xf/24/3600*1e3, yf, color='k', lw=0.5)
        for f in freqs:
            ax.axvline(f/24/3600*1e3, color='red', zorder=0, lw=3, alpha=0.3)
        ax.set_xlim(xf.min()/24/3600*1e3, xf.max()/24/3600*1e3)
        ax.set_ylim(yf.min(),1.2*yf.max())
        ax.set_ylabel('Power', fontsize=20)
        ax.set_yticks([])
        ax.set_xlabel('Frequency [mHz]', fontsize=20)

        for i, sf in enumerate(np.sort(freqs)[::-1]):
            ax.text(xf.min()/24/3600*1e3+0.1, yf.max()*(1.1-0.1*i), f'{0:.2f} min'.format(24*60/sf), fontsize=16)


        ax = plt.subplot2grid(shape=(5,10), loc=(3,0), rowspan=2, colspan=7)
        ax.plot(xcor*24, acf_emp-acf_mod, c='lightgrey')
        ax.set_xlim(xcor.min()*24,xcor.max()*24)
        ax.set_xticks(np.arange(0,xcor.max()*24,2))
        ax.set_xlabel('Lag time [hours]', fontsize=20)
        ax.set_ylabel('Residuals', fontsize=20)

        return fig


    # ## Generate empirical ACF and filter high-frequency ringing


    print("\nGenerating empirical autocorrelation function...\n")
    print("Season data types:", season_dtype, "\n")

    # set cutoff between low and high frequency signals
    fcut = 2/lcit
    fmin = 2/(5*(DURS.max()+lcit))

    # short cadency Nyquist freqency
    fnyq = 1/(2*scit)

    # now estimate the ACF
    acf_lag = []
    acf_emp = []
    acf_mod = []
    acf_freqs = []

    for z in range(4):
        if season_dtype[z] == 'none':
            acf_lag.append(None)
            acf_emp.append(None)
            acf_mod.append(None)
            acf_freqs.append(None)

        else:
            if season_dtype[z] == 'short':
                npts = int(np.min([5*(1/24+DURS.max()),2/3*periods.min()])/scit)
                use = sc.season == z
                m_ = sc.mask[use]

                if np.sum(use) > 0:
                    t_ = sc.time[use][~m_]
                    f_ = sc.flux[use][~m_]
                    c_ = sc.cadno[use][~m_]

            if season_dtype[z] == 'long':
                npts = int(np.min([5*(1/24+DURS.max()),2/3*periods.min()])/lcit)
                use = lc.season == z
                m_ = lc.mask[use]

                if np.sum(use) > 0:
                    t_ = lc.time[use][~m_]
                    f_ = lc.flux[use][~m_]
                    c_ = lc.cadno[use][~m_]

            if np.sum(use) == 0:
                acf_lag.append(None)
                acf_emp.append(None)
                acf_mod.append(None)
                acf_freqs.append(None)

            else:
                # generate the empirical acf (if generate_acf fails, use very low amplitude white noise)
                try:
                    xcor, acor = noise.generate_acf(t_, f_, c_, npts)
                except Exception:
                    try:
                        npts = int(2/3*npts)
                        xcor, acor = noise.generate_acf(t_, f_, c_, npts)
                    except Exception:
                        xcor = 1 + np.arange(npts, dtype='float')
                        acor = np.random.normal(size=len(xcor))*np.std(f_)*np.finfo(float).eps

                if season_dtype[z] == 'long':
                    xcor = xcor*lcit
                    method = 'smooth'
                    window_length = 3

                if season_dtype[z] == 'short':
                    xcor = xcor*scit
                    method = 'savgol'
                    window_length = None

                # model the acf
                acor_emp, acor_mod, xf, yf, freqs = noise.model_acf(xcor, acor, fcut, fmin=fmin,
                                                                    method=method, window_length=window_length)

                # make some plots
                fig = plot_acf(xcor, acor_emp, acor_mod, xf, yf, freqs, TARGET, z)
                fig.savefig(os.path.join(FIGURE_DIR, TARGET + f'_ACF_season_{z}.png'), bbox_inches='tight')
                if IPLOT:plt.show()
                else: plt.close()

                # filter out high-frequency components in short cadence data
                if season_dtype[z] == 'short':
                    fring = list(freqs[(freqs > fcut)*(freqs < fnyq)])
                    bw = 1/(lcit-scit) - 1/(lcit+scit)

                    if len(fring) > 0:
                        # apply the notch filter
                        flux_filtered = detrend.filter_ringing(sc, 15, fring, bw)

                        # search for addtional ringing frequencies
                        try:
                            xcor, acor = noise.generate_acf(t_, flux_filtered[use][~m_], c_, npts)
                            xcor = xcor*scit
                        except Exception:
                            pass

                        new_freqs = noise.model_acf(xcor, acor, fcut, fmin=fmin, method='savgol')[4]
                        new_fring = new_freqs[(new_freqs > fcut)*(new_freqs < fnyq)]

                        for nf in new_fring:
                            if np.sum(np.abs(fring-nf) < bw) == 0:
                                fring.append(nf)

                        # re-apply the notch filter with the new list of ringing frequencies
                        flux_filtered = detrend.filter_ringing(sc, 15, fring, bw)

                        # update the LiteCurve
                        sc.flux[use] = flux_filtered[use]
                        f_ = sc.flux[use][~m_]

                    # re-run the ACF modeling on the filtered lightcurve
                    try:
                        xcor, acor = noise.generate_acf(t_, f_, c_, npts)
                        xcor = xcor*scit
                    except Exception:
                        pass

                    acor_emp, acor_mod, xf, yf, freqs = noise.model_acf(xcor, acor, fcut, fmin=fmin, method='savgol')

                # add to list
                acf_lag.append(xcor)
                acf_emp.append(acor_emp)
                acf_mod.append(acor_mod)
                acf_freqs.append(freqs)


    # ## Save filtered lightcurves


    print("\nSaving detrended lightcurves...\n")

    if lc is not None:
        lc.to_fits(TARGET, os.path.join(RESULTS_DIR, f'{TARGET}_lc_filtered.fits'), cadence='LONG')
    else:
        print("No long cadence data")

    if sc is not None:
        sc.to_fits(TARGET, os.path.join(RESULTS_DIR, f'{TARGET}_sc_filtered.fits'), cadence='SHORT')
    else:
        print("No short cadence data")


    # ## Generate synthetic noise


    print("\nGenerating synthetic noise...\n")

    synth_time = []
    synth_red  = []
    synth_white = []

    for z in range(4):
        print(f"SEASON {z} - {season_dtype[z]}")

        if season_dtype[z] == 'none':
            synth_time.append(None)
            synth_red.append(None)
            synth_white.append(None)

        else:
            if season_dtype[z] == 'short':
                npts = np.min([int(2*DURS.max()/scit), len(acf_emp[z])])
                use = sc.season == z
                m_ = sc.mask[use]

                if np.sum(use) > 0:
                    t_ = sc.time[use][~m_]
                    f_ = sc.flux[use][~m_]

            if season_dtype[z] == 'long':
                npts = np.min([int(5*DURS.max()/lcit), len(acf_emp[z])])
                use = lc.season == z
                m_ = lc.mask[use]

                if np.sum(use) > 0:
                    t_ = lc.time[use][~m_]
                    f_ = lc.flux[use][~m_]

            if np.sum(use) == 0:
                synth_time.append(None)
                synth_red.append(None)
                synth_white.append(None)

            else:
                if season_dtype[z] == 'long':
                    vector_length = 5*len(acf_lag[z])
                if season_dtype[z] == 'short':
                    vector_length = 2*len(acf_lag[z])

                # pull and split high/low frequencies
                freqs = np.copy(acf_freqs[z])

                low_freqs  = freqs[freqs <= fcut]
                high_freqs = freqs[freqs > fcut]

                # generate some synthetic correlated noise
                clipped_acf = (acf_mod[z][:npts])*np.linspace(1,0,npts)

                x, red_noise, white_noise = noise.generate_synthetic_noise(acf_lag[z][:npts], clipped_acf,
                                                                           vector_length, np.std(f_))

                # hacky fix for zero red noise
                if np.var(red_noise) == 0:
                    red_noise = 1e-6*boxcar_smooth(white_noise, 5)

                # add to list
                synth_time.append(x)
                synth_red.append(red_noise)
                synth_white.append(white_noise)

                # plot the noise
                plt.figure(figsize=(20,5))
                plt.plot(x, white_noise + red_noise, '.', c='lightgrey')
                plt.plot(x, red_noise, c='r', lw=4, label=f"{TARGET}, SEASON {z}")
                plt.axhline(DEPTHS.max(), c='k', ls=':', lw=2)
                plt.axhline(DEPTHS.min(), c='k', ls='--', lw=2)
                plt.axhline(-DEPTHS.min(), c='k', ls='--', lw=2)
                plt.axhline(-DEPTHS.max(), c='k', ls=':', lw=2)
                plt.xlim(x.min(),x.max())
                plt.ylim(np.percentile(white_noise,1), np.percentile(white_noise,99))
                plt.xlabel("Time [days]", fontsize=24)
                plt.ylabel("Flux", fontsize=24)
                plt.xticks(fontsize=14)
                plt.yticks(fontsize=14)
                plt.legend(fontsize=20, loc='upper right', framealpha=1)
                #plt.savefig(os.path.join(FIGURE_DIR, TARGET + f'_synthetic_noise_season_{z}.png'), bbox_inches='tight')
                if IPLOT:
                    plt.show()
                else:
                    plt.close()


    # ## Fit a GP to the synthetic noise


    print("\nFitting a GP to synthetic noise...\n")

    gp_priors = []

    for z in range(4):
        print(f"SEASON {z}")

        if season_dtype[z] == 'none':
            gp_priors.append(None)

        else:
            srz = synth_red[z]

            # pull and split high/low frequencies
            freqs = np.copy(acf_freqs[z])

            if freqs is not None:

                low_freqs  = freqs[freqs <= fcut]
                high_freqs = freqs[freqs > fcut]

                if len(low_freqs) > 0:
                    lf = low_freqs[0]
                else:
                    lf = None

                if len(high_freqs) > 0:
                    warnings.warn("there are remaining high-frequency noise components")

            else:
                lf = None


            # fit a GP model to the synthetic noise
            try:
                gp_model = noise.build_sho_model(synth_time[z],
                                                 srz + np.random.normal(srz)*np.std(srz)*0.1,
                                                 var_method = 'fit',
                                                 fmax = 2/lcit,
                                                 f0 = lf
                                                )

                with gp_model:
                    gp_map = gp_model.test_point

                    for mv in gp_model.vars:
                        gp_map = pmx.optimize(start=gp_map, vars=[mv], verbose=VERBOSE)

                    gp_map = pmx.optimize(start=gp_map, verbose=VERBOSE)

                    try:
                        gp_trace = pmx.sample(tune=6000, draws=1500, start=gp_map, chains=2, target_accept=0.9, progressbar=VERBOSE)
                    except Exception:
                        gp_trace = pmx.sample(tune=12000, draws=1500, start=gp_map, chains=2, target_accept=0.95, progressbar=VERBOSE)

            except Exception:
                gp_model = noise.build_sho_model(synth_time[z],
                                                 srz + np.random.normal(srz)*np.std(srz)*0.1,
                                                 var_method = 'local',
                                                 fmax = 2/lcit,
                                                 f0 = lf,
                                                 Q0 = 1/np.sqrt(2)
                                                )

                with gp_model:
                    gp_map = gp_model.test_point

                    for mv in gp_model.vars:
                        gp_map = pmx.optimize(start=gp_map, vars=[mv], verbose=VERBOSE)

                    gp_map = pmx.optimize(start=gp_map, verbose=VERBOSE)

                    try:
                        gp_trace = pmx.sample(tune=12000, draws=1500, start=gp_map, chains=2, target_accept=0.95, progressbar=VERBOSE)
                    except Exception:
                        gp_trace = gp_map

            # track the priors
            gp_priors.append(noise.make_gp_prior_dict(gp_trace))


    # ## Save GP posteriors for use as priors during simulatenous transit fit


    print("\nSaving GP posteriors...\n")

    for z in range(4):
        if gp_priors[z] is not None:
            for k in gp_priors[z].keys():
                gp_priors[z][k] = list(gp_priors[z][k])

            fname_out = os.path.join(RESULTS_DIR, f'{TARGET}_shoterm_gp_priors_{z}.txt')

            with open(fname_out, 'w') as file:
                json.dump(gp_priors[z], file)


    # ## Exit program


    print("")
    print("+"*shutil.get_terminal_size().columns)
    print(f"Analysis of autocorrelated noise complete {datetime.now().strftime('%d-%b-%Y at %H:%M:%S')}")
    print("Total runtime = %.1f min" %((timer()-global_start_time)/60))
    print("+"*shutil.get_terminal_size().columns)

if __name__ == '__main__':
    main()
