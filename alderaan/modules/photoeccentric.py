__all__ = ['TransitModel',
           'RBDTransitModel',
           'TTVTransitModel',
           'OptimizationTTVFitter',
           'CrossCorrelationTTVFitter',
          ]

import os
import sys

from copy import deepcopy
import io
import numpy as np
import time


import pandas as pd
import astropy.constants as c
from astropy.table import Table


pi = np.pi
BIGG = 6.6743 * 10**(-8)


def parse_results(results, epic_id, planet_index=0):
    """
    CURRENTLY HARD CODED
    This might need to be moved to a 'results.py' in the future
    """

    # if mission == K2:
    k2planets = Table.read("alderaan/examples/catalogs/k2_condensed_planets.csv")
    k2_hostrow = k2planets[(k2planets['epic_hostname'] == epic_id)] # & (toi_catalog['pl_orbper_rank'] == float(planet_index+1))]

    results_dir = 'alderaan/examples/outputs/quicklook/develop/EPIC-210508766/'
    ttvs_file = 'alderaan/examples/outputs/quicklook/develop/EPIC-210508766/ttvs/'

    ttimes = Table.read(
        ttvs_file,
        format='ascii',
        delimiter='\t',
        names=['index', 'observed_time', 'model_time', 'uncertainty', 'flag']

    transit_index = ttimes['index'].value
    model_times = ttimes['model_time'].value

    c0_name = f'C0_{planet_index}'
    c1_name = f'C1_{planet_index}'
    ror_name = f'ROR_{planet_index}'
    impact_name = f'IMPACT_{planet_index}'
    dur_name = f'DUR14_{planet_index}'

    if c0_name not in samples.colnames or c1_name not in samples.colnames:
        print(f"    Warning: missing ephemeris columns for planet {planet_index}, skipping")
        continue

    if ror_name not in samples.colnames or impact_name not in samples.colnames or dur_name not in samples.colnames:
        print(f"    Warning: missing transit parameters for planet {planet_index}, skipping")
        continue

    C0 = samples[c0_name].value
    C1 = samples[c1_name].value

    centered_index = transit_index - transit_index[-1] // 2
    LegX = centered_index / (transit_index[-1] / 2)
    Leg0 = np.ones_like(LegX)
    ephem = model_times + np.outer(C0, Leg0) + np.outer(C1, LegX)
    t0, P = poly.polyfit(transit_index, ephem.T, 1)

    period = P
    rprs = samples[ror_name]
    impact = samples[impact_name]
    duration = samples[dur_name]

    planet_result = {
        'period': P,
        'epoch': t0,
        'samples': samples,
        'ttimes': ttimes,
    }

    return planet_result


def calc_aRs(P, rho):
    """
    P : period [days]
    rho : stellar density [g/cm3]
    """
    P_   = P*86400.       # [seconds]
    rho_ = rho*1000.      # [kg/m3]
    G    = c.G.value

    return ((G*P_**2*rho_)/(3*pi))**(1./3)


def calc_rho_star(P, T14, b, ror, ecc, omega):
    '''
    Inverting T14 equation from Winn 2010 
    
    Args:
        P: period in units of days
        T14: duration in units of days
        b: impact parameter
        ror: radius ratio
        ecc: eccentricity
        omega: argument of periastron in radians
    Out:
        rho_star: stellar density in units of g/cc
    '''
    per = P * 86400.
    dur = T14 * 86400.

    con = (3*pi) / (BIGG * per**2)
    num = (1+ror)**2 - b**2
    arg = (pi*dur/per) * (1+ecc*np.sin(omega)) / np.sqrt(1-ecc**2)
    den = np.sin(arg)**2
    
    return con * (num/den + b**2) ** 1.5


def imp_sample_rhostar(period, dur, rprs, impact, rho_star, norm=True, return_log=False, ecut=None, params=[]):
    '''
    Perform importance sampling from {IMPACT, ROR, PERIOD, DUR14} to {ECC, OMEGA}
    
    Args
    ----
    samples [dataframe]: pandas dataframe of sampled data which includes: IMPACT, ROR, PERIOD, DUR14
    rho_star [tuple]: values of the true stellar density and its uncertainty
    norm [bool]: True to normalize weights before output (default=True)
    return_log [bool]: True to return ln(weights) instead of weights (default=False)
    ecut [float]: upper bound on the ecc prior between (0,1); default None will set to a/Rs * (1-e) > 1
    params [list]: list of values to be used as parameters for the indicated distribution
    
    Output:
    weights [array]: importance sampling weights
    data [dataframe]: pandas dataframe containing all input and derived data, including: 
                      ECC: random values drawn from 0 to 'ecut' according to 'distr' and 'params'
                      OMEGA: random values drawn from -pi/2 to 3pi/2 (with transit obs prior if 'ew_obs_prior'=True)
                      IMPACT: inputs values
                      ROR: inputs values
                      PERIOD: inputs values
                      DUR14: inputs values
                      RHOSTAR: derived values
                      WEIGHTS (or LN_WT): importance weights
    '''
    P   = period
    T14 = dur    
    ror = rprs
    b   = impact
    
    N = len(b)

    if ecut is None:
        ecut = 1 - 1/np.mean(calc_aRs(P, rho_star[0]))

    ecc = np.random.uniform(0., ecut, N)
    omega = np.random.uniform(-0.5*np.pi, 1.5*np.pi, N)
        
    rho_samp = calc_rho_star(P, T14, b, ror, ecc, omega)
    log_weights = -np.log(rho_star[1]) - 0.5*np.log(2*pi) - 0.5 * ((rho_samp - rho_star[0]) / rho_star[1]) ** 2
    
    # flag weights that are NaN-valued or below machine precision
    bad = np.isnan(log_weights) + (log_weights < np.log(np.finfo(float).eps))

    if np.sum(bad)/len(bad) < 0.05:
        raise ValueError("Fraction of viable samples is below 5%")
    
    # prepare outputs
    data = pd.DataFrame()
    data['PERIOD']  = P[~bad]
    data['ROR']     = ror[~bad]
    data['IMPACT']  = b[~bad]
    data['DUR14']   = T14[~bad]
    data['ECC']     = ecc[~bad]
    data['OMEGA']   = omega[~bad]
    data['RHOSTAR'] = rho_samp[~bad]

    if return_log:       
        data['LN_WT'] = log_weights[~bad]
        return log_weights, data

    else:
        weights = np.exp(log_weights[~bad] - np.max(log_weights[~bad]))
        
        if norm:
            weights /= np.sum(weights)
        data['WEIGHTS'] = weights
        return weights, data

fs = np.vstack((np.array(d['PERIOD']), np.array(d['ROR']), np.array(d['IMPACT']), np.array(d['DUR14']))).T #, np.array(d['LD_Q1']), np.array(d['LD_Q2']))).T

def plot_ecc_corner(d, path):

    fs = np.vstack((np.array(d['OMEGA']), np.array(d['ECC']))).T
    omega_range = np.percentile(np.array(d['OMEGA']), [1,99])
    ecc_range = np.percentile(np.array(d['ECC']), [1,99])
    range = [omega_range, ecc_range]

    plt.clf()
    corner.corner(fs, labels=['w', 'e'], show_titles=True, plot_contours=True, range=range);
    plt.suptitle('TOI ' + str(toi_plrow['toi'].value[0]), fontsize=30)
    plt.savefig(path + 'ew_corner.png')
    plt.close()