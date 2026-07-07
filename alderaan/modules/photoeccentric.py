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
import matplotlib.pyplot as plt
import numpy.polynomial.polynomial as poly

import corner



import pandas as pd
import astropy.constants as c
from astropy.table import Table


pi = np.pi
BIGG = 6.6743 * 10**(-8)


def parse_results(results_fits_format, target_id, results_dir, planet_index=0):
    """
    Parse results for a given target identifier and planet index.
    
    Args:
        results_fits_format: FITS format results data
        target_id: target identifier, e.g. 'EPIC-210508766', 'K01234', or 'TOI-700'
        planet_index: Planet index (default 0)
    """
    target_id = str(target_id)

    ttvs_file = os.path.join(results_dir, f'{target_id}_{planet_index:02d}_quick.ttvs')
    

    ttimes = Table.read(
        ttvs_file,
        format='ascii',
        delimiter='\t',
        names=['index', 'observed_time', 'model_time', 'uncertainty', 'flag'])

    transit_index = ttimes['index'].value
    model_times = ttimes['model_time'].value

    c0_name = f'C0_{planet_index}'
    c1_name = f'C1_{planet_index}'
    ror_name = f'ROR_{planet_index}'
    impact_name = f'IMPACT_{planet_index}'
    dur_name = f'DUR14_{planet_index}'

    C0 = results_fits_format[c0_name].value
    C1 = results_fits_format[c1_name].value

    centered_index = transit_index - transit_index[-1] // 2
    LegX = centered_index / (transit_index[-1] / 2)
    Leg0 = np.ones_like(LegX)
    ephem = model_times + np.outer(C0, Leg0) + np.outer(C1, LegX)
    t0, P = poly.polyfit(transit_index, ephem.T, 1)

    period = P
    rprs = results_fits_format[ror_name]
    impact = results_fits_format[impact_name]
    duration = results_fits_format[dur_name]

    planet_result = {
        'period': P,
        'epoch': t0,
        'rprs': rprs,
        'impact': impact,
        'duration': duration,
        'samples': results_fits_format,
        'ttimes': ttimes,
    }


    return planet_result


def calc_aRs(P, rho):
    """
    P : period [days]
    rho : stellar density [solar density]
    """
    P_ = P * 86400.       # [seconds]
    G = c.G.value
    rho_ = rho * 1.408
    
    
    return ((G*P_**2*rho_)/(3*pi))**(1.0/3)


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
    
    # print('rho_star', con * (num/den + b**2) ** 1.5)
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


def plot_ecc_corner(d, path, planet_index):
    """
    Plot eccentricity-omega corner plot.
    
    Args:
        d: DataFrame with OMEGA and ECC columns
        path: Directory path or file path for saving the plot
    """
    fs = np.vstack((np.array(d['OMEGA']), np.array(d['ECC']))).T
    omega_range = np.percentile(np.array(d['OMEGA']), [1,99])
    ecc_range = np.percentile(np.array(d['ECC']), [1,99])
    range = [omega_range, ecc_range]

    os.makedirs(path, exist_ok=True)
    
    output_file = path + 'ew_corner_' + str(planet_index) + '.png'

    plt.clf()
    corner.corner(fs, labels=['w', 'e'], show_titles=True, plot_contours=True, range=range);
    plt.savefig(output_file)
    plt.close()