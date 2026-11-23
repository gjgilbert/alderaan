__all__ = ["build_transit_model", "remove_known_transits", "inject_synthetic_transits"]


import aesara_theano_fallback.tensor as T
import numpy as np
import exoplanet as exo
import pymc3 as pm

from .constants import *



def build_transit_model(planets, data, limbdark, texp, oversample):
    """
    Docstring
    """
    npl = len(planets)
    
    tts  = [None]*npl
    inds = [None]*npl
    ror  = np.zeros(npl)
    b    = np.zeros(npl)
    dur  = np.zeros(npl)

    for n, p in enumerate(planets):
        tts[n]  = p.tts
        inds[n] = p.index
        ror[n] = np.sqrt(p.depth)
        b[n]   = p.impact
        dur[n] = p.duration
        
    starrystar = exo.LimbDarkLightCurve([limbdark[0],limbdark[1]])

    orbit = exo.orbits.TTVOrbit(transit_times=tts, 
                                transit_inds=inds, 
                                ror=ror,
                                b=b, 
                                duration=dur
                               )
    
    model_flux = [None]*len(data)
    
    for i, d in enumerate(data):
        light_curve = starrystar.get_light_curve(orbit=orbit, 
                                                 r=ror, 
                                                 t=d.time, 
                                                 oversample=oversample, 
                                                 texp=texp,
                                                )
        
        model_flux[i] = pm.math.sum(light_curve, axis=-1) + T.ones(len(d.time))
        
    return model_flux


def remove_known_transits(planets, data, limbdark, texp, oversample):
    """
    Docstring
    """
    model_flux = build_transit_model(planets, data, limbdark, texp, oversample)
    
    for i, d in enumerate(data):
        d.flux /= model_flux[i].eval()
        
    return data


def inject_synthetic_transits(planets, data, limbdark, texp, oversample):
    """
    Docstring
    """
    model_flux = build_transit_model(planets, data, limbdark, texp, oversample)
    
    for i, d in enumerate(data):
        d.flux *= model_flux[i].eval()
        
    return data