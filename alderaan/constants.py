import astropy.constants as apc
import numpy as np

pi = np.pi

RJRE = (apc.R_jup / apc.R_earth).value  # R_jup/R_earth
RSRE = (apc.R_sun / apc.R_earth).value  # R_sun/R_earth
RSRJ = RSRE / RJRE  # R_sun/R_jup

MJME = (apc.M_jup / apc.M_earth).value  # M_jup/M_earth
MSME = (apc.M_sun / apc.M_earth).value  # M_sun/M_earth
MSMJ = MSME / MJME  # M_sun/M_jup

RSAU = (apc.R_sun / apc.au).value  # solar radius [AU]
RHOSUN_GCM3 = (
    3 * apc.M_sun / (4 * pi * apc.R_sun**3)
).value / 1000  # solar density [g/cm^3]

kepler_lcit = 29.4243885 / 60 / 24  # Kepler long cadence integration time + readout time [days]
kepler_scit = 58.848777 / 3600 / 24 # Kepler short cadence integration time + readout time [days]