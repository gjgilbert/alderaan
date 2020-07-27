import numpy as np
import astropy.constants as apc

# define constants
pi = np.pi

RJRE = (apc.R_jup/apc.R_earth).value       # R_jup/R_earth
RSRE = (apc.R_sun/apc.R_earth).value       # R_sun/R_earth
RSRJ = RSRE/RJRE                           # R_sun/R_jup

MJME = (apc.M_jup/apc.M_earth).value       # M_jup/M_earth
MSME = (apc.M_sun/apc.M_earth).value       # M_sun/M_earth
MSMJ = MSME/MJME                           # M_sun/M_jup

RSAU = (apc.R_sun/apc.au).value            # solar radius [AU]

LCIT = 29.42239340627566    # Kepler long cadence integration time [min]
SCIT = 58.84478681255132    # Kepler short cadence integration time [sec]