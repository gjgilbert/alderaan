##########################
# - Download from MAST - #
##########################

# This script downloads lightcurves for a single target star from the MAST archive using lightkurve.py
# To use, specify the KOI identifier (e.g. 'K00137') and a directory <PRIMARY_DIR> to place the downloaded fits files into. 
# Lightkurve creates a file structure based on each object's KIC identifier. 
# These files can then be read back in for detrending and modeling in other scripts.

import argparse
import lightkurve as lk
import numpy as np
import warnings

# parse inputs
parser = argparse.ArgumentParser(description="Inputs for ALDERAAN transit fiting pipeline")

parser.add_argument("--mission", default=None, type=str, required=True,
                    help="Mission name")
parser.add_argument("--target", default=None, type=str, required=True,
                    help="Target name; see ALDERAAN documentation for acceptable formats")
parser.add_argument("--primary_dir", default=None, type=str, required=True,
                    help="Primary directory for project; should end in '/'")

args = parser.parse_args()
MISSION     = args.mission
TARGET      = args.target
PRIMARY_DIR = args.primary_dir

# directory in which to place MAST downloads
DOWNLOAD_DIR = PRIMARY_DIR + 'MAST_downloads/'

# make a target name lightkurve and MAST can understand
MAST_TARGET = 'KOI-'+ str(int(TARGET[1:]))

print("")
print(MAST_TARGET)

##########################
# - SHORT CADENCE DATA - #
##########################

print('downloading short cadence data from MAST')

# this creates a LightCurveCollection of KeplerLightCurves
sc_searchresult = lk.search_lightcurve(MAST_TARGET, cadence='short', mission='Kepler')

if len(sc_searchresult) > 0:
    sc_rawdata = sc_searchresult.download_all(download_dir=DOWNLOAD_DIR)
else:
    print("...no short cadence data found")
    sc_rawdata = []

kic_ids = []
sc_quarters = []

for i, scrd in enumerate(sc_rawdata):
    kic_ids.append(scrd.meta['KEPLERID'])
    sc_quarters.append(scrd.meta['QUARTER'])
    
# check that all lightcurves are from the same object
if len(kic_ids) > 0:
    if np.sum(np.array(kic_ids) != kic_ids[0]):
        raise ValueError("Search results returned data from multiple objects")

# here's the list of quarters w/ short cadence data
sc_quarters = np.sort(np.unique(sc_quarters))

# make a list of quarters where short cadence flux is unavailable
qlist = np.arange(18, dtype='int')
keep  = ~np.isin(qlist, np.unique(sc_quarters))
lc_quarters = list(qlist[keep])

#########################
# - LONG CADENCE DATA - #
#########################

print('downloading long cadence data from MAST')

# this creates a LightCurveCollection of KeplerLightCurves
lc_searchresult = lk.search_lightcurve(MAST_TARGET, cadence='long', mission='Kepler', quarter=lc_quarters)

if len(lc_searchresult) > 0:
    lc_rawdata = lc_searchresult.download_all(download_dir=DOWNLOAD_DIR)
else:
    print("...no long cadence data found")
    lc_rawdata = [] 

kic_ids = []

for i, lcrd in enumerate(lc_rawdata):
    kic_ids.append(lcrd.meta['KEPLERID'])
    
# check that all lightcurves are from the same object
if len(kic_ids) > 0:
    if np.sum(np.array(kic_ids) != kic_ids[0]):
        raise ValueError("Search results returned data from multiple objects")