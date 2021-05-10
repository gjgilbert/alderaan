#!/usr/bin/env python
# coding: utf-8

# This script downloads lightcurves for a single target from the MAST archive using Lightkurve. To use, specify the KOI identifier (e.g. 'K00137') and a directory (PRIMARY_DIR) to place the downloaded fits files into. Lightkurve creates a file structure based on each object's KIC identifier. These files can then be read back in using lk.search.open() in the script "main_pipeline.py".
# 
# This two-step procedure is necessary because the University of Chicago midway.rcc server does not allow connections to the internet on compute nodes. However, internet access is granted on the login node, so this script ("download_from_MAST.py") should first be run on the login node, after which the data reduction pipeline ("main_pipeline.py") can be run on the compute node using a slurm batch request.

# In[1]:


import numpy as np

import csv
import sys
import os
import importlib as imp
import glob
import warnings
import argparse

import lightkurve as lk

from alderaan.constants import *
from alderaan.utils import *
from alderaan.LiteCurve import *
import alderaan.io as io

# flush buffer to avoid mixed outputs from progressbar
sys.stdout.flush()

# turn off FutureWarnings
warnings.filterwarnings('ignore', category=FutureWarning)


# # Manually set I/O parameters

# In[2]:


# select mission, target, and paths
#MISSION = "Kepler"
#TARGET  = "K00351"
#PRIMARY_DIR = '/Users/research/projects/alderaan/'


# In[3]:


# here's where we parse the inputs
try:
    parser = argparse.ArgumentParser(description="Inputs for ALDERAAN transit fiting pipeline")
    parser.add_argument("--mission", default=None, type=str, required=True,                         help="Mission name")
    parser.add_argument("--target", default=None, type=str, required=True,                         help="Target name; see ALDERAAN documentation for acceptable formats")
    parser.add_argument("--primary_dir", default=None, type=str, required=True,                         help="Primary directory path for accessing lightcurve data and saving outputs")

    args = parser.parse_args()
    MISSION     = args.mission
    TARGET      = args.target
    PRIMARY_DIR = args.primary_dir
    
except:
    pass


# In[4]:


# directory in which to place MAST downloads
DOWNLOAD_DIR = PRIMARY_DIR + 'MAST_downloads/'

# make a target name lightkurve and MAST can understand
MAST_TARGET = 'KOI-'+ str(int(TARGET[1:]))


print("")
print(MAST_TARGET)


# # Download the data from MAST

# In[5]:


# download the SHORT CADENCE data -- this creates a LightCurveFileCollection
print('downloading short cadence data from MAST')
sc_rawdata = lk.search_lightcurvefile(MAST_TARGET, cadence='short',                                       mission='Kepler').download_all(download_dir=DOWNLOAD_DIR)


# In[6]:


# clean up the SHORT CADENCE data
try:
    sc_data = io.cleanup_lkfc(sc_rawdata, KIC)

    # identify which quarters are found in short cadence
    sc_qlist = []
    for i, scq in enumerate(sc_data):
        sc_qlist.append(scq.quarter)
    
except:
    sc_data = None
    sc_qlist = []
    
    
# make a list of quarters where short cadence flux is unavailable
qlist = np.arange(18)
keep  = ~np.isin(qlist, np.unique(sc_qlist))
lc_qlist = qlist[keep]


# In[7]:


# download the LONG CADENCE data -- this creates a LightCurveFileCollection
print('downloading long cadence data from MAST')
lc_rawdata = lk.search_lightcurvefile(MAST_TARGET, cadence='long', quarter=lc_qlist,                                       mission='Kepler').download_all(download_dir=DOWNLOAD_DIR)    


# In[ ]:




