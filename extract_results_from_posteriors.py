#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np

import astropy
from   astropy.io import fits as pyfits
import csv
import sys
import os
import glob
from   timeit import default_timer as timer
import warnings

from alderaan.constants import *
import alderaan.io as io

# turn off FutureWarnings
warnings.filterwarnings('ignore', category=FutureWarning)

# start program timer
global_start_time = timer()


# # Set preliminaries

# In[2]:


# select mission, target, and paths
MISSION = "Simulated"
PRIMARY_DIR = '/home/gjgilbert/projects/alderaan/'
TARGET_FILE = PRIMARY_DIR + "Temp/target_list-sim-BIG-eccentric.txt"
OUTPUT_FILE = PRIMARY_DIR + "Catalogs/injection_and_recovery_results-BIG-ecc-v1.csv"


# In[3]:


TRACE_DIR = PRIMARY_DIR + "Traces/"

# locations of trace .fits files (IN) and .csv catalog (OUT)
if MISSION == "Simulated":
    trace_files = glob.glob(TRACE_DIR + "S*/*_transit_shape.fits")
    csv_outfile = OUTPUT_FILE
    

# list of targets
with open(TARGET_FILE) as tfile:
    target_list = tfile.read().splitlines()


# In[4]:


# only read results from objects on target_list
sim_ids = []

for i, tf in enumerate(trace_files):
    sim_ids.append(tf[-25:-19])

keep = np.isin(sim_ids, target_list)
trace_files = list(np.array(trace_files)[keep])


# # Build dictionary or results

# In[8]:


results = {}

results["koi_id"] = []
results["npl"] = []

results["rstar"] = []
results["mstar"] = []
results["limbdark_1"] = []
results["limbdark_2"] = []

results["epoch"] = []
results["period"] = []

results["prad"] = []
results["impact"] = []

for z in range(4):
    results["logsw4_{0}".format(z)] = []
    results["logw0_{0}".format(z)] = []
    results["logq_{0}".format(z)] = []


# In[9]:


for i, tf in enumerate(trace_files):
    with pyfits.open(tf) as trace:
        header  = trace[0].header
        hdulist = pyfits.HDUList(trace)
        
        print(header["TARGET"])
        
        
        # target/sampler info
        KOI_ID = "K" + header["TARGET"][1:]
        NDRAWS, NPL = trace['RP'].shape
    
        # stellar parameters
        RSTAR  = trace['RSTAR'].data
        MSTAR  = trace['MSTAR'].data
        U      = trace['U'].data
        U1, U2 = U[:,0], U[:,1]
    
        # planetary parameters
        T0   = trace['T0'].data
        P    = trace['P'].data
        RP   = trace['RP'].data * RSRE    # [R_earth]
        B    = trace['B'].data
        

        # GP parameters
        LOGSW4 = np.zeros((NDRAWS,4))
        LOGW0  = np.zeros((NDRAWS,4))
        LOGQ   = np.zeros((NDRAWS,4))

        for z in range(4):
            try: LOGSW4[:,z] = trace['LOGSW4_{0}'.format(z)].data
            except: pass

            try: LOGW0[:,z] = trace['LOGW0_{0}'.format(z)].data
            except: pass

            try: LOGQ[:,z] = trace['LOGQ_{0}'.format(z)].data
            except: pass
        
        
        for npl in range(NPL):
            results["koi_id"].append(KOI_ID)
            results["npl"].append(NPL)
            
            results["rstar"].append(np.percentile(RSTAR, 50))
            
            try: results["mstar"].append(np.percentile(MSTAR, 50))
            except: pass

            results["limbdark_1"].append(np.percentile(U1, 50))
            results["limbdark_2"].append(np.percentile(U2, 50))

            results["epoch"].append(np.percentile(T0[:,npl], 50))
            results["period"].append(np.percentile(P[:,npl], 50))

            results["prad"].append(np.percentile(RP[:,npl], 50))
            #results["prade_err3m"].append(np.percentile(RP[:,npl],  0.135))
            #results["prade_err2m"].append(np.percentile(RP[:,npl],  2.275))
            #results["prade_err1m"].append(np.percentile(RP[:,npl], 15.865))
            #results["prade_err1p"].append(np.percentile(RP[:,npl], 84.135))
            #results["prade_err2p"].append(np.percentile(RP[:,npl], 97.725))
            #results["prade_err3p"].append(np.percentile(RP[:,npl], 99.865))

            results["impact"].append(np.percentile(B[:,npl], 50))
            #results["impact_err3m"].append(np.percentile(B[:,npl],  0.135))
            #results["impact_err2m"].append(np.percentile(B[:,npl],  2.275))
            #results["impact_err1m"].append(np.percentile(B[:,npl], 15.865))
            #results["impact_err1p"].append(np.percentile(B[:,npl], 84.135))
            #results["impact_err2p"].append(np.percentile(B[:,npl], 97.725))
            #results["impact_err3p"].append(np.percentile(B[:,npl], 99.865))
            
            
            for z in range(4):
                results["logsw4_{0}".format(z)].append(np.median(LOGSW4[:,z]))
                results["logw0_{0}".format(z)].append(np.median(LOGSW4[:,z]))
                results["logq_{0}".format(z)].append(np.median(LOGSW4[:,z]))


# In[7]:


for k in results.keys():
    if k != "koi_id":
        results[k] = np.array(results[k], dtype="float")


for k in results.keys():
    if k != "koi_id":
        results[k] = np.round(results[k], 5) 


# # Write out the results catalog

# In[ ]:


WRITENEW = True
if WRITENEW:
    with open(csv_outfile, "w") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(results.keys())
        writer.writerows(zip(*results.values()))