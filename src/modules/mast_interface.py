import numpy as np
import pandas as pd

DR25_CATALOG = '/Users/research/projects/alderaan/Catalogs/kepler_dr25_gaia_dr2_crossmatch.csv'
dr25_catalog = pd.read_csv(DR25_CATALOG, index_col=0)

KOI_IDs = ['K00003', 'K00586', 'K00922', 'K00929']

KIC_IDs = []

for i, koi in enumerate(KOI_IDs):
    use = dr25_catalog.koi_id == koi
    
    kic = str(dr25_catalog.loc[use, 'kic_id'].values[0])
    
    KIC_IDs.append(kic)


targets = ' '.join(KIC_IDs)
args = '-c long --cmdtype wget'

command = 'python /Users/research/projects/alderaan/bin/get_kepler_data.py {0} {1}'.format(targets, args)

print(command)
#python /Users/research/projects/alderaan/bin/get_kepler_data.py 10748390 9570741 8826878 9141746 -c long --cmdtype wget

# kplr010748390-2009131105131_llc.fits


#### TUESDAY MORNING INSTRUCTIONS
# user supplied K01234 (KOI_ID) and target file directory
# use catalog to convert K01234 (KOI_ID) -> 12345678 (KIC_ID)
# python get_kepler_data.py 9141746 -c long --cmdtype wget   | creates get_kepler_data.sh
# get_kepler_data.sh | each line is a wget command for an expects kplr*.fits file
# parse get_kepler_data.sh to determine what .fits files are expected
# check for these files on disk
# if they don't exist, download them to desired directory using wget commands
# post: run a check that all files successfully downloaded and make another attempt if needed
# log success/failure
