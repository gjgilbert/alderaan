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