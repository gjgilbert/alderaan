import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.io import parse_holczer16_catalog

filepath = '/data/user/gjgilbert/projects/alderaan/Catalogs/holczer_2016_kepler_ttvs.txt'
koi_id = 'K00148'
num_planets = 3

holczer = parse_holczer16_catalog(filepath, koi_id, num_planets)

for n, ephem in enumerate(holczer):
    print(n)

    if ephem is not None:
        print(ephem.period)