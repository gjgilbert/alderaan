import os
import sys

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if base_path not in sys.path:
    sys.path.insert(0, base_path)

import argparse
from src.recipes.subrecipes import startup
#from src.recipes.subrecipes import load_kepler_data


def main():
    print("starting main recipe")

    startup.execute()






if __name__ == '__main__':
    main()