import os
import sys

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if base_path not in sys.path:
    sys.path.insert(0, base_path)

import argparse
from alderaan.recipes.subrecipes import startup
from types import SimpleNamespace

def main():
    print("starting main recipe")

    context = startup.execute()







if __name__ == '__main__':
    main()