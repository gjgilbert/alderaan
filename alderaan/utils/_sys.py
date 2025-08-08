import os
import sys
import gc
import matplotlib.pyplot as plt


def _system_cleanup():
    sys.stdout.flush()
    sys.stderr.flush()
    plt.close('all')
    gc.collect()