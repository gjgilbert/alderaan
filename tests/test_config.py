import os
import sys
from pathlib import Path

base_path = Path(__file__).resolve().parents[1]
if str(base_path) not in sys.path:
    sys.path.insert(0, str(base_path))

print(base_path)
