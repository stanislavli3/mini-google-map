"""
src/_path_bootstrap.py
======================
Shared path setup. Importing this module:
  1. Adds every `src/<subdir>/` to `sys.path`, so cross-subdir sibling
     imports work as if all source files lived in a single flat folder.
  2. Exposes `PROJECT_ROOT` as the absolute path to the repo root —
     entry-point scripts should `os.chdir(PROJECT_ROOT)` so that
     relative paths in the code (e.g. `'sf_road_network.graphml'`,
     `'Trajectories'`, `'submission.csv'`) resolve consistently no matter
     where the script was invoked from.

Usage at the top of any source file:

    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
    import _path_bootstrap as _pb
    PROJECT_ROOT = _pb.PROJECT_ROOT
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))   # this file lives in src/
PROJECT_ROOT = os.path.dirname(_HERE)

for _sub in os.listdir(_HERE):
    _p = os.path.join(_HERE, _sub)
    if os.path.isdir(_p) and not _sub.startswith(("_", ".")):
        if _p not in sys.path:
            sys.path.insert(0, _p)
