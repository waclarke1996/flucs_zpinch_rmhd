#!/usr/bin/env python3
"""
Run an offline tool script unchanged (as __main__) and keep the local variables of its main() at return.
Usage: capture_tool.py OUT.pkl TOOL.py [tool args ...]
Saves the numpy arrays / numbers / small lists and dicts of main()'s locals with pickle.
"""
import os
import pickle
import runpy
import sys

import numpy as np

out, tool, args = sys.argv[1], sys.argv[2], sys.argv[3:]
captured = {}


def prof(frame, event, arg):
    if event == "return" and frame.f_code.co_name == "main" and os.path.samefile(frame.f_code.co_filename, tool):
        captured.update(frame.f_locals)


def keep(v):
    if isinstance(v, (np.ndarray, float, int, np.floating, np.integer, str)):
        return True
    if isinstance(v, (list, tuple)) and len(v) < 10000:
        try:
            pickle.dumps(v)
            return True
        except Exception:
            return False
    if isinstance(v, dict):
        try:
            pickle.dumps(v)
            return True
        except Exception:
            return False
    return False


sys.argv = [tool, *args]
sys.setprofile(prof)
try:
    runpy.run_path(tool, run_name="__main__")
finally:
    sys.setprofile(None)
    sys.stdout = sys.__stdout__
data = {}
for k, v in captured.items():
    if hasattr(v, "get") and type(v).__module__.startswith("cupy"):
        v = v.get()
    if keep(v):
        data[k] = v
with open(out, "wb") as fh:
    pickle.dump(data, fh)
print(f"captured {len(data)} locals of {os.path.basename(tool)} -> {out}")
