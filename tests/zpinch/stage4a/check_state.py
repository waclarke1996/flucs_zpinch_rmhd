#!/usr/bin/env python3
"""
check_state.py -- validate a FLUCS restart file before the chain relies on it (Stage 4a storage chain).

    check_state.py FILE --input INPUT.toml [--t-min T] [--t-max T]
exit 0 and prints "OK t=<current_time> dt=<current_dt>" if ALL hold:
  * the file opens as netCDF and has current_time, current_dt, fields_real, fields_imag;
  * the field arrays have shape (4, nz, nx, ny//2 + 1) of the input's padded grid;
  * every field value is finite and current_dt > 0;
  * t_min <= current_time <= t_max (when given).
otherwise prints "BAD <reason>" and exits 1. Reads the arrays in z-chunks (memory ~ one chunk).
"""
import argparse
import re
import sys

import numpy as np


def dims_from_input(path):
    txt = open(path).read()
    sec = re.search(r"^\[dimensions\](.*?)(?=^\[)", txt, re.S | re.M).group(1)
    get = lambda k: int(re.search(rf"^\s*{k}\s*=\s*(\d+)", sec, re.M).group(1))  # noqa: E731
    return get("nx"), get("ny"), get("nz")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--input", required=True)
    ap.add_argument("--t-min", type=float, default=-np.inf)
    ap.add_argument("--t-max", type=float, default=np.inf)
    a = ap.parse_args()
    try:
        from netCDF4 import Dataset
        nx, ny, nz = dims_from_input(a.input)
        want = (4, nz, nx, ny // 2 + 1)
        with Dataset(a.file) as ds:
            t = float(ds["current_time"][...])
            dt = float(ds["current_dt"][...])
            for v in ("fields_real", "fields_imag"):
                var = ds[v]
                if tuple(var.shape) != want:
                    return bad(f"{v} shape {tuple(var.shape)} != {want}")
                for z0 in range(0, nz, 64):
                    if not np.all(np.isfinite(var[:, z0:z0 + 64])):
                        return bad(f"{v} has non-finite values (z {z0}..)")
    except Exception as e:  # noqa: BLE001 - any failure to read = invalid file
        return bad(f"cannot read: {type(e).__name__}: {e}")
    if not (np.isfinite(t) and dt > 0):
        return bad(f"t = {t}, dt = {dt}")
    if not (a.t_min <= t <= a.t_max):
        return bad(f"time {t!r} outside [{a.t_min!r}, {a.t_max!r}]")
    print(f"OK t={t!r} dt={dt!r}")
    return 0


def bad(msg):
    print(f"BAD {msg}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
