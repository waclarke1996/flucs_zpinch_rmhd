#!/usr/bin/env python3
"""
compress_state.py -- losslessly compress a FLUCS restart/state file in place (Stage 4 disk policy).

    compress_state.py FILE [--level 1]

The padded spectral arrays are ~70 % exact zeros (2/3 dealiasing): zlib level 1 + shuffle reduces a T1 state
from 273 MB to 79 MB in ~6 s. Steps (crash-safe; FILE is never missing or partial):
  1. copy FILE -> FILE.ztmp (NETCDF4, every variable with ndim > 0 zlib + shuffle; attributes, groups kept);
  2. re-open both and compare every variable BITWISE (chunked over the second axis) and the global attributes;
  3. os.replace(FILE.ztmp, FILE) (atomic rename in the same folder).
A file whose field variables are already compressed is left alone (exit 0, "already compressed").
Prints "OK <before> MB -> <after> MB" or "BAD <reason>" (exit 1; FILE untouched, FILE.ztmp removed).
FLUCS reads restarts with netCDF4, so a compressed file is a valid restart_from (tested in lanetest).
"""
import argparse
import os
import sys

import numpy as np
from netCDF4 import Dataset


def _copy(src, dst, level):
    for k in src.ncattrs():
        dst.setncattr(k, src.getncattr(k))
    for n, d in src.dimensions.items():
        dst.createDimension(n, None if d.isunlimited() else len(d))
    for n, v in src.variables.items():
        z = v.ndim > 0 and v.dtype != str
        kw = dict(zlib=True, complevel=level, shuffle=True) if z else {}
        o = dst.createVariable(n, v.datatype, v.dimensions, **kw)
        for k in v.ncattrs():
            o.setncattr(k, v.getncattr(k))
        v.set_auto_maskandscale(False)
        o.set_auto_maskandscale(False)
        if v.ndim >= 2 and v.shape[1] > 64:
            for i in range(0, v.shape[1], 64):
                o[:, i:i + 64] = v[:, i:i + 64]
        else:
            o[...] = v[...]
    for n, g in src.groups.items():
        _copy(g, dst.createGroup(n), level)


def _same(a, b, path=""):
    if sorted(a.ncattrs()) != sorted(b.ncattrs()):
        return f"attributes differ at {path or '/'}"
    for k in a.ncattrs():
        if not np.array_equal(np.asarray(a.getncattr(k)), np.asarray(b.getncattr(k))):
            return f"attribute {path}{k} differs"
    if sorted(a.variables) != sorted(b.variables):
        return f"variable lists differ at {path or '/'}"
    for n, v in a.variables.items():
        w = b.variables[n]
        v.set_auto_maskandscale(False)
        w.set_auto_maskandscale(False)
        if v.shape != w.shape or v.dtype != w.dtype:
            return f"{path}{n}: shape/dtype differ"
        if v.ndim >= 2 and v.shape[1] > 64:
            for i in range(0, v.shape[1], 64):
                x, y = np.asarray(v[:, i:i + 64]), np.asarray(w[:, i:i + 64])
                if x.tobytes() != y.tobytes():
                    return f"{path}{n}: data differ (chunk {i})"
        elif v.dtype == str:
            if str(v[...]) != str(w[...]):
                return f"{path}{n}: string differs"
        elif np.asarray(v[...]).tobytes() != np.asarray(w[...]).tobytes():
            return f"{path}{n}: data differ"
    for n, g in a.groups.items():
        if n not in b.groups:
            return f"group {path}{n} missing"
        r = _same(g, b.groups[n], f"{path}{n}/")
        if r:
            return r
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--level", type=int, default=1)
    a = ap.parse_args()
    f, tmp = a.file, a.file + ".ztmp"
    before = os.path.getsize(f) / 1e6
    try:
        with Dataset(f) as ds:
            if "fields_real" in ds.variables and ds["fields_real"].filters().get("zlib"):
                print(f"OK already compressed ({before:.1f} MB)")
                return 0
        with Dataset(f) as src, Dataset(tmp, "w", format="NETCDF4") as dst:
            _copy(src, dst, a.level)
        with Dataset(f) as x, Dataset(tmp) as y:
            r = _same(x, y)
        if r:
            raise RuntimeError(f"verification failed: {r}")
        os.replace(tmp, f)
    except Exception as e:  # noqa: BLE001
        if os.path.exists(tmp):
            os.remove(tmp)
        print(f"BAD {type(e).__name__}: {e}")
        return 1
    print(f"OK {before:.1f} MB -> {os.path.getsize(f) / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
