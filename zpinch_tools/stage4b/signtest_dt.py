#!/usr/bin/env python3
"""dt scaling of the zp_shells dynamical sign test (compare_4b.py v3, same formulas) for one or more run folders.
  python signtest_dt.py verify/V3_signtest verify/V3c_signtest_dt5e-5_rtx
Residual = max_l,steps |dE_l/dt (central difference) - sum_j T[j,l]| / max|T|, for the mag/kin and the Elsasser sets.
A central difference of the shell energies has an O(dt^2) error: halving dt should cut the residual ~4x."""
import os
import sys
import warnings

import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.expanduser("~/flucs_work/flucs/src"))
from flucs.systems.zpinch_fourier.zpinch_readers import load_diag  # noqa: E402

MK = ("Tuu", "TBu", "TBB", "TuB", "TBparBpar", "TuparBpar", "Tuparupar", "TBparupar")
EL = ("zplus", "zminus", "zparplus", "zparminus")


def rel(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300))


for run in sys.argv[1:]:
    t, Sh = load_diag(f"{run}/output.shells.nc", "zp_shells")
    _, Z = load_diag(f"{run}/output.0d.nc", "energy", ["E"])
    t = np.asarray(t)
    out = [f"{os.path.basename(run.rstrip('/'))}: {len(t)} records, dt {np.unique(np.round(np.diff(t), 12))}, "
           f"E rel. change {abs(Z['E'][-1] - Z['E'][0]) / Z['E'][0]:.1e}"]
    for st in ("kperp", "kz"):
        E = sum(Sh[f"{st}_E_{p}"] for p in ("uperp", "bperp", "dbpar", "upar"))
        dEdt = (E[2:] - E[:-2]) / (t[2:] - t[:-2])[:, None]
        Tm = sum(Sh[f"{st}_{n}"] for n in MK).sum(axis=1)[1:-1]
        Te = 0.5 * sum(Sh[f"{st}_{n}"] for n in EL).sum(axis=1)[1:-1]
        act = np.abs(Tm) > 1e-3 * np.abs(Tm).max()
        sgn = float(np.mean(np.sign(dEdt[act]) == np.sign(Tm[act])))
        out.append(f"{st}: magkin {rel(dEdt, Tm):.2e}, Elsasser {rel(dEdt, Te):.2e}, sign agreement {sgn:.3f} "
                   f"({len(Tm)} steps)")
    print("; ".join(out))
