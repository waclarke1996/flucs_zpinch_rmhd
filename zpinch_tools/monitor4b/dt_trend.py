#!/usr/bin/env python3
"""Stage 4b monitoring: dt trend and cost of running chains from FINISHED segments (monitor4b/<run>/tNNNN symlinks).
  python dt_trend.py RUN [RUN ...]
Per run: min dt per 10 R/vA (compact), measured steps/RvA = integral of 1/dt (0d records, 0.1 R/vA) over the whole
run and over the last 100 R/vA, <Q/Q_MHD> and <E> over the last 50 R/vA. Read-only on the run outputs."""
import glob, os, sys, warnings
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.expanduser("~/flucs_work/flucs/src"))
from flucs.systems.zpinch_fourier.zpinch_readers import load_diag  # noqa: E402

for run in sys.argv[1:]:
    f0 = sorted(glob.glob(os.path.join(run, "t[0-9]*", "output.0d.nc")))
    t, d = load_diag(f0, "energy", ["Q_over_QMHD", "E"])
    import netCDF4
    tt, dt = [], []
    for f in f0:
        with netCDF4.Dataset(f) as nc:
            tt.append(nc.groups["0"]["time"][:]); dt.append(nc.groups["0"]["dt"][:])
    tt, dt = np.concatenate(tt), np.concatenate(dt)
    k = np.argsort(tt); tt, dt = tt[k], dt[k]
    o = np.argsort(t); t = np.asarray(t)[o]
    qq, E = np.asarray(d["Q_over_QMHD"])[o], np.asarray(d["E"])[o]
    bins = (tt // 10).astype(int)
    ub = np.unique(bins); mins = np.array([dt[bins == b].min() for b in ub])
    seg = " ".join(f"{mins[(ub >= 10 * j) & (ub < 10 * j + 10)].mean() * 1e4:.2f}" for j in range(int(ub.max()) // 10 + 1) if ((ub >= 10 * j) & (ub < 10 * j + 10)).sum() >= 5)
    steps = lambda m: np.trapz(1 / dt[m], tt[m]) / (tt[m][-1] - tt[m][0])
    last100, last50 = tt >= tt[-1] - 100, t >= t[-1] - 50
    name = os.path.basename(run.rstrip("/"))
    print(f"{name}: t_end {t[-1]:.0f}; steps/RvA all {steps(tt >= 0):.0f}, last100 {steps(last100):.0f}; "
          f"last50 <Q/Q_MHD> {qq[last50].mean():.3f} <E> {E[last50].mean():.3g}; <dt_min/10RvA> per 100 RvA (1e-4): " + seg)
