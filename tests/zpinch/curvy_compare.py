"""Stage 3 helpers: read FLUCS ZPinchFourier output and curvy3d .nc output (read-only) on a common footing.

All quantities are BOX AVERAGES. curvy3d's saved sums are half the box average; the conversion uses Will's
reference/curvy3d/zpinch_runtools.py (imported read-only, no bytecode written there).
"""
import os
import pathlib as pl
import sys
import warnings

sys.dont_write_bytecode = True
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
warnings.filterwarnings("ignore")
REF = pl.Path.home() / "flucs_work" / "reference" / "curvy3d"
sys.path.insert(0, str(REF))
import netCDF4  # noqa: E402
import numpy as np  # noqa: E402
import zpinch_runtools as zr  # noqa: E402

INVARIANTS = ("zplus", "zminus", "Zpplus", "Zpminus")


def read_flucs(run_dir):
    ds = netCDF4.Dataset(f"{run_dir}/output.0d.nc")
    g = ds["0"]
    e = g["energy"]
    v = {k: np.asarray(e[k][:], dtype=float) for k in e.variables}
    v["t"] = np.asarray(g["time"][:], dtype=float)
    for k in ("dt", "step"):
        if k in g.variables:
            v[k] = np.asarray(g[k][:], dtype=float)
    # drop duplicated time entries (forced final save)
    t, idx = np.unique(v["t"], return_index=True)
    return {k: (a[idx] if np.ndim(a) and len(a) == len(v["t"]) else a) for k, a in v.items()}


def read_curvy(path):
    """Box-average series of a curvy3d run: t, Q/Q_MHD, Q_es, Q_em, E, invariants W (box), dt."""
    ds = netCDF4.Dataset(path, "r")
    cwd = os.getcwd()
    os.chdir(pl.Path.home() / "flucs_work" / "runs")  # no curvy3d_solver.options there
    try:
        P = zr.params_from_dataset(ds)
    finally:
        os.chdir(cwd)
    t, q_es, q_em = zr.heat_flux_series(ds, P)
    n = len(t)
    out = {"P": P, "t": t, "Q_es": q_es, "Q_em": q_em, "QQ": (q_es + q_em) / zr.q_mhd(P)}
    for name in INVARIANTS:
        out[name] = 2.0 * np.asarray(ds[name][:n], dtype=float)  # saved = W/2
    out["E"] = 0.5 * sum(out[nm] for nm in INVARIANTS)
    out["dt"] = np.asarray(ds["dt_series"][:n], dtype=float)
    out["ds"] = ds
    return out


def mean_err(t, x, t0, t1):
    """Mean and autocorrelation-corrected error over [t0, t1] (uniform resampling at the median spacing)."""
    m = (t >= t0) & (t <= t1)
    tt, xx = t[m], x[m]
    dt = np.median(np.diff(tt))
    tu = np.arange(tt[0], tt[-1] + 0.5 * dt, dt)
    mean, err, tau, neff = zr.mean_and_error(np.interp(tu, tt, xx))
    return mean, err, tau * dt, neff
