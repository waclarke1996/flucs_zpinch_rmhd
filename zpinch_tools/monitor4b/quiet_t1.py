#!/usr/bin/env python3
"""Stage 4b monitoring: characterise the quiescent FINAL T1 runs of prod4 (read-only on the run outputs).
  python quiet_t1.py [ROOT=~/flucs_work/runs/prod4] > quiet_t1.txt
Runs: queue/done/* with tier T1 and <Q/Q_MHD> (t >= 60) < 1e-3. Per run: Q/Q_MHD peak (t, value), t_quiet = first
record after which Q/Q_MHD stays < 1e-3 (0d, 0.1 R/vA), means over the final 200 R/vA of zp_scalars E_ky0_frac,
E_kx0_frac, E_ky0kz0_frac, S_zon0, and the two largest bins of E_kx, E_ky, E_kz (zp_spectra) at the last record
(fraction of the record's E) and the split of that E into uperp/bperp/dbpar/upar."""
import glob, os, sys, warnings
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.expanduser("~/flucs_work/flucs/src"))
from flucs.systems.zpinch_fourier.zpinch_readers import load_diag  # noqa: E402

root = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/flucs_work/runs/prod4")
runs = sorted(n.split("_", 1)[1] for n in os.listdir(os.path.join(root, "queue", "done"))
              if not n.endswith(".owner") and "-T1-" in n)


PARTS = ("uperp", "bperp", "dbpar", "upar")


def series(files, diag, names=None):
    t, d = load_diag(files, diag, names)
    o = np.argsort(np.asarray(t))
    return np.asarray(t)[o], {k: np.asarray(v)[o] for k, v in d.items() if np.ndim(v) >= 1 and len(v) == len(o)}


def top2(k, e, etot, fmt):
    i = np.argsort(e)[::-1][:2]
    return ",".join(f"{fmt(k[j])}({e[j] / etot:.2f})" for j in i)


print(f"{'run':7s} {'kP':>5s} {'kb':>6s} {'<Q>':>8s} {'Qpeak':>8s} {'t_pk':>6s} {'t_q<1e-3':>8s} {'Eend':>7s} "
      f"{'ky0':>5s} {'kx0':>5s} {'ky0kz0':>6s} {'Szon0':>5s}  top E_kx | E_ky | E_kz (fraction of E) | uperp/bperp/dbpar/upar")
for r in runs:
    run = os.path.join(root, r)
    if not os.path.exists(os.path.join(run, "FINAL")):
        continue
    f0 = sorted(glob.glob(os.path.join(run, "t[0-9]*", "output.0d.nc")))
    f1 = sorted(glob.glob(os.path.join(run, "t[0-9]*", "output.1d.nc")))
    t, e = series(f0, "energy", ["Q_over_QMHD", "E"])
    _, s = series(f0, "zp_scalars")
    q, E = e["Q_over_QMHD"], e["E"]
    qm = q[t >= 60].mean()
    if not qm < 1e-3:
        continue
    ip = np.argmax(q)
    hi = np.nonzero(q >= 1e-3)[0]
    tq = "never" if len(hi) == 0 else f"{t[min(hi[-1] + 1, len(t) - 1)]:.0f}"
    late = t >= t[-1] - 200
    m = {k: s[k][late].mean() for k in ("E_ky0_frac", "E_kx0_frac", "E_ky0kz0_frac", "S_zon0")}
    t1, S = load_diag(f1[-1:], "zp_spectra", ["E_kx", "E_ky", "E_kz"] + [f"E_{p}_kx" for p in PARTS])
    j = int(np.argmax(np.asarray(t1)))
    ex, ey, ez = (np.asarray(S[n])[j] for n in ("E_kx", "E_ky", "E_kz"))
    et = ey.sum()
    split = "/".join(f"{np.asarray(S[f'E_{p}_kx'])[j].sum() / et:.2f}" for p in PARTS)
    pt = __import__("json").load(open(os.path.join(run, "point.json")))
    print(f"{r[:6]:7s} {pt['kP']:5.2f} {pt['kb']:6.4f} {qm:8.1e} {q[ip]:8.1e} {t[ip]:6.1f} {tq:>8s} {E[late].mean():7.3g} "
          f"{m['E_ky0_frac']:5.3f} {m['E_kx0_frac']:5.3f} {m['E_ky0kz0_frac']:6.3f} {m['S_zon0']:5.3f}  "
          f"{top2(np.asarray(S['kx']), ex, et, lambda v: f'{v:g}')} | {top2(np.asarray(S['ky']), ey, et, lambda v: f'{v:g}')} | "
          f"{top2(np.asarray(S['kz']), ez, et, lambda v: f'{v:g}')} | {split}")
