#!/usr/bin/env python3
"""Stage 3e part 2: time-weighted <Q/Q_MHD> in sub-windows (window check) for the stage3e runs, curvy3d A2-Y on the
same window with zr.mean_and_error, and A3a f_high in 200-R/vA windows (from two_state_series.npz). -> e_blocks.txt"""
import sys, glob, os, numpy as np
R = os.path.expanduser("~/flucs_work/runs"); sys.path.insert(0, f"{R}/tools")
import two_state as ts, zpinch_runtools as zr  # noqa: E402
W = [(100, 500), (200, 500), (100, 200), (200, 300), (300, 400), (400, 500)]
for r in ["A2-T1-k8", "A2-T2-k8", "A4-T2-k8", "A3b-T2-k8", "A3a-T2-k8"]:
    t, q, E, kx = ts.concat([tuple(ts.flucs_piece(f, 0.0)) for f in sorted(glob.glob(f"{R}/stage3e/{r}/t*/output.0d.nc"))])
    g = np.arange(100, t[-1], 0.1); qq = np.interp(g, t, q)
    print(r, " ".join(f"[{a},{b}] {qq[(g >= a) & (g < b)].mean():.3f}" for a, b in W))
p = ts.curvy_piece("A2-Y", 0.0); g = np.arange(100, 500.0001, 0.1)
m, e, tau, _ = zr.mean_and_error(np.interp(g, p[0], p[1])); print(f"C A2-Y [100,500] {m:.3f} +- {e:.3f} (tau_int {tau * 0.1:.1f})")
d = np.load(f"{R}/stage3e/two_state_series.npz", allow_pickle=True)
for r in ["F:A3a-T2-k8", "C:A3a-Y+Yc2", "F:A3a-Z-k8-C300"]:
    t, l = d[r + "|t"], d[r + "|l"]   # l: 1 high, -1 low, 0 unclassified
    f = [((l[m] == 1).sum() / max(((l[m] == 1) | (l[m] == -1)).sum(), 1)) for a in np.arange(max(100, t.min()), t.max() - 199, 100)
         for m in [(t >= a) & (t < a + 200)]]
    nb = np.array(f[::2]); print(r, "f_high 200-windows", np.round(f, 2).tolist(),
                                 f"| non-overlapping mean {nb.mean():.2f} +- {nb.std(ddof=1) / np.sqrt(len(nb)):.2f}")
