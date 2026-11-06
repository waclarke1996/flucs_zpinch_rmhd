"""Stage 3b (4): A3a-Y-k8 continued to t = 500 (S30 t = 0-167 + S30c restart), vs curvy3d A3a-Y.

python check_S33c.py <A3a-Y-k8-S30 dir> <A3a-Y-k8-S30c dir> <curvy3d A3a-Y .nc> [out.png]
<Q/Q_MHD> (autocorrelation-corrected error) over t = 191-500 (and sub-windows) and burst list
(Q/Q_MHD > 5: start, peak, peak time, min dt in the burst).
"""
import sys

import numpy as np

from curvy_compare import mean_err, read_curvy, read_flucs

A, B = read_flucs(sys.argv[1]), read_flucs(sys.argv[2])
F = {k: np.concatenate([A[k][A["t"] < B["t"][0]], B[k]]) for k in ("t", "Q_over_QMHD", "dt", "E")}
C = read_curvy(sys.argv[3])
F["QQ"] = F["Q_over_QMHD"]
print(f"FLUCS t = {F['t'][0]:.1f}-{F['t'][-1]:.2f}, curvy3d t = {C['t'][0]:.1f}-{C['t'][-1]:.2f}")
for t0, t1 in ((100, 200), (191, 500), (191, 350), (350, 500)):
    line = f"<Q/Q_MHD> t={t0}-{t1}:"
    for name, X in (("FLUCS", F), ("curvy3d", C)):
        if X["t"][-1] >= t0 + 20:
            m = mean_err(X["t"], X["QQ"], t0, min(t1, X["t"][-1]))
            line += f"  {name} {m[0]:.3f} +- {m[1]:.3f} (tau {m[2]:.1f}, to t={min(t1, X['t'][-1]):.0f})"
    print(line)


def bursts(t, q, dt, thr=5.0):
    above = q > thr
    out, i = [], 0
    while i < len(t):
        if above[i]:
            j = i
            while j < len(t) and above[j]:
                j += 1
            k = i + np.argmax(q[i:j])
            out.append((t[i], t[j - 1], q[k], t[k], dt[i:j].min()))
            i = j
        else:
            i += 1
    return out


for name, X in (("FLUCS", F), ("curvy3d", C)):
    b = bursts(X["t"], X["QQ"], X["dt"])
    print(f"{name}: {len(b)} bursts with Q/Q_MHD > 5; max Q/Q_MHD {X['QQ'].max():.1f} @ t="
          f"{X['t'][np.argmax(X['QQ'])]:.1f}; min dt {X['dt'][X['t'] > 1].min():.2e}")
    for (ta, tb, qp, tp, dtm) in b:
        print(f"   t {ta:6.1f}-{tb:6.1f}  peak {qp:7.1f} @ {tp:6.1f}  min dt {dtm:.2e}")

if len(sys.argv) > 4:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for name, X, col in (("FLUCS", F, "C0"), ("curvy3d", C, "C3")):
        ax[0].semilogy(X["t"], np.abs(X["QQ"]), col, lw=0.8, label=name)
        ax[1].semilogy(X["t"], X["dt"], col, lw=0.8)
    ax[0].axvspan(191, 500, color="0.9")
    ax[0].set_ylabel("|Q/Q_MHD|")
    ax[1].set_ylabel("dt")
    ax[1].set_xlabel("t")
    ax[0].legend()
    fig.tight_layout()
    fig.savefig(sys.argv[4], dpi=110)
    print("plot:", sys.argv[4])
