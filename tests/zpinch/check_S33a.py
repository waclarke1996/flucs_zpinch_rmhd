"""S3.3a: runaway timeline, FLUCS vs curvy3d. Usage: check_S33a.py FLUCS_RUN_DIR CURVY_NC OUT_PNG"""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from curvy_compare import mean_err, read_curvy, read_flucs  # noqa: E402

F = read_flucs(sys.argv[1])
C = read_curvy(sys.argv[2])
if len(C["t"]) == 0:
    # .nc never written (run stopped before its first save): parse the progress lines of the .log next to it.
    # Same conversion as zpinch_runtools.heat_flux_series (factor 2, solver prefactors).
    import re

    P = C["P"]
    txt = open(sys.argv[2][:-3] + ".log").read()
    t = np.array([float(x) for x in re.findall(r"Heat flux at most recent time to\s+([\d.]+)", txt)])
    hf = np.array([float(x) for x in re.findall(r"Heat flux at most recent time to\s+[\d.]+ is\s+(\S+)", txt)])
    hfe = np.array([float(x) for x in re.findall(r"EM heat flux flux at most recent time to\s+[\d.]+ is\s+(\S+)", txt)])
    dtl = np.array([float(x) for x in re.findall(r"Current dt is\s+(\S+)\s+at step", txt)])[: len(t)]
    q = (2 * hf * (1 + P.kappaBeta) / (P.kappaP * (1 + P.kappaBeta) - 2.0) + 2 * hfe / P.kappaP) / P.Ly**2
    C = {"t": t, "QQ": q / (2 * P.kappaBeta * P.kappaP) ** 1.5, "dt": dtl}
    print("curvy3d .nc empty -> parsed log:", " ".join(f"t={a:.2f}: Q/Q_MHD {b:.2f} dt {c:.2e};"
                                                         for a, b, c in zip(t, C["QQ"], dtl)))
    m = re.search(r"Timed\s+(\d+)\s+steps, which took\s+(\S+)\s+s", txt)
    if m:
        print(f"curvy3d timing from log: {1e3 * float(m.group(2)) / int(m.group(1)):.2f} ms/step")
for name, t, q, dt in (("FLUCS", F["t"], F["Q_over_QMHD"], F["dt"]), ("curvy3d", C["t"], C["QQ"], C["dt"])):
    s = f"{name:8s} t_end {t[-1]:.2f} (Q/Q_MHD {q[-1]:.3e}, dt {dt[-1]:.2e}); first t with Q/Q_MHD >"
    for thr in (0.1, 1, 5, 20, 124, 1000):
        i = np.nonzero(q > thr)[0]
        s += f" {thr:g}: {t[i[0]]:.2f}" if len(i) else f" {thr:g}: -"
    print(s)
    for a, b in ((15, 25), (20, 30)):
        if t[-1] >= b and np.sum((t >= a) & (t <= b)) > 10:
            m = mean_err(t, q, a, b)
            print(f"   <Q/Q_MHD> t={a}-{b}: {m[0]:.3f} +- {m[1]:.3f}; mean dt {np.mean(dt[(t >= a) & (t <= b)]):.2e}")
    for tt in (30, 32, 34, 36, 36.4, 38, 40):
        if t[-1] >= tt:
            i = np.argmin(np.abs(t - tt))
            print(f"   t={t[i]:.2f}: Q/Q_MHD {q[i]:.3e}, dt {dt[i]:.2e}")
i = np.nonzero(F["Q_over_QMHD"] > 1)[0]
if len(i):
    i0 = i[0]
    print(f"FLUCS budget at end: |dEdt - (inj - D)|/inj median {np.median(np.abs(F['budget_error'][i0:] / F['injection'][i0:])):.2e}"
          f" (1-step FD), D/inj at end {F['D'][-1] / F['injection'][-1]:.3f}, E_end {F['E'][-1]:.3e}")
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].semilogy(C["t"], C["QQ"], label="curvy3d")
ax[0].semilogy(F["t"], F["Q_over_QMHD"], "--", label="FLUCS")
ax[0].set_ylabel("Q/Q_MHD")
ax[0].legend()
ax[1].semilogy(C["t"], C["dt"])
ax[1].semilogy(F["t"], F["dt"], "--")
ax[1].set_ylabel("dt")
for a in ax:
    a.set_xlabel("t")
    a.set_xlim(0, max(F["t"][-1], 45))
fig.tight_layout()
fig.savefig(sys.argv[3], dpi=110)
print("saved", sys.argv[3])
