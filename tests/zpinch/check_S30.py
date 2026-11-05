"""S3.0: same-IC replay of curvy3d A2-R0 to t = 60. Usage: check_S30.py FLUCS_RUN_DIR CURVY_NC OUT_PNG"""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from curvy_compare import INVARIANTS, mean_err, read_curvy, read_flucs  # noqa: E402

run, nc, png = sys.argv[1:4]
F = read_flucs(run)
C = read_curvy(nc)
tmax = F["t"][-1]
mc = C["t"] <= tmax + 1e-9
tc = C["t"][mc]
P = C["P"]
Ly2 = P.Ly ** 2

# t = 0 normalisation
print(f"t=0: Qhat_es F {F['Qhat_es'][0]:.6e} C {C['Q_es'][0] * Ly2:.6e} rel {F['Qhat_es'][0] / (C['Q_es'][0] * Ly2) - 1:.2e}")
print(f"t=0: Qhat_em F {F['Qhat_em'][0]:.6e} C {C['Q_em'][0] * Ly2:.6e} rel {F['Qhat_em'][0] / (C['Q_em'][0] * Ly2) - 1:.2e}")
print(f"t=0: E F {F['E'][0]:.6e} C {C['E'][0]:.6e} rel {F['E'][0] / C['E'][0] - 1:.2e} (ky=0 Hermitian projection)")
for n in INVARIANTS:
    print(f"t=0: {n} rel {F[n][0] / C[n][0] - 1:.2e}", end="; ")
print()


def on_c(key):
    return np.interp(tc, F["t"], F[key])


qF = np.exp(np.interp(tc, F["t"], np.log(np.abs(F["Q_over_QMHD"]))))
qC = np.abs(C["QQ"][mc])
lr = np.log(qF / qC)
grow = (tc >= 0.5) & (tc < tc[np.argmax(qC > 1e-2)])  # t=0: Q = 0 exactly (real-valued IC)
print(f"linear phase (t >= 0.5, Q/Q_MHD < 1e-2, t {tc[grow][0]:.1f}-{tc[grow][-1]:.1f}): max |Q_F/Q_C - 1| = "
      f"{np.max(np.abs(qF[grow] / qC[grow] - 1)):.2e}")
bad = np.nonzero(np.abs(lr) > 0.1)[0]
print(f"first t with |ln(Q_F/Q_C)| > 0.1: {tc[bad[0]] if len(bad) else 'never':}")
for name, q in (("FLUCS", qF), ("curvy3d", qC)):
    i1 = np.argmax(q > 1e-2)
    win = (tc >= tc[i1]) & (tc <= tc[i1] + 10)
    ip = np.argmax(np.where(win, q, -1))
    print(f"{name:8s}: Q/Q_MHD crosses 1e-2 at t = {tc[i1]:.2f}; first peak {q[ip]:.4f} at t = {tc[ip]:.2f}")
for t0, t1 in ((40, 60), (30, 60)):
    mF = mean_err(F["t"], F["Q_over_QMHD"], t0, t1)
    mC = mean_err(tc, C["QQ"][mc], t0, t1)
    print(f"<Q/Q_MHD> t={t0}-{t1}: FLUCS {mF[0]:.4f} +- {mF[1]:.4f} (tau {mF[2]:.2f}); "
          f"curvy3d {mC[0]:.4f} +- {mC[1]:.4f} (tau {mC[2]:.2f})")
    for key in ("E", *INVARIANTS):
        a = mean_err(F["t"], F[key], t0, t1)[0]
        b = mean_err(tc, C[key][mc], t0, t1)[0]
        print(f"   <{key}> F {a:.4e} C {b:.4e} ratio {a / b:.3f}")
# dt
for t0, t1 in ((0, 20), (20, 40), (40, 60)):
    mf = (F["t"] >= t0) & (F["t"] < t1)
    dtF = np.mean(F["dt"][mf])  # sampled every 5 steps
    print(f"mean dt t={t0}-{t1}: FLUCS {dtF:.3e}  curvy3d (dt_series mean) {np.mean(C['dt'][mc][(tc >= t0) & (tc < t1)]):.3e}")
# E, invariants agreement over growth
for key in ("E", *INVARIANTS):
    r = on_c(key) / C[key][mc]
    print(f"{key}: max |F/C - 1| for t<15: {np.max(np.abs(r[tc < 15] - 1)):.2e}, t<{tc[grow][-1]:.0f}: "
          f"{np.max(np.abs(r[tc <= tc[grow][-1]] - 1)):.2e}")

fig, ax = plt.subplots(3, 2, figsize=(11, 10), sharex=True)
ax = ax.ravel()
ax[0].semilogy(tc, qC, label="curvy3d A2-R0")
ax[0].semilogy(F["t"], np.abs(F["Q_over_QMHD"]), "--", label="FLUCS")
ax[0].set_ylabel("Q/Q_MHD")
ax[0].legend()
ax[1].plot(tc, qF / qC)
ax[1].set_ylim(0, 2)
ax[1].set_ylabel("Q_F / Q_C")
ax[2].semilogy(tc, C["E"][mc])
ax[2].semilogy(F["t"], F["E"], "--")
ax[2].set_ylabel("E")
for i, key in enumerate(INVARIANTS):
    a = ax[3] if i < 2 else ax[4]
    a.semilogy(tc, C[key][mc], label=f"C {key}")
    a.semilogy(F["t"], F[key], "--", label=f"F {key}")
    a.legend(fontsize=7)
    a.set_ylabel("W (box avg)")
ax[5].semilogy(tc, C["dt"][mc], label="curvy3d")
if "dt" in F:
    ax[5].semilogy(F["t"], F["dt"], "--", label="FLUCS")
ax[5].set_ylabel("dt")
ax[5].legend()
for a in ax[4:]:
    a.set_xlabel("t")
fig.tight_layout()
fig.savefig(png, dpi=110)
print("saved", png)
