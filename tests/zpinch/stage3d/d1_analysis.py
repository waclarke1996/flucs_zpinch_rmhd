"""Stage 3d D1: square-box (Lx=Ly=2pi) A2 runs k8/k4: Q/Q_MHD(t), E_kx0/E(t), window means with autocorrelation-corrected errors."""
import sys, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "/mnt/users/clarkew/flucs_work/flucs/tests/zpinch"); from zp_common import read0d
R = "/mnt/users/clarkew/flucs_work/runs/stage3d/"
SEG = [f"t{50*i:04d}" for i in range(1, 11)]
def load(run):
    vs = [read0d(R + run + "/" + s) for s in SEG]
    t = np.concatenate([v["t"] for v in vs]); o = np.argsort(t); t, i = np.unique(t[o], return_index=True)
    g = lambda k: np.concatenate([v[k] for v in vs])[o][i]
    tg = np.arange(0, 500.0 + 1e-9, 0.1)
    return tg, np.interp(tg, t, g("Q_over_QMHD")), np.interp(tg, t, g("E")), np.interp(tg, t, g("E_kx0")), np.interp(tg, t, g("dt"))
def tau_int(x, dt=0.1):
    x = x - x.mean(); n = len(x); c = np.correlate(x, x, "full")[n-1:] / (x.var() * np.arange(n, 0, -1))
    s = 0.5
    for k in range(1, n // 4):          # Sokal window: stop at M >= 5 tau
        s += c[k]
        if k >= 5 * 2 * s: break
    return 2 * s * dt                   # integrated autocorrelation time (R/vA); err = std*sqrt(tau/T)
def stats(name, t, q, E, k0, a, b):
    m = (t >= a) & (t <= b); x = q[m]; T = b - a; tau = tau_int(x)
    err = x.std() * np.sqrt(tau / T)
    bl = np.array([y.mean() for y in np.array_split(x, 4)])
    print(f"{name} [{a:.0f},{b:.0f}] <Q/QMHD> {x.mean():.3f} +- {err:.3f} (tau_int {tau:.1f}, N_eff {T/tau:.0f}; 4-block "
          f"{' '.join(f'{y:.3f}' for y in bl)}, sd/2 {bl.std(ddof=1)/2:.3f}) med {np.median(x):.3f} "
          f"<E> {E[m].mean():.1f} <E_kx0/E> {np.mean(k0[m]/E[m]):.2f} (E-wtd {k0[m].mean()/E[m].mean():.2f})")
D = {r: load(r) for r in ["A2-SQ-k8", "A2-SQ-k4"]}
for r, (t, q, E, k0, dt) in D.items():
    print(r)
    for a in range(0, 500, 50):
        m = (t >= a) & (t < a + 50)
        print(f"   t{a:3d}-{a+50:3d}: Q {q[m].mean():.3f} (max {q[m].max():.2f}) E {E[m].mean():6.1f} kx0/E {np.mean((k0/E)[m]):.2f} dtmin {dt[m].min():.1e}")
    print(f"   max Q/QMHD {q.max():.2f} at t {t[q.argmax()]:.1f}; max E {E.max():.0f}")
    for a, b in [(100, 500), (150, 500), (200, 500), (300, 500)]: stats(r, t, q, E, k0, a, b)
fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
for (r, (t, q, E, k0, dt)), c in zip(D.items(), ["tab:red", "tab:blue"]):
    ax[0].semilogy(t, q, lw=0.6, c=c, label=r); ax[1].plot(t, k0 / E, lw=0.6, c=c)
for v, l, ls in [(0.116, "curvy3d A2-Y (4pi box, p4) 0.116", "--"), (0.078, "curvy3d A2-R0 (4pi box, p2) 0.078", ":")]:
    ax[0].axhline(v, c="k", ls=ls, lw=1, label=l)
ax[0].set_ylabel("Q/Q_MHD"); ax[1].set_ylabel("E_kx0 / E"); ax[1].set_xlabel("t (R/vA)"); ax[1].set_ylim(0, 1)
ax[0].legend(fontsize=8); ax[0].set_title("D1: A2 in the square box Lx=Ly=2pi (FLUCS, 85x85x171)", fontsize=9, loc="left")
fig.tight_layout(); fig.savefig(R + "d1_Qt.png", dpi=110)
np.savez(R + "d1_series.npz", **{f"{r}|{k}": x for r, s in D.items() for k, x in zip("tqEkd", s)})
