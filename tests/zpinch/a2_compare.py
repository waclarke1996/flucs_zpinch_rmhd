"""Stage 3c A2: FLUCS C300 runs (restart from curvy3d A3a-Y t=300) vs curvy3d A3a-Y+Yc2, t = 300-600."""
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = "/mnt/users/clarkew/flucs_work/runs/stage3c/"
z = np.load(R + "two_state_series.npz")
get = lambda k: [z[f"{k}|{v}"] for v in "tqEl"]
C = get("C:A3a-Y+Yc2")
def win(s, a=300, b=600):
    m = (s[0] >= a - 1e-9) & (s[0] <= b + 1e-9); return [x[m] for x in s]
Cw = win(C)
def st(name, s):
    t, q, E, l = s
    h, lo = l == 1, l == -1
    print(f"{name:24s} f_high {h.mean():.2f} f_low {lo.mean():.2f} Q_H {np.mean(q[h]):.2f} Q_L {np.mean(q[lo]):.2f} "
          f"medQ_H {np.median(q[h]):.2f} medQ_L {np.median(q[lo]):.2f} E_H {np.mean(E[h]):.0f} E_L {np.mean(E[lo]):.0f} <Q> {q.mean():.2f}")
st("C:A3a-Y(+Yc2) 300-600", Cw)
fig, ax = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True)
cols = {"F:A3a-Y-k8-C300": "tab:red", "F:A3a-Y-k8-C300-cfl025": "tab:green"}
ax[0].semilogy(Cw[0], Cw[1], lw=0.6, c="k", label="curvy3d A3a-Y+Yc2")
ax[1].semilogy(Cw[0], Cw[2], lw=0.6, c="k")
F = {}
for k, c in cols.items():
    s = win(get(k)); F[k] = s; st(k, s)
    ax[0].semilogy(s[0], s[1], lw=0.6, c=c, label=k[2:]); ax[1].semilogy(s[0], s[2], lw=0.6, c=c)
    # divergence from curvy3d (same 0.1 grid)
    qc = np.interp(s[0], Cw[0], Cw[1])
    d = np.abs(np.log(s[1] / qc))
    i = np.argmax(d > 0.1)
    print(f"   first |ln Q_F/Q_C| > 0.1 at t = {s[0][i]:.1f}; >0.5 at t = {s[0][np.argmax(d > 0.5)]:.1f}; "
          f"state agreement (both classified) {np.mean((s[3] == np.interp(s[0], Cw[0], Cw[3]))[(s[3] != 0)]):.2f}")
a, b = F["F:A3a-Y-k8-C300"], F["F:A3a-Y-k8-C300-cfl025"]
n = min(len(a[0]), len(b[0])); d = np.abs(np.log(a[1][:n] / b[1][:n]))
print(f"cfl0.5 vs 0.25: first |ln Q ratio| > 0.1 at t = {a[0][np.argmax(d > 0.1)]:.1f}")
for x in ax: x.axhline(1.5 if x is ax[0] else 400, ls=":", c="tab:red"); x.axhline(0.8 if x is ax[0] else 250, ls=":", c="tab:blue")
ax[0].set_ylabel("Q/Q_MHD"); ax[1].set_ylabel("E (box avg)"); ax[1].set_xlabel("t (R/vA)"); ax[0].legend(fontsize=8)
ax[0].set_title("A2: FLUCS from curvy3d A3a-Y t=300 (k8) vs curvy3d A3a-Y+Yc2", fontsize=9, loc="left")
fig.tight_layout(); fig.savefig(R + "a2_Qt.png", dpi=110)
# early divergence detail
for k in cols:
    s = F[k]; qc = np.interp(s[0], Cw[0], Cw[1])
    print(k[2:], " ".join(f"t{tt}:{np.interp(tt, s[0], s[1]):.2f}/{np.interp(tt, Cw[0], Cw[1]):.2f}" for tt in (300.5, 301, 302, 303, 305, 310, 320)))
