"""Stage 3b analysis: streamer growth, secondary growth of E(kx != 0), NL vs linear control.

python check_S3b.py <runs/stage3b>   -> table on stdout, <dir>/S3b.png

Rates: gamma = amplitude rate of the streamer (0.5 d ln E_kx0/dt); sigma = ENERGY growth rate of
E(kx != 0) (d ln E_kxn0/dt; amplitude rate sigma/2). Theory gamma_diss = gamma - nu_perp kperp^2p - nu_z kz^2p.
"""
import sys
import pathlib as pl

import numpy as np

from make_S3b import CASES, ES
from zp_common import POINTS, gamma_theory, read0d

d = pl.Path(sys.argv[1])
PARTS = ("uperp", "bperp", "upar", "dbpar")


def fit_rate(t, y, t0, t1):
    m = (t >= t0) & (t <= t1) & (y > 0)
    if m.sum() < 5:
        return np.nan
    return np.polyfit(t[m], np.log(y[m]), 1)[0]


def first_time(t, cond):
    i = np.flatnonzero(cond)
    return t[i[0]] if len(i) else np.nan


def local_rate(t, y, dt_s=0.2):
    """d ln y / dt by central difference over +-dt_s/2 (interpolated)."""
    ly = np.log(np.maximum(y, 1e-300))
    return (np.interp(t + dt_s / 2, t, ly) - np.interp(t - dt_s / 2, t, ly)) / dt_s


rows, curves, extra = [], {}, []
for case, (pt, mode, dims, nu, power) in CASES.items():
    kb, kP = POINTS[pt]
    kperp2 = mode[0] ** 2 + mode[1] ** 2
    g_th = gamma_theory(kb, kP, *mode) - nu[0] * kperp2 ** power - nu[1] * (mode[2] ** 2) ** power
    for es in ES:
        name = f"{case}-S{es:g}"
        try:
            v, vl = read0d(d / name), read0d(d / f"{name}-lin")
        except (FileNotFoundError, OSError):
            print(f"{name}: missing output")
            continue
        curves[name] = (v, vl)
        t, tl = v["t"], vl["t"]
        frac = v["E_kxn0"] / v["E"]
        t10 = first_time(t, frac >= 0.1)
        t_eq = first_time(t, v["E_kxn0"] >= v["E_kx0"])
        t_end = t[-1]
        # streamer: clean window while E(kx != 0) < 1e-3 E, t >= 0.5
        t_clean = first_time(t, frac > 1e-3)
        t_c1 = min(t_clean if np.isfinite(t_clean) else t_end, t_end) - 0.2
        gam_nl = 0.5 * fit_rate(t, v["E_kx0"], 0.5, t_c1)
        gam_lin = 0.5 * fit_rate(tl, vl["E_kx0"], 0.5, tl[-1])
        # secondary: fit E_kxn0 over the 3 time units before t10 (or before the end); same window in the control
        t_b = t10 if np.isfinite(t10) else t_end
        win = (max(t_b - 3.0, 1.0), t_b)
        sig_nl = fit_rate(t, v["E_kxn0"], *win)
        sig_lin = fit_rate(tl, vl["E_kxn0"], *win)
        sig_lin_late = fit_rate(tl, vl["E_kxn0"], 15.0, tl[-1])
        e_kx0_win = np.interp(win[0], t, v["E_kx0"])
        # local rate sigma(t) = d ln E_kxn0/dt over dt_s = 0.2 (NL), peak before t_b and at fixed E_kx0
        lr = local_rate(t, v["E_kxn0"])
        pre = (t >= 0.5) & (t <= t_b)
        i_pk = np.flatnonzero(pre)[np.argmax(lr[pre])] if pre.any() else 0
        sig_pk, E_pk = lr[i_pk], v["E_kx0"][i_pk]
        sig_at = {}
        for e_ref in (30.0, 100.0, 300.0):
            j = first_time(np.arange(len(t)), (v["E_kx0"] >= e_ref) & (t <= t_b))
            sig_at[e_ref] = lr[int(j)] if np.isfinite(j) else np.nan
        lr_l = local_rate(tl, vl["E_kxn0"])
        extra.append((name, sig_pk, t[i_pk], E_pk, sig_pk / np.sqrt(E_pk), sig_at,
                      np.interp(t[i_pk], tl, lr_l)))
        f0 = [v[f"f_kx0_{p}"][0] for p in PARTS]
        rows.append((name, gam_nl, gam_lin, g_th, sig_nl, sig_lin, sig_lin_late, win, e_kx0_win,
                     t10, t_eq, t_end, f0, v["dt"][-1]))

print(f"{'run':9s} {'gam_NL':>8s} {'gam_lin':>8s} {'gam_th':>8s} | {'sig_NL':>7s} {'sig_lin':>7s} "
      f"{'(2g_lin)':>8s} {'window':>11s} {'E_kx0':>8s} | {'t10%':>6s} {'t_eq':>6s} {'t_end':>6s} "
      f"{'dt_end':>8s} | f_kx0(0) u_perp b_perp u_par dB_par")
for (name, gnl, gl, gth, snl, sl, sll, win, ek, t10, teq, tend, f0, dte) in rows:
    print(f"{name:9s} {gnl:8.5f} {gl:8.5f} {gth:8.5f} | {snl:7.3f} {sl:7.3f} {sll:8.3f} "
          f"{win[0]:5.1f}-{win[1]:<5.1f} {ek:8.2e} | {t10:6.2f} {teq:6.2f} {tend:6.2f} {dte:8.1e} | "
          + " ".join(f"{x:.3f}" for x in f0))

print("\nLocal secondary rate sigma(t) = d ln E_kxn0/dt (NL); peak before t10% (or end), and at E_kx0 = 30/100/300")
print(f"{'run':9s} {'sig_pk':>7s} {'t_pk':>6s} {'E_kx0_pk':>9s} {'sig/sqrtE':>9s} | {'s@30':>6s} {'s@100':>6s} {'s@300':>6s} | lin@t_pk")
for (name, spk, tpk, epk, sr, sat, slin) in extra:
    print(f"{name:9s} {spk:7.2f} {tpk:6.2f} {epk:9.2e} {sr:9.3f} | {sat[30.0]:6.2f} {sat[100.0]:6.2f} {sat[300.0]:6.2f} | {slin:6.2f}")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, len(CASES), figsize=(4 * len(CASES), 3.6), sharey=True)
    for ax, case in zip(axs, CASES):
        for es, col in zip(ES, ("C0", "C3")):
            name = f"{case}-S{es:g}"
            if name not in curves:
                continue
            v, vl = curves[name]
            ax.semilogy(v["t"], v["E_kx0"], col, lw=1.5, label=f"E_s={es:g} kx=0")
            ax.semilogy(v["t"], v["E_kxn0"], col, ls="--", lw=1.5, label="kx!=0 NL")
            ax.semilogy(vl["t"], vl["E_kxn0"], col, ls=":", lw=1, label="kx!=0 lin")
        pt, mode = CASES[case][:2]
        ax.set_title(f"{case}: {pt} {mode}")
        ax.set_xlabel("t")
        ax.set_ylim(1e-9, 1e7)
    axs[0].set_ylabel("energy")
    axs[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(d / "S3b.png", dpi=110)
    print(f"plot: {d / 'S3b.png'}")
except ImportError:
    pass
