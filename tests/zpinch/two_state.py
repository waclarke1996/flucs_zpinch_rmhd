#!/usr/bin/env python3
"""
two_state.py -- two-state (low / high) analysis of Q/Q_MHD(t) for curvy3d and FLUCS A3a-Y-family runs (Stage 3c A1).

Series on a common 0.1 grid:  curvy3d: zr.heat_flux_series (box average) and E = f_E (zplus+zminus+Zpplus+Zpminus)/2
(f_E calibrated against the snapshot box energy, printed);  FLUCS: energy diagnostic (Q_over_QMHD, E, E_kx0),
interpolated.  Continuations are concatenated (a later piece replaces an earlier one from its first time on).

Classification (t >= T0 = 100, after the linear phase):
  1. qm(t) = running MEDIAN of Q/Q_MHD over W = 5 R/vA (centred).  The median ignores bursts shorter than ~W/2
     (bursts in both codes last ~1-2 R/vA, peak 5-13): a running mean would let one burst lift a low stretch over
     the threshold for several R/vA.
  2. raw label: high if qm > 1.5, low if qm < 0.8, else undecided.
  3. a high/low stretch counts only if it lasts >= TMIN = 5 R/vA ("sustained"); shorter ones -> undecided.
  4. episodes: consecutive sustained stretches of the same state, separated only by undecided time, are merged
     (the gap is given to that state); gaps between different states are "transition" (not counted in either).
Per run: fraction of classified time in high, <Q/Q_MHD> and <E> (raw series) in each state, number of switches
(low<->high episode changes), residence times (episode lengths; '>' = cut by the start/end of the data).
Cross-check: the same with E (high E > 400, low E < 250) instead of Q; agreement = fraction of time with the same
label among the times both classify.
"""
import glob
import os
import sys

import numpy as np
from netCDF4 import Dataset

sys.path.insert(0, os.path.expanduser("~/flucs_work/reference/curvy3d"))
import zpinch_runtools as zr  # noqa: E402

D = "/mnt/users/clarkew/data/rmhd_zpinch/output/3D/BP2_anchors_v2"
R = os.path.expanduser("~/flucs_work/runs")
DT, T0, W, TMIN = 0.1, 100.0, 5.0, 5.0
QH, QL, EH, EL = 1.5, 0.8, 400.0, 250.0

RUNS = {   # name: list of (kind, path, time offset)
    "C:A3a-Y+Yc2": [("c", "A3a-Y", 0.0), ("c", "A3a-Yc2", 500.0)],
    "C:A3a-YP4": [("c", "A3a-YP4", 0.0)],
    "C:A3a-Z": [("c", "A3a-Z", 0.0)],
    "C:A3a-YHR2": [("c", "A3a-YHR2", 0.0)],
    "F:A3a-Y-k8-S30+S30c": [("f", f"{R}/stage3/A3a-Y-k8-S30/output.0d.nc", 0.0),
                            ("f", f"{R}/stage3/A3a-Y-k8-S30c/output.0d.nc", 0.0)],
    **{f"F:{r}": [("f", f"{R}/stage3c/{r}/t{s:04d}/output.0d.nc", 300.0) for s in range(350, 601, 50)]
       for r in ("A3a-Y-k8-C300", "A3a-Y-k8-C300-cfl025")},
    "F:A3a-Z-k8-C300": [("f", f"{R}/stage3d/A3a-Z-k8-C300/t{s:04d}/output.0d.nc", 300.0) for s in range(350, 1001, 50)],
    "F:A3a-T2-k8": [("f", f"{R}/stage3e/A3a-T2-k8/t{s:04d}/output.0d.nc", 0.0) for s in range(50, 1201, 50)],   # Stage 3e E2
}
OUT = sys.argv[1] if len(sys.argv) > 1 else "stage3c"   # output folder under runs/ (Stage 3d: stage3d)


def curvy_piece(rid, off):
    f = glob.glob(f"{D}/*_RKARS343_{rid}.nc")[0]
    with Dataset(f) as ds:
        P = zr.params_from_dataset(ds)
        t, qe, qm = zr.heat_flux_series(ds, P)
        n = len(t)
        E = sum(np.ma.filled(ds[p][:n].astype(float), np.nan) for p in ("zplus", "zminus", "Zpplus", "Zpminus")) / 2
        # calibrate E series against the box energy of the last snapshot
        it = zr.last_saved_snapshot(ds)
        F = {k: np.array(ds[k + "_rfft_r"][it], float) + 1j * np.array(ds[k + "_rfft_i"][it], float)
             for k in ("Phi", "deltaBpar", "Psi", "upar")}
    nz, nx, hy = F["Phi"].shape
    kx = np.fft.fftfreq(nx, 1.0 / nx) * 2 * np.pi / P.Lx
    ky = np.arange(hy) * 2 * np.pi / P.Ly
    k2 = (kx[:, None] ** 2 + ky[None, :] ** 2)[None]
    w = np.where(np.arange(hy) > 0, 2.0, 1.0)[None, None, :]
    Eb = float(np.sum(w * (0.5 * k2 * (abs(F["Phi"]) ** 2 + abs(F["Psi"]) ** 2) + 0.5 * abs(F["upar"]) ** 2
                           + 0.5 * (1 + 1 / P.kappaBeta) * abs(F["deltaBpar"]) ** 2)))
    j = int(round(it * P.save_time / P.save_time_hf))
    fE = Eb / E[j] if j < n else np.nan
    return t + off, (qe + qm) / zr.q_mhd(P), E, fE, it * P.save_time + off


def flucs_piece(path, off):
    with Dataset(path) as ds:
        G = ds[sorted(ds.groups, key=int)[-1]]
        e = G["energy"]
        out = [np.asarray(G["time"][:], float) + off, np.asarray(e["Q_over_QMHD"][:], float),
               np.asarray(e["E"][:], float),
               np.asarray(e["E_kx0"][:], float) if "E_kx0" in e.variables else None]
    return out


def concat(pieces):
    """pieces: list of (t, *series); later pieces replace earlier ones from their first time on."""
    out = []
    for p in pieces:
        out = [tuple(None if a is None else a[q[0] < p[0][0]] for a in q) for q in out]
        out.append(p)
    res = []
    for i in range(len(out[0])):
        if all(q[i] is None for q in out):
            res.append(None)
        else:
            res.append(np.concatenate([np.full(len(q[0]), np.nan) if q[i] is None else q[i] for q in out]))
    return res


def running_median(x, n):
    h = n // 2
    xp = np.pad(x, h, mode="edge")
    from numpy.lib.stride_tricks import sliding_window_view
    return np.median(sliding_window_view(xp, 2 * h + 1), axis=1)


def classify(t, x, hi, lo):
    """labels +1 high / -1 low / 0 undecided-or-transition after steps 1-4; returns labels, episodes."""
    xm = running_median(x, int(round(W / DT)) + 1)
    raw = np.where(xm > hi, 1, np.where(xm < lo, -1, 0))
    lab = np.zeros_like(raw)
    i, n = 0, len(raw)
    runs = []
    while i < n:
        j = i
        while j + 1 < n and raw[j + 1] == raw[i]:
            j += 1
        if raw[i] != 0 and t[j] - t[i] + DT >= TMIN:
            runs.append([raw[i], i, j])
        i = j + 1
    eps = []
    for s, i, j in runs:
        if eps and eps[-1][0] == s:
            eps[-1][2] = j               # merge over undecided gap
        else:
            eps.append([s, i, j])
    for s, i, j in eps:
        lab[i:j + 1] = s
    return lab, eps


def stats(name, t, q, E, lab, eps, fkx=None):
    tot = DT * np.count_nonzero(lab != 0)
    th, tl = DT * np.count_nonzero(lab == 1), DT * np.count_nonzero(lab == -1)
    res = []
    for k, (s, i, j) in enumerate(eps):
        cut = ">" if (i == 0 or j == len(t) - 1) else ""
        res.append(f"{'HL'[int(s < 0)]}{cut}{t[j] - t[i] + DT:.0f}")
    m = lambda a, s: np.nanmean(a[lab == s]) if np.any(np.isfinite(a[lab == s])) else np.nan
    sw = sum(1 for a, b in zip(eps[:-1], eps[1:]) if a[0] != b[0])
    line = (f"{name:22s} {t[0]:5.0f}-{t[-1]:5.0f} {th / tot if tot else np.nan:6.2f} {th:6.0f} {tl:6.0f} "
            f"{tot / (t[-1] - t[0] + DT):5.2f}  {m(q, 1):5.2f} {m(q, -1):5.2f}  {m(E, 1):6.0f} {m(E, -1):6.0f} "
            f"{'' if fkx is None else f'{m(fkx, 1):5.2f} {m(fkx, -1):5.2f}':11s} {sw:3d}  {' '.join(res)}")
    return line


def main():
    hdr = (f"{'run':22s} {'t range':11s} {'f_high':>6s} {'T_H':>6s} {'T_L':>6s} {'f_cls':>5s}  {'Q_H':>5s} {'Q_L':>5s}  "
           f"{'E_H':>6s} {'E_L':>6s} {'kx0/E H L':11s} {'sw':>3s}  episodes (state, length; '>' = cut)")
    print(f"T0 = {T0}, running median W = {W}, sustained >= {TMIN}; Q thresholds high > {QH}, low < {QL}")
    print(hdr)
    lines_e = []
    store = {}
    for name, pieces in RUNS.items():
        ps, notes = [], []
        for kind, p, off in pieces:
            if kind == "f" and not os.path.exists(p):   # segment not run (yet)
                continue
            if kind == "c":
                t, q, E, fE, ts = curvy_piece(p, off)
                notes.append(f"{p}: E-series factor (snapshot t={ts:g}) {fE:.6f}")
                ps.append((t, q, E * fE, None))
            else:
                ps.append(tuple(flucs_piece(p, off)))
        if not ps:
            print(f"{name:22s} no data")
            continue
        t, q, E, kx = concat(ps)
        tg = np.arange(np.ceil(max(t[0], T0) / DT) * DT, t[-1] + 1e-9, DT)
        ok = np.isfinite(q) & np.isfinite(E)
        qg, Eg = np.interp(tg, t[ok], q[ok]), np.interp(tg, t[ok], E[ok])
        fk = None if kx is None else np.interp(tg, t[ok], kx[ok] / E[ok], left=np.nan)
        if fk is not None:
            fk[tg < t[ok][np.isfinite(kx[ok])][0]] = np.nan
        if len(tg) < 100:
            print(f"{name:22s} too short after T0 = {T0} (t_end {t[-1]:.1f})")
            continue
        lab, eps = classify(tg, qg, QH, QL)
        print(stats(name, tg, qg, Eg, lab, eps, fk))
        labE, epsE = classify(tg, Eg, EH, EL)
        both = (lab != 0) & (labE != 0)
        lines_e.append(stats(name, tg, qg, Eg, labE, epsE, fk) +
                       f"   agree(Q,E) {np.mean(lab[both] == labE[both]) if both.any() else np.nan:.3f}")
        lines_e[-1] += "   [" + "; ".join(notes) + "]" if notes else ""
        store[name] = (tg, qg, Eg, lab)
    print("\nCross-check: classified by E (high > %g, low < %g)" % (EH, EL))
    print(hdr)
    for s in lines_e:
        print(s)
    np.savez_compressed(os.path.join(R, OUT, "two_state_series.npz"),
                        **{f"{k}|{v}": a for k, arrs in store.items() for v, a in zip("tqEl", arrs)})
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(len(store), 1, figsize=(10, 2.0 * len(store)), sharex=True)
        for a, (k, (tg, qg, Eg, lab)) in zip(ax, store.items()):
            a.semilogy(tg, qg, lw=0.4, color="0.6")
            a.semilogy(tg, running_median(qg, int(round(W / DT)) + 1), lw=1.0, color="k")
            for s, c in ((1, "tab:red"), (-1, "tab:blue")):
                a.fill_between(tg, 0.05, 30, where=lab == s, color=c, alpha=0.15, lw=0)
            a.axhline(QH, ls=":", c="tab:red"); a.axhline(QL, ls=":", c="tab:blue")
            a.set_ylim(0.05, 30); a.set_ylabel("Q/Q_MHD"); a.set_title(k, fontsize=9, loc="left")
        ax[-1].set_xlabel("t (R/vA)")
        fig.tight_layout()
        fig.savefig(os.path.join(R, OUT, "two_state.png"), dpi=110)
    except Exception as ex:  # plotting is optional
        print("plot failed:", ex)


if __name__ == "__main__":
    main()
