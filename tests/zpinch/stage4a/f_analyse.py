#!/usr/bin/env python3
"""Stage 4a: analysis of one FLUCS production run from its RUN-TIME outputs only (no 3D states, no conversion).
Successor of runs/stage3e/e_analyse.py (same series / two-state / pass-fail logic and criteria).

  python f_analyse.py RUN [--t0 auto|T] [--win-frac 0.8] [--at-times t1,t2,..]   -> RUN/analysis.txt (also stdout)

--t0 auto (default): statistics from t0 = t_sat + 50 (rounded up to 10), t_sat = first time Q/Q_MHD reaches half its
median over the second half of the run (t_saturation(); the campaign extension rule uses the same).
--at-times: spectra, zp_scalars and k_par only from the records at these times (|dt| < 0.05), e.g. the snapshot
times of e_analyse.py, for a like-for-like comparison (maps are window averages and stay as they are).

RUN = folder with segments t????/ (or a single run folder). Uses
  output.0d.nc   energy (Q/Q_MHD, E, E_kx0, dt) + zp_scalars       -> series, runaway, two-state labels (tools/two_state.py)
  output.1d.nc   zp_spectra (1 R/vA)                               -> state-conditioned E(k_perp), E(|k_z|): outer-third
                                                                      energy and pile-up (energy PER MODE, shells 2 / 4)
  output.maps.nc zp_maps (5-R/vA window averages)                  -> per state (windows with >= win-frac of their
                 classified time in one state): dissipation_map.py quantities from the unit-bin (k_perp, |k_z|) maps
                 (the same rint(k_perp) shells as dissipation_map.py, so the per-shell table is identical; k_d is
                 resolved to unit shells instead of per mode): f_par, outer-third D (k_perp, k_z), k_d, kz_rms(k_d),
                 alpha (tilt), k_perp,cut, k_perp,tilt, k_z*, and the shell where kz_rms reaches k_z*
  output.kpar.nc zp_kpar (5 R/vA)                                  -> k_par(k_perp) table per state (kpar_spectrum.py)
Pass/fail per state as e_analyse.py.  Interpreter: ~/flucs_work/.venv/bin/python (CPU)."""
import argparse
import glob
import os
import re
import sys

import numpy as np

HOME = os.path.expanduser("~/flucs_work")
sys.path.insert(0, f"{HOME}/reference/curvy3d")
sys.path.insert(0, f"{HOME}/runs/tools")
sys.path.insert(0, f"{HOME}/flucs/src")
import zpinch_runtools as zr      # noqa: E402
import two_state as ts            # noqa: E402
from flucs_to_curvy3d import parse_toml   # noqa: E402
from flucs.systems.zpinch_fourier.zpinch_readers import load_diag, kpar_table   # noqa: E402

PASS_EOUT, PASS_DOUT, KZ_FRAC = 1e-2, 0.5, 0.8
ST = {1: "H", -1: "L", 0: "-"}


def growth_rate_max(kb, kP):     # as dissipation_map.py / kpar_spectrum.py
    cc = 1 + 1 / kb
    N2 = 2 * kb / (1 + kb) * (kP * (1 + kb) - 2)
    N02 = 2 * kb * kP
    gam = 0.0
    for kzz in (0.0, 1.0):
        r = np.roots([cc, cc * N2 - (1 + cc) * kzz ** 2, kzz ** 2 * (kzz ** 2 - N02)])
        gam = max(gam, np.sqrt(max(0.0, -min(r.real))))
    return gam


def t_saturation(t, q, frac=0.5):
    """first time Q reaches frac x median(Q over the second half of the series); nan if never."""
    t, q = np.asarray(t, float), np.asarray(q, float)
    ok = np.isfinite(q)
    t, q = t[ok], q[ok]
    late = np.median(q[t >= t[0] + 0.5 * (t[-1] - t[0])])
    i = np.flatnonzero(q >= frac * late)
    return float(t[i[0]]) if i.size and late > 0 else np.nan


def auto_t0(t, q):
    ts_ = t_saturation(t, q)
    return ts_, (np.ceil((ts_ + 50.0) / 10.0) * 10.0 if np.isfinite(ts_) else np.nan)


def files(segs, name):
    return [f"{s}/{name}" for s in segs if os.path.exists(f"{s}/{name}")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--t0", default="auto")
    ap.add_argument("--win-frac", type=float, default=0.8)
    ap.add_argument("--at-times", default="")
    a = ap.parse_args()
    RUN = os.path.abspath(a.run.rstrip("/"))
    name = os.path.basename(RUN)
    segs = sorted(glob.glob(f"{RUN}/t[0-9][0-9][0-9][0-9]")) or [RUN]
    out = []

    def say(s=""):
        print(s)
        out.append(s)

    inp = parse_toml(open(f"{segs[0]}/input.toml").read())
    dim, par, hyp = inp["dimensions"], inp["parameters"], inp["hyperdissipation"]
    Lx, Ly, Lz = float(dim.get("Lx", 4 * np.pi)), float(dim.get("Ly", 2 * np.pi)), float(dim.get("Lz", 2 * np.pi))
    kb, kp = float(par["kb"]), float(par["kP"])
    Kx, Ky, Kz = ((int(dim[k]) - 1) // 3 for k in ("nx", "ny", "nz"))
    p, q = int(hyp["kperp_power"]), int(hyp["kz_power"])
    nup, nuz = float(hyp["kperp"]), float(hyp["kz"])
    say(f"== {name}: kb {kb} kP {kp}; padded {dim['nx']} x {dim['ny']} x {dim['nz']}; nu_perp {nup} (k^{2 * p}), "
        f"nu_z {nuz} (k^{2 * q})")

    # ---------------- 1. series + two-state ----------------
    f0 = files(segs, "output.0d.nc")
    t, d0 = load_diag(f0, "energy", ["Q_over_QMHD", "E", "E_kx0"])
    _, dt0 = load_diag(f0, "zp_scalars")
    o = np.argsort(t, kind="stable")
    keep = np.concatenate([np.diff(t[o]) > 0, [True]])
    o = o[keep]
    t, qq, E, kx0 = t[o], d0["Q_over_QMHD"][o], d0["E"][o], d0["E_kx0"][o]
    from netCDF4 import Dataset
    dts = np.concatenate([np.asarray(Dataset(f)[g]["dt"][:], float) for f in f0 for g in sorted(Dataset(f).groups, key=int)])
    early = open(f"{RUN}/EARLY_STOP").read().strip() if os.path.exists(f"{RUN}/EARLY_STOP") else ""
    steps, wall = 0, 0.0
    for s in segs:
        lg = f"{s}/run.log"
        m = re.search(r"given in (.*) \((\d+) steps\)", open(lg).read()) if os.path.exists(lg) else None
        if m:
            steps += int(m.group(2))
            for unit, f in (("hour", 3600), ("minute", 60), ("second", 1)):
                mm = re.search(rf"(\d+) {unit}", m.group(1))
                wall += int(mm.group(1)) * f if mm else 0
    say(f"segments {len(f0)}/{len(segs)}, t {t[0]:.1f}-{t[-1]:.1f}; {steps} steps, wall {wall / 3600:.2f} h "
        f"({1e3 * wall / max(steps, 1):.1f} ms/step); min dt {np.nanmin(dts):.2e}; EARLY_STOP: {early or 'none'}")
    runaway = bool(early) or np.nanmin(dts) < 1e-5
    t_sat, t0_auto = auto_t0(t, qq)
    t0 = max(t0_auto if a.t0 == "auto" else float(a.t0), t[0])
    say(f"saturation t_sat {t_sat:.1f} (Q first >= 0.5 x late median) -> auto t0 {t0_auto:.0f}; using t0 = {t0:.1f}")
    at = np.array([float(x) for x in a.at_times.split(",") if x.strip()])

    def near(tt):
        tt = np.asarray(tt, float)
        if not at.size:
            return np.ones(tt.shape, bool)
        return np.min(np.abs(tt[:, None] - at[None, :]), axis=1) < 0.05
    tg = np.arange(np.ceil(t0 / ts.DT) * ts.DT, t[-1] + 1e-9, ts.DT)
    ok = np.isfinite(qq) & np.isfinite(E)
    qg, Eg = np.interp(tg, t[ok], qq[ok]), np.interp(tg, t[ok], E[ok])
    fk = np.interp(tg, t[ok], kx0[ok] / E[ok])
    lab = np.zeros(len(tg), int)
    if len(tg) > 20:
        mq, eq, tau, _ = zr.mean_and_error(qg)
        say(f"t >= {tg[0]:.0f}: <Q/Q_MHD> {mq:.3f} +- {eq:.3f} (tau_int {tau * ts.DT:.1f}), median {np.median(qg):.3f}, "
            f"<E> {Eg.mean():.0f}, <E_kx0/E> {np.nanmean(fk):.3f}")
        lab, eps = ts.classify(tg, qg, ts.QH, ts.QL)
        say("two-state: run t-range f_high T_H T_L f_cls Q_H Q_L E_H E_L kx0/E(H L) switches episodes")
        say(ts.stats(name, tg, qg, Eg, lab, eps, fk))
    else:
        say(f"series too short after t0 = {t0} for statistics")

    def label_at(tt):
        if len(tg) == 0 or tt < tg[0] - 1e-6 or tt > tg[-1] + 1e-6:
            return 0
        return int(lab[np.argmin(np.abs(tg - tt))])

    def window_label(ta, tb):
        sel = (tg > ta + 1e-9) & (tg <= tb + 1e-9)
        if not sel.any():
            return 0
        L = lab[sel]
        for st in (1, -1):
            if np.mean(L == st) >= a.win_frac:
                return st
        return 0

    # zp_scalars state means
    s0 = (t >= t0) & near(t)
    ts0 = t[s0]
    lab0 = np.array([label_at(x) for x in ts0])
    sc = {k: np.asarray(v)[o][s0] for k, v in dt0.items() if np.ndim(v) == 1 and len(v) == len(d0["E"])}
    bands = sorted(k for k in sc if k.startswith("kzrms_"))
    say("\nzp_scalars, state means (t >= t0): st  n  b_rms u_rms E_ky0/E E_ky0kz0/E S_zon0 S_zon,all W_rms cfl_max | "
        + " ".join(b.replace("kzrms_", "kz[") + "]" for b in bands))
    for st in (1, -1, 0):
        m = lab0 == st
        if m.sum() == 0:
            continue
        g = lambda k: np.nanmean(sc[k][m])
        say(f"   {ST[st]} {m.sum():5d} {np.sqrt(np.nanmean(sc['b_rms'][m] ** 2)):6.2f} {g('u_rms'):5.2f} "
            f"{g('E_ky0_frac'):.3f} {g('E_ky0kz0_frac'):.4f} {g('S_zon0'):.3f} {g('S_zon_all'):.3f} {g('W_rms'):.2f} "
            f"{np.nanmax(sc['cfl_max'][m]):.3f} | " + " ".join(f"{g(b):5.1f}" for b in bands))

    # ---------------- 2. state-conditioned 1D spectra ----------------
    # mode counts (w-weighted) per unit bin for the energy-per-mode pile-up test
    mx = np.arange(-Kx, Kx + 1)
    my = np.arange(0, Ky + 1)
    mz = np.arange(-Kz, Kz + 1)
    KX, KY = np.meshgrid(mx * 2 * np.pi / Lx, my * 2 * np.pi / Ly, indexing="ij")
    wxy = np.where(KY > 0, 2.0, 1.0)
    kperp2d = np.sqrt(KX ** 2 + KY ** 2)
    kpm = min(Kx * 2 * np.pi / Lx, Ky * 2 * np.pi / Ly)
    kzm = Kz * 2 * np.pi / Lz
    nkz_modes = len(mz)
    dkp, dkz = 2.0, 4.0
    # modes counted in the SAME bins as the energy: unit bins rint(k_perp / dk_y) (zp_spectra), then shells
    # rint(unit / 2) (counting by rint(k_perp / 2) directly put energy and mode counts in different shells)
    pbin = np.rint(np.rint(kperp2d / (2 * np.pi / Ly)) / dkp).astype(int)
    cnt_p = np.bincount(pbin.ravel(), (wxy * nkz_modes).ravel())
    cnt_xy = (wxy).sum()
    t1, S1 = load_diag(files(segs, "output.1d.nc"), "zp_spectra")
    kp_c, kz_c = S1["kperp"], S1["kz"]

    def pile(spec, kmax, dk):
        kk = np.arange(len(spec)) * dk
        sel = (kk > 2 / 3 * kmax) & (kk <= kmax + 1e-9) & (spec > 0)
        sp = spec[sel]
        if len(sp) < 3:
            return np.nan
        return float(max(sp[i] / sp[:i].min() for i in range(1, len(sp))))

    spec_rows = {}
    sel1 = (t1 >= t0) & near(t1)
    l1 = np.array([label_at(x) for x in t1])
    for st in (1, -1, 0):
        m = sel1 & (l1 == st)
        if not m.any():
            continue
        Ep = S1["E_kperp"][m].mean(axis=0)
        Ez = S1["E_kz"][m].mean(axis=0)
        Et = Ep.sum()
        pb = np.rint(kp_c / dkp).astype(int)
        Ep2 = np.bincount(pb, Ep, len(cnt_p)) / np.maximum(cnt_p, 1)
        zb = np.rint(kz_c / dkz).astype(int)
        cz = np.where(kz_c > 0, 2.0, 1.0) * cnt_xy           # modes per |k_z| value: +-k_z
        Ez2 = np.bincount(zb, Ez) / np.maximum(np.bincount(zb, cz), 1)
        spec_rows[st] = dict(n=int(m.sum()), Eop=Ep[kp_c > 2 / 3 * kpm].sum() / Et, Eoz=Ez[kz_c > 2 / 3 * kzm].sum() / Et,
                             pp=pile(Ep2, kpm, dkp), pz=pile(Ez2, kzm, dkz), E=Et)
    say(f"\n1D spectra per state (1-R/vA records; k_perp,max {kpm:.1f}, k_z,max {kzm:.0f}; outer third beyond 2/3; pile = "
        "max e(k)/min e(k'<k), e = energy per mode, shells 2 / 4; > 1 = rising)")
    say("   st  n      E  Eout_perp  Eout_z  pile_perp pile_z")
    for st, r in spec_rows.items():
        say(f"   {ST[st]} {r['n']:4d} {r['E']:7.0f} {r['Eop']:10.2e} {r['Eoz']:8.2e} {r['pp']:9.2f} {r['pz']:6.2f}")

    # ---------------- 3. state-conditioned maps (dissipation_map quantities) ----------------
    gam = growth_rate_max(kb, kp)
    k_cut = (gam / nup) ** (1 / (2 * p)) if nup > 0 else np.inf
    kz_star = (gam / nuz) ** (1 / (2 * q)) if nuz > 0 else np.inf
    fm = files(segs, "output.maps.nc")
    dm = {}
    if fm:
        tm, M = load_diag(fm, "zp_maps")
        W = float(np.median(np.diff(tm))) if len(tm) > 1 else 5.0
        lm = np.array([window_label(x - W, x) for x in tm])
        kpc, kzc = M["kperp"], M["kz"]
        for st in (1, -1, 0):
            m = (tm - W >= t0 - 1e-9) & (lm == st)
            if not m.any():
                continue
            wts = M["n_samples"][m]
            avg = {k: np.tensordot(wts, M[f"{k}_map"][m].astype(float), axes=1) / wts.sum()
                   for k in ("E", "inj", "D_perp", "D_par")}
            Es, Dp, Dz = avg["E"].sum(1), avg["D_perp"].sum(1), avg["D_par"].sum(1)
            D = Dp + Dz
            Dt = D.sum()
            with np.errstate(invalid="ignore", divide="ignore"):
                kzr = np.sqrt((avg["E"] * kzc[None, :] ** 2).sum(1) / Es)
            cum = np.cumsum(D) / Dt
            i_d = int(np.searchsorted(cum, 0.5))
            tw = np.concatenate([np.arange(x - W, x + 1e-9, ts.DT) for x in tm[m]])
            b2 = np.interp(tw, t, dt0["b_rms"][o] ** 2)
            b_rms = np.sqrt(b2.mean())
            low = [i for i in range(1, len(kpc)) if 1 <= kpc[i] <= 4 and np.isfinite(kzr[i])]
            alpha = float(np.median([kzr[i] / (b_rms * kpc[i]) for i in low])) if low else np.nan
            ktilt = kz_star / (alpha * b_rms) if alpha > 0 else np.inf
            cross = next((float(kpc[i]) for i in range(len(kpc)) if Es[i] > 0 and kzr[i] >= kz_star), np.inf)
            Dz_k = (avg["D_perp"] + avg["D_par"]).sum(0)
            dm[st] = dict(n=int(m.sum()), b=b_rms, fpar=Dz.sum() / Dt, Dop=D[kpc > 2 / 3 * kpm].sum() / Dt,
                          Doz=Dz_k[kzc > 2 / 3 * kzm].sum() / Dt, kd=kpc[i_d], kzd=kzr[i_d], alpha=alpha,
                          ktilt=ktilt, cross=cross, inj_frac_low=avg["inj"][kpc < 1.5].sum() / avg["inj"].sum())
        say(f"\nmaps per state ({W:g}-R/vA windows, state if >= {a.win_frac:g} of the window classified so): "
            f"gamma {gam:.3f}, k_perp,cut {k_cut:.1f}, k_z* {kz_star:.1f}")
        say("   st  n  b_rms f_par D_out p/z  k_d kz_rms(k_d) alpha k_tilt k(kz*) inj(k<1.5)")
        for st, d in dm.items():
            say(f"   {ST[st]} {d['n']:3d} {d['b']:6.2f} {d['fpar']:.2f} {d['Dop']:.2f}/{d['Doz']:.2f} {d['kd']:4.0f} "
                f"{d['kzd']:8.1f} {d['alpha']:.3f} {d['ktilt']:6.1f} {d['cross']:6.0f} {d['inj_frac_low']:.2f}")

    # ---------------- 4. k_par per state ----------------
    fk_ = files(segs, "output.kpar.nc")
    if fk_:
        tk, _ = load_diag(fk_, "zp_kpar", ["b2"])
        lk = np.array([label_at(x) for x in tk])
        say("\nk_par per state (zp_kpar records; energy-weighted over fields: kz_rms / k_par / k_par,< / k_par,wrong ; chi_Phi)")
        for st in (1, -1, 0):
            sel = (tk >= t0) & (lk == st) & near(tk)
            if not sel.any():
                continue
            T = kpar_table(fk_, mask=lambda tt, s=sel: s)
            say(f"   {ST[st]} n={T['n']} b_rms {T['b_rms']:.2f}: " + "  ".join(
                f"[{lo:g}-{hi:g}] {T['kz_rms'][i]:.1f}/{T['k_par'][i]:.1f}/{T['k_par_large'][i]:.1f}/"
                f"{T['k_par_wrong'][i]:.1f} chi {T['chi'][i]:.2f}"
                for i, (lo, hi) in enumerate(T["shells"]) if np.isfinite(T["kz_rms"][i])))

    # ---------------- 5. pass/fail ----------------
    lim = KZ_FRAC * kzm / np.sqrt(3)
    say(f"\npass/fail per state (runaway: {'FAIL' if runaway else 'pass'}); criteria: kz_rms(k_d) < {KZ_FRAC} k_z,max/sqrt3 "
        f"= {lim:.1f}; E_out < {PASS_EOUT:g} and pile <= 1 (both dirs); D_out < {PASS_DOUT} (warn); "
        "k_perp(kz_rms = k_z*) >= k_perp,cut")
    for st in (1, -1, 0):
        r, d = spec_rows.get(st), dm.get(st)
        if r is None and d is None:
            continue
        eo = r is not None and max(r["Eop"], r["Eoz"]) < PASS_EOUT and not (r["pp"] > 1) and not (r["pz"] > 1)
        kzd = d["kzd"] if d else np.nan
        do = d is not None and max(d["Dop"], d["Doz"]) < PASS_DOUT
        cr = d["cross"] if d else np.nan
        say(f"   {ST[st]}: kz_rms(k_d) {kzd:.1f} {'pass' if kzd < lim else 'FAIL'} | E_out/pile "
            f"{'pass' if eo else 'FAIL'} | D_out {'ok' if do else 'warn'} | k(kz*) {cr:.0f} vs k_cut {k_cut:.1f} "
            f"{'pass' if cr >= k_cut else 'FAIL'}")
    open(f"{RUN}/analysis.txt", "w").write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
