#!/usr/bin/env python3
"""Stage 3e part 2: pass/fail analysis of one FLUCS segment-chain run (folder RUN with t*/ segments, run.sh chain).

  python e_analyse.py RUN [--t-offset 0] [--t0 100] [--no-dmap] [--keep-nc]

1. Series (all t*/output.0d.nc): runaway check (EARLY_STOP file, min dt), <Q/Q_MHD>, <E>, <E_kx0/E> for t >= t0 with
   autocorrelation-corrected errors (zpinch_runtools.mean_and_error); two-state classification (runs/tools/two_state.py,
   same thresholds) -> f_high, Q_H, Q_L, E_H, E_L, kx0/E per state, switches, episodes.
2. States (each t*/restart.nc, t >= t0), direct: E, b_rms, E_kx0/E, E_ky0/E, E_ky0kz0/E, zonal-shear rms (k_y=k_z=0),
   outer-third energy fraction in k_perp (k_perp > 2/3 k_perp,max) and in |k_z| (> 2/3 k_z,max), pile-up = the
   shell spectrum E(k_perp) or E(|k_z|) rises towards the edge inside the outer third (max E(k)/min over smaller k there).
3. dissipation_map.py (reference/curvy3d, fixed version) on the converted states, one call per contiguous group of
   same-state snapshots (t >= t0); parsed: b_rms, f_par, outer-third D in k_perp / k_z, k_d and kz_rms at k_d,
   k_perp,cut, k_perp,tilt, k_z*, and the lab k_perp shell where kz_rms first reaches k_z* (cut-off 2).
4. Pass/fail table per state (H, L, or 'all' if the run never classifies). Criteria (provisional, see PASS_*):
   runaway none; kz_rms(k_d) < 0.8 k_z,max/sqrt(3); outer-third E < 1e-2 both directions and no pile-up;
   outer-third D < 0.5 both directions (warn only); k_perp(kz_rms = k_z*) >= k_perp,cut (nu_z does not cut first).
Output: RUN/analysis.txt (also stdout; with --no-dmap the dissipation columns are nan/FAIL); converted states in RUN_state/ (deleted unless --keep-nc; the dmap .txt kept).
Interpreter: ~/flucs_work/.venv/bin/python (CPU)."""
import argparse
import glob
import os
import re
import subprocess
import sys

import numpy as np
from netCDF4 import Dataset

HOME = os.path.expanduser("~/flucs_work")
sys.path.insert(0, f"{HOME}/reference/curvy3d")
sys.path.insert(0, f"{HOME}/runs/tools")
import zpinch_runtools as zr      # noqa: E402
import two_state as ts            # noqa: E402
from flucs_to_curvy3d import parse_toml   # noqa: E402

PY = sys.executable   # run with ~/flucs_work/.venv/bin/python (has tomli; the curvy3d interpreter has no TOML parser)
NAMES = ("Phi", "Psi", "deltaBpar", "upar")
PASS_EOUT, PASS_DOUT, KZ_FRAC = 1e-2, 0.5, 0.8

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--t-offset", type=float, default=0.0)
ap.add_argument("--t0", type=float, default=100.0)
ap.add_argument("--no-dmap", action="store_true")
ap.add_argument("--keep-nc", action="store_true")
ap.add_argument("--redo", action="store_true", help="recompute dmap .txt files that already exist")
a = ap.parse_args()
RUN = os.path.abspath(a.run.rstrip("/"))
name = os.path.basename(RUN)
segs = sorted(glob.glob(f"{RUN}/t[0-9][0-9][0-9][0-9]"))
out_lines = []


def say(s=""):
    print(s)
    out_lines.append(s)


inp = parse_toml(open(f"{segs[0]}/input.toml").read())
dim, par, hyp = inp["dimensions"], inp["parameters"], inp["hyperdissipation"]
Lx, Ly, Lz = float(dim.get("Lx", 4 * np.pi)), float(dim.get("Ly", 2 * np.pi)), float(dim.get("Lz", 2 * np.pi))
kb, kp = float(par["kb"]), float(par["kP"])
say(f"== {name}: kb {kb} kP {kp}; padded {dim['nx']} x {dim['ny']} x {dim['nz']}; L {Lx:.3f} {Ly:.3f} {Lz:.3f}; "
    f"nu_perp {hyp['kperp']} (p {hyp['kperp_power']}), nu_z {hyp['kz']} (q {hyp['kz_power']})")

# ---------------- 1. series ----------------
done = [s for s in segs if os.path.exists(f"{s}/output.0d.nc")]
pieces = [tuple(ts.flucs_piece(f"{s}/output.0d.nc", a.t_offset)) for s in done]
t, q, E, kx = ts.concat(pieces)
dts = np.concatenate([np.asarray(Dataset(f"{s}/output.0d.nc")[sorted(Dataset(f"{s}/output.0d.nc").groups, key=int)[-1]]["dt"][:], float)
                      for s in done])
early = open(f"{RUN}/EARLY_STOP").read().strip() if os.path.exists(f"{RUN}/EARLY_STOP") else ""
steps = 0
wall = 0.0
for s in done:
    m = re.search(r"given in (.*) \((\d+) steps\)", open(f"{s}/run.log").read()) if os.path.exists(f"{s}/run.log") else None
    if m:
        steps += int(m.group(2))
        h = re.search(r"(\d+) hour", m.group(1)); mi = re.search(r"(\d+) minute", m.group(1)); se = re.search(r"(\d+) second", m.group(1))
        wall += (int(h.group(1)) * 3600 if h else 0) + (int(mi.group(1)) * 60 if mi else 0) + (int(se.group(1)) if se else 0)
say(f"segments {len(done)}/{len(segs)}, t {t[0]:.1f}-{t[-1]:.1f}; {steps} steps, wall {wall / 3600:.2f} h "
    f"({1e3 * wall / max(steps, 1):.1f} ms/step); min dt {np.nanmin(dts):.2e}; EARLY_STOP: {early or 'none'}")
runaway = bool(early) or np.nanmin(dts) < 1e-5
w = t >= a.t0
tg = np.arange(np.ceil(max(t[0], ts.T0, a.t0) / ts.DT) * ts.DT, t[-1] + 1e-9, ts.DT)
ok = np.isfinite(q) & np.isfinite(E)
qg, Eg = np.interp(tg, t[ok], q[ok]), np.interp(tg, t[ok], E[ok])
fk = np.interp(tg, t[ok], kx[ok] / E[ok]) if kx is not None else None
if len(tg) > 20:
    mq, eq, tau, _ = zr.mean_and_error(qg)
    say(f"t >= {tg[0]:.0f}: <Q/Q_MHD> {mq:.3f} +- {eq:.3f} (tau_int {tau * ts.DT:.1f}), median {np.median(qg):.3f}, "
        f"<E> {Eg.mean():.0f}, <E_kx0/E> {np.nanmean(fk):.3f}")
    lab, eps = ts.classify(tg, qg, ts.QH, ts.QL)
    say("two-state: run t-range f_high T_H T_L f_cls Q_H Q_L E_H E_L kx0/E(H L) switches episodes")
    say(ts.stats(name, tg, qg, Eg, lab, eps, fk))
else:
    lab = np.zeros_like(tg, dtype=int)
    say(f"series too short after t0 = {a.t0} for statistics")


def label_at(tt):
    if len(tg) == 0 or tt < tg[0] - 1e-6 or tt > tg[-1] + 1e-6:
        return 0
    return int(lab[np.argmin(np.abs(tg - tt))])


# ---------------- 2. states, direct ----------------
def state_file(s):
    """Stage 4 chains keep the validated state as state.nc; Stage 3e chains as restart.nc."""
    for nm in ("state.nc", "restart.nc"):
        if os.path.exists(f"{s}/{nm}"):
            return f"{s}/{nm}"
    return None


states = []
shape = None
finished = [s for s in done if os.path.exists(f"{s}/run.log") and "given in" in open(f"{s}/run.log").read()]
for s in finished:   # a running segment's restart.nc is an intermediate write
    f = state_file(s)
    if f is None:
        continue
    with Dataset(f) as ds:
        tt = float(ds["current_time"][...]) + a.t_offset
        if tt < a.t0 - 1e-6:
            continue
        X = np.array(ds["fields_real"][:]) + 1j * np.array(ds["fields_imag"][:])
    F = dict(zip(NAMES, X))
    if shape != F["Phi"].shape:
        shape = F["Phi"].shape
        nz, nx, hy = shape
        kxv = np.fft.fftfreq(nx, d=1.0 / nx) * 2 * np.pi / Lx
        kyv = np.arange(hy) * 2 * np.pi / Ly
        kzv = np.fft.fftfreq(nz, d=1.0 / nz) * 2 * np.pi / Lz
        KZ, KX, KY = np.meshgrid(kzv, kxv, kyv, indexing="ij")
        k2 = KX ** 2 + KY ** 2
        kperp = np.sqrt(k2)
        wgt = np.where(KY > 0, 2.0, 1.0)
        kpm = min((dim["nx"] - 1) // 3 * 2 * np.pi / Lx, (dim["ny"] - 1) // 3 * 2 * np.pi / Ly)
        kzm = (dim["nz"] - 1) // 3 * 2 * np.pi / Lz
        dkp = 2.0   # pile-up shells: width 2 in k_perp (k_y integer, k_x step 0.5: finer bins are noisy), 4 in |k_z|
        dkz = 4.0
        pbin = np.rint(kperp / dkp).astype(int)
        zbin = np.rint(np.abs(KZ) / dkz).astype(int)
    p2 = np.abs(F["Phi"]) ** 2 * wgt
    eb = 0.5 * k2 * np.abs(F["Psi"]) ** 2 * wgt
    e = 0.5 * k2 * p2 + eb + 0.5 * (np.abs(F["upar"]) ** 2 + (1 + 1 / kb) * np.abs(F["deltaBpar"]) ** 2) * wgt
    Et = e.sum()
    Ep = np.bincount(pbin.ravel(), e.ravel()) / np.maximum(np.bincount(pbin.ravel(), wgt.ravel()), 1)   # energy per mode
    Ez = np.bincount(zbin.ravel(), e.ravel()) / np.maximum(np.bincount(zbin.ravel(), wgt.ravel()), 1)

    def pile(spec, kmax, dk):
        kk = np.arange(len(spec)) * dk
        sel = (kk > 2 / 3 * kmax) & (kk <= kmax + 1e-9) & (spec > 0)
        sp = spec[sel]
        if len(sp) < 3:
            return np.nan
        return float(max(sp[i] / sp[:i].min() for i in range(1, len(sp))))   # > 1: rises towards the edge

    states.append(dict(t=tt, l=label_at(tt), E=Et, b=np.sqrt(2 * eb.sum()), kx0=e[KX == 0].sum() / Et,
                       ky0=e[KY == 0].sum() / Et, ky0kz0=e[(KY == 0) & (KZ == 0)].sum() / Et,
                       S0=np.sqrt((k2 ** 2 * p2)[(KY == 0) & (KZ == 0)].sum()),
                       Eop=e[kperp > 2 / 3 * kpm].sum() / Et, Eoz=e[np.abs(KZ) > 2 / 3 * kzm].sum() / Et,
                       pp=pile(Ep, kpm, dkp), pz=pile(Ez, kzm, dkz)))
ST = {1: "H", -1: "L", 0: "-"}
if not states:
    say(f"no 3D states with t >= {a.t0} yet")
    open(f"{RUN}/analysis.txt", "w").write("\n".join(out_lines) + "\n")
    sys.exit(0)
say(f"\nstates (direct; k_perp,max {kpm:.1f}, k_z,max {kzm:.0f}; outer third = beyond 2/3 of these; "
    f"pile = max e(k)/min e(k'<k), e = energy PER MODE in shells of width 2 (k_perp) / 4 (k_z) in the outer third; > 1 = rising)")
say("     t st      E  b_rms kx0/E ky0/E ky0kz0/E S_zon0  Eout_perp  Eout_z  pile_perp pile_z")
for r in states:
    say(f"{r['t']:6.0f} {ST[r['l']]} {r['E']:7.0f} {r['b']:6.2f} {r['kx0']:.3f} {r['ky0']:.3f} {r['ky0kz0']:8.4f} "
        f"{r['S0']:6.3f} {r['Eop']:10.2e} {r['Eoz']:8.2e} {r['pp']:9.2f} {r['pz']:6.2f}")

# ---------------- 3. dissipation_map per same-state group ----------------
dm = []
if not a.no_dmap and states:
    sd = f"{RUN}_state"
    os.makedirs(sd, exist_ok=True)
    nc = f"{sd}/{name}.nc"
    rs = [state_file(s) for s in finished if state_file(s)]
    def convert():
        if os.path.exists(nc):
            return
        subprocess.run([PY, f"{HOME}/runs/tools/flucs_to_curvy3d.py", *rs, "--out", nc, "--run-id", name,
                        "--t-offset", str(a.t_offset), "--series", ",".join(f"{s}/output.0d.nc" for s in done)],
                       check=True, stdout=subprocess.DEVNULL)

    groups = []
    for r in states:
        if groups and groups[-1][0] == r["l"]:
            groups[-1][2] = r["t"]; groups[-1][3] += 1
        else:
            groups.append([r["l"], r["t"], r["t"], 1])
    for st, ta, tb, n in groups:
        tag = f"{ST[st]}_{ta:.0f}" + (f"-{tb:.0f}" if tb > ta else "")
        txt = f"{sd}/dmap_{tag}.txt"
        if a.redo or not os.path.exists(txt):
            convert()
            res = subprocess.run([PY, f"{HOME}/reference/curvy3d/dissipation_map.py", nc, "--tmin", f"{ta - 1:.0f}", "--tmax", f"{tb + 1:.0f}",
                                  "--out", f"{sd}/dmap_{tag}.png"], capture_output=True, text=True)
            open(txt, "w").write(res.stdout + res.stderr[-2000:])
        s = open(txt).read()
        g = lambda pat, i=1: float(re.search(pat, s, re.M).group(i)) if re.search(pat, s, re.M) else np.nan
        tab = s[s.find(" k_perp       E(k)"):s.find("(inj/D > 1")]
        shells = re.findall(r"^\s+([\d.]+)\s+[\d.e+-]+\s+[\d.e+-]+\s+[\d.e+-]+\s+[-\d.e+]+\s+[-\d.e+]+\s+[\d.]+\s+([\d.]+)\s",
                            tab, re.M)
        kzs = g(r"k_z\*/\(alpha b\) = ([\d.]+)/")
        cross = next((float(k) for k, z in shells if float(z) >= kzs), np.inf)
        dm.append(dict(st=st, tag=tag, n=n, b=g(r"^\s+mean\s+[\d.e+]+\s+([\d.]+)", 1) if n > 1 else g(r"^\s+[\d.]+\s+[\d.e+]+\s+([\d.]+)"),
                       fpar=g(r"f_par = D_par / D = ([\d.]+)"),
                       Dop=g(r"k_perp: energy [\d.e+-]+, dissipation ([\d.e+-]+);"), Doz=g(r"k_z: energy [\d.e+-]+, dissipation ([\d.e+-]+)"),
                       kd=g(r"k_d = ([\d.]+): kz_rms"), kzd=g(r"k_d = [\d.]+: kz_rms = ([\d.]+)"),
                       kcut=g(r"k_perp,cut\s+= ([\d.]+)"), ktilt=g(r"k_perp,tilt = k_z\*/\(alpha b\) = [\d.]+/\(alpha b\) = ([\d.]+)"),
                       kzs=kzs, cross=cross, kzmax=g(r"k_z,max = ([\d.]+)"), need=g(r"needed .*~ ([\d.]+) \(now")))
    if not a.keep_nc and os.path.exists(nc):
        os.remove(nc)

# ---------------- 4. pass/fail per state ----------------
say(f"\npass/fail per state (runaway: {'FAIL' if runaway else 'pass'}); criteria: kz_rms(k_d) < {KZ_FRAC} k_z,max/sqrt3; "
    f"E_out < {PASS_EOUT:g} and pile <= 1 (both dirs); D_out < {PASS_DOUT} (warn); k_perp(kz_rms=k_z*) >= k_perp,cut")
say("st  n  b_rms  E kx0/E ky0/E | kz_rms(k_d)/limit   | E_out p/z (pile p/z)      | D_out p/z  | f_par | k_cut k_tilt k(kz*) | kz,max need")
for st in (1, -1, 0):
    S = [r for r in states if r["l"] == st]
    G = [d for d in dm if d["st"] == st]
    if not S:
        continue
    mean = lambda L, k: np.nansum([d[k] * d["n"] for d in L]) / max(sum(d["n"] for d in L if np.isfinite(d[k])), 1)
    sm = lambda k: np.mean([r[k] for r in S])
    pmax = lambda k: np.nanmax([r[k] for r in S])
    lim = KZ_FRAC * kzm / np.sqrt(3)
    kzd = mean(G, "kzd") if G else np.nan
    eo = max(sm("Eop"), sm("Eoz")) < PASS_EOUT and pmax("pp") <= 1.0 and pmax("pz") <= 1.0
    do = max(mean(G, "Dop"), mean(G, "Doz")) < PASS_DOUT if G else False
    cr = min(d["cross"] for d in G) if G else np.nan
    kc = mean(G, "kcut") if G else np.nan
    say(f"{ST[st]:2s} {len(S):2d} {sm('b'):6.2f} {sm('E'):5.0f} {sm('kx0'):.2f} {sm('ky0'):.3f} | "
        f"{kzd:5.1f}/{lim:5.1f} {'pass' if kzd < lim else 'FAIL'} | {sm('Eop'):.1e}/{sm('Eoz'):.1e} ({pmax('pp'):.2f}/{pmax('pz'):.2f}) "
        f"{'pass' if eo else 'FAIL'} | {mean(G, 'Dop'):.2f}/{mean(G, 'Doz'):.2f} {'ok' if do else 'warn'} | "
        f"{mean(G, 'fpar'):.2f} | {kc:5.1f} {mean(G, 'ktilt'):5.1f} {cr:5.0f} {'pass' if cr >= kc else 'FAIL'} | "
        f"{kzm:.0f} {mean(G, 'need'):.0f}")
for d in dm:
    say(f"   dmap {d['tag']}: b {d['b']:.2f} f_par {d['fpar']:.2f} D_out {d['Dop']:.2f}/{d['Doz']:.2f} k_d {d['kd']:.1f} "
        f"kz_rms(k_d) {d['kzd']:.0f} k_cut {d['kcut']:.1f} k_tilt {d['ktilt']:.1f} k_z* {d['kzs']:.1f} k(kz*) {d['cross']:.0f}")
open(f"{RUN}/analysis.txt", "w").write("\n".join(out_lines) + "\n")
