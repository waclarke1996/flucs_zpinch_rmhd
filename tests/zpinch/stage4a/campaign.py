#!/usr/bin/env python3
"""
Stage 4a STEP 5: production-scan campaign tooling (DRY RUN: writes CSV / plan files only, submits nothing).

    python campaign.py points                -> campaign_points.csv (+ printed summary)
    python campaign.py plan [--speed FILE]   -> campaign_plan.txt (job list, GPU-h per GPU type, wall days, disk)
    python campaign.py make RUN --point ID [--t-end auto|T]      -> new chain folder RUN (run.sh, point.json, tEEEE/)
    python campaign.py make RUN --kP X --kb Y --tier T1 [--grid nx,ny,nz --nu-perp A --nu-z B --cadence test ...]
    python campaign.py extend RUN --t-end T                     -> append segments up to T (refuses after EARLY_STOP)
    python campaign.py assess RUN                               -> extension-rule verdict for a finished chain

Chains: segments of SEG = 100 R/vA (multiple of every save_time), white noise 1e-5 seed 257 (1e-3 is NOT linear on
these grids: |X_k| = amplitude on every mode gives E(0) = 0.6 at 1e-5 on T2, i.e. 6e3 at 1e-3, above saturation).
Initial length (rule 8): t_init = ceil_SEG(T_SAT_C / gamma_damped + 50 + T_STAT): saturation (Q first reaches half
its late median) was at t_sat = (12.5-17.3) / gamma_damped for the five Stage 3e anchors -> T_SAT_C = 15.
Extension rule (assess; statistics always counted from the MEASURED saturation, t_s = t_sat + 50 rounded up to 10):
    EARLY_STOP -> final (runaway). Else extend by EXT (rounded to segments) while any of
      T_stat = t_end - t_s < T_STAT;  rel. error of <Q/Q_MHD> (tau_int) > 10 %;  |drift_z| > 3 and relative drift
      > 10 % (zpinch_runtools.drift_z);  two-state (both labels present) with < 10 switches;
    up to t_end = t_s + T_STAT_MAX.

Scan grid: kappaP = 0.4 x 2^(n/4), n = 0..10;  kappaBeta = (0.05/sqrt 2) x 2^(m/2), m = 0..11.
Production = linearly unstable points:
    interchange   kappaP (1 + kappaBeta) > 2
    m = 1         2 kappaBeta kappaP > 1      ("supercritical")
regime: 'supercritical' (m=1 unstable; tier T2), 'interchange' (only interchange; tier T1).
Growth rates from the SPEC dispersion relation on the box's allowed modes (kx = 0.5 i, ky = j >= 1, kz = k,
|i| <= Kx, j <= Ky, |k| <= Kz of the tier's grid):  c w^4 + w^2 [kap c N^2 - (1+c) kz^2] + kz^2 (kz^2 - kap N0^2) = 0,
kap = ky^2/kperp^2, c = 1 + 1/kb, N^2 = 2 kb/(1+kb) (kP(1+kb) - 2), N0^2 = 2 kb kP.  gamma_ideal = max Im w;
gamma_damped = max (Im w - nu_perp kperp^8 - nu_z kz^8) (the same hyperdissipation acts on all four fields, so it
shifts every eigenvalue of a mode by exactly its rate). Points with gamma_damped < 0.05 are flagged 'slow'.
"""
import argparse
import csv
import json
import os
import shutil

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TIERS = {
    "T1": dict(nx=256, ny=128, nz=256, nu_z=7.5e-14),
    "T2": dict(nx=256, ny=128, nz=512, nu_z=5.9e-16),
    "T3": dict(nx=256, ny=128, nz=768, nu_z=3.4e-17),
}
NU_PERP = 4e-11
T_INIT = 500.0
SEG = 100.0
T_SAT_C, T_STAT, T_STAT_MAX, EXT = 15.0, 400.0, 1500.0, 500.0
AMP, SEED = 1e-5, 257
CHAIN = os.path.join(HERE, "chain_run.sh")
ENERGY = '{name = "energy", options = {invariants = true, stop_dt_min = 1e-05, stop_persist_time = 5.0}}'
# (output name, save_time, sample_time or None, diagnostics); "test" = scaled cadence for tiny-grid chain tests
CADENCE = {
    "production": [("0d", 0.1, None, [ENERGY, '{name = "zp_scalars"}']),
                   ("1d", 1.0, None, ['{name = "zp_spectra"}', '{name = "zp_profiles"}']),
                   ("maps", 5.0, 0.5, ['{name = "zp_maps"}']), ("kpar", 5.0, None, ['{name = "zp_kpar"}']),
                   ("slices", 10.0, None, ['{name = "zp_slices"}']), ("kyt", 1.0, None, ['{name = "zp_kytransfer"}']),
                   ("kytavg", 25.0, 1.0, ['{name = "zp_streamer_kzkx"}']), ("shells", 5.0, None, ['{name = "zp_shells"}'])],
    "test": [("0d", 0.1, None, [ENERGY, '{name = "zp_scalars"}']),
             ("1d", 1.0, None, ['{name = "zp_spectra"}', '{name = "zp_profiles"}']),
             ("maps", 2.5, 0.5, ['{name = "zp_maps"}']), ("kpar", 2.5, None, ['{name = "zp_kpar"}']),
             ("slices", 5.0, None, ['{name = "zp_slices"}']), ("kyt", 1.0, None, ['{name = "zp_kytransfer"}']),
             ("kytavg", 5.0, 1.0, ['{name = "zp_streamer_kzkx"}']), ("shells", 2.5, None, ['{name = "zp_shells"}'])],
}


def grid_points():
    for n in range(11):
        for m in range(12):
            kP = 0.4 * 2 ** (n / 4)
            kb = 0.05 / np.sqrt(2) * 2 ** (m / 2)
            yield n, m, kP, kb


def growth(kP, kb, tier):
    T = TIERS[tier]
    Kx, Ky, Kz = ((T[k] - 1) // 3 for k in ("nx", "ny", "nz"))
    kx = 0.5 * np.arange(-Kx, Kx + 1)
    ky = np.arange(1, Ky + 1, dtype=float)
    kz = np.arange(0, Kz + 1, dtype=float)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
    k2 = KX**2 + KY**2
    kap = KY**2 / k2
    c = 1 + 1 / kb
    N2 = 2 * kb / (1 + kb) * (kP * (1 + kb) - 2)
    N02 = 2 * kb * kP
    a, b, cc = c, kap * c * N2 - (1 + c) * KZ**2, KZ**2 * (KZ**2 - kap * N02)
    disc = np.sqrt((b * b - 4 * a * cc).astype(complex))
    gam = np.zeros(k2.shape)
    for s in (+1, -1):
        w = np.sqrt((-b + s * disc) / (2 * a))
        gam = np.maximum(gam, np.abs(w.imag))
    damp = NU_PERP * k2**4 + T["nu_z"] * KZ**8
    i = np.unravel_index(np.argmax(gam), gam.shape)
    j = np.unravel_index(np.argmax(gam - damp), gam.shape)
    return (float(gam[i]), (KX[i], KY[i], KZ[i]), float((gam - damp)[j]), (KX[j], KY[j], KZ[j]))


def points(write=True):
    rows = []
    for n, m, kP, kb in grid_points():
        inter = kP * (1 + kb) > 2
        m1 = 2 * kb * kP > 1
        if not (inter or m1):
            continue
        regime = "supercritical" if m1 else "interchange"
        tier = "T2" if m1 else "T1"
        g, mode, gd, moded = growth(kP, kb, tier)
        rows.append(dict(run_id=f"P{n:02d}B{m:02d}-{tier}-k8", n=n, m=m, kappaP=round(kP, 6),
                         kappaBeta=round(kb, 6), regime=regime, tier=tier, interchange_unstable=int(inter),
                         m1_unstable=int(m1), N0sq=round(2 * kb * kP, 5), gamma_ideal=round(g, 4),
                         mode_ideal="%g/%g/%g" % mode, gamma_damped=round(gd, 4), mode_damped="%g/%g/%g" % moded,
                         slow=int(gd < 0.05), t_init=T_INIT))
    if write:
        with open(os.path.join(HERE, "campaign_points.csv"), "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)
    return rows


def t_initial(gamma_damped, seg=SEG):
    return float(np.ceil((T_SAT_C / gamma_damped + 50.0 + T_STAT) / seg) * seg)


def segment_toml(cfg, tfinal, prev):
    """input.toml of one chain segment; prev = previous segment name (restart from ../prev/state.nc) or None."""
    outs = []
    for name, save, sample, diags in CADENCE[cfg["cadence"]]:
        outs.append(f"[output.{name}]\nsave_time = {save!r}\n" + (f"sample_time = {sample!r}\n" if sample else "")
                    + f"diags = [{', '.join(diags)}]\n")
    rst = (f'restart_from = "../{prev}/state.nc"\n' if prev else "") + \
        "write_restart_file = true\nwrite_steps = 1000000000  # no periodic restarts: only the final (validated) one\n"
    init = (f'method = "white_noise"{"  # ignored: restart" if prev else ""}\namplitude = {cfg["amplitude"]!r}\n'
            f'rand_seed = {cfg["seed"]}\n')
    return f"""# {cfg['run_id']} (campaign.py chain segment; restart chain run by run.sh = chain_run.sh)
[setup]
system = "ZPinchFourier"
solver = "FourierSolver"
precision = "double"
linear = false

[dimensions]
nx = {cfg['nx']}
ny = {cfg['ny']}
nz = {cfg['nz']}

[parameters]
kb = {cfg['kb']!r}
kP = {cfg['kP']!r}

[timestepping]
method = "ssprk3e"

[time]
dt_max = 0.01
tfinal = {tfinal!r}  # duration of this run/segment
max_cfl = 0.5

[hyperdissipation]
kperp = {cfg['nu_perp']!r}
kperp_power = 4
kz = {cfg['nu_z']!r}
kz_power = 4

[init]
{init}
[output]
write_steps = 500

{chr(10).join(outs)}
[restart]
{rst}"""


def seg_names(run):
    return sorted(d for d in os.listdir(run) if len(d) == 5 and d[0] == "t" and d[1:].isdigit())


def add_segments(run, cfg, t_end):
    segs = seg_names(run)
    t_last = int(segs[-1][1:]) if segs else 0
    prev = segs[-1] if segs else None
    seg = cfg["seg"]
    for _, save, _, _ in CADENCE[cfg["cadence"]]:
        if abs(seg / save - round(seg / save)) > 1e-9:
            raise SystemExit(f"segment length {seg} is not a multiple of save_time {save}")
    new = []
    t = t_last
    while t + 1e-9 < t_end:
        t += seg
        name = f"t{int(round(t)):04d}"
        os.makedirs(os.path.join(run, name))
        with open(os.path.join(run, name, "input.toml"), "w") as fh:
            fh.write(segment_toml(cfg, float(seg), prev))
        new.append(name)
        prev = name
    shutil.copy(CHAIN, os.path.join(run, "run.sh"))
    return new


def make(a):
    run = os.path.abspath(a.run)
    if os.path.exists(run):
        raise SystemExit(f"{run} exists: refusing to overwrite (use extend)")
    if a.point:
        r = next((r for r in points(write=False) if r["run_id"].startswith(a.point)), None)
        if r is None:
            raise SystemExit(f"no campaign point {a.point}")
        kP, kb, tier, rid = float(r["kappaP"]), float(r["kappaBeta"]), r["tier"], r["run_id"]
    else:
        kP, kb, tier = a.kP, a.kb, a.tier
        rid = a.run_id or os.path.basename(run)
    T = dict(TIERS[tier])
    if a.grid:
        T["nx"], T["ny"], T["nz"] = (int(x) for x in a.grid.split(","))
    gd = growth(kP, kb, tier)[2]          # on the tier's production grid
    cfg = dict(run_id=rid, kP=kP, kb=kb, tier=tier, nx=T["nx"], ny=T["ny"], nz=T["nz"],
               nu_perp=a.nu_perp if a.nu_perp is not None else NU_PERP,
               nu_z=a.nu_z if a.nu_z is not None else T["nu_z"], seg=a.seg, cadence=a.cadence,
               amplitude=AMP, seed=SEED, gamma_damped=round(gd, 4))
    t_end = t_initial(gd, a.seg) if a.t_end == "auto" else float(a.t_end)
    cfg["t_init"] = t_end
    os.makedirs(run)
    with open(os.path.join(run, "point.json"), "w") as fh:
        json.dump(cfg, fh, indent=1)
    new = add_segments(run, cfg, t_end)
    print(f"{run}: {rid} kP {kP:.4f} kb {kb:.4f} {tier} grid {cfg['nx']}x{cfg['ny']}x{cfg['nz']} gamma_d {gd:.3f}; "
          f"segments {new[0]}..{new[-1]} (t_end {t_end:g})")


def extend(a):
    run = os.path.abspath(a.run)
    if os.path.exists(os.path.join(run, "EARLY_STOP")):
        raise SystemExit(f"{run}: EARLY_STOP present (runaway): not extended")
    cfg = json.load(open(os.path.join(run, "point.json")))
    new = add_segments(run, cfg, float(a.t_end))
    print(f"{run}: added {len(new)} segments {new[0] if new else ''}..{new[-1] if new else ''}; "
          f"run.sh updated; resubmit the folder")


def assess(a):
    """extension-rule verdict for a chain (CPU, from output.0d.nc)."""
    import sys
    sys.path.insert(0, os.path.expanduser("~/flucs_work/reference/curvy3d"))
    sys.path.insert(0, os.path.expanduser("~/flucs_work/runs/tools"))
    sys.path.insert(0, os.path.expanduser("~/flucs_work/flucs/src"))
    import two_state as ts
    import zpinch_runtools as zr
    from f_analyse import auto_t0
    from flucs.systems.zpinch_fourier.zpinch_readers import load_diag
    run = os.path.abspath(a.run)
    cfg = json.load(open(os.path.join(run, "point.json")))
    segs = seg_names(run)
    done = [x for x in segs if os.path.exists(os.path.join(run, x, "SEG_OK"))]
    if os.path.exists(os.path.join(run, "EARLY_STOP")):
        print(f"{cfg['run_id']}: EARLY_STOP ({open(os.path.join(run, 'EARLY_STOP')).read().strip()}) -> final")
        return
    if len(done) < len(segs):
        print(f"{cfg['run_id']}: {len(done)}/{len(segs)} segments done -> resubmit (not finished)")
        return
    t, d = load_diag([os.path.join(run, x, "output.0d.nc") for x in done], "energy", ["Q_over_QMHD"])
    o = np.argsort(t, kind="stable")
    t, q = t[o], d["Q_over_QMHD"][o]
    t_sat, t_s = auto_t0(t, q)
    t_end = float(t[-1])
    tg = np.arange(t_s, t_end + 1e-9, ts.DT)
    qg = np.interp(tg, t, q)
    reasons = []
    if t_end - t_s < T_STAT - 1e-6:
        reasons.append(f"T_stat {t_end - t_s:.0f} < {T_STAT:g}")
    if len(tg) > 20:
        m, e, _, _ = zr.mean_and_error(qg)
        if e / m > 0.10:
            reasons.append(f"rel. error {e / m:.2f} > 0.10")
        z = zr.drift_z(qg)
        h = len(qg) // 2
        rd = abs(qg[h:].mean() - qg[:h].mean()) / m
        if abs(z) > 3 and rd > 0.10:
            reasons.append(f"drift z {z:.1f}, {rd:.0%}")
        lab, eps = ts.classify(tg, qg, ts.QH, ts.QL)
        sw = sum(1 for x, y in zip(eps[:-1], eps[1:]) if x[0] != y[0])
        if (lab == 1).any() and (lab == -1).any() and sw < 10:
            reasons.append(f"two-state, {sw} switches < 10")
        info = f"<Q/Q_MHD> {m:.3f} +- {e:.3f}, switches {sw}"
    else:
        info = "too short for statistics"
    cap = t_s + T_STAT_MAX
    if reasons and t_end < cap - 1e-6:
        target = t_s + T_STAT if t_end - t_s < T_STAT else t_end + EXT
        new_end = float(np.ceil(min(target, cap) / cfg["seg"] - 1e-9) * cfg["seg"])
        print(f"{cfg['run_id']}: t_sat {t_sat:.1f}, stats from {t_s:g} to {t_end:.0f}; {info}; EXTEND to {new_end:g} "
              f"({'; '.join(reasons)}):  python campaign.py extend {run} --t-end {new_end:g}")
    else:
        print(f"{cfg['run_id']}: t_sat {t_sat:.1f}, stats from {t_s:g} to {t_end:.0f}; {info}; FINAL"
              + (f" (cap t_s + {T_STAT_MAX:g} reached; open: {'; '.join(reasons)})" if reasons else ""))


def steps_per_rva(n0sq):
    """time steps per R/vA vs drive N0^2 = 2 kb kP: Stage 3e anchors A2/A4 (1.28): 1180-1280/1120, A3b (1.81) 2065,
    A3a (2.56) 2825 (ssprk3e, CFL 0.5, same k_perp grid in T1-T3). Linear fit; beyond 2.56 an extrapolation."""
    return max(1100.0, 1270.0 * n0sq - 425.0)


def plan(a):
    rows = points(write=False)
    sp = {}
    for line in open(a.speed):
        if line.startswith("#") or not line.strip():
            continue
        tier, gpu, ms = line.split()[:3]
        sp[(tier, gpu)] = float(ms)
    gpus = sorted({g for _, g in sp}, key=lambda g: ["h200", "a100", "rtx"].index(g) if g in
                  ("h200", "a100", "rtx") else 9)
    slots = dict(h200=4, a100=2, rtx=2)
    disk_state = dict(T1=0.27, T2=0.55, T3=0.82)      # GB apparent per kept state.nc (T1/T2 measured)
    disk_out = 0.175                                   # GB per 1000 R/vA run-time outputs (T2 measured, 10 R/vA)
    jobs, out = [], []
    for r in rows:
        tier, n0 = r["tier"], float(r["N0sq"])
        tin = t_initial(float(r["gamma_damped"]))
        nst = steps_per_rva(n0) * tin
        h = {g: nst * sp[(tier, g)] / 3.6e6 for g in gpus if (tier, g) in sp}
        jobs.append(dict(r, t_init=tin, steps=nst, h=h, extrap=n0 > 2.56))
    tot = {g: sum(j["h"].get(g, 0) for j in jobs) for g in gpus}
    out.append(f"# campaign plan (campaign.py plan, speed file {a.speed}); {len(jobs)} points, initial lengths only "
               f"(extensions: + up to {EXT:g} R/vA each when the rule asks; budget +50 % below)")
    out.append("# GPU-h per point = steps_per_rva(N0^2) x t_init x ms/step; '*' = N0^2 > 2.56 (extrapolated cost)")
    out.append("run_id            tier  N0^2  gamma_d t_init  steps/RvA " + " ".join(f"{g + '_h':>7s}" for g in gpus))
    for j in jobs:
        out.append(f"{j['run_id']:17s} {j['tier']}  {float(j['N0sq']):5.2f}  {float(j['gamma_damped']):.3f} "
                   f"{j['t_init']:6.0f} {steps_per_rva(float(j['N0sq'])):9.0f}{'*' if j['extrap'] else ' '} "
                   + " ".join(f"{j['h'].get(g, np.nan):7.1f}" for g in gpus))
    out.append("TOTAL GPU-h if all on one type: " + ", ".join(f"{g} {tot[g]:.0f}" for g in gpus))
    # schedule: longest-first onto the fastest free slot (each point = one job = one chain)
    free = [(0.0, g, k) for g in gpus for k in range(slots.get(g, 0))]
    used = {g: 0.0 for g in gpus}
    for j in sorted(jobs, key=lambda j: -j["h"][gpus[0]]):
        best = min(free, key=lambda s: s[0] + j["h"][s[1]])
        free.remove(best)
        free.append((best[0] + j["h"][best[1]], best[1], best[2]))
        used[best[1]] += j["h"][best[1]]
    mk = max(f[0] for f in free)
    out.append(f"SCHEDULE (slots {slots}, every slot free, longest first): wall-clock {mk / 24:.1f} days "
               f"(+50 % extensions: {1.5 * mk / 24:.1f} days); GPU-h used " + ", ".join(f"{g} {used[g]:.0f}" for g in gpus))
    h200_only = sorted((j["h"]["h200"] for j in jobs), reverse=True)
    lanes = [0.0] * slots["h200"]
    for h in h200_only:
        lanes[lanes.index(min(lanes))] += h
    out.append(f"H200 only (4 slots): {max(lanes) / 24:.1f} days")
    dk = sum(disk_state[j["tier"]] + disk_out * j["t_init"] / 1000 for j in jobs)
    out.append(f"DISK: final state + run-time outputs {dk:.0f} GB apparent (+50 % extensions {1.5 * dk:.0f} GB); "
               f"one extra state per running job during the swap; summary .nc ~ state/4 each if made with snapshot")
    txt = "\n".join(out) + "\n"
    open(os.path.join(HERE, "campaign_plan.txt"), "w").write(txt)
    print(txt)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["points", "plan", "make", "extend", "assess"])
    ap.add_argument("run", nargs="?")
    ap.add_argument("--speed", default=os.path.join(HERE, "speed.txt"))
    ap.add_argument("--point", default=None)
    ap.add_argument("--kP", type=float)
    ap.add_argument("--kb", type=float)
    ap.add_argument("--tier", default="T2")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--grid", default=None, help="nx,ny,nz (tests)")
    ap.add_argument("--nu-perp", type=float, default=None)
    ap.add_argument("--nu-z", type=float, default=None)
    ap.add_argument("--seg", type=float, default=SEG)
    ap.add_argument("--cadence", default="production", choices=list(CADENCE))
    ap.add_argument("--t-end", default="auto")
    a = ap.parse_args()
    if a.what in ("make", "extend", "assess", "plan"):
        {"make": make, "extend": extend, "assess": assess, "plan": plan}[a.what](a)
        return
    rows = points()
    ns = sum(r["regime"] == "supercritical" for r in rows)
    print(f"{len(rows)} unstable points: {ns} supercritical (T2), {len(rows) - ns} interchange-only (T1); "
          f"slow (gamma_damped < 0.05): {[r['run_id'] for r in rows if r['slow']]}")
    for r in rows:
        print(f"{r['run_id']:16s} kP {r['kappaP']:7.4f} kb {r['kappaBeta']:7.4f} {r['regime']:13s} "
              f"N0^2 {r['N0sq']:6.3f} gamma {r['gamma_ideal']:.3f} ({r['mode_ideal']}) "
              f"damped {r['gamma_damped']:.3f} ({r['mode_damped']}){'  SLOW' if r['slow'] else ''}")


if __name__ == "__main__":
    main()
