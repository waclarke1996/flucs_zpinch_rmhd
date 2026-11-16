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
T0_OFFSET = 50.0       # statistics from t_sat + T0_OFFSET (rounded up to 10), as f_analyse --t0 auto
# tiny-grid tests only: CAMPAIGN_TEST_RULE="T_STAT,T_STAT_MAX,EXT,T0_OFFSET" overrides the extension rule
if os.environ.get("CAMPAIGN_TEST_RULE"):
    T_STAT, T_STAT_MAX, EXT, T0_OFFSET = (float(x) for x in os.environ["CAMPAIGN_TEST_RULE"].split(","))
LANE = os.path.join(HERE, "lane_worker.sh")
# on-disk sizes (ZFS compresses the padded zeros of a state; outputs are incompressible): measured 2026-11-16
DISK_STATE = dict(T1=0.080, T2=0.159, T3=0.24)      # GB on disk per state.nc (T3 scaled)
DISK_OUT = dict(T1=0.0079, T2=0.0110, T3=0.0141)    # GB per 100 R/vA, all run-time outputs (T1 measured, T2/T3 scaled)
DISK_SNAP = dict(T1=0.038, T2=0.079, T3=0.118)     # GB: float32 snapshot in a summary .nc (optional)
DISK_FIXED = 0.003                                 # GB per run: summary without snapshot, logs, inputs
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
               amplitude=AMP, seed=SEED, gamma_damped=round(gd, 4), auto_extend=bool(a.auto_extend))
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
    from f_analyse import t_saturation
    from flucs.systems.zpinch_fourier.zpinch_readers import load_diag
    run = os.path.abspath(a.run)
    cfg = json.load(open(os.path.join(run, "point.json")))
    segs = seg_names(run)
    done = [x for x in segs if os.path.exists(os.path.join(run, x, "SEG_OK"))]
    if os.path.exists(os.path.join(run, "EARLY_STOP")):
        return _verdict(a, run, cfg, "FINAL", None, f"{cfg['run_id']}: EARLY_STOP "
                        f"({open(os.path.join(run, 'EARLY_STOP')).read().strip()}) -> FINAL")
    if len(done) < len(segs):
        print(f"{cfg['run_id']}: {len(done)}/{len(segs)} segments done -> resubmit (not finished)")
        if getattr(a, "apply", False):
            raise SystemExit(2)
        return
    t, d = load_diag([os.path.join(run, x, "output.0d.nc") for x in done], "energy", ["Q_over_QMHD"])
    o = np.argsort(t, kind="stable")
    t, q = t[o], d["Q_over_QMHD"][o]
    t_sat = t_saturation(t, q)
    t_end = float(t[-1])
    if not np.isfinite(t_sat):
        if t_end >= T_STAT_MAX - 1e-6:
            return _verdict(a, run, cfg, "FINAL", None, f"{cfg['run_id']}: not saturated by t = {t_end:.0f}; FINAL (cap)")
        new_end = float(np.ceil(min(t_end + EXT, T_STAT_MAX) / cfg["seg"] - 1e-9) * cfg["seg"])
        return _verdict(a, run, cfg, "EXTEND", new_end, f"{cfg['run_id']}: not saturated by t = {t_end:.0f}; "
                        f"EXTEND to {new_end:g}")
    t_s = float(np.ceil((t_sat + T0_OFFSET) / 10.0) * 10.0)
    tg = np.arange(t_s, t_end + 1e-9, ts.DT)
    qg = np.interp(tg, t, q)
    reasons = []
    if t_end - t_s < T_STAT - 1e-6:
        reasons.append(f"T_stat {t_end - t_s:.0f} < {T_STAT:g}")
    if len(tg) > 20:
        m, e, _, _ = zr.mean_and_error(qg)
        if e / m > 0.10:
            reasons.append(f"rel. error {e / m:.2f} > 0.10")
        z, rd = zr.drift_z(qg)          # (z, relative change of the half means): zpinch_runtools convention
        if np.isfinite(z) and abs(z) > 3 and abs(rd) > 0.10:
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
        return _verdict(a, run, cfg, "EXTEND", new_end, f"{cfg['run_id']}: t_sat {t_sat:.1f}, stats from {t_s:g} to "
                        f"{t_end:.0f}; {info}; EXTEND to {new_end:g} ({'; '.join(reasons)})")
    return _verdict(a, run, cfg, "FINAL", None, f"{cfg['run_id']}: t_sat {t_sat:.1f}, stats from {t_s:g} to {t_end:.0f}; "
                    f"{info}; FINAL" + (f" (cap t_s + {T_STAT_MAX:g} reached; open: {'; '.join(reasons)})" if reasons else ""))


def _verdict(a, run, cfg, what, new_end, msg):
    """print the verdict; with --apply (used by chain_run.sh on the node): EXTEND -> add the segments up to new_end
    and exit 10; FINAL -> write RUN/FINAL and exit 0. Every verdict is appended to RUN/assess.log."""
    import datetime
    import sys
    print(msg)
    with open(os.path.join(run, "assess.log"), "a") as fh:
        fh.write(f"{datetime.datetime.now().isoformat(timespec='seconds')} {msg}\n")
    if not getattr(a, "apply", False):
        if what == "EXTEND":
            print(f"   -> python campaign.py extend {run} --t-end {new_end:g}")
        return
    if what == "EXTEND":
        new = add_segments(run, cfg, new_end)
        if new:
            print(f"   applied: added {len(new)} segments {new[0]}..{new[-1]}")
            sys.exit(10)
        msg += " -- no segment added (target not beyond the last segment): FINAL"
        print(msg)
    with open(os.path.join(run, "FINAL.tmp"), "w") as fh:
        fh.write(msg + "\n")
    os.replace(os.path.join(run, "FINAL.tmp"), os.path.join(run, "FINAL"))


def steps_per_rva(n0sq):
    """time steps per R/vA vs drive N0^2 = 2 kb kP: Stage 3e anchors A2/A4 (1.28): 1180-1280/1120, A3b (1.81) 2065,
    A3a (2.56) 2825 (ssprk3e, CFL 0.5, same k_perp grid in T1-T3). Linear fit; beyond 2.56 an extrapolation."""
    return max(1100.0, 1270.0 * n0sq - 425.0)


def read_speed(path):
    sp = {}
    for line in open(path):
        if line.startswith("#") or not line.strip():
            continue
        tier, gpu, ms = line.split()[:3]
        sp[(tier, gpu)] = float(ms)
    return sp


GPU_N = dict(h200=4, a100=2, rtx=2)             # physical GPUs with Will's priority (sinfo 2026-11-16)
CLASSES = dict(h200="T2 T1", a100="T2 T1", rtx="T1")   # which to-do lists a worker on that GPU type takes, in order


def jobs_with_cost(sp):
    jobs = []
    for r in points(write=False):
        tier, n0 = r["tier"], float(r["N0sq"])
        tin = t_initial(float(r["gamma_damped"]))
        nst = steps_per_rva(n0) * tin
        h = {g: nst * sp[(tier, g)] / 3.6e6 for g in GPU_N if (tier, g) in sp}
        jobs.append(dict(r, t_init=tin, steps=nst, h=h, extrap=n0 > 2.56))
    return jobs


def simulate_lanes(jobs, ext=1.0, gpu_n=GPU_N):
    """the lane workers' policy (lane_worker.sh): each GPU, when free, claims the longest remaining run (by H200
    cost) from its first non-empty class list (CLASSES) and runs it to the end (in-chain extensions: duration x ext).
    Returns makespan (h), per-GPU [(run_id, start, end)], per-type time at which its lists are exhausted."""
    import heapq
    todo = {c: sorted([j for j in jobs if j["tier"] == c], key=lambda j: -j["h"]["h200"]) for c in ("T1", "T2")}
    free = [(0.0, g, k) for g in gpu_n for k in range(gpu_n[g])]
    heapq.heapify(free)
    lanes = {(g, k): [] for g in gpu_n for k in range(gpu_n[g])}
    end = {}
    while free:
        t, g, k = heapq.heappop(free)
        j = next((todo[c].pop(0) for c in CLASSES[g].split() if todo[c]), None)
        if j is None:
            end[(g, k)] = t
            continue
        d = j["h"][g] * ext
        lanes[(g, k)].append((j["run_id"], t, t + d))
        heapq.heappush(free, (t + d, g, k))
    return max(end.values()), lanes, end


def plan(a):
    sp = read_speed(a.speed)
    jobs = jobs_with_cost(sp)
    gpus = list(GPU_N)
    tot = {g: sum(j["h"].get(g, 0) for j in jobs) for g in gpus}
    out = [f"# campaign plan (campaign.py plan, speed file {a.speed}); {len(jobs)} points, initial lengths "
           f"(in-chain extensions: + {EXT:g} R/vA per step of the rule; '+50 %' rows assume 1.5 x every run)",
           "# GPU-h per point = steps_per_rva(N0^2) x t_init x ms/step; '*' = N0^2 > 2.56 (extrapolated cost); "
           "disk_GB = on-disk per run at t_init (state + outputs + summary without snapshot)",
           "run_id            tier  N0^2  gamma_d t_init  steps/RvA " + " ".join(f"{g + '_h':>7s}" for g in gpus)
           + "  disk_GB"]
    for j in jobs:
        dk = DISK_STATE[j["tier"]] + DISK_OUT[j["tier"]] * j["t_init"] / 100 + DISK_FIXED
        out.append(f"{j['run_id']:17s} {j['tier']}  {float(j['N0sq']):5.2f}  {float(j['gamma_damped']):.3f} "
                   f"{j['t_init']:6.0f} {steps_per_rva(float(j['N0sq'])):9.0f}{'*' if j['extrap'] else ' '} "
                   + " ".join(f"{j['h'].get(g, np.nan):7.1f}" for g in gpus) + f"  {dk:7.3f}")
    out.append("TOTAL GPU-h if all on one type: " + ", ".join(f"{g} {tot[g]:.0f}" for g in gpus)
               + f"; lower bound with all {sum(GPU_N.values())} GPUs busy: "
               f"{tot['h200'] / sum(GPU_N[g] * tot['h200'] / tot[g] for g in gpus) / 24:.2f} days")
    out.append("")
    out.append(f"LANES (queue policy): one GPU job = one worker (lane_worker.sh) that claims runs from shared to-do lists "
               f"queue/todo/T2, T1 (longest first) and runs each chain to its end ON THE NODE, extensions included, "
               f"without releasing the GPU; it exits only when no eligible run is left. h200/a100 take T2 then T1; "
               f"rtx only T1 (fp64 6.6x slower). Workers submitted: h200 {GPU_N['h200']}+1, a100 {GPU_N['a100']}+1, "
               f"rtx {GPU_N['rtx']}+1 = {sum(GPU_N.values()) + 3} jobs (queue.sh limits 8/4/4, total 16).")
    for ext in (1.0, 1.5):
        mk, lanes, end = simulate_lanes(jobs, ext)
        out.append(f"-- lane simulation, run lengths x {ext:g}: wall-clock {mk / 24:.2f} days (all GPUs free at start)")
        for (g, k), L in lanes.items():
            h = sum(e - s for _, s, e in L)
            out.append(f"   {g}#{k}: {len(L):2d} runs, {h:5.1f} h, ends {end[(g, k)] / 24:4.2f} d: "
                       + " ".join(r.split('-')[0] + ('' if r.endswith('T2-k8') else '(T1)') for r, _, _ in L))
    mk1, _, end1 = simulate_lanes(jobs, 1.0)
    out.append(f"PENDING JOBS: 3 at all times (the '+1' worker of each type) until that type's lists are empty "
               f"(x1: h200 {min(v for (g, _), v in end1.items() if g == 'h200') / 24:.2f} d, "
               f"a100 {min(v for (g, _), v in end1.items() if g == 'a100') / 24:.2f} d, "
               f"rtx {min(v for (g, _), v in end1.items() if g == 'rtx') / 24:.2f} d); a freed GPU (worker exit/crash, "
               f"or a non-priority job ending on a GPU we do not hold yet) goes to it at once; it then claims what is "
               f"left or exits within seconds. A crashed worker's claim is adopted by the next worker (sacct state).")
    out.append("If non-priority jobs hold some GPUs at submission (6 h, not preemptible), those lanes start up to 6 h late.")
    out.append("")
    dk = {x: sum(DISK_STATE[j["tier"]] + DISK_OUT[j["tier"]] * x * j["t_init"] / 100 + DISK_FIXED for j in jobs)
          for x in (1.0, 1.5)}
    snap = sum(DISK_SNAP[j["tier"]] for j in jobs)
    peak = sum(2 * DISK_STATE["T2"] for _ in range(sum(GPU_N.values())))
    out.append(f"DISK (on disk, ZFS-compressed; apparent sizes ~2-3x larger for states): kept per run = one state.nc + "
               f"outputs + summary .nc WITHOUT snapshot: {dk[1.0]:.1f} GB (x1.5 lengths {dk[1.5]:.1f} GB); "
               f"+ {snap:.1f} GB if every summary carries the float32 snapshot; transient while running "
               f"<= {peak:.1f} GB (previous state + new restart per running job). Per run: T1 "
               f"{DISK_STATE['T1'] + DISK_OUT['T1'] * 5.5 + DISK_FIXED:.2f}, T2 "
               f"{DISK_STATE['T2'] + DISK_OUT['T2'] * 5.2 + DISK_FIXED:.2f} GB at ~550/520 R/vA.")
    # per-output breakdown: MB on disk per 100 R/vA (T1 measured on realtest A2-T1-k8 t 0-500, du; T2 = T1 x the
    # growth of the record size with nz: maps / kytavg x 2 (k_z bins), 1d / shells x 1.1, others unchanged)
    per100 = [("0d (0.1)", 0.62, 0.62), ("1d spectra+profiles (1)", 2.42, 2.66), ("maps (5, 0.5-windows)", 1.39, 2.78),
              ("kpar (5)", 0.04, 0.04), ("kyt (1)", 0.88, 0.88), ("kytavg streamer (25)", 1.35, 2.70),
              ("shells (5)", 0.26, 0.29), ("slices (10)", 0.98, 0.98)]
    L = dict(T1=550, T2=520)
    out.append("")
    out.append("DISK PER RUN (MB on disk; run lengths T1 550, T2 520 R/vA = mean t_init; apparent sizes in brackets)")
    out.append(f"  {'item':28s} {'T1 /100RvA':>10s} {'T2 /100RvA':>10s} {'T1 run':>8s} {'T2 run':>8s}")
    for n, x1, x2 in per100:
        out.append(f"  {n:28s} {x1:10.2f} {x2:10.2f} {x1 * L['T1'] / 100:8.1f} {x2 * L['T2'] / 100:8.1f}")
    o1, o2 = (sum(x[i] for x in per100) for i in (1, 2))
    out.append(f"  {'run-time outputs, total':28s} {o1:10.2f} {o2:10.2f} {o1 * 5.5:8.1f} {o2 * 5.2:8.1f}")
    out.append(f"  {'state.nc (one, float64)':28s} {'':10s} {'':10s} {'80 (273)':>8s} {'159 (545)':>8s}")
    out.append(f"  {'summary .nc, no snapshot':28s} {'':10s} {'':10s} {2.1:8.1f} {2.1:8.1f}")
    out.append(f"  {'summary snapshot (float32)':28s} {'':10s} {'':10s} {38.2:8.1f} {79.0:8.1f}")
    out.append(f"  {'logs, inputs, timing':28s} {'':10s} {'':10s} {0.5:8.1f} {0.5:8.1f}")
    t1 = 80 + o1 * 5.5 + 2.6
    t2 = 159 + o2 * 5.2 + 2.6
    out.append(f"  TOTAL kept, no snapshot: T1 {t1:.0f} MB, T2 {t2:.0f} MB (with snapshot {t1 + 38:.0f} / {t2 + 79:.0f}); "
               f"T2 doubled by one extension (+500): {t2 + o2 * 5:.0f} MB; T2 at the cap (~1600 R/vA): "
               f"{159 + o2 * 16 + 2.6:.0f} MB")
    out.append("  CUTS: (1, adopted) summary without snapshot: -38/-79 MB, regenerable any time from state.nc "
               "(flucs_summary_nc.py RUN, ~1 min). (2) zlib on state.nc: 0 on disk (ZFS already compresses the padded "
               "zeros: 273 -> 80 MB); only smaller apparent size / copies (tools/compress_state.py, lossless, tested). "
               "(3) optional, only if the quota is tighter than 0.3 GB/run at the cap: slices every 20 (-0.5 MB/100) "
               "and kytavg windows 50 (-1.4/-2.7 MB/100). Not proposed: coarser maps (state-conditioned dissipation "
               "maps need windows << two-state residence times) or fewer 0d/1d records (analysis inputs). Only ONE "
               "state.nc per run (chain policy) and no other 3D data.")
    txt = "\n".join(out) + "\n"
    open(os.path.join(HERE, "campaign_plan.txt"), "w").write(txt)
    print(txt)


def lanes(a):
    """create a production (or test) root: ROOT/<run_id>/ chains (auto_extend), ROOT/queue/todo/<tier>/NN_<run_id>
    (NN = rank by H200 cost, longest first), and fresh worker folders ROOT/workers/W<stamp>-<gpu>-<k>/ (run.sh =
    lane_worker.sh, lane.cfg). Prints the queue.sh commands. Existing run folders / queue entries are kept."""
    import time
    root = os.path.abspath(a.run)
    sp = read_speed(a.speed)
    jobs = sorted(jobs_with_cost(sp), key=lambda j: -j["h"]["h200"])
    if a.only:
        keep = a.only.split(",")
        jobs = [j for j in jobs if any(j["run_id"].startswith(k) for k in keep)]
    q = os.path.join(root, "queue")
    for d in ("todo/T1", "todo/T2", "claimed", "done", "failed"):
        os.makedirs(os.path.join(q, d), exist_ok=True)
    known = {e.split("_", 1)[1] for d in ("todo/T1", "todo/T2", "claimed", "done", "failed")
             for e in os.listdir(os.path.join(q, d)) if "_" in e and not e.endswith(".owner")}
    for rank, j in enumerate(jobs):
        rd = os.path.join(root, j["run_id"])
        if not os.path.exists(rd):
            make(argparse.Namespace(run=rd, point=j["run_id"], kP=None, kb=None, tier=None, run_id=None, grid=a.grid,
                                    nu_perp=a.nu_perp, nu_z=a.nu_z, seg=a.seg, cadence=a.cadence, t_end=a.t_end,
                                    auto_extend=True))
        if j["run_id"] not in known:
            with open(os.path.join(q, "todo", j["tier"], f"{rank:02d}_{j['run_id']}"), "w") as fh:
                fh.write(j["run_id"] + "\n")
    stamp = time.strftime("%m%d%H%M")
    cmds = []
    for spec in a.workers.split(","):
        g, n = spec.split(":")
        for _ in range(int(n)):
            w = os.path.join(root, "workers", f"W{stamp}-{len(cmds):02d}-{g}")
            os.makedirs(w)
            shutil.copy(LANE, os.path.join(w, "run.sh"))
            with open(os.path.join(w, "lane.cfg"), "w") as fh:
                fh.write(f'ROOT="{root}"\nCLASSES="{CLASSES[g]}"\nGPU="{g}"\n')
            cmds.append(f"~/flucs_work/queue.sh submit --gpu {g} {w}")
    print(f"{root}: {len(jobs)} runs queued ({sum(j['tier'] == 'T2' for j in jobs)} T2, "
          f"{sum(j['tier'] == 'T1' for j in jobs)} T1); {len(cmds)} worker folders:")
    print("\n".join(cmds))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["points", "plan", "make", "extend", "assess", "lanes"])
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
    ap.add_argument("--apply", action="store_true", help="assess: apply the verdict (extend in place / write FINAL)")
    ap.add_argument("--auto-extend", action="store_true", help="make: chain applies the extension rule itself")
    ap.add_argument("--only", default="", help="lanes: comma list of run_id prefixes")
    ap.add_argument("--workers", default="h200:5,a100:3,rtx:3", help="lanes: worker folders per GPU type")
    a = ap.parse_args()
    if a.what in ("make", "extend", "assess", "plan", "lanes"):
        {"make": make, "extend": extend, "assess": assess, "plan": plan, "lanes": lanes}[a.what](a)
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
