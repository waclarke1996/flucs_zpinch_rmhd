#!/usr/bin/env python3
"""
Stage 4a chain test: state of a chain folder after one test phase, and record continuity of every time-based
output over the finished segments.
    inspect_chain.py CHAIN PHASE [--expect "t0005:ok,state;t0010:-ok,state,partial"]
Per segment: SEG_OK / state.nc / restart.nc (validated or not) / last run.log line. Expectations (comma list per
segment; '-' = must NOT hold): ok (SEG_OK), state (state.nc present), partial (a restart.nc that check_state rejects).
Continuity (finished segments, in order): for every output.<name>.nc, record times strictly increasing, no duplicate
grid point, every grid point k*save_time from the first to the last record present, each record within
[k save_time, k save_time + 2 dt_max]; windowed outputs: n_samples = save_time/sample_time (1 for a t = 0 record);
energy dEdt NaN exactly at step-0 records (t = 0 of a fresh run); final state at most 2 dt_max after the last 0d
record (a record within 1e-9 of a grid point may be one step before the end). Prints PASS/FAIL lines; exit 1 on any FAIL.
"""
import glob
import os
import re
import subprocess
import sys

import numpy as np
from netCDF4 import Dataset

sys.path.insert(0, os.path.expanduser("~/flucs_work/flucs/src"))
from flucs.systems.zpinch_fourier.zpinch_readers import load_diag  # noqa: E402

PY = os.path.expanduser("~/flucs_work/.venv/bin/python")
CHK = os.path.expanduser("~/flucs_work/runs/tools/check_state.py")
chain, phase = sys.argv[1], sys.argv[2]
expect = {}
if len(sys.argv) > 4 and sys.argv[3] == "--expect":
    for item in sys.argv[4].split(";"):
        seg, conds = item.split(":")
        expect[seg] = conds.split(",")
nfail = 0


def res(name, ok, extra=""):
    global nfail
    nfail += not ok
    print(f"  [{phase}] {name:60s} {'PASS' if ok else 'FAIL'} {extra}")


segs = sorted(d for d in os.listdir(chain) if re.fullmatch(r"t\d{4}", d))
print(f"== {phase}: {chain}  EARLY_STOP={'yes' if os.path.exists(f'{chain}/EARLY_STOP') else 'no'}")
for s in segs:
    d = f"{chain}/{s}"
    ok, st, rs = (os.path.exists(f"{d}/{x}") for x in ("SEG_OK", "state.nc", "restart.nc"))
    partial = False
    if rs:
        r = subprocess.run([PY, CHK, f"{d}/restart.nc", "--input", f"{d}/input.toml"], capture_output=True, text=True)
        partial = r.returncode != 0
    last = ""
    if os.path.exists(f"{d}/run.log"):
        lines = [x for x in open(f"{d}/run.log", errors="ignore").read().splitlines() if x.strip()]
        last = lines[-1][:70] if lines else ""
    others = sorted(os.path.basename(x) for x in glob.glob(f"{d}/restart.*"))
    print(f"   {s}: SEG_OK {'Y' if ok else '-'} ({open(f'{d}/SEG_OK').read().split()[0] if ok else ''}) "
          f"state {'Y' if st else '-'} restart {'partial' if partial else ('valid' if rs else '-')} {others} | {last}")
    for c in expect.get(s, []):
        neg = c.startswith("-")
        key = c.lstrip("-")
        val = {"ok": ok, "state": st, "partial": partial}[key]
        res(f"{s} {'not ' if neg else ''}{key}", val != neg)

done = [s for s in segs if os.path.exists(f"{chain}/{s}/SEG_OK")]
if done:
    dtmax = 0.01
    outs = sorted({os.path.basename(p)[7:-3] for s in done for p in glob.glob(f"{chain}/{s}/output.*.nc")})
    txt = open(f"{chain}/{done[0]}/input.toml").read()
    for o in outs:
        m = re.search(rf"^\[output\.{o}\]\s*\nsave_time = ([0-9.e+-]+)\s*\n(sample_time = ([0-9.e+-]+))?", txt, re.M)
        if not m:
            continue
        save = float(m.group(1))
        sample = float(m.group(3)) if m.group(3) else None
        files = [f"{chain}/{s}/output.{o}.nc" for s in done]
        missing = [f for f in files if not os.path.exists(f)]
        ts = []
        nsamp = []
        for f in files:
            if not os.path.exists(f):
                continue
            with Dataset(f) as ds:
                for g in sorted(ds.groups, key=int):
                    ts.append(np.asarray(ds.groups[g]["time"][:], float))
                    if sample:
                        gg = ds.groups[g]
                        dg = [x for x in gg.groups][0]
                        nsamp.append(np.asarray(gg.groups[dg]["n_samples"][:], float)
                                     if "n_samples" in gg.groups[dg].variables else np.full(len(ts[-1]), np.nan))
        t = np.concatenate(ts)
        k = np.floor(t / save + 1e-9).astype(int)
        off = t - k * save
        okk = (not missing and np.all(np.diff(t) > 0) and len(np.unique(k)) == len(k)
               and np.array_equal(k, np.arange(k[0], k[0] + len(k))) and np.all((off >= -1e-9) & (off <= 2 * dtmax)))
        res(f"records {o:7s} save {save:g}: {len(t)} recs t {t[0]:.3f}..{t[-1]:.3f}", okk,
            f"(max offset {off.max():.2e}{'; MISSING ' + str(missing) if missing else ''})")
        if sample and nsamp:
            n = np.concatenate(nsamp)
            want = np.where(t < 1e-9, 1.0, round(save / sample))
            if np.all(np.isnan(n)):
                print(f"  [{phase}] {o}: no n_samples variable (window count not stored)")
            else:
                res(f"window samples {o}: n = save/sample (1 at t = 0)", bool(np.all(n == want)),
                    f"(n {np.unique(n)})")
    # dEdt NaN exactly at t = 0 (step 0) of the fresh run, finite elsewhere
    t, E = load_diag([f"{chain}/{s}/output.0d.nc" for s in done], "energy", ["dEdt"])
    nan_at = t[~np.isfinite(E["dEdt"])]
    res("energy dEdt NaN only at the step-0 record (t = 0)", len(nan_at) == (1 if t[0] == 0 else 0)
        and (len(nan_at) == 0 or nan_at[0] == 0.0), f"(NaN at {nan_at[:3]})")
    # last done segment: state time = (time of the state it started from) + tfinal, overshoot <= 2 dt_max
    # (FLUCS final_time = init_time + tfinal, so the end time drifts by <= one dt per segment; records are on an
    # absolute grid, so the drift is harmless) and no 0d record missed (state - last 0d record < 0d save_time)
    for s in done[-1:]:
        if os.path.exists(f"{chain}/{s}/state.nc"):
            with Dataset(f"{chain}/{s}/state.nc") as ds:
                tst = float(ds["current_time"][...])
            i = segs.index(s)
            ts0 = float(open(f"{chain}/{segs[i-1]}/SEG_OK").read().split()[0]) if i > 0 else 0.0
            tf = float(re.search(r"^tfinal\s*=\s*([0-9.e+-]+)", open(f"{chain}/{s}/input.toml").read(), re.M).group(1))
            t0d, _ = load_diag(f"{chain}/{s}/output.0d.nc", "energy", ["E"])
            sv0 = float(re.search(r"^\[output\.0d\]\s*\nsave_time = ([0-9.e+-]+)", txt, re.M).group(1))
            res(f"{s}: t_start+tf <= state t <= +2dt_max, < last 0d + save ({ts0:.4f}+{tf:g}, {tst:.4f}, {t0d[-1]:.4f})",
                ts0 + tf - 1e-6 <= tst <= ts0 + tf + 2 * dtmax and 0 <= tst - t0d[-1] < sv0)
print(f"== {phase}: FAIL count {nfail}")
sys.exit(1 if nfail else 0)
