#!/usr/bin/env python3
"""Stage 4b run-time diagnostic overhead at production cadence (overhead/O1_T1, O2_T2: 10 R/vA from turbulent
states, all outputs, H200). Per diag method: ms/call (zp_timing_<output>.txt, GPU-synchronised) x production calls
per R/vA (execute 1/save_time, sample 1/sample_time); step cost = (wall - diag time) / steps, steps per R/vA from
the run. overhead = diag time per R/vA / step time per R/vA."""
import glob
import os
import re

H = os.path.dirname(os.path.abspath(__file__))
for d in sorted(glob.glob(os.path.join(H, "overhead", "O*"))):
    log = open(os.path.join(d, "run.log")).read()
    m = re.search(r"given in (.*) \((\d+) steps\)", log)
    if not m:
        print(os.path.basename(d), "no 'given in' line")
        continue
    wall = sum(float(x) * {"hour": 3600, "minute": 60, "second": 1}[u.rstrip("s")]
               for x, u in re.findall(r"([\d.]+) (hours?|minutes?|seconds?)", m.group(1)))
    steps = int(m.group(2))
    txt = open(os.path.join(d, "input.toml")).read()
    tfinal = float(re.search(r"^tfinal = ([\d.]+)", txt, re.M).group(1))
    cad = {}
    for blk in re.split(r"(?m)^(?=\[)", txt):
        mo = re.match(r"\[output\.(\w+)\]", blk)
        if not mo:
            continue
        o, body = mo.group(1), blk
        sv = re.search(r"save_time = ([\d.]+)", body)
        sm = re.search(r"sample_time = ([\d.]+)", body)
        cad[o] = (float(sv.group(1)), float(sm.group(1)) if sm else None)
    rows, diag_s = [], 0.0
    for f in sorted(glob.glob(os.path.join(d, "zp_timing_*.txt"))):
        o = os.path.basename(f)[10:-4]
        for line in open(f):
            if line.startswith("#") or not line.strip():
                continue
            name, calls, tot, ms = line.split()[:4]
            diag_s += float(tot)
            per = 1 / cad[o][0] if name.endswith(".execute") else (1 / cad[o][1] if cad[o][1] else 0)
            rows.append((o, name, float(ms), per, float(ms) * per))
    ms_step = 1e3 * (wall - diag_s) / steps
    spr = steps / tfinal
    step_ms_per_rva = ms_step * spr
    tot = sum(r[4] for r in rows)
    print(f"== {os.path.basename(d)}: {steps} steps in {wall:.0f} s over {tfinal:g} R/vA -> {spr:.0f} steps/RvA, "
          f"{ms_step:.2f} ms/step without diags; diag time in this run {diag_s:.1f} s")
    by_out = {}
    for o, name, ms, per, msr in rows:
        by_out[o] = by_out.get(o, 0.0) + msr
        print(f"   {o:7s} {name:28s} {ms:9.2f} ms/call x {per:5.2f}/RvA = {msr:8.1f} ms/RvA "
              f"({100 * msr / step_ms_per_rva:5.2f} %)")
    print("   per output: " + ", ".join(f"{o} {100 * v / step_ms_per_rva:.2f} %" for o, v in by_out.items()))
    print(f"   TOTAL production overhead {100 * tot / step_ms_per_rva:.2f} % of the time-step cost")
