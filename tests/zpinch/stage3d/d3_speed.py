#!/usr/bin/env python3
"""Stage 3d D3: wall time per step and mean dt, FLUCS vs curvy3d, Y grid (padded 256 x 128 x 256 = 171 x 85 x 171 modes),
k^8, max_cfl 0.5, H200 NVL. curvy3d: wall_hours from registry_v2.csv; steps = integral of dt_series (0.1 cadence:
sum 0.1/dt, trapezoid in 1/dt), checked against the last 'Current dt is ... at step N' in the .log (a lower bound,
printed every 20000 steps). FLUCS: 'flucs given in ... (N steps)' of each 3c segment (output.log)."""
import glob
import os
import re

import netCDF4
import numpy as np
import pandas as pd

D = "/mnt/users/clarkew/data/rmhd_zpinch/output/3D/BP2_anchors_v2"
REG = "/mnt/users/clarkew/source/curvy3d_etg_github/rmhd_zpinch/campaigns/BP2/registry_v2.csv"
R = os.path.expanduser("~/flucs_work/runs")
reg = pd.read_csv(REG, comment="#").set_index("run_id")
rows = []
for rid in ("A3a-Y", "A3a-Yc2", "A3a-YP4", "A3a-Y-seed", "A2-Y", "A3a-Z"):
    nc = glob.glob(f"{D}/*_RKARS343_{rid}.nc")[0]
    with netCDF4.Dataset(nc) as ds:
        dt = np.ma.filled(ds["dt_series"][:].astype(float), np.nan)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    inv = 1.0 / dt
    steps = float(np.sum(0.5 * (inv[1:] + inv[:-1])) * 0.1)
    T = 0.1 * (len(dt) - 1)
    log = open(nc[:-3] + ".log").read()
    nlog = max(int(n) for n in re.findall(r"at step\s+(\d+)", log))
    wh = float(reg.loc[rid, "wall_hours"])
    rows.append(("curvy3d " + rid, int(reg.loc[rid, "nz"]), T, steps, nlog, wh, wh * 3.6e6 / steps, T / steps))
for run in ("A3a-Y-k8-C300", "A3a-Y-k8-C300-cfl025"):
    n = s = 0
    for f in sorted(glob.glob(f"{R}/stage3c/{run}/t0*/output.log")):
        m = re.search(r"given in (?:(\d+) hours?, )?(\d+) minutes?, (\d+) seconds? \((\d+) steps\)", open(f).read())
        h, mi, se, st = (int(x) if x else 0 for x in m.groups())
        s += 3600 * h + 60 * mi + se
        n += st
    T = 300.0
    rows.append(("FLUCS " + run, 171, T, n, n, s / 3600, s * 1e3 / n, T / n))
print(f"{'run':32s} {'nz':>4s} {'T':>6s} {'steps':>9s} {'log>=':>9s} {'wall h':>7s} {'ms/step':>8s} {'mean dt':>9s} {'s per R/vA':>10s}")
for r in rows:
    print(f"{r[0]:32s} {r[1]:4d} {r[2]:6.0f} {r[3]:9.0f} {r[4]:9d} {r[5]:7.2f} {r[6]:8.2f} {r[7]:9.2e} {r[5]*3600/r[2]:10.1f}")
