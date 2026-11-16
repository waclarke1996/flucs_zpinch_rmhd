#!/usr/bin/env python3
"""Unit test of the ZPTimedOutputNC trigger (record/sample grid), CPU only, with the real _grid_point_after.
Emulates execute(): steps of size dt from t0, some landing within eps BELOW a grid point (the 2026-11-16 duplicate
bug). Checks: one record per grid point (no duplicates, none missing), each record at the first step with
t + eps >= point, n samples per window = save/sample; restart from a state whose time is within eps below a grid
point that the previous segment already recorded does not re-record it."""
import math, os, sys
import importlib.util
spec = importlib.util.spec_from_file_location("zo_fn", os.path.expanduser(
    "~/flucs_work/flucs/src/flucs/systems/zpinch_fourier/zpinch_output.py"))
src = open(spec.origin).read()
ns = {"math": math}
a = src.index("def _grid_point_after"); b = src.index("class ZPTimedOutputNC")
exec(src[a:b], ns)
after = ns["_grid_point_after"]
nfail = 0
def check(name, ok):
    global nfail; nfail += not ok; print(f"  {name:70s} {'PASS' if ok else 'FAIL'}")

def emulate(times, save, sample, fresh):
    eps = lambda t: 1e-9 * max(1.0, abs(t))
    t0 = times[0]
    nr = 0.0 if fresh else after(t0, save)
    nsm = 0.0 if fresh else after(t0, sample)
    recs, nwin, win = [], [], 0
    for t in times:
        if t + eps(t) >= nsm:
            win += 1; nsm = after(t, sample)
        if t + eps(t) >= nr:
            nr = after(t, save)
            if win == 0: continue
            recs.append(t); nwin.append(win); win = 0
    return recs, nwin

save, sample, dt = 1.0, 0.5, 8.6e-4
for T in (108.0, 500.0, 1e4):
    # steps that hit T - 0.4 eps exactly, then continue
    eps = 1e-9 * T
    pre = [T - 0.4 * eps - dt * i for i in range(3000, 0, -1)]
    post = [T - 0.4 * eps + dt * i for i in range(0, 3000)]
    times = pre + post
    recs, nwin = emulate(times, save, sample, fresh=False)
    ks = [round(t / save) for t in recs]
    check(f"T={T:g}: one record per grid point near T", len(ks) == len(set(ks)) and ks == list(range(ks[0], ks[0] + len(ks))))
    check(f"T={T:g}: record of T taken at the step T - 0.4 eps", any(abs(r - (T - 0.4 * eps)) < 1e-12 for r in recs))
    check(f"T={T:g}: window counts = save/sample (first partial window excepted)", all(n == 2 for n in nwin[1:]))
    # restart from the state at T - 0.4 eps (the previous segment recorded T there): T not recorded again
    r2, _ = emulate(post, save, sample, fresh=False)
    check(f"T={T:g}: restart from T - 0.4 eps does not re-record T", all(abs(r - T) > 0.5 for r in r2[:1]) and round(r2[0]) == T + 1)
# fresh run: t = 0 recorded, then every grid point
times = [i * 7.3e-3 for i in range(0, 2000)]
recs, nwin = emulate(times, 0.1, 0.05, fresh=True)
ks = [round(t / 0.1) for t in recs]
check("fresh: t=0 recorded, all grid points once", recs[0] == 0.0 and ks == list(range(len(ks))) and nwin[0] == 1)
check("fresh: windows 2 samples after t=0", all(n == 2 for n in nwin[1:]))
# exact grid hit and rounding cases
check("after(108-1e-7*1e-2,1)=109", after(108 - 1e-9, 1.0) == 109.0)
check("after(0.3,0.1) > 0.3+eps", after(0.30000000000000004, 0.1) > 0.3 + 1e-9 and after(0.3, 0.1) < 0.41)
print("FAIL count", nfail); sys.exit(1 if nfail else 0)
