#!/usr/bin/env python3
"""Stage 4b monitoring: progress of runs/prod4 and projected completion (read-only).
  python project.py [L_T2 ...]      (default 1600 1100: final length assumed for T2 runs not yet FINAL)
Cost per R/vA = plan steps/RvA (campaign_plan.txt) x RATIO (measured / plan, monitoring 1-2) x ms/step of the GPU type
(measured: h200 19.1 ms, a100 44.5 ms on T2). Lanes: the running workers finish their claimed run (from its last
validated segment), then claim from todo/T2 in list order (longest first), as lane_worker.sh does. GPUs: the workers
running now (pending spares are assumed never to start: sondhigpu has 4 H200)."""
import glob, os, re, sys
from datetime import datetime, timedelta, timezone

ROOT = os.path.expanduser("~/flucs_work/runs/prod4")
PLAN = os.path.expanduser("~/flucs_work/runs/stage4a/campaign_plan.txt")
RATIO, MS = 0.95, dict(h200=19.1e-3, a100=44.5e-3)
plan = {}
for line in open(PLAN):
    m = re.match(r"(P\d\dB\d\d-T[12]-k8)\s+(T[12])\s+([\d.]+)\s+[\d.]+\s+(\d+)\s+(\d+)\*?", line)
    if m:
        plan[m[1]] = dict(tier=m[2], t_init=int(m[4]), spr=int(m[5]))


def done_t(run):
    ok = [int(os.path.basename(os.path.dirname(p))[1:]) for p in glob.glob(f"{ROOT}/{run}/t*/SEG_OK")]
    return max(ok, default=0)


def state(run):
    for st in ("done", "claimed", "failed"):
        if glob.glob(f"{ROOT}/queue/{st}/*_{run}"):
            return st
    return "todo"


runs = sorted(plan)
st = {r: state(r) for r in runs}
dn = {r: done_t(r) for r in runs}
fin = {r: os.path.exists(f"{ROOT}/{r}/FINAL") for r in runs}
for tier in ("T1", "T2"):
    rs = [r for r in runs if plan[r]["tier"] == tier]
    cnt = {s: sum(st[r] == s for r in rs) for s in ("done", "claimed", "todo", "failed")}
    print(f"{tier}: done {cnt['done']}, running {cnt['claimed']}, todo {cnt['todo']}, failed {cnt['failed']}; "
          f"R/vA validated {sum(dn[r] for r in rs)} (sum t_init {sum(plan[r]['t_init'] for r in rs)}; "
          f"FINAL lengths {sorted(set(dn[r] for r in rs if fin[r]))})")
now = datetime.now(timezone.utc)
owners = {}
for f in glob.glob(f"{ROOT}/queue/claimed/*.owner"):
    run = os.path.basename(f)[:-6].split("_", 1)[1]
    owners[run] = open(f).read().split()[0].rsplit("-", 1)[1]
todo = [n.split("_", 1)[1] for n in sorted(os.listdir(f"{ROOT}/queue/todo/T2"))]
for L in [int(x) for x in sys.argv[1:]] or [1600, 1100]:
    hours = lambda r, gpu, t0: max(L - t0, 0) * plan[r]["spr"] * RATIO * MS[gpu] / 3600
    lanes = [[hours(r, g, dn[r]), g, [r]] for r, g in owners.items()]
    for r in todo:
        ln = min(lanes, key=lambda x: x[0])
        ln[0] += hours(r, ln[1], 0); ln[2].append(r)
    end = max(x[0] for x in lanes)
    print(f"-- T2 runs end at {L} R/vA: last lane ends in {end / 24:.1f} d ({(now + timedelta(hours=end)):%Y-%m-%d %H:%M} UTC); "
          f"remaining {sum(x[0] * (1 if x[1] == 'h200' else MS['h200'] / MS['a100']) for x in lanes):.0f} h200-h")
    for t, g, rs in sorted(lanes, key=lambda x: -x[0]):
        print(f"   {g}: {len(rs)} runs, {t:6.1f} h: {' '.join(r[:6] for r in rs)}")
