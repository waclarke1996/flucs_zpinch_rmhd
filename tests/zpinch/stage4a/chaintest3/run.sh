#!/bin/bash
# Stage 4a chain test, third version: chaintest2 rerun after the 2026-11-16 timed-output trigger fix (duplicate records); same phases.
# Tiny grid 48x24x48, A2 parameters, 5-R/vA segments, test cadence. ./chain generated on the login node with
#   campaign.py make chain --kP 2.2627417 --kb 0.28284271 --tier T1 --grid 48,24,48 --nu-perp 2e-6 --nu-z 1e-8
#                          --seg 5 --cadence test --t-end 20 --run-id A2-tiny
# Each phase is followed by inspect_chain.py (expected state + record continuity).
set -u
PY=~/flucs_work/.venv/bin/python
C=~/flucs_work/runs/stage4a/campaign.py
T=$(pwd)
R=$T/chain
I="$PY $T/inspect_chain.py $R"
step() { echo; echo "######## $* ($(date +%T))"; }
cd "$R" || exit 2

step "P1 first segment, stop after validation (valid restart.nc, not renamed, no SEG_OK)"
CHAIN_TEST_STOP_AT=after_validate bash run.sh; echo "chain rc=$?"
$I P1 --expect "t0005:-ok,-state"

step "P2 resubmit (t0005 rerun); job killed (SIGKILL, whole process group) as soon as t0010 has started; then a partial restart.nc left in t0010"
setsid bash run.sh > "$T/p2_chain.log" 2>&1 &
pg=$!
for i in $(seq 1 3000); do [ -f t0010/output.0d.nc ] && break; sleep 0.2; done
kill -9 -- -"$pg"; wait "$pg" 2>/dev/null
sleep 2
grep "== \|validated\|deleted" "$T/p2_chain.log"
head -c 20000 t0005/state.nc > t0010/restart.nc       # emulate a restart write cut off mid-file
$I P2 --expect "t0005:ok,state;t0010:-ok,-state,partial"

step "P3 resubmit: t0010 rerun from t0005/state.nc (partial file removed); stop after SEG_OK, before the deletion"
CHAIN_TEST_STOP_AT=after_ok bash run.sh; echo "chain rc=$?"
$I P3 --expect "t0005:ok,state;t0010:ok,state,-partial"

step "P4 resubmit: left-over t0005/state.nc removed; t0015 runs; stop after the rename (state.nc, no SEG_OK)"
CHAIN_TEST_STOP_AT=after_rename bash run.sh; echo "chain rc=$?"
$I P4 --expect "t0005:ok,-state;t0010:ok,state;t0015:-ok,state"

step "P5 resubmit: t0015 rerun from t0010/state.nc, then t0020"
bash run.sh; echo "chain rc=$?"
$I P5 --expect "t0010:ok,-state;t0015:ok,-state;t0020:ok,state"

step "P6 extension to t = 30 (campaign.py extend), resubmit"
$PY "$C" extend "$R" --t-end 30
grep restart_from t0025/input.toml
bash run.sh; echo "chain rc=$?"
$I P6 --expect "t0020:ok,-state;t0025:ok,-state;t0030:ok,state"

step "P7 resubmit a finished chain: nothing runs"
bash run.sh; echo "chain rc=$?"
$I P7 --expect "t0030:ok,state"

step "P8 EARLY_STOP present: chain does nothing, extend refuses"
echo "test" > EARLY_STOP
bash run.sh; echo "chain rc=$?"
$PY "$C" extend "$R" --t-end 35; echo "extend rc=$?"
rm -f EARLY_STOP
ls -d t????
step "done"
