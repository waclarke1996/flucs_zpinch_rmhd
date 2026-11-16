#!/bin/bash
# Stage 4a lane test (tiny grid 48x24x48, test cadence, 5-R/vA segments, t_init 10): the production path end to end.
# ./root made on the login node with
#   campaign.py lanes lanetest/root --only P10B04,P09B07,P08B08 --grid 48,24,48 --nu-perp 2e-6 --nu-z 1e-8 --seg 5
#                     --cadence test --t-end 10 --workers h200:1,rtx:1,h200:1
# Extension rule shortened for the tiny runs: CAMPAIGN_TEST_RULE = T_STAT 10, T_STAT_MAX 40, EXT 10, T0_OFFSET 5.
#  L1 fake orphan: 01_P09B07 claimed by a job that has ENDED (13134354, COMPLETED) -> must be adopted.
#  L2 workers 00 (h200 classes "T2 T1") and 01 (rtx class "T1") run AT THE SAME TIME (two processes, one GPU):
#     every run claimed exactly once, chains extend in place until FINAL, summary written, queue -> done/.
#  L3 worker 02 (h200) starts when nothing is left: exits at once.
#  L4 restart from a zlib-compressed state.nc (tools/compress_state.py): P10B04 extended by one segment.
set -u
export CAMPAIGN_TEST_RULE="10,40,10,5"
PY=~/flucs_work/.venv/bin/python
T=$(pwd); R=$T/root; Q=$R/queue
step() { echo; echo "######## $* ($(date +%T))"; }

step "L1 fake orphan"
mv "$Q/todo/T2/01_P09B07-T2-k8" "$Q/claimed/01_P09B07-T2-k8"
echo "Wfake 13134354 fakehost" > "$Q/claimed/01_P09B07-T2-k8.owner"
find "$Q" -type f | sort | sed "s|$Q/||"

step "L2 two workers at once"
W0=$(ls -d "$R"/workers/*-00-h200); W1=$(ls -d "$R"/workers/*-01-rtx); W2=$(ls -d "$R"/workers/*-02-h200)
( cd "$W0" && bash run.sh > run.log 2>&1 ) &
( cd "$W1" && bash run.sh > run.log 2>&1 ) &
wait
for w in "$W0" "$W1"; do echo "--- $(basename "$w")"; cat "$w/run.log"; done
find "$Q" -type f | sort | sed "s|$Q/||"

step "L3 a worker with nothing left"
( cd "$W2" && bash run.sh > run.log 2>&1 ); echo "worker rc=$?"; cat "$W2/run.log"

step "L4 restart from a compressed state"
run=$R/P10B04-T1-k8
last=$(cd "$run" && ls -d t[0-9][0-9][0-9][0-9] | sort | tail -1)
$PY ~/flucs_work/runs/tools/compress_state.py "$run/$last/state.nc" 2>/dev/null
mv "$run/FINAL" "$run/FINAL.before_L4"
$PY ~/flucs_work/runs/stage4a/campaign.py extend "$run" --t-end $(( 10#${last#t} + 5 )) 2>/dev/null
( cd "$run" && bash run.sh ); echo "chain rc=$?"

step "done"
