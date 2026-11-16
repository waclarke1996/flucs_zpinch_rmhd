#!/bin/bash
# Stage 4a real short run: production chain (campaign.py make, chain_run.sh, production cadence, all diagnostics)
# for A2 on T1 (same physics, noise and seed as stage3e/A2-T1-k8), t = 0-300, then extension to 500 in the same job.
# KEEP_STATES=1 keeps every state.nc (for e_analyse.py and the comparison with stage3e/A2-T1-k8/t0500/restart.nc).
PY=~/flucs_work/.venv/bin/python
cd A2-T1-k8 || exit 2
echo "######## chain to t = 300 ($(date))"
KEEP_STATES=1 bash run.sh; rc=$?; echo "chain rc=$rc"
[ $rc -eq 0 ] || exit $rc
echo "######## extension to t = 500 ($(date))"
$PY ~/flucs_work/runs/stage4a/campaign.py extend . --t-end 500
KEEP_STATES=1 bash run.sh; echo "chain rc=$?"
