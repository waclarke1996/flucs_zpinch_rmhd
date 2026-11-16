#!/bin/bash
# zpinch_tools/sync.sh -- refresh the repo copies of the Z-pinch project's non-data files (backup / visibility).
# COPIES (never moves): the originals in ~/flucs_work stay where they are and keep working.
#   zpinch_docs/   <- ~/flucs_work/{DIAGNOSTICS,RUNNING_FLUCS,PROGRESS,NOTES_flucs,SPEC,PLAN}.md
#   zpinch_tools/  <- files from ~/flucs_work/runs/, same relative path (zpinch_tools/tools/..., stage4a/..., stage4b/...)
# Text only: *.nc, *.png, run.log / chain.log and any file > 1 MB are refused. Run before every commit:
#   bash ~/flucs_work/flucs/zpinch_tools/sync.sh && git -C ~/flucs_work/flucs add zpinch_docs zpinch_tools
set -u
W=~/flucs_work
R=$W/runs
REPO=$W/flucs
DOCS="DIAGNOSTICS.md RUNNING_FLUCS.md PROGRESS.md NOTES_flucs.md SPEC.md PLAN.md"
# paths relative to runs/ (globs allowed)
TOOLS="
tools/*.py
stage4a/chain_run.sh stage4a/lane_worker.sh stage4a/campaign.py stage4a/f_analyse.py stage4a/campaign_plan.txt
stage4a/speed.txt
stage4a/verify/run.sh stage4a/verify/compare_stage4a.py stage4a/verify/capture_tool.py
stage4a/verify/stub/curvy3d_linear_etg.py stage4a/verify/flucs_run/input.toml stage4a/verify/verify.txt
stage4a/verify/ref/dmap.txt stage4a/verify/ref/kpar.txt
stage4a/review/check_transfers.py stage4a/review/check_profiles.py stage4a/review/test_timed_trigger.py
stage4a/chaintest2/run.sh stage4a/chaintest2/inspect_chain.py stage4a/chaintest3a/run.sh
stage4a/lanetest2/run.sh
stage3e/e_analyse.py
stage4b/compare_4b.py stage4b/make_inputs.py stage4b/test_kperp_bins.py stage4b/overhead_4b.py
stage4b/verify/verify.txt stage4b/lanetest3/run.sh
stage4b/verify/V1_spec_T1/input.toml stage4b/verify/V2_lx8pi/input.toml stage4b/verify/V3_signtest/input.toml
stage4b/overhead/O1_T1/input.toml stage4b/overhead/O2_T2/input.toml
"
ok=0; bad=0
copy() {   # src dst
    local s=$1 d=$2
    case "$s" in *.nc|*.png|*/run.log|*/chain.log) echo "  REFUSED (data/log): $s"; bad=$((bad + 1)); return ;; esac
    [ -f "$s" ] || { echo "  MISSING: $s"; bad=$((bad + 1)); return; }
    [ "$(stat -c %s "$s")" -le 1048576 ] || { echo "  REFUSED (> 1 MB): $s"; bad=$((bad + 1)); return; }
    mkdir -p "$(dirname "$d")" && cp -p "$s" "$d" && ok=$((ok + 1))
}
for f in $DOCS; do copy "$W/$f" "$REPO/zpinch_docs/$f"; done
nd=$ok
for pat in $TOOLS; do
    for s in $R/$pat; do copy "$s" "$REPO/zpinch_tools/${s#$R/}"; done
done
echo "sync.sh: $nd docs -> zpinch_docs/, $((ok - nd)) files -> zpinch_tools/; $bad refused/missing"
[ $bad -eq 0 ]
