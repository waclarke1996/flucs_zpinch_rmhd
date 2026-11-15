#!/bin/bash
# Stage 4 production chain (template; copied into each RUN folder as run.sh by the campaign generator).
# Runs the segments t????/ of this folder in order (each 100 R/vA, restarting from the previous one).
# Storage policy (Stage 4a STEP 2):
#   * after a segment finishes with exit 0 and has written its restart.nc, the PREVIOUS segment's restart.nc is
#     deleted (only the latest 3D state is kept; the final one stays);
#   * no initial / periodic 3D outputs (no output.3d; the periodic restart overwrites the segment's own file);
#   * segments that already finished (run.log ends with "flucs given in" and restart.nc exists) are skipped, so a
#     resubmitted job continues the chain (extension to 1000/1500 R/vA = add segments and resubmit).
# An early stop (runaway rule) ends the chain: RUN/EARLY_STOP.
FLUCS=~/flucs_work/.venv/bin/flucs
here=$(pwd)
git -C ~/flucs_work/flucs rev-parse HEAD > "$here/flucs_git.txt" 2>/dev/null
prev=""
for seg in t[0-9][0-9][0-9][0-9]; do
    if [ -f "$here/$seg/restart.nc" ] && grep -q "flucs given in" "$here/$seg/run.log" 2>/dev/null; then
        echo "== $seg already done"
        prev=$seg
        continue
    fi
    cd "$here/$seg" || exit 2
    rm -f output.*.nc zp_timing_*.txt          # a crashed earlier attempt: start the segment afresh
    echo "== start $seg $(date)"
    "$FLUCS" --run --io_path . > run.log 2>&1
    rc=$?
    echo "== exit $rc $seg $(date) | $(grep -o 'given in.*' run.log | tail -1)"
    [ $rc -eq 0 ] || { echo "segment $seg failed (exit $rc): chain stopped"; exit $rc; }
    [ -f restart.nc ] || { echo "segment $seg wrote no restart.nc: chain stopped"; exit 4; }
    if [ -n "$prev" ] && [ -f "$here/$prev/restart.nc" ]; then
        rm -f "$here/$prev/restart.nc" && echo "   deleted $prev/restart.nc"
    fi
    prev=$seg
    if grep -q "Early stop" run.log; then
        echo "segment $seg: $(grep 'Early stop' run.log | tail -1) -> chain stopped"
        echo "$seg $(grep 'Early stop' run.log | tail -1)" > "$here/EARLY_STOP"
        exit 0
    fi
done
echo "== all segments done $(date)"
