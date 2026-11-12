#!/bin/bash
# Stage 3d: run this folder's 50-R/vA restart segments t*/ in order (same command as queue.sh uses for input.toml
# folders). If a segment ends with an early stop (runaway / dt rule), do NOT run the later segments (they would only
# restart from the stopped state and stop again after another 5 R/vA at tiny dt).
FLUCS=~/flucs_work/.venv/bin/flucs
here=$(pwd)
for seg in t[0-9][0-9][0-9][0-9]; do
    cd "$here/$seg" || exit 2
    echo "== start $seg $(date)"
    "$FLUCS" --run --io_path . > run.log 2>&1
    rc=$?
    echo "== exit $rc $seg $(date) | $(grep -o 'given in.*' run.log | tail -1)"
    [ $rc -eq 0 ] || { echo "segment $seg failed (exit $rc): chain stopped"; exit $rc; }
    if grep -q "Early stop" run.log; then
        echo "segment $seg: $(grep 'Early stop' run.log | tail -1) -> chain stopped"
        echo "$seg $(grep 'Early stop' run.log | tail -1)" > "$here/EARLY_STOP"
        exit 0
    fi
done
echo "== all segments done $(date)"
