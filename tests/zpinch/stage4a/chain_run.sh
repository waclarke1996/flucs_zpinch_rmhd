#!/bin/bash
# Stage 4 production chain (copied into each RUN folder as run.sh by campaign.py make / extend).
# Runs the segments tEEEE/ of this folder in order (EEEE = nominal end time); segment k restarts from
# ../t<prev>/state.nc (its input.toml says so; checked here).
#
# Storage policy: only the latest VALIDATED 3D state is kept. A run never lacks a valid restart:
#   1. FLUCS writes the segment's final state to restart.nc in the segment folder (= temporary name: never used by
#      another segment until validated; a crash leaves at most a partial restart.nc that is ignored and removed).
#   2. tools/check_state.py checks restart.nc: opens, grid shape, all values finite, and the time
#      t_start + tfinal <= t <= t_start + tfinal + 2 dt_max (t_start = time of the state it started from; an early
#      stop only needs t_start <= t).
#   3. mv restart.nc state.nc (atomic rename in the same folder), then SEG_OK (end time) is written atomically.
#   4. only then the previous segment's state.nc is deleted (never a state outside this folder).
# A resubmitted job continues: segments with SEG_OK are skipped; the first unfinished one is started afresh from the
# last validated state (its partial outputs / restart files are removed). Left-over older states (a kill between
# steps 3 and 4) are deleted at the start. KEEP_STATES=1 in the environment keeps every state (tests only).
# An early stop (runaway rule) ends the chain: RUN/EARLY_STOP (a resubmitted job then does nothing).
# Test hook: CHAIN_TEST_STOP_AT=after_validate|after_rename|after_ok exits (rc 99) at that point of the first
# segment that is run, to emulate a job killed there.
FLUCS=~/flucs_work/.venv/bin/flucs
PY=~/flucs_work/.venv/bin/python
CHK=~/flucs_work/runs/tools/check_state.py
here=$(pwd)
keep=${KEEP_STATES:-0}

val() { grep -E "^\s*$1\s*=" "$2" | head -1 | sed -E 's/^[^=]*=\s*//; s/\s*#.*$//; s/^"(.*)"$/\1/'; }

[ -f "$here/EARLY_STOP" ] && { echo "== EARLY_STOP present ($(cat "$here/EARLY_STOP")): nothing to do"; exit 0; }
segs=( $(cd "$here" && ls -d t[0-9][0-9][0-9][0-9] 2>/dev/null | sort) )
[ ${#segs[@]} -gt 0 ] || { echo "no segments t????/ in $here"; exit 2; }

# last finished segment, and clean-up of states older than it
last=""
for s in "${segs[@]}"; do [ -f "$here/$s/SEG_OK" ] && last=$s; done
if [ -n "$last" ]; then
    [ -f "$here/$last/state.nc" ] || { echo "ERROR: last finished segment $last has no state.nc"; exit 5; }
    if [ "$keep" != 1 ]; then
        for s in "${segs[@]}"; do
            [ "$s" = "$last" ] && break
            [ -f "$here/$s/state.nc" ] && rm -f "$here/$s/state.nc" && echo "== removed left-over $s/state.nc"
        done
    fi
fi

prev=""
for seg in "${segs[@]}"; do
    d="$here/$seg"
    if [ -f "$d/SEG_OK" ]; then
        echo "== $seg already done ($(cat "$d/SEG_OK"))"
        prev=$seg
        continue
    fi
    inp="$d/input.toml"
    rf=$(val restart_from "$inp")
    if [ -n "$prev" ] && [ "$rf" != "../$prev/state.nc" ]; then
        echo "ERROR: $seg/input.toml restart_from = '$rf', expected '../$prev/state.nc': chain stopped"; exit 6
    fi
    if [ -n "$rf" ]; then
        rs=$(cd "$d" && $PY $CHK "$rf" --input "$inp") || { echo "ERROR: start state $rf of $seg invalid: $rs"; exit 7; }
        t0=$(echo "$rs" | sed -E 's/^OK t=([^ ]*) .*/\1/')
    else
        t0=0
    fi
    tf=$(val tfinal "$inp"); dtm=$(val dt_max "$inp")
    cd "$d" || exit 2
    rm -f output.*.nc zp_timing_*.txt restart.nc restart.temp.nc restart.backup.*.nc state.nc SEG_OK.tmp STOP
    echo "$(date -Is) $seg $(git -C ~/flucs_work/flucs rev-parse HEAD 2>/dev/null)" >> "$here/flucs_git.txt"
    echo "== start $seg $(date) (from t = $t0)"
    "$FLUCS" --run --io_path . > run.log 2>&1
    rc=$?
    echo "== exit $rc $seg $(date) | $(grep -o 'given in.*' run.log | tail -1)"
    [ $rc -eq 0 ] || { echo "segment $seg failed (exit $rc): chain stopped"; exit $rc; }
    early=$(grep "Early stop" run.log | tail -1)
    if [ -n "$early" ]; then
        tmin=$t0
    else
        tmin=$(awk -v a="$t0" -v b="$tf" 'BEGIN{printf "%.17g", a + b - 1e-6}')
    fi
    tmax=$(awk -v a="$t0" -v b="$tf" -v c="$dtm" 'BEGIN{printf "%.17g", a + b + 2 * c}')
    rs=$($PY $CHK restart.nc --input "$inp" --t-min "$tmin" --t-max "$tmax")
    [ $? -eq 0 ] || { echo "segment $seg: restart.nc invalid ($rs): chain stopped, previous state kept"; exit 8; }
    t1=$(echo "$rs" | sed -E 's/^OK t=([^ ]*) .*/\1/')
    echo "   validated restart.nc: t = $t1 (allowed [$tmin, $tmax])"
    [ "${CHAIN_TEST_STOP_AT:-}" = after_validate ] && { echo "TEST: stop after_validate"; exit 99; }
    mv -f restart.nc state.nc || { echo "rename failed: chain stopped"; exit 9; }
    [ "${CHAIN_TEST_STOP_AT:-}" = after_rename ] && { echo "TEST: stop after_rename"; exit 99; }
    echo "$t1 $(date -Is)" > SEG_OK.tmp && mv -f SEG_OK.tmp SEG_OK
    [ "${CHAIN_TEST_STOP_AT:-}" = after_ok ] && { echo "TEST: stop after_ok"; exit 99; }
    if [ -n "$prev" ] && [ "$keep" != 1 ] && [ -f "$here/$prev/state.nc" ]; then
        rm -f "$here/$prev/state.nc" && echo "   deleted $prev/state.nc"
    fi
    prev=$seg
    if [ -n "$early" ]; then
        echo "segment $seg: $early -> chain stopped"
        echo "$seg $early" > "$here/EARLY_STOP"
        exit 0
    fi
done
echo "== all segments done $(date)"
