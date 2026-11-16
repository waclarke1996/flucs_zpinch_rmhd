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
# In-chain extension (point.json "auto_extend": true, production lanes): when every segment is done, the extension
# rule is applied ON THE NODE (campaign.py assess --apply): EXTEND adds segments and the chain continues without
# releasing the GPU; FINAL writes RUN/FINAL. At FINAL / EARLY_STOP the summary .nc (no snapshot) is written.
# A run with RUN/FINAL does nothing. Without auto_extend the chain ends after its last segment, as before.
FLUCS=~/flucs_work/.venv/bin/flucs
PY=~/flucs_work/.venv/bin/python
CHK=~/flucs_work/runs/tools/check_state.py
CAMP=~/flucs_work/runs/stage4a/campaign.py
SUMM=~/flucs_work/runs/tools/flucs_summary_nc.py
export PYTHONDONTWRITEBYTECODE=1        # never write __pycache__ (e.g. into the read-only curvy3d folder)
here=$(pwd)
keep=${KEEP_STATES:-0}
auto=$(grep -c '"auto_extend": true' "$here/point.json" 2>/dev/null)

githash() {   # without git (not installed on every node): .git/HEAD -> ref file or packed-refs
    local g=~/flucs_work/flucs/.git h r
    h=$(cat "$g/HEAD" 2>/dev/null) || { echo "?"; return; }
    case "$h" in
        "ref: "*) r=${h#ref: }
                  if [ -f "$g/$r" ]; then cat "$g/$r"; else awk -v r="$r" '$2 == r {print $1}' "$g/packed-refs"; fi ;;
        *) echo "$h" ;;
    esac
}

val() { grep -E "^\s*$1\s*=" "$2" | head -1 | sed -E 's/^[^=]*=\s*//; s/\s*#.*$//; s/^"(.*)"$/\1/'; }

[ -f "$here/FINAL" ] && { echo "== FINAL present ($(cat "$here/FINAL")): nothing to do"; exit 0; }
[ -f "$here/EARLY_STOP" ] && [ "$auto" != 1 ] && { echo "== EARLY_STOP present ($(cat "$here/EARLY_STOP")): nothing to do"; exit 0; }

finalize() {   # summary .nc without snapshot (failure is reported, not fatal: it can be redone on the login node)
    $PY $SUMM "$here" --no-snapshot > "$here/summary.log" 2>&1 \
        && echo "== summary: $(grep -o 'wrote [^:]*' "$here/summary.log" | tail -1)" \
        || echo "== summary FAILED (see summary.log; redo: flucs_summary_nc.py $here --no-snapshot)"
}

while true; do
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
    [ -f "$here/EARLY_STOP" ] && break      # (auto_extend resubmission after an early stop: go to the verdict)
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
    echo "$(date -Is) $seg $(githash) $(hostname)" >> "$here/flucs_git.txt"
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
        break
    fi
done
[ "$auto" = 1 ] || { [ -f "$here/EARLY_STOP" ] || echo "== all segments done $(date)"; exit 0; }
# auto_extend: apply the extension rule here (EARLY_STOP -> FINAL)
$PY $CAMP assess "$here" --apply 2>&1 | grep -v -i "cupy\|warn" | sed 's/^/   assess: /'
arc=${PIPESTATUS[0]}
if [ "$arc" = 10 ]; then
    echo "== extended in place $(date): continuing"
    continue
fi
[ "$arc" = 0 ] && [ -f "$here/FINAL" ] || { echo "ERROR: assess --apply failed (rc $arc): chain stopped"; exit 11; }
echo "== FINAL $(date): $(cat "$here/FINAL")"
finalize
exit 0
done
