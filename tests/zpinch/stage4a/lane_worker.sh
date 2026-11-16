#!/bin/bash
# Stage 4 lane worker: copied as run.sh into ROOT/workers/W<stamp>-<gpu>-<k>/ by `campaign.py lanes` (with lane.cfg:
# ROOT, CLASSES, GPU). One GPU job = one worker = one "lane": it keeps the GPU and runs campaign chains back to back.
#
#   loop: claim a run  ->  ROOT/<run_id>/run.sh (chain_run.sh: segments, in-chain extension rule, FINAL, summary)
#         until no eligible run is left (then exit: the GPU is released only when there is no work for it).
#
# Claiming (shared NFS folder ROOT/queue; both GPU nodes see it):
#   todo/<tier>/NN_<run_id>  (NN = rank by H200 cost, longest first)  --mv-->  claimed/NN_<run_id> (+ .owner)
#   `mv` is an atomic rename: two workers can never claim the same entry. Lists are taken in CLASSES order
#   (h200/a100: "T2 T1", rtx: "T1"). Finished: claimed -> done/ (FINAL present) or failed/ (chain error).
# Orphans: a claimed entry whose owner job has ENDED according to sacct (crash, node failure, scancel) is adopted
#   (atomic mv of its .owner file); the chain resumes from its last validated state. A claim is never adopted
#   while sacct does not positively report a terminal state for the owner job.
# Safety: two consecutive failed runs in this worker -> the worker stops (exit 3), so a broken GPU/node cannot mark
#   the whole list failed. ROOT/queue/STOP_WORKERS: every worker exits after its current run (drain).
set -u
here=$(pwd)
# shellcheck disable=SC1091
. "$here/lane.cfg"
Q=$ROOT/queue
export PATH=$PATH:/usr/local/shared/slurm/bin
export PYTHONDONTWRITEBYTECODE=1
me="$(basename "$here") ${SLURM_JOB_ID:-nojob} $(hostname)"
log() { echo "$(date -Is) $*"; }

job_ended() {   # 0 only if sacct reports a terminal state for job $1
    local st
    st=$(sacct -n -X -j "$1" -o State 2>/dev/null | head -1 | awk '{print $1}')
    case "$st" in
        COMPLETED|FAILED|CANCELLED*|TIMEOUT|NODE_FAIL|OUT_OF_MEMORY|PREEMPTED|BOOT_FAIL|DEADLINE) return 0 ;;
    esac
    return 1
}

eligible() {    # entry name -> is its tier in CLASSES?
    local c
    for c in $CLASSES; do case "$1" in *-"$c"-*) return 0 ;; esac; done
    return 1
}

claim() {       # prints the claimed entry; log lines go to stderr
    local e o oj c
    for e in $(ls "$Q/claimed" 2>/dev/null | grep -v '\.owner' | sort); do
        o="$Q/claimed/$e.owner"
        [ -f "$o" ] && eligible "$e" || continue
        oj=$(awk '{print $2}' "$o")
        if job_ended "$oj" && mv "$o" "$o.adopt.$$" 2>/dev/null; then
            echo "$me" > "$o" && rm -f "$o.adopt.$$"
            log "adopted orphan $e (owner job $oj has ended)" >&2
            echo "$e"; return 0
        fi
    done
    for c in $CLASSES; do
        for e in $(ls "$Q/todo/$c" 2>/dev/null | sort); do
            if mv "$Q/todo/$c/$e" "$Q/claimed/$e" 2>/dev/null; then
                echo "$me" > "$Q/claimed/$e.owner"
                echo "$e"; return 0
            fi
        done
    done
    return 1
}

log "worker $me: GPU $GPU, classes '$CLASSES', root $ROOT"
nfail=0
while true; do
    [ -f "$Q/STOP_WORKERS" ] && { log "STOP_WORKERS present: exit"; exit 0; }
    e=$(claim) || { log "no eligible run left: exit (GPU released)"; exit 0; }
    rid=$(cat "$Q/claimed/$e")
    run=$ROOT/$rid
    log "claimed $e -> $run"
    ( cd "$run" && echo "######## $(date -Is) $me" >> chain.log && bash run.sh >> chain.log 2>&1 )
    rc=$?
    if [ $rc -eq 0 ] && [ -f "$run/FINAL" ]; then
        mv "$Q/claimed/$e" "$Q/done/$e" && mv "$Q/claimed/$e.owner" "$Q/done/$e.owner"
        log "done $e: $(cat "$run/FINAL")"
        nfail=0
    else
        mv "$Q/claimed/$e" "$Q/failed/$e" && mv "$Q/claimed/$e.owner" "$Q/failed/$e.owner"
        log "FAILED $e (rc $rc): see $run/chain.log"
        nfail=$((nfail + 1))
        [ $nfail -ge 2 ] && { log "two consecutive failures: worker stops"; exit 3; }
    fi
done
