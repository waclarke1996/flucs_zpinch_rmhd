#!/bin/bash
# Stage 4a verification (GPU job): FLUCS run with all run-time diagnostics from the A3a T2 seed (t reset to 0), then
# the offline tools on the same state, then the comparison -> verify.txt.
FLUCS=~/flucs_work/.venv/bin/flucs
PY=~/flucs_work/.venv/bin/python
C=~/flucs_work/reference/curvy3d
SEED=~/flucs_work/runs/stage3e/A3a-T2-k8/t1200/restart.nc
export PYTHONDONTWRITEBYTECODE=1 HDF5_USE_FILE_LOCKING=FALSE MPLBACKEND=Agg
here=$(pwd)
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

if [ -e "$here/flucs_run/output.0d.nc" ]; then
    echo "== flucs run exists, skipped"
else
    echo "== flucs run $(date)"
    cd "$here/flucs_run" && "$FLUCS" --run --io_path . > run.log 2>&1
    echo "== flucs exit $? $(date) | $(grep -o 'given in.*' run.log | tail -1)"
fi
cd "$here"
rm -f ref/seed.nc

echo "== convert $(date)"
$PY ../../tools/flucs_to_curvy3d.py "$SEED" --out ref/seed.nc --run-id A3a-T2-seed \
    --series flucs_run/output.0d.nc --save-time 100000 > ref/convert.log 2>&1 || { tail ref/convert.log; exit 3; }

echo "== dissipation_map $(date)"
$PY capture_tool.py ref/dmap.pkl $C/dissipation_map.py ref/seed.nc --out ref/dmap.png > ref/dmap.log 2>&1
tail -n 1 ref/dmap.log
echo "== kpar_spectrum --gpu $(date)"
$PY capture_tool.py ref/kpar.pkl $C/kpar_spectrum.py ref/seed.nc --gpu --per-field --out ref/kpar.png > ref/kpar.log 2>&1
tail -n 1 ref/kpar.log

for s in elsasser:shells_els magnetic_kinetic:shells_mk; do
    mkdir -p ref/${s#*:}
    echo "== hydra_shell_plotter_${s%%:*}_gpu.py $(date)"
    (cd ref/${s#*:} && PYTHONPATH="$here/stub:$C" $PY $C/hydra_shell_plotter_${s%%:*}_gpu.py ../seed.nc double 0 transfer \
        > plotter.log 2>&1; echo "exit $?"; tail -n 2 plotter.log)
done

echo "== compare $(date)"
$PY compare_stage4a.py > verify.txt 2>&1
tail -n 3 verify.txt
echo "== done $(date)"
