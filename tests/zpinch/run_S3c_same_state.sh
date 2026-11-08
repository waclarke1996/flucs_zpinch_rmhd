#!/bin/bash
# dissipation_map on every snapshot t >= 100 of the curvy3d A3a-Y family + FLUCS states; kpar_spectrum on a subset.
cd /mnt/users/clarkew/flucs_work/runs/stage3c/same_state
PY=/mnt/users/clarkew/flucs_work/.venv/bin/python; R=/mnt/users/clarkew/flucs_work/reference/curvy3d
D=/mnt/users/clarkew/data/rmhd_zpinch/output/3D/BP2_anchors_v2
f() { ls $D/*_RKARS343_A3a-$1.nc; }
dm() { $PY $R/dissipation_map.py "$1" --tmin $3 --tmax $3 --out dmap_$2_t$3.png > dmap_$2_t$3.txt 2>&1; }
kp() { $PY $R/kpar_spectrum.py "$1" --tmin $3 --tmax $3 --pad 1.5 --out kpar_$2_t$3.png > kpar_$2_t$3.txt 2>&1; }
for T in 100 150 200 250 300 350 400 450 500; do dm $(f Y) A3a-Y $T; dm $(f YP4) A3a-YP4 $T; done
for T in 500 1000; do dm $(f Yc2) A3a-Yc2 $T; done      # Yc2 own time: 500 -> t = 1000, 1000 -> t = 1500
for T in 100 150 200 250 300; do dm $(f Z) A3a-Z $T; done
for T in 100 200 300 400 500; do dm $(f YHR2) A3a-YHR2 $T; done
dm ../A3a-Y-k8-S30_t167.nc A3a-Y-k8-S30 167 ; dm ../A3a-Y-k8-S30c_t500.nc A3a-Y-k8-S30c 500
# kpar: high-state and low-state examples
kp $(f Y) A3a-Y 150; kp $(f Y) A3a-Y 300; kp $(f YP4) A3a-YP4 250; kp $(f YP4) A3a-YP4 400; kp $(f YP4) A3a-YP4 450
kp $(f Z) A3a-Z 300; kp $(f Z) A3a-Z 250; kp $(f YHR2) A3a-YHR2 200; kp $(f YHR2) A3a-YHR2 300; kp $(f Yc2) A3a-Yc2 500
kp ../A3a-Y-k8-S30_t167.nc A3a-Y-k8-S30 167
echo all done
