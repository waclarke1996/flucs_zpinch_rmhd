#!/usr/bin/env python3
"""Stage 4b CPU test of the Paper 1 k_perp bin rule (zpinch_runtime_diags.kperp_bin_edges / kperp_bin_index)."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.expanduser("~/flucs_work/flucs/src"))
from flucs.systems.zpinch_fourier.zpinch_runtime_diags import kperp_bin_edges, kperp_bin_index  # noqa: E402
nf = 0
def chk(name, ok):
    global nf; nf += not ok; print(f"  {name:72s} {'PASS' if ok else 'FAIL'}")
for Lx in (4 * np.pi, 8 * np.pi, 2 * np.pi):
    dkx, dky = 2 * np.pi / Lx, 1.0
    e = kperp_bin_edges(dkx, dky, 60.0)
    b = lambda k: int(kperp_bin_index(np.array([k]), e)[0])  # noqa: E731
    chk(f"Lx={Lx/np.pi:g}pi: e0 = 0.5 min(dkx,dky) = {0.5*min(dkx,dky):g}, e1 = 1.50001", np.isclose(e[0], 0.5 * min(dkx, dky)) and np.isclose(e[1], 1.50001))
    chk(f"Lx={Lx/np.pi:g}pi: k_perp = 0 -> bin 0", b(0.0) == 0)
    chk(f"Lx={Lx/np.pi:g}pi: (a) k_x = 2pi/Lx = {dkx:g}, k_y = 0 -> bin 1", b(dkx) == 1)
    chk(f"Lx={Lx/np.pi:g}pi: (b) k_perp 1.5 -> 1, 2.5 -> 2, 1.50002 -> 2, 41.5 -> 41", (b(1.5), b(2.5), b(1.50002), b(41.5)) == (1, 2, 2, 41))
    chk(f"Lx={Lx/np.pi:g}pi: integers j -> bin j (1..42)", all(b(float(j)) == j for j in range(1, 43)))
    # every k_perp of the box once (T1 grid: |mx| <= 85, my <= 42): bins contiguous from 0, edges cover max
    mx, my = np.meshgrid(np.arange(-85, 86), np.arange(0, 43), indexing="ij")
    kp = np.sqrt((mx * dkx) ** 2 + (my * dky) ** 2)
    e = kperp_bin_edges(dkx, dky, kp.max())
    i = kperp_bin_index(kp, e)
    lo = np.concatenate(([-1.0], e))[i]
    chk(f"Lx={Lx/np.pi:g}pi: every box k_perp (max {kp.max():.2f}) in its bin (e_(i-1), e_i], bins 0..{len(e)-1} all used",
        i.max() < len(e) and np.all((kp > lo) & (kp <= e[i])) and len(np.unique(i)) == len(e))
# difference to np.rint at half-integers (Lx = 4 pi): k_x = 0.5 -> rint 0 (old), new 1
print("  rint vs new at k_perp 0.5, 1.5, 2.5, 3.5:", [int(np.rint(k)) for k in (0.5, 1.5, 2.5, 3.5)], [int(kperp_bin_index(np.array([k]), kperp_bin_edges(0.5, 1, 10))[0]) for k in (0.5, 1.5, 2.5, 3.5)])
print("FAIL count", nf); sys.exit(1 if nf else 0)
