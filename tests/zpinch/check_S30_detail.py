"""S3.0 detail: F/C ratios of Q and E at selected times; FLUCS dt columns. Usage: run_dir curvy_nc"""
import sys

import netCDF4
import numpy as np
from curvy_compare import read_curvy, read_flucs

F = read_flucs(sys.argv[1])
C = read_curvy(sys.argv[2])
g = netCDF4.Dataset(f"{sys.argv[1]}/output.0d.nc")["0"]
print("group vars:", list(g.variables), list(g.groups))
print(open(f"{sys.argv[1]}/output.time.txt").read().splitlines()[-2:])
tc = C["t"]
for t in (0.5, 1, 2, 3, 5, 8, 10, 12, 15, 17, 19, 20, 21, 22, 23):
    i = np.argmin(np.abs(tc - t))
    qf = np.interp(tc[i], F["t"], F["Q_over_QMHD"])
    ef = np.interp(tc[i], F["t"], F["E"])
    print(f"t={tc[i]:5.1f} Q/QMHD C {C['QQ'][i]:.3e} F/C {qf / C['QQ'][i]:.5f}  E C {C['E'][i]:.4e} F/C-1 {ef / C['E'][i] - 1:+.2e}")
