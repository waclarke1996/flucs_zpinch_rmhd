"""T0: linear operator of ZPinchFourier vs the SPEC.md quartic on every grid mode.
Usage: python check_T0.py <scratch dir> [n]
For each point: characteristic polynomial of L (= -compute_linear_matrix_reference)
vs lambda^4 - (b/c) lambda^2 + cc/c (omega = i lambda), coefficient error relative
to the largest coefficient; max Re(eig L) vs gamma from the quartic; SPEC table."""
import sys

import numpy as np
from zp_common import POINTS, SPEC_MODES, SPEC_TABLE, gamma_theory, write_input

from flucs.input import FlucsInput

scratch, n = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 64
worst = 0.0
for name, (kb, kP) in POINTS.items():
    d = write_input(f"{scratch}/T0_{name}", kb=kb, kP=kP, n=n, linear=True)
    solver, system = FlucsInput(d / "input.toml").create_solver_system()
    system.setup()
    system.compile_cupy_module()
    M = system.compute_linear_matrix_reference()
    L = -np.moveaxis(M, (0, 1), (-2, -1))  # (..., 4, 4)
    kz, kx, ky = (np.broadcast_to(a, L.shape[:-2]) for a in system.get_broadcast_wavenumbers())
    sel = system.get_solved_grid_mask().astype(bool) & ((kx**2 + ky**2) > 0)
    L, kx, ky, kz = L[sel], kx[sel], ky[sel], kz[sel]
    # characteristic polynomial coefficients of each 4x4 L
    cp_num = np.array([np.poly(m) for m in L])
    c = 1 + 1 / kb
    kap = ky**2 / (kx**2 + ky**2)
    N2 = 2 * kb / (1 + kb) * (kP * (1 + kb) - 2)
    N02 = 2 * kb * kP
    b = kap * c * N2 - (1 + c) * kz**2
    cc = kz**2 * (kz**2 - kap * N02)
    cp_th = np.stack([np.ones_like(b), 0 * b, -b / c, 0 * b, cc / c], axis=1)
    scale = np.max(np.abs(cp_th), axis=1)
    err_cp = np.max(np.abs(cp_num - cp_th), axis=1) / scale
    g_num = np.max(np.linalg.eigvals(L).real, axis=1)
    g_th = np.array([gamma_theory(kb, kP, a, bb, z) for a, bb, z in zip(kx, ky, kz)])
    unst = g_th > 1e-6
    err_g = np.max(np.abs(g_num[unst] - g_th[unst]) / g_th[unst]) if unst.any() else 0.0
    stab_g = np.max(np.abs(g_num[~unst])) if (~unst).any() else 0.0
    tab = [gamma_theory(kb, kP, *m) for m in SPEC_MODES]
    err_tab = max(abs(g - (r or 0.0)) for g, r in zip(tab, SPEC_TABLE[name]))
    worst = max(worst, err_cp.max(), err_g)
    print(f"{name}: modes {sel.sum()}, charpoly max rel err {err_cp.max():.1e}, "
          f"unstable {unst.sum()} max rel err gamma {err_g:.1e}, "
          f"stable max|Re eig| {stab_g:.1e}, SPEC table max abs err {err_tab:.1e}")
    print("   table:", " ".join(f"{g:.5f}" if g > 0 else "st" for g in tab))
print(f"T0 {'PASS' if worst < 1e-10 else 'FAIL'}: worst rel err {worst:.1e}")
