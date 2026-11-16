#!/usr/bin/env python3
"""
Stage 4a review (6): mean-field profiles from zp_profiles (spectral k_y = k_z = 0 and k_x = k_z = 0 lines, read with
zpinch_readers.mean_profiles) against profiles computed directly from the full 3D state (restart.nc -> real space
on the padded grid with numpy irfftn, derivatives spectral on the full grid, then (y,z)- or (x,z)-averages).
Also: Hermitian symmetry of the k_y = 0 plane of the state.
Usage: check_profiles.py RESTART.nc OUTPUT.1d.nc [record index, default 0]
"""
import os
import sys

import numpy as np
from netCDF4 import Dataset

sys.path.insert(0, os.path.expanduser("~/flucs_work/flucs/src"))
from flucs.systems.zpinch_fourier.zpinch_readers import mean_profiles  # noqa: E402

rst, out1d = sys.argv[1], sys.argv[2]
rec = int(sys.argv[3]) if len(sys.argv) > 3 else 0
Lx, Ly, Lz = 4 * np.pi, 2 * np.pi, 2 * np.pi
with Dataset(rst) as ds:
    t_rst = float(ds["current_time"][...])
    X = {n: np.asarray(ds["fields_real"][i]) + 1j * np.asarray(ds["fields_imag"][i])
         for i, n in enumerate(("Phi", "Psi"))}
nz, nx, hy = X["Phi"].shape
ny = 2 * (hy - 1)
kx = np.fft.fftfreq(nx, 1.0 / nx) * 2 * np.pi / Lx
ky = np.arange(hy) * 2 * np.pi / Ly
kz = np.fft.fftfreq(nz, 1.0 / nz) * 2 * np.pi / Lz
N = nx * ny * nz
print(f"state t = {t_rst}, padded grid (nz, nx, ny) = ({nz}, {nx}, {ny})")

# Hermitian symmetry of the k_y = 0 plane: X(-kz, -kx, 0) = X(kz, kx, 0)*
P0 = X["Phi"][:, :, 0]
neg = P0[(-np.arange(nz)) % nz][:, (-np.arange(nx)) % nx]
print(f"k_y=0 plane Hermitian error |X(-k) - X(k)*|/max|X|: Phi {np.max(np.abs(neg - P0.conj())) / np.max(np.abs(P0)):.2e}")


def real(F):
    return np.fft.irfftn(F, s=(nz, nx, ny), axes=(0, 1, 2)) * N


KX = kx[None, :, None]
KY = ky[None, None, :]
direct = {
    "uy_x": real(1j * KX * X["Phi"]).mean(axis=(0, 2)),
    "dBy_x": real(1j * KX * X["Psi"]).mean(axis=(0, 2)),
    "shear_x": real(-KX**2 * X["Phi"]).mean(axis=(0, 2)),
    "jz_x": real(-KX**2 * X["Psi"]).mean(axis=(0, 2)),
    "ux_y": real(-1j * KY * X["Phi"]).mean(axis=(0, 1)),
    "dBx_y": real(-1j * KY * X["Psi"]).mean(axis=(0, 1)),
    "shear_y": real(KY**2 * X["Phi"]).mean(axis=(0, 1)),         # d_y u_x = -d_y^2 Phi
    "jz_y": real(-KY**2 * X["Psi"]).mean(axis=(0, 1)),           # d_y^2 Psi
    "Phi_x": real(X["Phi"]).mean(axis=(0, 2)),
    "Phi_y": real(X["Phi"]).mean(axis=(0, 1)),
}
P = mean_profiles(out1d, nx=nx, ny=ny)
print(f"profile record {rec}: t = {P['t'][rec]}")
nfail = 0
for k, v in direct.items():
    a = P[k][rec]
    err = np.max(np.abs(a - v)) / np.max(np.abs(v))
    ok = err < 1e-10
    nfail += not ok
    print(f"  {k:8s} rms {np.sqrt(np.mean(v**2)):10.4g}  rel err {err:.2e} {'PASS' if ok else 'FAIL'}")
print(f"FAIL count: {nfail}")
sys.exit(1 if nfail else 0)
