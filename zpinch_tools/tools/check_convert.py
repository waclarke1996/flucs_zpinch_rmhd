"""Full-precision check of a converted file: E and Q/Q_MHD with dissipation_map.py's formulas vs FLUCS 0d at the
same time.  usage: check_convert.py CONVERTED.nc INDEX FLUCS_output.0d.nc"""
import sys, os
import numpy as np
from netCDF4 import Dataset
sys.path.insert(0, os.path.expanduser("~/flucs_work/reference/curvy3d"))
import zpinch_runtools as zr
fn, it, f0 = sys.argv[1], int(sys.argv[2]), sys.argv[3]
with Dataset(fn) as ds:
    P = zr.params_from_dataset(ds)
    F = {n: np.array(ds[n + "_rfft_r"][it], float) + 1j * np.array(ds[n + "_rfft_i"][it], float)
         for n in ("Phi", "deltaBpar", "Psi", "upar")}
    t_hf, qe, qm = zr.heat_flux_series(ds, P)
nz, nx, hy = F["Phi"].shape
kx = np.fft.fftfreq(nx, d=1.0 / nx) * 2 * np.pi / P.Lx
ky = np.arange(hy) * 2 * np.pi / P.Ly
kz = np.fft.fftfreq(nz, d=1.0 / nz) * 2 * np.pi / P.Lz
KZ, KX, KY = np.meshgrid(kz, kx, ky, indexing="ij")
k2 = KX ** 2 + KY ** 2
w = np.where(KY > 0, 2.0, 1.0)
c = 1 + 1 / P.kappaBeta
e = 0.5 * k2 * (np.abs(F["Phi"]) ** 2 + np.abs(F["Psi"]) ** 2) * w + 0.5 * (np.abs(F["upar"]) ** 2 + c * np.abs(F["deltaBpar"]) ** 2) * w
inj = w * (P.kappaP * (1 + P.kappaBeta) * np.real(1j * KY * F["Phi"] * np.conj(F["deltaBpar"]))
           - P.kappaP * P.kappaBeta * np.real(1j * KY * F["Psi"] * np.conj(F["upar"])))
E, Q = e.sum(), inj.sum() / P.kappaP / P.Ly ** 2 / zr.q_mhd(P)
with Dataset(f0) as d:
    g = d[sorted(d.groups, key=int)[-1]]
    tF, EF, QF = float(g["time"][-1]), float(g["energy"]["E"][-1]), float(g["energy"]["Q_over_QMHD"][-1])
qs = (qe[-1] + qm[-1]) / zr.q_mhd(P)
print(f"index {it} (t = {it*P.save_time:g}); FLUCS t = {tF:.6f}")
print(f"E   converted {E:.12e}  FLUCS {EF:.12e}  rel {abs(E/EF-1):.1e}")
print(f"Q/QMHD conv.  {Q:.12e}  FLUCS {QF:.12e}  rel {abs(Q/QF-1):.1e}")
print(f"heat_flux_series last sample t = {t_hf[-1]:.2f}: Q/QMHD {qs:.12e}  rel {abs(qs/QF-1):.1e}")
