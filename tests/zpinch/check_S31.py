"""S3.1 / S3.3: saturated Q/Q_MHD and k_perp spectrum, FLUCS vs a curvy3d run.

Usage: check_S31.py FLUCS_RUN_DIR CURVY_NC T0 T1 OUT_PNG [CT0 CT1]
  T0..T1   averaging window for FLUCS (and also applied to curvy3d)
  CT0 CT1  optional extra window for curvy3d alone (e.g. its own reference window)
Spectrum: Alfvenic E(k) = 1/2 k^2 (|Phi|^2 + |Psi|^2) per unit k, box average, bins of width 2pi/Ly centred on
integers (as reference/curvy3d/spectrum_check.py). FLUCS floor-shells of width 0.5: bin n = shells 2n-1 + 2n.
"""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import netCDF4  # noqa: E402
import numpy as np  # noqa: E402
from curvy_compare import mean_err, read_curvy, read_flucs  # noqa: E402

run, nc, t0, t1, png = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
extra = [float(x) for x in sys.argv[6:8]]
F = read_flucs(run)
C = read_curvy(nc)
P = C["P"]
print(f"FLUCS: t_end {F['t'][-1]:.2f}, steps saved {len(F['t'])}, mean dt {np.mean(F['dt']):.4e}")
m = mean_err(F["t"], F["Q_over_QMHD"], t0, t1)
print(f"FLUCS   <Q/Q_MHD> t={t0:g}-{t1:g}: {m[0]:.4f} +- {m[1]:.4f} (tau_int {m[2]:.2f}, n_eff {m[3]:.0f})")
for a, b in [(t0, t1)] + ([tuple(extra)] if extra else []):
    if C["t"][-1] >= b - 1e-6:
        mc = mean_err(C["t"], C["QQ"], a, b)
        print(f"curvy3d <Q/Q_MHD> t={a:g}-{b:g}: {mc[0]:.4f} +- {mc[1]:.4f} (tau_int {mc[2]:.2f}, n_eff {mc[3]:.0f})")
    else:
        print(f"curvy3d ends at t = {C['t'][-1]:.1f} < {b:g}")
w = (F["t"] >= t0) & (F["t"] <= t1)
print(f"FLUCS   <E> {np.mean(F['E'][w]):.4e}, <D/inj> {np.mean(F['D'][w]) / np.mean(F['injection'][w]):.4f}, "
      f"max Q/Q_MHD {np.max(F['Q_over_QMHD']):.3e} at t = {F['t'][np.argmax(F['Q_over_QMHD'])]:.2f}")
wc = (C["t"] >= t0) & (C["t"] <= t1)
if wc.any():
    print(f"curvy3d <E> {np.mean(C['E'][wc]):.4e}, mean dt {np.mean(C['dt'][wc]):.4e}; FLUCS mean dt {np.mean(F['dt'][w]):.4e}")


def slope_fit(k, E, k0=2, k1=7):
    s = (k >= k0) & (k <= k1) & (E > 0)
    return np.polyfit(np.log(k[s]), np.log(E[s]), 1)[0], np.mean(np.gradient(np.log(E), np.log(k))[s])


# FLUCS spectrum
g = netCDF4.Dataset(f"{run}/output.1d.nc")["0"]
ts = np.asarray(g["time"][:], dtype=float)
sp = g["energy_1d/kperp_spectra"]
ws = (ts >= t0) & (ts <= t1)
EA = np.mean(np.asarray(sp["E_uperp"][:], dtype=float)[ws] + np.asarray(sp["E_bperp"][:], dtype=float)[ws], axis=0)
nb = len(EA) // 2
kF = np.arange(1, nb)
EF = np.array([EA[2 * n - 1] + EA[2 * n] for n in kF])  # dk = 1
print(f"FLUCS spectrum: {ws.sum()} snapshots in t={t0:g}-{t1:g}")

# curvy3d spectrum (time average from start_spectral_saves), same logic as spectrum_check.py
ds = C["ds"]
phi2 = np.array(ds["Phi_spec_calc"][:], dtype=float)
psi2 = np.array(ds["Psi_spec_calc"][:], dtype=float)
nx, hy = phi2.shape
kx = np.fft.fftfreq(nx, d=1.0 / nx) * 2 * np.pi / P.Lx
ky = np.arange(hy) * 2 * np.pi / P.Ly
k2 = kx[:, None] ** 2 + ky[None, :] ** 2
ea = 0.5 * k2 * (phi2 + psi2) * 2.0
dk = 2 * np.pi / P.Ly
ib = np.rint(np.sqrt(k2) / dk).astype(int)
EC = np.bincount(ib.ravel(), ea.ravel()) / dk
kC = np.arange(len(EC)) * dk
EC, kC = EC[1:], kC[1:]
ssp = getattr(P, "start_spectral_saves", np.nan)
for name, k, E in (("FLUCS", kF, EF), (f"curvy3d (avg from t={ssp})", kC, EC)):
    fit, loc = slope_fit(k, E)
    ipk = np.argmax(E[: int(P.ny // 2)])
    print(f"{name:28s}: peak k = {k[ipk]:g}, slope k=2-7 fit {fit:.2f}, mean local {loc:.2f}")
n = min(len(EF), len(EC), 30)
r = EF[:n] / EC[:n]
print("E_F/E_C (k=1..12):", " ".join(f"{x:.2f}" for x in r[:12]))
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(C["t"], C["QQ"], lw=0.6, label="curvy3d")
ax[0].plot(F["t"], F["Q_over_QMHD"], lw=0.6, label="FLUCS")
ax[0].set_yscale("log")
ax[0].set_xlabel("t")
ax[0].set_ylabel("Q/Q_MHD")
ax[0].legend()
ax[1].loglog(kC, EC, label=f"curvy3d (t>={ssp})")
ax[1].loglog(kF, EF, "--", label=f"FLUCS t={t0:g}-{t1:g}")
kk = np.array([2.0, 7.0])
ax[1].loglog(kk, EF[1] * (kk / 2) ** (-5 / 3), "k:", label="-5/3")
ax[1].set_xlabel("k_perp")
ax[1].set_ylabel("E_A(k)")
ax[1].legend()
fig.tight_layout()
fig.savefig(png, dpi=110)
print("saved", png)
