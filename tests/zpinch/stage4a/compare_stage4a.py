#!/usr/bin/env python3
"""
Stage 4a verification: run-time diagnostics (FLUCS outputs, record at t = 0 of flucs_run/, = the seed state)
against the offline tools / curvy3d formulas on the SAME state (ref/seed.nc, converted from the seed restart.nc):
  dissipation_map.py (captured locals: ref/dmap.pkl), kpar_spectrum.py --gpu (ref/kpar.pkl), the two shell plotters
  (ref/shells_els/*.csv, ref/shells_mk/*.csv; time mean / Ly^2 over the single snapshot), and numpy re-implementations
  (curvy3d compact layout, CPU FFTs) of e1_zonal.py and of the curvy3d solver's subcrit/streamer formulas.
Prints one line per check: max |a - b| / max |b| (rel) and PASS (< tol) / FAIL.
"""
import os
import pickle
import sys

import numpy as np
from netCDF4 import Dataset

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.expanduser("~/flucs_work/flucs/src"))
from flucs.systems.zpinch_fourier.zpinch_readers import load_diag, mean_profiles  # noqa: E402

sys.path.insert(0, os.path.expanduser("~/flucs_work/reference/curvy3d"))
from curvy3dinput import Curvy3DInput  # noqa: E402

RUN = os.path.join(HERE, "flucs_run")
REF = os.path.join(HERE, "ref")
results = []


def check(diag, name, a, b, tol=1e-10):
    a, b = np.asarray(a, dtype=complex if np.iscomplexobj(a) or np.iscomplexobj(b) else float), np.asarray(b)
    if a.shape != b.shape:
        results.append((diag, name, np.nan, f"FAIL shape {a.shape} vs {b.shape}"))
        return
    scale = np.max(np.abs(b)) if b.size else 0.0
    err = np.max(np.abs(a - b)) / scale if scale > 0 else np.max(np.abs(a - b))
    results.append((diag, name, err, "PASS" if err < tol else "FAIL"))


def rec0(out, diag):
    t, d = load_diag(os.path.join(RUN, f"output.{out}.nc"), diag)
    assert abs(t[0]) < 1e-12, f"{out}/{diag}: first record at t = {t[0]}"
    return t, {k: (v[0] if (np.ndim(v) > 0 and len(v) == len(t) and k not in STATIC) else v) for k, v in d.items()}


STATIC = {"kperp", "ky", "kz", "field", "shell", "shell_hi", "kx_line", "ky_line", "x", "y", "ky_mode",
          "ky_profile", "kz_c", "kx_c", "kperp_from", "kperp_to", "kz_from", "kz_to", "kperp_K_boundaries",
          "kz_K_boundaries", "kperp_edges", "kz_edges", "turb_wavenumber_measurements",
          "turb_alfvenic_comps_measurements"}

# ---------------- reference state (curvy3d compact layout) ----------------
with Dataset(os.path.join(REF, "seed.nc")) as ds:
    P = Curvy3DInput(input_str=ds["input"][:].data.tobytes().decode())
    F = {n: np.array(ds[n + "_rfft_r"][0], float) + 1j * np.array(ds[n + "_rfft_i"][0], float)
         for n in ("Phi", "Psi", "deltaBpar", "upar")}
nz, nx, hy = F["Phi"].shape
ny = 2 * hy - 1
kb, kP, c = P.kappaBeta, P.kappaP, 1 + 1 / P.kappaBeta
Lx, Ly, Lz = P.Lx, P.Ly, P.Lz
mx = np.rint(np.fft.fftfreq(nx, 1.0 / nx)).astype(int)
mz = np.rint(np.fft.fftfreq(nz, 1.0 / nz)).astype(int)
kx1, ky1, kz1 = mx * 2 * np.pi / Lx, np.arange(hy) * 2 * np.pi / Ly, mz * 2 * np.pi / Lz
KZ, KX, KY = np.meshgrid(kz1, kx1, ky1, indexing="ij")
k2 = KX**2 + KY**2
kperp = np.sqrt(k2)
w = np.where(KY > 0, 2.0, 1.0)
a2 = {n: np.abs(v) ** 2 for n, v in F.items()}
eparts = [0.5 * w * k2 * a2["Phi"], 0.5 * w * k2 * a2["Psi"], 0.5 * w * c * a2["deltaBpar"], 0.5 * w * a2["upar"]]
e = sum(eparts)
E = e.sum()
Q_MHD = (2 * kb * kP) ** 1.5
ux, dbx = -1j * KY * F["Phi"], -1j * KY * F["Psi"]
q_es = -(1 + kb) * (F["deltaBpar"] * np.conj(ux)).real * w
q_em = kb * (F["upar"] * np.conj(dbx)).real * w
print(f"seed: modes (nz,nx,hy) = {F['Phi'].shape}; Pars nx ny nz = {P.nx} {P.ny} {P.nz}; E = {E:.6g}")

# ---------------- (a) scalars ----------------
_, S = rec0("0d", "zp_scalars")
_, En = rec0("0d", "energy")
zon, zon0 = KY == 0, (KY == 0) & (KZ == 0)
vort2 = w * k2**2 * a2["Phi"]
check("a", "E (energy diag)", En["E"], E)
check("a", "E_kx0_frac", S["E_kx0_frac"], e[KX == 0].sum() / E)
check("a", "E_ky0_frac", S["E_ky0_frac"], e[zon].sum() / E)
check("a", "E_ky0kz0_frac", S["E_ky0kz0_frac"], e[zon0].sum() / E)
check("a", "S_zon0", S["S_zon0"], np.sqrt(vort2[zon0].sum()))
check("a", "S_zon_all", S["S_zon_all"], np.sqrt(vort2[zon].sum()))
check("a", "W_rms", S["W_rms"], np.sqrt(vort2.sum()))
check("a", "b_rms", S["b_rms"], np.sqrt((w * k2 * a2["Psi"]).sum()))
check("a", "u_rms", S["u_rms"], np.sqrt((w * k2 * a2["Phi"]).sum()))
check("a", "upar_rms", S["upar_rms"], np.sqrt((w * a2["upar"]).sum()))
check("a", "dbpar_rms", S["dbpar_rms"], np.sqrt((w * a2["deltaBpar"]).sum()))
check("a", "Q_es/Q_MHD", S["Q_es_over_QMHD"], q_es.sum() / Ly**2 / Q_MHD)
check("a", "Q_em/Q_MHD", S["Q_em_over_QMHD"], q_em.sum() / Ly**2 / Q_MHD)
check("a", "Q_es+em = energy Q/Q_MHD", S["Q_es_over_QMHD"] + S["Q_em_over_QMHD"], En["Q_over_QMHD"])
edges = [1, 2, 4, 8, 16, 32]
for lo, hi in zip(edges[:-1], edges[1:]):
    m = (kperp >= lo) & (kperp < hi)
    check("a", f"kzrms band {lo}-{hi}", S[f"kzrms_{lo}_{hi}"], np.sqrt((KZ**2 * e)[m].sum() / e[m].sum()))
D = pickle.load(open(os.path.join(REF, "dmap.pkl"), "rb"))
sn = D["snaps"][0]
check("a", "dmap snapshot E", En["E"], sn["E"])
check("a", "dmap snapshot b_rms", S["b_rms"], sn["b"])
check("a", "dmap snapshot E_kx0/E", S["E_kx0_frac"], sn["fkx0"])
check("a", "dmap snapshot Q/Q_MHD (inj)", S["Q_es_over_QMHD"] + S["Q_em_over_QMHD"], sn["q"])
results.append(("a", f"cfl {S['cfl']:.3f} cfl_max {S['cfl_max']:.3f} dt_min {S['dt_min']:.3g} (t=0: info)", 0.0, "INFO"))

# ---------------- (b) spectra vs dissipation_map.py ----------------
_, Sp = rec0("1d", "zp_spectra")
nb = int(D["nb"])
check("b", "E_kperp vs dmap E_s", Sp["E_kperp"], D["E_s"])
check("b", "D_perp_kperp vs dmap Dp_s", Sp["D_perp_kperp"], D["Dp_s"])
check("b", "D_par_kperp vs dmap Dz_s", Sp["D_par_kperp"], D["Dz_s"])
check("b", "inj_kperp vs dmap I_s", Sp["inj_kperp"], D["I_s"])
with np.errstate(invalid="ignore", divide="ignore"):
    kzr = np.sqrt(Sp["kz2E_kperp"] / Sp["E_kperp"])
ok = np.isfinite(D["kzrms_s"])
check("b", "kz_rms(kperp) vs dmap kzrms_s", kzr[ok], D["kzrms_s"][ok])
ib = np.rint(kperp).astype(int)
for i, nm in enumerate(("E_uperp", "E_bperp", "E_dbpar", "E_upar")):
    check("b", f"{nm}_kperp", Sp[f"{nm}_kperp"], np.bincount(ib.ravel(), eparts[i].ravel(), nb))
check("b", "Qhat_kperp", Sp["Qhat_kperp"], np.bincount(ib.ravel(), (q_es + q_em).ravel(), nb))
iky = np.rint(KY / (2 * np.pi / Ly)).astype(int)
for nm, arr in (("Qhat", q_es + q_em), ("Qhat_es", q_es), ("Qhat_em", q_em), ("E", e)):
    check("b", f"{nm}_ky", Sp[f"{nm}_ky"], np.bincount(iky.ravel(), arr.ravel(), hy))
ikz = np.abs(np.rint(KZ / (2 * np.pi / Lz))).astype(int)
nkz = ikz.max() + 1
for nm, arr in (("E", D["e"]), ("inj", D["inj"]), ("D_perp", D["dperp"]), ("D_par", D["dpar"])):
    check("b", f"{nm}_kz (dmap per-mode arrays)", Sp[f"{nm}_kz"], np.bincount(ikz.ravel(), arr.ravel(), nkz))

# ---------------- (c) maps ----------------
_, M = rec0("maps", "zp_maps")
b2 = (ib * nkz + ikz).ravel()
for nm, arr in (("E", D["e"]), ("inj", D["inj"]), ("D_perp", D["dperp"]), ("D_par", D["dpar"])):
    ref = np.bincount(b2, arr.ravel(), nb * nkz).reshape(nb, nkz)
    check("c", f"{nm}_map (float32)", M[f"{nm}_map"], ref, tol=1e-6)
check("c", "E_map summed over kz = E_kperp", M["E_map"].astype(float).sum(axis=1), Sp["E_kperp"], tol=1e-6)
results.append(("c", f"n_samples in t=0 record = {M['n_samples']:g}", 0.0, "INFO"))

# ---------------- (d) kpar vs kpar_spectrum.py ----------------
_, Kp = rec0("kpar", "zp_kpar")
K = pickle.load(open(os.path.join(REF, "kpar.pkl"), "rb"))
tool_fields = list(K["fields"])                 # ("Phi", "Psi", "upar", "deltaBpar")
order = [tool_fields.index(n) for n in ("Phi", "Psi", "upar", "deltaBpar")]
mine = [0, 1, 3, 2]                             # FLUCS order Phi Psi dBpar upar -> tool order
for nm in ("den", "lab", "full", "wrong", "large", "E", "vort2"):
    check("d", f"kpar acc {nm}", Kp[nm][mine], K["acc"][nm][order])
check("d", "kpar b2", Kp["b2"], K["b2_sum"])
check("d", "kpar shells", np.stack((Kp["shell"], Kp["shell_hi"]), 1), np.array(K["shells"]))
results.append(("d", f"kpar norm_err runtime {Kp['norm_err']:.1e} tool {K['norm_err']:.1e}", 0.0, "INFO"))

# ---------------- (e) profiles + reader ----------------
_, Pr = rec0("1d", "zp_profiles")
order_x = np.argsort(mx)
names = ("Phi", "Psi", "deltaBpar", "upar")
check("e", "line_x (ky=kz=0)", Pr["line_x"], np.stack([F[n][0, order_x, 0] for n in names]), tol=1e-14)
check("e", "line_y (kx=kz=0)", Pr["line_y"], np.stack([F[n][0, 0, :] for n in names]), tol=1e-14)
check("e", "kx_line", Pr["kx_line"], kx1[order_x], tol=1e-14)
prof = mean_profiles(os.path.join(RUN, "output.1d.nc"), nx=nx, ny=ny, Lx=Lx, Ly=Ly)
N3 = nz * nx * ny
real3 = lambda X: np.fft.irfftn(X, s=(nz, nx, ny)) * N3
phi3 = real3(F["Phi"])
check("e", "reader Phi_x = <Phi>_yz(x)", prof["Phi_x"][0], phi3.mean(axis=(0, 2)))
check("e", "reader Phi_y = <Phi>_xz(y)", prof["Phi_y"][0], phi3.mean(axis=(0, 1)))
check("e", "reader uy_x = <d_x Phi>_yz", prof["uy_x"][0], real3(1j * KX * F["Phi"]).mean(axis=(0, 2)))
check("e", "reader dBx_y = <-d_y Psi>_xz", prof["dBx_y"][0], real3(-1j * KY * F["Psi"]).mean(axis=(0, 1)))
check("e", "reader shear_x = <d_x^2 Phi>_yz", prof["shear_x"][0], real3(-KX**2 * F["Phi"]).mean(axis=(0, 2)))
check("e", "reader shear_y = <d_y u_x>_xz", prof["shear_y"][0],
      real3(1j * KY * (-1j * KY * F["Phi"])).mean(axis=(0, 1)))

# ---------------- (f) slices ----------------
_, Sl = rec0("slices", "zp_slices")
check("f", "Phi_z0 (float32)", Sl["Phi_z0"], phi3[0], tol=1e-6)
check("f", "dBpar_z0 (float32)", Sl["dBpar_z0"], real3(F["deltaBpar"])[0], tol=1e-6)

# ---------------- (g) ky transfers: curvy3d formulas, literal (3D FFT on the padded grid, pick (0,0,ky)) ----
_, Ky = rec0("kyt", "zp_kytransfer")
npz_, npx_, npy_ = P.padded_nz, P.padded_nx, P.padded_ny
iz_p = np.where(np.arange(nz) < (nz + 1) // 2, np.arange(nz), np.arange(nz) + npz_ - nz)
ix_p = np.where(np.arange(nx) < (nx + 1) // 2, np.arange(nx), np.arange(nx) + npx_ - nx)


def padded_real(X):
    out = np.zeros((npz_, npx_, npy_ // 2 + 1), complex)
    out[iz_p[:, None, None], ix_p[None, :, None], np.arange(hy)[None, None, :]] = X
    return np.fft.irfftn(out, s=(npz_, npx_, npy_)) * (npz_ * npx_ * npy_)


def line00(r):      # Fourier coefficients (kz=0, kx=0, ky) of a real padded field, normalised as curvy3d (/N)
    return np.fft.rfftn(r)[0, 0, :hy] / r.size


dxPhi, Psi_r = padded_real(1j * KX * F["Phi"]), padded_real(F["Psi"])
dxPsi, B_r, u_r = padded_real(1j * KX * F["Psi"]), padded_real(F["deltaBpar"]), padded_real(F["upar"])
P0, P1, P2 = line00(dxPhi * Psi_r), line00(dxPhi * u_r), line00(dxPsi * B_r)
del Psi_r, dxPsi, B_r, u_r
nsub = len(Ky["dBx_sq"])
kyv = ky1[1:nsub + 1]
psi, uu = F["Psi"][0, 0, 1:nsub + 1], F["upar"][0, 0, 1:nsub + 1]
XNL0, XNL2, XNL3 = 1j * kyv * P0[1:nsub + 1], 1j * kyv * P1[1:nsub + 1], -1j * kyv * P2[1:nsub + 1]
nu, pp = P.perpvisc, P.perpvisc_power
check("g", "dBx_sq", Ky["dBx_sq"], 0.5 * kyv**2 * np.abs(psi) ** 2)
check("g", "dBx_transfer", Ky["dBx_transfer"], -(kyv**2 * np.conj(psi) * XNL0).real)
check("g", "dBx_diss", Ky["dBx_diss"], nu * kyv ** (2 * pp + 2) * np.abs(psi) ** 2)
check("g", "upar_sq", Ky["upar_sq"], 0.5 * np.abs(uu) ** 2)
check("g", "upar_transfer", Ky["upar_transfer"], -(np.conj(uu) * (XNL2 + XNL3)).real)
check("g", "upar_inj", Ky["upar_inj"], (np.conj(uu) * kb * kP * (-1j * kyv) * psi).real)
check("g", "upar_diss", Ky["upar_diss"], nu * kyv ** (2 * pp) * np.abs(uu) ** 2)
check("g", "upar_reynolds_transfer", Ky["upar_reynolds_transfer"], -(np.conj(uu) * XNL2).real)
check("g", "upar_maxwell_transfer", Ky["upar_maxwell_transfer"], -(np.conj(uu) * XNL3).real)
results.append(("g", f"dtdBx_sq/dtupar_sq finite: {np.all(np.isfinite(Ky['dtdBx_sq']))} (backward FD; no ref)", 0.0, "INFO"))
check("g", "dBx_profile", Ky["dBx_profile"], -1j * ky1 * F["Psi"][0, 0, :])
check("g", "upar_profile", Ky["upar_profile"], F["upar"][0, 0, :])
tm = np.ones_like(KX)
tm[0, 0, :] = 0.0                                         # turb_mask: no kx = kz = 0 line
T = [padded_real(X * tm) for X in (1j * KX * F["Phi"], -1j * KY * F["Phi"], F["deltaBpar"],
                                   1j * KX * F["Psi"], -1j * KY * F["Psi"], F["upar"])]
check("g", "turb_uperp_sq_profile", Ky["turb_uperp_sq_profile"], line00(T[0] ** 2 + T[1] ** 2))
check("g", "turb_dBpar_sq_profile", Ky["turb_dBpar_sq_profile"], line00(T[2] ** 2))
check("g", "turb_dBperp_sq_profile", Ky["turb_dBperp_sq_profile"], line00(T[3] ** 2 + T[4] ** 2))
check("g", "turb_upar_sq_profile", Ky["turb_upar_sq_profile"], line00(T[5] ** 2))
del T
wm = np.ones((nz, nx, hy))
wm[:, P.half_nx:, 0] = 0
wm[P.half_nz:, 0, 0] = 0
wm[0, 0, 0] = 0
wt = 2 * wm * tm
cw = []
for q in (k2 * a2["Phi"], a2["deltaBpar"], k2 * a2["Psi"], a2["upar"]):
    d0 = (wt * q).sum()
    cw += [(wt * KX**2 * q).sum() / d0, (wt * KY**2 * q).sum() / d0, (wt * k2 * q).sum() / d0]
check("g", "turb_char_wavenumbers", Ky["turb_char_wavenumbers"], np.array(cw))
comps = [(wt * k2 * a2["Phi"]).sum(), (wt * KY**2 * a2["Phi"]).sum(), (wt * KX**2 * a2["Phi"]).sum(),
         (wt * k2 * a2["Psi"]).sum(), (wt * KY**2 * a2["Psi"]).sum(), (wt * KX**2 * a2["Psi"]).sum()]
check("g", "turb_alfvenic_comps", Ky["turb_alfvenic_comps"], np.array(comps))
_, St = rec0("kytavg", "zp_streamer_kzkx")
kxs = np.broadcast_to(kx1, (nz, nx))
Ph, Ps, Bp, up = F["Phi"], F["Psi"], F["deltaBpar"], F["upar"]
pair = lambda a, b: np.sum(np.conj(a[:, :, :-1]) * b[:, :, 1:] - a[:, :, 1:] * np.conj(b[:, :, :-1]), axis=2)
check("g", "dBx_streamer_transfer_kzkx (f32)", St["dBx_streamer_transfer_kzkx"],
      (kxs * np.conj(Ps[0, 0, 1]) * pair(Ps, Ph)).real, tol=1e-6)
check("g", "upar_streamer_transfer_reynolds_kzkx (f32)", St["upar_streamer_transfer_reynolds_kzkx"],
      (kxs * np.conj(up[0, 0, 1]) * pair(up, Ph)).real, tol=1e-6)
check("g", "upar_streamer_transfer_maxwell_kzkx (f32)", St["upar_streamer_transfer_maxwell_kzkx"],
      -(kxs * np.conj(up[0, 0, 1]) * pair(Bp, Ps)).real, tol=1e-6)

# ---------------- (h) shell transfers vs the two plotters ----------------
_, Sh = rec0("shells", "zp_shells")
for st in ("kperp", "kz"):
    for d, names in (("shells_els", ("zplus", "zminus", "zparplus", "zparminus")),
                     ("shells_mk", ("Tuu", "TBu", "TBB", "TuB", "TBparBpar", "TuparBpar", "Tuparupar", "TBparupar"))):
        for nm in names:
            p = os.path.join(REF, d, f"{st}_{nm}_transfer_data.csv")
            if not os.path.exists(p):
                results.append(("h", f"{st} {nm}: {p} missing", np.nan, "FAIL"))
                continue
            ref = np.atleast_2d(np.loadtxt(p, delimiter=",")) * Ly**2
            check("h", f"{st} {nm} [{ref.shape[0]} shells]", Sh[f"{st}_{nm}"], ref, tol=1e-9)
    tot = sum(Sh[f"{st}_{n}"] for n in ("zplus", "zminus", "zparplus", "zparminus"))
    results.append(("h", f"{st}: sum over j,l of Elsasser transfers = {tot.sum():.3e} "
                         f"(vs max |T| {np.abs(Sh[f'{st}_zplus']).max():.3e}; should be ~0)", 0.0, "INFO"))

# ---------------- report ----------------
nf = 0
for diag, name, err, st in results:
    print(f"({diag}) {name:60s} {err:10.2e}  {st}")
    nf += st == "FAIL"
print(f"\n{len(results)} checks, {nf} FAIL")
