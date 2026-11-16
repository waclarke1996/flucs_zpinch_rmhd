#!/usr/bin/env python3
"""
Stage 4b verification (CPU, login node) of the outputs of verify/V1-V3 (job 13136371). Prints PASS/FAIL lines.
 V1 (A2-T1 turbulent state at t = 0 after reset_time; records t = 0, 0.1, 0.2):
   C1 per bin, every binning: (zplus + zminus)/2 = E_uperp + E_bperp, (Zpplus + Zpminus)/2 = E_dbpar + E_upar
   C2 bin sums = 0d invariants (energy diag zplus, zminus, Zpplus, Zpminus) and E, at the same records
   C3 sums equal over k_perp, k_y, |k_x|, |k_z| (E, the four W, Qhat, Qhat_es, Qhat_em where binned)
   C4 zp_kxky (t = 0 window, n = 1): sum over k_y = the |k_x| spectrum (+-k_x folded), sum over k_x = the k_y spectrum
   C5 zp_kxky at t = 0 = numpy from the seed state (independent per-mode formula, FLUCS layout)
   C6 zp_kxky vs curvy3d's X_spec_calc = sum_kz |wavenumber_mask X|^2 (solver l.406-409 mask, l.2650-2658 formula)
      on the seed converted by flucs_to_curvy3d.py: k_y > 0: ours = k_perp^2 curvy (W_zeta), = curvy (W_Z);
      k_y = 0: ours(k_x) + ours(-k_x) = k_perp^2 curvy(k_x) for k_x > 0, curvy(k_x < 0, 0) = 0
 V2 (tiny, L_x = 8 pi, fresh run; last record t = 1 = final state):
   bins follow the box (dk_x 0.25, first edge 0.125, k_x = 0.25 in bin 1); spectra = numpy from the final state
 V3 (kP = 0, nu = 0, dt = 1e-4, zp_shells every step): dE_l/dt (central difference of the shell energies) =
   sum_j T[j, l] (kin/mag set, and half the Elsasser set), per shell type; total energy constant
"""
import os
import subprocess
import sys

import numpy as np
from netCDF4 import Dataset

sys.path.insert(0, os.path.expanduser("~/flucs_work/flucs/src"))
from flucs.systems.zpinch_fourier.zpinch_readers import load_diag  # noqa: E402
from flucs.systems.zpinch_fourier.zpinch_runtime_diags import kperp_bin_edges, kperp_bin_index  # noqa: E402

H = os.path.dirname(os.path.abspath(__file__))
V1, V2, V3 = (os.path.join(H, "verify", d) for d in ("V1_spec_T1", "V2_lx8pi", "V3_signtest"))
SEED = os.path.join(H, "../stage4a/realtest/A2-T1-k8/t0500/state.nc")
PY = os.path.expanduser("~/flucs_work/.venv/bin/python")
nf = 0


def chk(name, val, tol):
    global nf
    ok = bool(np.isfinite(val) and val <= tol)
    nf += not ok
    print(f"  {name:88s} {val:9.2e} [{tol:.0e}] {'PASS' if ok else 'FAIL'}")


def rel(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300))


def compact_state(path, nx, ny, nz):
    """solved modes of a FLUCS state: dict field -> complex (nz_c, nx_c, hy_c) in FFT order, and (mz, mx, my)"""
    with Dataset(path) as ds:
        X = np.asarray(ds["fields_real"][:]) + 1j * np.asarray(ds["fields_imag"][:])
    Kx, Ky, Kz = (nx - 1) // 3, (ny - 1) // 3, (nz - 1) // 3
    ix = np.r_[0:Kx + 1, nx - Kx:nx]
    iz = np.r_[0:Kz + 1, nz - Kz:nz]
    C = X[:, iz][:, :, ix][:, :, :, :Ky + 1]
    mx = np.r_[0:Kx + 1, -Kx:0]
    mz = np.r_[0:Kz + 1, -Kz:0]
    return C, mz, mx, np.arange(Ky + 1)


def per_mode(C, kx, ky, kb):
    """w-weighted per-mode quantities on (nz, nx, hy) arrays (independent of the diag code)"""
    c = 1 + 1 / kb
    s = np.sqrt(c)
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    k2 = (KX**2 + KY**2)[None]
    w = np.where(KY > 0, 2.0, 1.0)[None]
    P, S, B, U = C
    a = lambda z: np.abs(z) ** 2  # noqa: E731
    q = (-(1 + kb) * (B * np.conj(-1j * KY[None] * P)).real + kb * (U * np.conj(-1j * KY[None] * S)).real) * w
    out = dict(E_uperp=0.5 * w * k2 * a(P), E_bperp=0.5 * w * k2 * a(S), E_dbpar=0.5 * w * c * a(B),
               E_upar=0.5 * w * a(U), zplus=0.5 * w * k2 * a(P + S), zminus=0.5 * w * k2 * a(P - S),
               Zpplus=0.5 * w * a(U + s * B), Zpminus=0.5 * w * a(U - s * B), Qhat=q)
    out["E"] = out["E_uperp"] + out["E_bperp"] + out["E_dbpar"] + out["E_upar"]
    return out


def v1():
    print("== V1 spectra / maps on the A2-T1 turbulent state")
    t1, S = load_diag(f"{V1}/output.1d.nc", "zp_spectra")
    t0, Z = load_diag(f"{V1}/output.0d.nc", "energy", ["E", "zplus", "zminus", "Zpplus", "Zpminus"])
    print(f"  records 1d {t1}, 0d {t0[:3]}...")
    for b in ("kperp", "ky", "kx", "kz"):
        if f"E_uperp_{b}" in S:
            chk(f"C1 {b:5s}: (zplus+zminus)/2 = E_uperp+E_bperp per bin",
                rel((S[f"zplus_{b}"] + S[f"zminus_{b}"]) / 2, S[f"E_uperp_{b}"] + S[f"E_bperp_{b}"]), 1e-13)
            chk(f"C1 {b:5s}: (Zpplus+Zpminus)/2 = E_dbpar+E_upar per bin",
                rel((S[f"Zpplus_{b}"] + S[f"Zpminus_{b}"]) / 2, S[f"E_dbpar_{b}"] + S[f"E_upar_{b}"]), 1e-13)
        else:
            chk(f"C1 {b:5s}: (zplus+zminus+Zpplus+Zpminus)/2 = E per bin",
                rel((S[f"zplus_{b}"] + S[f"zminus_{b}"] + S[f"Zpplus_{b}"] + S[f"Zpminus_{b}"]) / 2, S[f"E_{b}"]), 1e-13)
    i0 = [int(np.argmin(np.abs(t0 - x))) for x in t1]
    for n in ("E", "zplus", "zminus", "Zpplus", "Zpminus"):
        chk(f"C2 sum_kperp {n} = 0d {n} (3 records)", rel(S[f"{n}_kperp"].sum(1), Z[n][i0]), 1e-12)
    for n in ("E", "zplus", "zminus", "Zpplus", "Zpminus", "Qhat", "Qhat_es", "Qhat_em"):
        ref = S[f"{n}_kperp"].sum(1)
        for b in ("ky", "kx", "kz"):
            if f"{n}_{b}" in S:
                chk(f"C3 sum_{b} {n} = sum_kperp {n}", rel(S[f"{n}_{b}"].sum(1), ref), 1e-12)
    tm, M = load_diag(f"{V1}/output.maps.nc", "zp_kxky")
    i = int(np.argmin(np.abs(tm)))
    kxs, kys = np.asarray(M["kx"]), np.asarray(M["ky"])
    print(f"  zp_kxky records {tm}, n_samples {M['n_samples']}, kx {kxs[0]:g}..{kxs[-1]:g} (d {kxs[1]-kxs[0]:g}), "
          f"ky 0..{kys[-1]:g}")
    j1 = int(np.argmin(np.abs(t1)))
    Kx = (len(kxs) - 1) // 2
    for n in ("E", "zplus", "zminus", "Zpplus", "Zpminus", "Qhat"):
        m = M[f"{n}_kxky"][i].astype(float)
        fold = np.concatenate(([m[Kx].sum()], (m[Kx + 1:] + m[Kx - 1::-1]).sum(1)))
        chk(f"C4 zp_kxky {n}: sum_ky (+-kx folded) = {n}_kx (t = 0)", rel(fold, S[f"{n}_kx"][j1]), 3e-6)
        chk(f"C4 zp_kxky {n}: sum_kx = {n}_ky (t = 0)", rel(m.sum(0), S[f"{n}_ky"][j1]), 3e-6)
    # C5 independent numpy from the seed
    nx, ny, nz, kb = 256, 128, 256, 0.28284271
    C, mz, mx, my = compact_state(SEED, nx, ny, nz)
    pm = per_mode(C, mx * 0.5, my * 1.0, kb)
    for n in ("E", "zplus", "zminus", "Zpplus", "Zpminus", "Qhat"):
        mine = pm[n].sum(0)                                            # (nx_c FFT order, hy)
        mine = np.roll(mine, Kx, axis=0)                               # -> kx = -Kx..Kx
        chk(f"C5 zp_kxky {n} (t = 0, float32) = numpy from the seed state", rel(M[f"{n}_kxky"][i], mine), 3e-6)
    # C6 curvy3d spec_calc on the converted seed
    cv = os.path.join(H, "verify", "seed_curvy.nc")
    if not os.path.exists(cv):
        subprocess.run([PY, os.path.expanduser("~/flucs_work/runs/tools/flucs_to_curvy3d.py"), SEED, "--out", cv,
                        "--run-id", "A2-T1-seed", "--save-time", "100000"], check=True, capture_output=True)
    with Dataset(cv) as ds:
        F = {}
        for f in ("Phi", "deltaBpar", "Psi", "upar"):
            a = np.asarray(ds[f + "_rfft_r"][:]) + 1j * np.asarray(ds[f + "_rfft_i"][:])
            it = int(np.argmax([np.abs(a[k]).sum() for k in range(a.shape[0])]))
            F[f] = a[it]
    nzc, nxc, hy = F["Phi"].shape
    mask = np.ones((nzc, nxc, hy))
    half_nx, half_nz = nxc // 2 + 1, nzc // 2 + 1                     # curvy3dinput.py l.311-313
    mask[:, half_nx:, 0] = 0
    mask[half_nz:, 0, 0] = 0
    mask[0, 0, 0] = 0
    s = np.sqrt(1 + 1 / kb)
    zeta_p, zeta_m = mask * F["Phi"] + mask * F["Psi"], mask * F["Phi"] - mask * F["Psi"]
    zpar_p, zpar_m = mask * F["upar"] + s * mask * F["deltaBpar"], mask * F["upar"] - s * mask * F["deltaBpar"]
    spec = {"zplus": np.sum(np.abs(zeta_p) ** 2, 0), "zminus": np.sum(np.abs(zeta_m) ** 2, 0),
            "Zpplus": np.sum(np.abs(zpar_p) ** 2, 0), "Zpminus": np.sum(np.abs(zpar_m) ** 2, 0)}
    mxc = np.where(np.arange(nxc) < half_nx, np.arange(nxc), np.arange(nxc) - nxc)
    KX, KY = np.meshgrid(mxc * 0.5, np.arange(hy) * 1.0, indexing="ij")
    K2 = KX**2 + KY**2
    for n in ("zplus", "zminus", "Zpplus", "Zpminus"):
        ours = M[f"{n}_kxky"][i].astype(float)                         # kx = -Kx..Kx
        ours_fft = np.roll(ours, -Kx, axis=0)                          # -> FFT order as curvy
        fac = K2 if n in ("zplus", "zminus") else np.ones_like(K2)
        cur = fac * spec[n]
        chk(f"C6 {n}_kxky = (k_perp^2) {n}_spec_calc at k_y > 0", rel(ours_fft[:, 1:], cur[:, 1:]), 3e-6)
        pos = np.arange(1, half_nx)
        pair = ours_fft[pos, 0] + ours_fft[nxc - pos, 0]
        chk(f"C6 {n}: ours(kx,0) + ours(-kx,0) = (k_perp^2) curvy(kx,0), kx > 0", rel(pair, cur[pos, 0]), 3e-6)
        chk(f"C6 {n}: curvy(kx < 0, k_y = 0) = 0 (its mask)", float(np.abs(spec[n][half_nx:, 0]).max()), 0.0)


def v2():
    print("== V2 tiny box L_x = 8 pi")
    t1, S = load_diag(f"{V2}/output.1d.nc", "zp_spectra")
    e = np.asarray(S["kperp_bin_edges"])
    print(f"  records {t1}; kperp edges {e[:4]} ...; kx centres {np.asarray(S['kx'])[:3]}")
    chk("bins follow the box: kx spacing = 2 pi / 8 pi = 0.25", abs(S["kx"][1] - S["kx"][0] - 0.25), 1e-14)
    chk("first edge = 0.5 min(dkx, dky) = 0.125, then 1.50001", rel(e[1:3], [0.125, 1.50001]), 1e-14)
    up = kperp_bin_edges(0.25, 1.0, 100.0)[:len(e) - 1]
    chk("stored edges = kperp_bin_edges(0.25, 1, .)", rel(e[1:], up), 1e-14)
    chk("(a) mode k_x = 0.25, k_y = 0 -> bin 1", abs(int(kperp_bin_index(np.array([0.25]), e[1:])[0]) - 1), 0)
    with Dataset(f"{V2}/restart.nc") as ds:
        ts = float(ds["current_time"][...])
    j = int(np.argmax(t1))
    chk(f"final state time {ts:.6f} = last record time {t1[j]:.6f}", abs(ts - t1[j]), 1e-12)
    C, mz, mx, my = compact_state(f"{V2}/restart.nc", 48, 24, 48)
    pm = per_mode(C, mx * 0.25, my * 1.0, 0.28284271)
    KX, KY = np.meshgrid(mx * 0.25, my * 1.0, indexing="ij")
    kp = np.broadcast_to(np.sqrt(KX**2 + KY**2)[None], C[0].shape)
    ib = kperp_bin_index(kp, e[1:])
    for n in ("E", "zplus", "Zpminus", "Qhat"):
        mine = np.bincount(ib.ravel(), pm[n].ravel(), len(e) - 1)
        chk(f"{n}_kperp (last record) = numpy with the Paper 1 bins from the final state", rel(S[f"{n}_kperp"][j], mine), 1e-12)
        ikx = np.broadcast_to(np.abs(mx)[None, :, None], C[0].shape)
        chk(f"{n}_kx = numpy |k_x| bins", rel(S[f"{n}_kx"][j], np.bincount(ikx.ravel(), pm[n].ravel())), 1e-12)


def v3():
    print("== V3 dynamical sign test of zp_shells (kP = 0, nu = 0, dt = 1e-4, every step)")
    t, Sh = load_diag(f"{V3}/output.shells.nc", "zp_shells")
    t0, Z = load_diag(f"{V3}/output.0d.nc", "energy", ["E"])
    print(f"  {len(t)} records, t {t[0]:.1e}..{t[-1]:.1e}, dt {np.unique(np.round(np.diff(t), 12))}; "
          f"total E rel. change {abs(Z['E'][-1] - Z['E'][0]) / Z['E'][0]:.1e}")
    mk = ("Tuu", "TBu", "TBB", "TuB", "TBparBpar", "TuparBpar", "Tuparupar", "TBparupar")
    for st in ("kperp", "kz"):
        E = sum(Sh[f"{st}_E_{p}"] for p in ("uperp", "bperp", "dbpar", "upar"))          # (nt, n)
        dEdt = (E[2:] - E[:-2]) / (t[2:] - t[:-2])[:, None]
        Tm = sum(Sh[f"{st}_{n}"] for n in mk).sum(axis=1)[1:-1]                          # sum over j -> (nt-2, n)
        Te = 0.5 * sum(Sh[f"{st}_{n}"] for n in ("zplus", "zminus", "zparplus", "zparminus")).sum(axis=1)[1:-1]
        act = np.abs(Tm) > 1e-3 * np.abs(Tm).max()
        sgn = float(np.mean(np.sign(dEdt[act]) == np.sign(Tm[act])))
        print(f"  {st}: shell energies {E[0].round(3)}")
        print(f"  {st}: sum_j T[j,l] (step 1) {Tm[0].round(4)}")
        print(f"  {st}: dE_l/dt          (step 1) {dEdt[0].round(4)}")
        chk(f"{st}: dE_l/dt = sum_j T_magkin[j, l] (all shells, {len(Tm)} steps), rel. to max |T|", rel(dEdt, Tm), 1e-3)
        chk(f"{st}: dE_l/dt = 1/2 sum_j T_elsasser[j, l]", rel(dEdt, Te), 1e-3)
        chk(f"{st}: sign agreement on active shells (1 - fraction)", 1 - sgn, 0.0)


for f in (v1, v2, v3):
    try:
        f()
    except Exception as ex:  # noqa: BLE001
        nf += 1
        print(f"  {f.__name__}: ERROR {type(ex).__name__}: {ex}")
print("FAIL count", nf)
sys.exit(1 if nf else 0)
