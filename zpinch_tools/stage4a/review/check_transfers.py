#!/usr/bin/env python3
"""
Stage 4a review (1): independent checks of the k_y-line transfers (zp_kytransfer, zp_streamer_kzkx) and of the
shell-to-shell matrices (zp_shells), from the run-time outputs only (CPU):
  K1 budget of the k_x = k_z = 0 line against the solver's own time evolution (no d_z terms there):
       dtdBx_sq = dBx_transfer - dBx_diss,   dtupar_sq = upar_transfer + upar_inj - upar_diss
     (dt* = one-step backward difference, O(dt) error); reynolds + maxwell = upar_transfer (exact).
  K2 half-plane x2 weights incl. the k_y = 0 plane: real-space (x,z)-means of the turbulent parts (Parseval, no
     weights) vs the weighted spectral sums (turb_alfvenic_comps), and turbulent + k_x = k_z = 0 line parts vs the
     energy diagnostic (independent code path, checked against the solver's invariants in Stage 2).
  K3 streamer maps: sum over (k_z, k_x) of the k_y = 1 transfer maps = the k_y = 1 line transfers (same window).
  K4 shells: antisymmetry T[j,l] = -T[l,j] of every self-transfer (Tuu TBB TBparBpar Tuparupar zplus zminus
     zparplus zparminus), exchange pairs TuB = -TBu^T and TBparupar = -TuparBpar^T, conservation (sum over all j,l
     of each conserving combination = 0), and the per-receiving-shell budget of the Elsasser set against the
     kinetic/magnetic set (different fields, different products): sum_j (z+ + z-)/2 = sum_j (Tuu+TBu+TBB+TuB) and
     sum_j (Z+ + Z-)/2 = sum_j (TBparBpar+TBparupar+Tuparupar+TuparBpar).
Usage: check_transfers.py RUN_DIR [RUN_DIR ...]   (prints one line per check, PASS/FAIL, tolerance in brackets)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.expanduser("~/flucs_work/flucs/src"))
from flucs.systems.zpinch_fourier.zpinch_readers import load_diag  # noqa: E402

nfail = 0


def rep(name, err, tol):
    global nfail
    ok = err < tol
    nfail += not ok
    print(f"  {name:58s} {err:9.2e} [{tol:.0e}] {'PASS' if ok else 'FAIL'}")


def rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    s = np.max(np.abs(b))
    return np.max(np.abs(a - b)) / s if s > 0 else np.max(np.abs(a - b))


def common(t1, t2):
    """indices of equal record times (same step -> bitwise equal times)."""
    i1, i2 = [], []
    for i, t in enumerate(t1):
        j = np.flatnonzero(t2 == t)
        if j.size:
            i1.append(i)
            i2.append(j[0])
    return np.array(i1, int), np.array(i2, int)


def run_checks(run):
    print(f"== {run}")
    p = lambda o: os.path.join(run, f"output.{o}.nc")  # noqa: E731
    if os.path.exists(p("kyt")):
        tk, K = load_diag(p("kyt"), "zp_kytransfer")
        print(f"  kyt records {len(tk)} (t {tk[0]:.4f} .. {tk[-1]:.4f})")
        # K1
        r1 = K["dtdBx_sq"] - (K["dBx_transfer"] - K["dBx_diss"])
        r2 = K["dtupar_sq"] - (K["upar_transfer"] + K["upar_inj"] - K["upar_diss"])
        s1 = np.max(np.abs(K["dBx_transfer"]), axis=1, keepdims=True)
        s2 = np.max(np.abs(K["upar_transfer"]) + np.abs(K["upar_inj"]), axis=1, keepdims=True)
        # the one-step backward difference is O(dt): a meaningful test only at production time steps
        from netCDF4 import Dataset
        with Dataset(p("0d")) as ds:
            dtm = float(np.median(np.concatenate([np.asarray(ds.groups[g]["dt"][:]) for g in ds.groups])))
        if dtm < 2e-3:
            rep(f"K1 dBx line budget max|resid|/max|T| (O(dt), dt {dtm:.1e})", np.nanmax(np.abs(r1) / s1), 2e-2)
            rep("K1 upar line budget max|resid|/max(|T|+|I|) (O(dt))", np.nanmax(np.abs(r2) / s2), 2e-2)
        else:
            print(f"  K1 INFO (dt {dtm:.1e} >= 2e-3: O(dt) residuals not tested): dBx "
                  f"{np.nanmax(np.abs(r1) / s1):.2e}, upar {np.nanmax(np.abs(r2) / s2):.2e}")
        # same budget with the transfer sign flipped must FAIL (sensitivity of K1)
        r1f = K["dtdBx_sq"] - (-K["dBx_transfer"] - K["dBx_diss"])
        print(f"  K1 (sensitivity: transfer sign flipped -> {np.nanmax(np.abs(r1f) / s1):.2e}, must be O(1))")
        rep("K1 reynolds + maxwell = upar_transfer",
            rel(K["upar_reynolds_transfer"] + K["upar_maxwell_transfer"], K["upar_transfer"]), 1e-12)
        # K2
        _, En = load_diag(p("0d"), "energy")
        te, _ = load_diag(p("0d"), "energy", names=["E"])
        tp, Pr = load_diag(p("1d"), "zp_profiles")
        ik, ie = common(tk, te)
        ik2, ip = common(tk, tp)
        assert len(ik) and np.array_equal(ik, ik2), "kyt / 0d / 1d records not on common steps"
        comps = K["turb_alfvenic_comps"][ik]
        rep("K2 <|u_perp,turb|^2> (real space) = sum w k^2|Phi|^2",
            rel(K["turb_uperp_sq_profile"][ik, 0].real, comps[:, 0]), 1e-10)
        rep("K2 <|dB_perp,turb|^2> (real space) = sum w k^2|Psi|^2",
            rel(K["turb_dBperp_sq_profile"][ik, 0].real, comps[:, 3]), 1e-10)
        rep("K2 comps: k^2 = kx^2 + ky^2 (u, b)",
            max(rel(comps[:, 1] + comps[:, 2], comps[:, 0]), rel(comps[:, 4] + comps[:, 5], comps[:, 3])), 1e-12)
        ky = Pr["ky_line"]
        wl = np.where(ky > 0, 2.0, 1.0)
        LY = Pr["line_y"][ip]                       # (n, 4, nky): Phi Psi dBpar upar at k_x = k_z = 0
        line = lambda f, k2: (wl * k2 * np.abs(LY[:, f]) ** 2).sum(axis=1)  # noqa: E731
        c = 1 + 1 / float(np.asarray(load_kb(run)))
        rep("K2 E_uperp(energy) = (turb + line)/2",
            rel(0.5 * (comps[:, 0] + line(0, ky**2)), En["E_uperp"][ie]), 1e-10)
        rep("K2 E_bperp(energy) = (turb + line)/2",
            rel(0.5 * (comps[:, 3] + line(1, ky**2)), En["E_bperp"][ie]), 1e-10)
        rep("K2 E_dbpar(energy) = c (turb + line)/2",
            rel(0.5 * c * (K["turb_dBpar_sq_profile"][ik, 0].real + line(2, 1)), En["E_dbpar"][ie]), 1e-10)
        rep("K2 E_upar(energy) = (turb + line)/2",
            rel(0.5 * (K["turb_upar_sq_profile"][ik, 0].real + line(3, 1)), En["E_upar"][ie]), 1e-10)
        rep("K2 line_y vs kyt upar_profile / dBx_profile (two diags, same modes)",
            max(rel(LY[:, 3], K["upar_profile"][ik]), rel(-1j * ky * LY[:, 1], K["dBx_profile"][ik])), 1e-12)
        # K2: Hermitian k_y = 0 plane on the k_x line (k_y = k_z = 0)
        kx = Pr["kx_line"]
        LX = Pr["line_x"]
        herm = np.max(np.abs(LX - np.conj(LX[..., ::-1]))) / np.max(np.abs(LX))
        assert np.allclose(kx, -kx[::-1])
        rep("K2 Hermitian k_x line: |X(-k) - X(k)*| / max|X|", herm, 1e-10)
        # K3
        if os.path.exists(p("kytavg")):
            ta, A = load_diag(p("kytavg"), "zp_streamer_kzkx")
            for i, t in enumerate(ta):
                n = int(A["n_samples"][i])
                if n == 1:
                    j = np.flatnonzero(tk == t)
                    if j.size:
                        j = j[0]
                        s = lambda nm: float(np.sum(A[nm][i], dtype=np.float64))  # noqa: E731
                        rep(f"K3 t={t:.3f} sum dBx_kzkx = dBx_transfer(ky=1) (f4 map)",
                            abs(s("dBx_streamer_transfer_kzkx") - K["dBx_transfer"][j, 0])
                            / abs(K["dBx_transfer"][j, 0]), 1e-5)
                        rep(f"K3 t={t:.3f} sum reynolds_kzkx = upar_reynolds(ky=1)",
                            abs(s("upar_streamer_transfer_reynolds_kzkx") - K["upar_reynolds_transfer"][j, 0])
                            / abs(K["upar_reynolds_transfer"][j, 0]), 1e-5)
                        rep(f"K3 t={t:.3f} sum maxwell_kzkx = upar_maxwell(ky=1)",
                            abs(s("upar_streamer_transfer_maxwell_kzkx") - K["upar_maxwell_transfer"][j, 0])
                            / abs(K["upar_maxwell_transfer"][j, 0]), 1e-5)
                else:
                    # window of n samples at sample_time spacing: compare with the kyt records in the window
                    print(f"  K3 t={t:.3f}: window mean of {n} samples (identity checked on single-sample records)")
    if os.path.exists(p("shells")):
        ts, S = load_diag(p("shells"), "zp_shells")
        print(f"  shells records {len(ts)}")
        for st in ("kperp", "kz"):
            T = lambda nm: S[f"{st}_{nm}"]  # noqa: E731
            sc = max(np.max(np.abs(T(n))) for n in ("Tuu", "TBB", "zplus", "zminus"))
            anti = max(np.max(np.abs(T(n) + np.swapaxes(T(n), 1, 2))) / np.max(np.abs(T(n)))
                       for n in ("Tuu", "TBB", "TBparBpar", "Tuparupar", "zplus", "zminus", "zparplus",
                                 "zparminus"))
            rep(f"K4 {st}: self-transfers antisymmetric (max over 8)", anti, 1e-10)
            rep(f"K4 {st}: TuB = -TBu^T", rel(T("TuB"), -np.swapaxes(T("TBu"), 1, 2)), 1e-10)
            rep(f"K4 {st}: TBparupar = -TuparBpar^T", rel(T("TBparupar"), -np.swapaxes(T("TuparBpar"), 1, 2)), 1e-10)
            tot = sum(T(n).sum(axis=(1, 2)) for n in ("Tuu", "TBB", "TuB", "TBu"))
            rep(f"K4 {st}: sum_jl (Tuu+TBB+TuB+TBu) / max|T|", np.max(np.abs(tot)) / sc, 1e-10)
            perp_e = 0.5 * (T("zplus") + T("zminus")).sum(axis=1)
            perp_m = (T("Tuu") + T("TBu") + T("TBB") + T("TuB")).sum(axis=1)
            rep(f"K4 {st}: per-shell perp budget Elsasser vs kin/mag", rel(perp_e, perp_m), 1e-10)
            par_e = 0.5 * (T("zparplus") + T("zparminus")).sum(axis=1)
            par_m = (T("TBparBpar") + T("TBparupar") + T("Tuparupar") + T("TuparBpar")).sum(axis=1)
            rep(f"K4 {st}: per-shell par budget Elsasser vs kin/mag", rel(par_e, par_m), 1e-10)
            nz = np.max(np.abs(perp_m)) / sc
            print(f"  K4 {st}: (net per-shell perp transfer / max|T| = {nz:.2e}: the budget test is not trivial)")


def load_kb(run):
    import tomli as tomllib
    with open(os.path.join(run, "input.toml"), "rb") as fh:
        return tomllib.load(fh)["parameters"]["kb"]


if __name__ == "__main__":
    for r in sys.argv[1:]:
        run_checks(r)
    print(f"FAIL count: {nfail}")
    sys.exit(1 if nfail else 0)
