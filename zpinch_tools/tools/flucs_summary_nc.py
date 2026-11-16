#!/usr/bin/env python3
"""
flucs_summary_nc.py -- per-run summary file in curvy3d format for a FLUCS segment-chain run (Stage 4a), so that
Will's build_registry.py and make_flux_table.py (and zpinch_runtools.heat_flux_series) read FLUCS runs unchanged.

    python flucs_summary_nc.py RUN [--out RUN/<run_id>.nc] [--run-id ID] [--f8] [--no-snapshot]

RUN = folder with segments t????/ (input.toml, output.0d.nc, run.log, restart.nc of the last segment), or one run
folder. Writes:
  input       curvy3d input string (template A3a-Y input + overrides: kappaP, kappaBeta, padded grid, box,
              hyperviscosity, dt, maxcfl, save_time = end time, save_time_hf, tfinal = end time, extra_comment = run id;
              with the zp_kytransfer output also subcrit_mode_balances 1, n_subcrit_modes, subcrit_mode_hfmult)
  heatflux, heatflux_em, dt_series [nt_hf]   curvy3d convention (half-space sums, solver prefactors), interpolated
              from the FLUCS 0d series onto t = 0, save_time_hf, ... (t = chain time; heat_flux_series() returns
              Q_es, Q_em = Qhat_es/Ly^2, Qhat_em/Ly^2 = the FLUCS box averages)
  <field>_rfft_r/_i [nt = 2, nz, nx, half_ny]   index 1 = final state (restart.nc of the last segment), index 0 empty
              (zeros = never written), so t_last_snapshot = end time. float32 unless --f8.
  log         written only if the chain finished (all segments exit 0, or an early stop) -> is_complete() True
  ky-transfer variables of curvy3d (dBx_sq ... [n_subcrit_modes, nt_subcrit], profiles and streamer maps = run
              averages; here the true average. NB curvy3d adds dBx_profile twice in the initial save block of
              solve() (solver l.2580/2583), once per later save: its value = mean + dBx(t_start)/N_saves) if present.
  attributes  run_id, source, git_commit (FLUCS hash of the run, from the input or run.log if found), gpu_name.
"""
import argparse
import contextlib
import glob
import io
import os
import re
import sys
import warnings

import numpy as np
from netCDF4 import Dataset

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.expanduser("~/flucs_work/reference/curvy3d")
DATA = "/mnt/users/clarkew/data/rmhd_zpinch/output/3D/BP2_anchors_v2"
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.expanduser("~/flucs_work/flucs/src"))
from flucs_to_curvy3d import parse_toml, read_state, to_compact  # noqa: E402
from flucs.systems.zpinch_fourier.zpinch_readers import duplicate_mask, load_diag, timed_period  # noqa: E402


def segments(run):
    segs = sorted(glob.glob(os.path.join(run, "t[0-9][0-9][0-9][0-9]")))
    return segs if segs else [run]


def series_0d(segs):
    """chain time series of Qhat_es, Qhat_em, dt (later segment wins on overlaps)."""
    t, es, em, dt = [], [], [], []
    for s in segs:
        p = os.path.join(s, "output.0d.nc")
        if not os.path.exists(p):
            continue
        with Dataset(p) as ds:
            for g in sorted(ds.groups, key=int):
                G = ds.groups[g]
                tt = np.asarray(G["time"][:], float)
                if not len(tt):
                    continue
                for i in range(len(t)):
                    k = t[i] < tt[0]
                    t[i], es[i], em[i], dt[i] = t[i][k], es[i][k], em[i][k], dt[i][k]
                t.append(tt)
                es.append(np.asarray(G["energy"]["Qhat_es"][:], float))
                em.append(np.asarray(G["energy"]["Qhat_em"][:], float))
                dt.append(np.asarray(G["dt"][:], float))
    t, es, em, dt = (np.concatenate(a) for a in (t, es, em, dt))
    o = np.argsort(t, kind="stable")
    t, es, em, dt = t[o], es[o], em[o], dt[o]
    keep = np.concatenate([np.diff(t) > 0, [True]])
    t, es, em, dt = t[keep], es[keep], em[keep], dt[keep]
    p0 = [os.path.join(s, "output.0d.nc") for s in segs if os.path.exists(os.path.join(s, "output.0d.nc"))]
    keep = duplicate_mask(t, timed_period(p0[0]) if p0 else None)    # duplicates of pre-2026-11-16 outputs
    return t[keep], es[keep], em[keep], dt[keep]


def state_file(seg):
    """validated final state of a Stage 4 chain segment (state.nc), else a Stage 3e restart.nc."""
    for nm in ("state.nc", "restart.nc"):
        if os.path.exists(os.path.join(seg, nm)):
            return os.path.join(seg, nm)
    return None


def finished(segs, run):
    if os.path.exists(os.path.join(run, "EARLY_STOP")):
        return True, "early stop: " + open(os.path.join(run, "EARLY_STOP")).read().strip()
    if os.path.exists(os.path.join(segs[-1], "SEG_OK")):
        return True, "all segments finished (SEG_OK)"
    last = os.path.join(segs[-1], "run.log")
    if os.path.exists(last) and "flucs given in" in open(last, errors="ignore").read():
        return True, "all segments finished"
    return False, "not finished"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--out", default=None)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--save-time-hf", type=float, default=0.1)
    ap.add_argument("--f8", action="store_true", help="final snapshot in double precision (default float32)")
    ap.add_argument("--no-snapshot", action="store_true")
    ap.add_argument("--template", default=None)
    a = ap.parse_args()
    run = os.path.abspath(a.run)
    rid = a.run_id or os.path.basename(run.rstrip("/"))
    out = a.out or os.path.join(run, f"{rid}_summary.nc")
    segs = segments(run)
    inp = parse_toml(open(os.path.join(segs[-1], "input.toml")).read())
    d, p, h, tm = inp["dimensions"], inp["parameters"], inp["hyperdissipation"], inp.get("time", {})

    t, es, em, dts = series_0d(segs)
    hf_dt = a.save_time_hf
    t_end = float(t[-1])
    thf = np.arange(0.0, t_end + 0.5 * hf_dt, hf_dt)
    inr = (thf >= t[0] - 0.5 * hf_dt) & (thf <= t_end + 0.5 * hf_dt)
    kP, kb = float(p["kP"]), float(p["kb"])
    pref = kP * (1 + kb) - 2.0
    # no interpolation across gaps of the 0d series (a missing segment): grid points farther than 1.5 hf_dt from
    # the nearest record are NaN
    jn = np.clip(np.searchsorted(t, thf), 1, len(t) - 1)
    gap = np.minimum(np.abs(t[jn] - thf), np.abs(t[jn - 1] - thf))
    inr &= gap <= 1.5 * hf_dt
    hf = np.where(inr, np.interp(thf, t, es) * pref / (2.0 * (1 + kb)), np.nan)
    hfe = np.where(inr, np.interp(thf, t, em) * kP / 2.0, np.nan)
    j = np.clip(np.searchsorted(t, thf), 0, len(t) - 1)
    dt_series = np.where(inr, dts[j], np.nan)

    snap = None
    if not a.no_snapshot:
        rp = state_file(segs[-1])
        if rp is None:
            raise SystemExit(f"no state.nc / restart.nc in {segs[-1]}: use --no-snapshot")
        X, t_state, _ = read_state(rp)
        snap, left = to_compact(X)
        if left != 0.0:
            raise SystemExit("non-zero modes outside the 2/3 set")
        if abs(t_state - t_end) > 0.5:
            print(f"warning: final state t = {t_state:.3f} vs end of series {t_end:.3f}")
        t_end_state = t_state
    else:
        t_end_state = t_end

    # ky-transfer output (optional)
    kyt_files = [os.path.join(s, "output.kyt.nc") for s in segs if os.path.exists(os.path.join(s, "output.kyt.nc"))]
    kyt = None
    if kyt_files:
        tk, kyt = load_diag(kyt_files, "zp_kytransfer")
        kyt_dt = float(f"{np.median(np.diff(tk)):.3g}") if len(tk) > 1 else 1.0    # nominal save_time
        # curvy3d's subcrit series has no time variable: record i is at t = i * subcrit_mode_hfmult * save_time_hf
        # from t = 0 (plot3d_rmhd_zpinch.py). Put the records on that grid; NaN where there is no record (e.g. the
        # segments before the ky-transfer output was switched on).
        ngrid = int(np.floor(tk[-1] / kyt_dt + 0.5)) + 1
        gi = np.rint(tk / kyt_dt).astype(int)
        if np.max(np.abs(tk - gi * kyt_dt)) > 0.1 * kyt_dt or len(np.unique(gi)) != len(gi):
            raise SystemExit("zp_kytransfer records are not on a regular time grid")
        for k in list(kyt):
            v = kyt[k]
            if np.ndim(v) >= 1 and k not in ("ky_mode", "ky_profile", "turb_wavenumber_measurements",
                                             "turb_alfvenic_comps_measurements"):
                g = np.full((ngrid,) + v.shape[1:], np.nan, dtype=v.dtype)
                g[gi] = v
                kyt[k] = g
        print(f"ky-transfer: {len(tk)} records on a grid of {ngrid} (t = i x {kyt_dt:g}, first record t = {tk[0]:.3f})")
        st_files = [os.path.join(s, "output.kytavg.nc") for s in segs
                    if os.path.exists(os.path.join(s, "output.kytavg.nc"))]
        streamer = load_diag(st_files, "zp_streamer_kzkx")[1] if st_files else None

    template = a.template or glob.glob(os.path.join(DATA, "*_RKARS343_A3a-Y.nc"))[0]
    with Dataset(template, "r") as tds:
        base = tds["input"][:].data.tobytes().decode()
    over = {
        "kappaP": kP, "kappaBeta": kb,
        "padded_nx": int(d["nx"]), "padded_ny": int(d["ny"]), "padded_nz": int(d["nz"]),
        "perpvisc": float(h["kperp"]), "perpvisc_power": int(h["kperp_power"]),
        "parvisc": float(h["kz"]), "parvisc_power": int(h["kz_power"]),
        "Ly": float(d.get("Ly", 2 * np.pi)), "Lz": float(d.get("Lz", 2 * np.pi)),
        "Lx_mult": float(d.get("Lx", 4 * np.pi)) / float(d.get("Ly", 2 * np.pi)),
        "dt": float(tm.get("dt_max", 0.01)), "maxcfl": float(tm.get("max_cfl", 0.5)),
        "save_time": float(t_end_state), "save_time_hf": hf_dt, "tfinal": float(t_end_state),
        "restart": 0, "extra_comment": rid, "comment": "FLUCS",
    }
    # the template's input string has explicit nx/ny/nz lines (A3a-Y grid): override with this run's mode counts
    Kx, Ky, Kz = (int(d["nx"]) - 1) // 3, (int(d["ny"]) - 1) // 3, (int(d["nz"]) - 1) // 3
    over.update({"nx": 2 * Kx + 1, "ny": 2 * Ky + 1, "nz": 2 * Kz + 1})
    if kyt is not None:
        over.update({"subcrit_mode_balances": 1, "n_subcrit_modes": int(kyt["dBx_sq"].shape[1]),
                     "subcrit_mode_hfmult": int(round(kyt_dt / hf_dt))})
    input_str = base.rstrip("\n") + "\n//----- flucs_summary_nc.py overrides -----//\n" + \
        "".join(f"{k} {v!r}\n" if isinstance(v, float) else f"{k} {v}\n" for k, v in over.items())
    sys.path.insert(0, REF)
    from curvy3dinput import Curvy3DInput
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        P = Curvy3DInput(input_str=input_str)

    done, why = finished(segs, run)
    # FLUCS commit of the run: RUN/flucs_git.txt (written by the Stage 4 chain script); the version string in run.log
    # is the pip install-time hash, not the branch state, so it is not used.
    git = ""
    gp = os.path.join(run, "flucs_git.txt")
    if os.path.exists(gp):
        m = re.search(r"[0-9a-f]{7,40}", open(gp).read())
        git = m.group(0) if m else ""

    tmp = out + ".tmp"
    with Dataset(tmp, "w") as ds:
        ds.setncattr("run_id", rid)
        ds.setncattr("source", f"FLUCS ZPinchFourier chain {run} ({len(segs)} segments), flucs_summary_nc.py")
        if git:
            ds.setncattr("git_commit", git)
        b = np.frombuffer(input_str.encode(), dtype="S1")
        ds.createDimension("input_dim", len(b))
        ds.createVariable("input", "S1", ("input_dim",))[:] = b
        ds.createDimension("nt_hf", len(thf))
        ds.createVariable("heatflux", "f8", ("nt_hf",))[:] = hf
        ds.createVariable("heatflux_em", "f8", ("nt_hf",))[:] = hfe
        ds.createVariable("dt_series", "f8", ("nt_hf",))[:] = dt_series
        if snap is not None:
            nz, nx, hy = snap["Phi"].shape
            ds.createDimension("nt", 2)
            ds.createDimension("nz", nz)
            ds.createDimension("nx", nx)
            ds.createDimension("half_ny", hy)
            ds.createDimension("ny", 2 * hy - 1)
            prec = "f8" if a.f8 else "f4"
            for n in ("Phi", "deltaBpar", "Psi", "upar"):
                for part, fn in (("_rfft_r", np.real), ("_rfft_i", np.imag)):
                    v = ds.createVariable(n + part, prec, ("nt", "nz", "nx", "half_ny"), zlib=True, complevel=1)
                    v[0] = np.zeros((nz, nx, hy))
                    v[1] = fn(snap[n])
        if kyt is not None:
            n_sub = kyt["dBx_sq"].shape[1]
            ds.createDimension("n_subcrit_modes", n_sub)
            ds.createDimension("nt_subcrit", kyt["dBx_sq"].shape[0])
            if "half_ny" not in ds.dimensions:
                ds.createDimension("half_ny", kyt["dBx_profile"].shape[1])
            for nm in ("dBx_sq", "dtdBx_sq", "dBx_transfer", "dBx_diss", "upar_sq", "dtupar_sq", "upar_transfer",
                       "upar_inj", "upar_diss", "upar_reynolds_transfer", "upar_maxwell_transfer"):
                ds.createVariable(nm, "f8", ("n_subcrit_modes", "nt_subcrit"))[:] = kyt[nm].T
            for nm in ("dBx_profile", "upar_profile", "turb_uperp_sq_profile", "turb_dBpar_sq_profile",
                       "turb_dBperp_sq_profile", "turb_upar_sq_profile"):
                m = np.nanmean(kyt[nm], axis=0)
                ds.createVariable(nm + "_r", "f8", ("half_ny",))[:] = m.real
                ds.createVariable(nm + "_i", "f8", ("half_ny",))[:] = m.imag
            ds.createDimension("turb_wavenumber_measurements", 12)
            ds.createDimension("turb_alfvenic_comps_measurements", 6)
            ds.createVariable("turb_char_wavenumbers", "f8", ("turb_wavenumber_measurements", "nt_subcrit"))[:] = \
                kyt["turb_char_wavenumbers"].T
            ds.createVariable("turb_alfvenic_comps", "f8", ("turb_alfvenic_comps_measurements", "nt_subcrit"))[:] = \
                kyt["turb_alfvenic_comps"].T
            if streamer is not None and "nz" in ds.dimensions:
                w = streamer["n_samples"]
                for nm in ("dBx_streamer_transfer_kzkx", "upar_streamer_transfer_reynolds_kzkx",
                           "upar_streamer_transfer_maxwell_kzkx"):
                    avg = np.tensordot(w, streamer[nm].astype(float), axes=1) / w.sum()
                    ds.createVariable(nm, "f8", ("nz", "nx"))[:] = avg
        if done:
            log = f"FLUCS chain {run}: {why}; {len(segs)} segments; end time {t_end_state:.4f}\n"
            lb = np.frombuffer(log.encode(), dtype="S1")
            ds.createDimension("log_dim", len(lb))
            ds.createVariable("log", "S1", ("log_dim",))[:] = lb
    os.replace(tmp, out)
    print(f"wrote {out}: nt_hf {len(thf)} (t 0-{thf[-1]:.1f}), final state t = {t_end_state:.3f}, "
          f"complete = {done} ({why}); Curvy3DInput nx ny nz = {P.nx} {P.ny} {P.nz}, Ly {P.Ly:.4f}"
          + (f"; ky-transfer {kyt['dBx_sq'].shape}" if kyt is not None else ""))


if __name__ == "__main__":
    main()
