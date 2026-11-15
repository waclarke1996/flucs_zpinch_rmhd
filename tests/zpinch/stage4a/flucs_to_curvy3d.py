#!/usr/bin/env python3
"""
flucs_to_curvy3d.py -- FLUCS ZPinchFourier states -> minimal curvy3d-format .nc (Stage 3c, A1).

Reads one or more FLUCS restart.nc files (fields_real/_imag [4, nz, nx, half_ny] padded layout, field order
Phi, Psi, deltaBpar, upar; f = sum_k X_k exp(i k.x), the same normalisation as curvy3d) and writes a file that
reference/curvy3d/{dissipation_map,kpar_spectrum}.py and zpinch_runtools read unchanged:
  * <field>_rfft_r/_i [nt, nz, nx, half_ny] in curvy3d's compact mode layout (K = (n_pad - 1) // 3 per direction,
    FFT order in z and x, k_y = 0..K_y), full k_y = 0 plane (as curvy3d's raw snapshots);
  * snapshot of time t at index round((t + t_offset) / save_time); unused indices are zero (= "never written",
    exactly as the solver's pre-fill, so zr.last_saved_snapshot works);
  * 'input' = the input string of a curvy3d template file + appended overrides (later lines win in Curvy3DInput):
    kappaP/kappaBeta, padded grid, hyperviscosity, save_time, save_time_hf, tfinal, extra_comment (run id);
  * optional heatflux/heatflux_em series from FLUCS output.0d.nc files (--series), in curvy3d's half-plane-sum
    convention, interpolated on t = 0, save_time_hf, ... (NaN outside the FLUCS data).
The inverse map is ZPinchFourier._init_curvy3d_snapshot (curvy3d -> FLUCS), copied mode by mode.

Usage:
  python flucs_to_curvy3d.py RUN/restart.nc [MORE/restart.nc ...] --out X.nc --run-id A3a-Y-k8-S30c
         [--series RUN/output.0d.nc,...] [--save-time 50] [--t-offset 0] [--template CURVY.nc]
Checks printed: energy outside the compact mode set (must be 0), time misplacement, Curvy3DInput parameters.
"""
import argparse
import contextlib
import glob
import io
import os
import sys
import warnings

import numpy as np
from netCDF4 import Dataset

REF = os.path.expanduser("~/flucs_work/reference/curvy3d")
DATA = "/mnt/users/clarkew/data/rmhd_zpinch/output/3D/BP2_anchors_v2"
FLUCS_ORDER = ("Phi", "Psi", "deltaBpar", "upar")


def parse_toml(s):
    try:
        import tomllib as tl
    except ImportError:
        try:
            import tomli as tl
        except ImportError:
            import toml
            return toml.loads(s)
    return tl.loads(s)


def compact_index(n_c, n_pad):
    i = np.arange(n_c)
    return np.where(i < (n_c + 1) // 2, i, i + (n_pad - n_c))


def read_state(path):
    with Dataset(path, "r") as ds:
        X = np.asarray(ds["fields_real"][:], dtype=float) + 1j * np.asarray(ds["fields_imag"][:], dtype=float)
        t = float(ds["current_time"][...])
        inp = parse_toml(str(ds["input_file"][...]))
    return X, t, inp


def to_compact(X):
    """padded FLUCS (4, nz, nx, hy_pad) -> dict name -> (nz_c, nx_c, hy_c); returns also leftover energy."""
    _, nz, nx, hyp = X.shape
    ny = 2 * (hyp - 1)
    Kz, Kx, Ky = (nz - 1) // 3, (nx - 1) // 3, (ny - 1) // 3
    iz, ix = compact_index(2 * Kz + 1, nz), compact_index(2 * Kx + 1, nx)
    sel = np.ix_(iz, ix, np.arange(Ky + 1))
    out, inside = {}, np.zeros(X.shape[1:], bool)
    inside[sel] = True
    left = max(float(np.sum(np.abs(X[f][~inside]) ** 2)) for f in range(4))
    for f, n in enumerate(FLUCS_ORDER):
        out[n] = X[f][sel]
    return out, left


def load_series(paths, t_offset):
    """files in run order; a later file (restart) replaces an earlier one from its first time on."""
    t, es, em = [], [], []
    for p in paths:
        with Dataset(p, "r") as ds:
            for g in sorted(ds.groups, key=int):
                G = ds.groups[g]
                tt = np.asarray(G["time"][:], float)
                for i in range(len(t)):                       # cut earlier pieces at this piece's start
                    k = t[i] < tt[0]
                    t[i], es[i], em[i] = t[i][k], es[i][k], em[i][k]
                t.append(tt)
                es.append(np.asarray(G["energy"]["Qhat_es"][:], float))
                em.append(np.asarray(G["energy"]["Qhat_em"][:], float))
    t, es, em = (np.concatenate(a) for a in (t, es, em))
    o = np.argsort(t, kind="stable")
    t, es, em = t[o] + t_offset, es[o], em[o]
    keep = np.concatenate([np.diff(t) > 0, [True]])        # drop duplicate times (restart overlaps)
    return t[keep], es[keep], em[keep]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("states", nargs="+", help="FLUCS restart.nc files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--series", default="", help="comma-separated FLUCS output.0d.nc files (heat-flux series)")
    ap.add_argument("--save-time", type=float, default=50.0)
    ap.add_argument("--save-time-hf", type=float, default=0.1)
    ap.add_argument("--t-offset", type=float, default=0.0, help="added to FLUCS time (e.g. 300 for a run "
                    "started from curvy3d's t = 300 snapshot)")
    ap.add_argument("--template", default=None, help="curvy3d .nc whose input string is used (default A3a-Y)")
    a = ap.parse_args()
    template = a.template or glob.glob(os.path.join(DATA, "*_RKARS343_A3a-Y.nc"))[0]

    snaps, inp0 = {}, None
    for p in a.states:
        X, t, inp = read_state(p)
        T = t + a.t_offset
        it = int(round(T / a.save_time))
        F, left = to_compact(X)
        print(f"{p}: t_flucs = {t:.6f}, t = {T:.6f} -> index {it} (misplacement {T - it * a.save_time:+.2e}); "
              f"max field energy outside compact set {left:.1e}")
        if left != 0.0:
            raise SystemExit("non-zero modes outside the 2/3 set: layout assumption wrong")
        if it in snaps:
            raise SystemExit(f"two states map to index {it}")
        snaps[it] = F
        inp0 = inp0 or inp
    shape = next(iter(snaps.values()))["Phi"].shape
    nt = max(snaps) + 1

    with Dataset(template, "r") as tds:
        base = tds["input"][:].data.tobytes().decode()
    d, p, h = inp0["dimensions"], inp0["parameters"], inp0["hyperdissipation"]
    over = {
        "kappaP": p["kP"], "kappaBeta": p["kb"],
        "padded_nx": d["nx"], "padded_ny": d["ny"], "padded_nz": d["nz"],
        "perpvisc": h["kperp"], "perpvisc_power": h["kperp_power"],
        "parvisc": h["kz"], "parvisc_power": h["kz_power"],
        # box (Stage 3d): FLUCS zpinch default Lx = 4pi, Ly = Lz = 2pi; curvy3d sets Lx = Lx_mult * Ly
        "Ly": float(d.get("Ly", 2 * np.pi)), "Lz": float(d.get("Lz", 2 * np.pi)),
        "Lx_mult": float(d.get("Lx", 4 * np.pi)) / float(d.get("Ly", 2 * np.pi)),
        "save_time": a.save_time, "save_time_hf": a.save_time_hf, "tfinal": (nt - 1) * a.save_time,
        "extra_comment": a.run_id,
        # Stage 4a fix: the template's input string has explicit nx/ny/nz lines (A3a-Y grid 171 x 85 x 171);
        # override with this state's mode counts (before, Curvy3DInput reported the template's nz for T2 files).
        "nx": 2 * ((int(d["nx"]) - 1) // 3) + 1, "ny": 2 * ((int(d["ny"]) - 1) // 3) + 1,
        "nz": 2 * ((int(d["nz"]) - 1) // 3) + 1,
    }
    input_str = base.rstrip("\n") + "\n//----- flucs_to_curvy3d.py overrides -----//\n" + \
        "".join(f"{k} {v!r}\n" if isinstance(v, float) else f"{k} {v}\n" for k, v in over.items())

    # check the parameters as the diagnostics will see them
    sys.path.insert(0, REF)
    from curvy3dinput import Curvy3DInput
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        P = Curvy3DInput(input_str=input_str)
    print(f"Curvy3DInput: kP {P.kappaP} kb {P.kappaBeta} L {P.Lx:.4f} {P.Ly:.4f} {P.Lz:.4f} "
          f"nu_perp {P.perpvisc} p {P.perpvisc_power} nu_z {P.parvisc} q {P.parvisc_power} "
          f"save_time {P.save_time} save_time_hf {P.save_time_hf}; modes (nz,nx,hy) {shape}")

    if os.path.exists(a.out):
        os.remove(a.out)
    with Dataset(a.out, "w") as ds:
        ds.createDimension("nt", nt)
        ds.createDimension("nz", shape[0])
        ds.createDimension("nx", shape[1])
        ds.createDimension("half_ny", shape[2])
        ds.createDimension("ny", 2 * shape[2] - 1)
        for k in ("nx", "ny", "nz"):
            ds.createDimension("padded_" + k, int(d[k]))
        b = np.frombuffer(input_str.encode(), dtype="S1")
        ds.createDimension("input_dim", len(b))
        ds.createVariable("input", "S1", ("input_dim",))[:] = b
        ds.setncattr("run_id", a.run_id)
        ds.setncattr("source", "FLUCS ZPinchFourier, converted by flucs_to_curvy3d.py: " + ", ".join(a.states))
        for n in ("Phi", "deltaBpar", "Psi", "upar"):
            vr = ds.createVariable(n + "_rfft_r", "f8", ("nt", "nz", "nx", "half_ny"), zlib=True, complevel=1)
            vi = ds.createVariable(n + "_rfft_i", "f8", ("nt", "nz", "nx", "half_ny"), zlib=True, complevel=1)
            for it in range(nt):
                Z = snaps[it][n] if it in snaps else np.zeros(shape, complex)
                vr[it] = Z.real
                vi[it] = Z.imag
        if a.series:
            ts, es, em = load_series([s for s in a.series.split(",") if s], a.t_offset)
            h2 = 0.5 * a.save_time_hf
            thf = np.arange(0.0, ts[-1] + h2, a.save_time_hf)
            inr = (thf >= ts[0] - h2) & (thf <= ts[-1] + h2)
            kP, kb = float(p["kP"]), float(p["kb"])
            hf = np.where(inr, np.interp(thf, ts, es) * (kP * (1 + kb) - 2.0) / (2.0 * (1 + kb)), np.nan)
            hfe = np.where(inr, np.interp(thf, ts, em) * kP / 2.0, np.nan)
            ds.createDimension("nt_hf", len(thf))
            ds.createVariable("heatflux", "f8", ("nt_hf",))[:] = hf
            ds.createVariable("heatflux_em", "f8", ("nt_hf",))[:] = hfe
            print(f"series: {len(ts)} FLUCS samples t = {ts[0]:.2f}-{ts[-1]:.2f} -> {inr.sum()} hf samples")
    print(f"wrote {a.out}: nt {nt}, snapshots at indices {sorted(snaps)}")


if __name__ == "__main__":
    main()
