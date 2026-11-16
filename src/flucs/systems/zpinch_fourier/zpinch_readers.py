"""
Readers for the Z-pinch run-time outputs (Stage 4a). CPU only (numpy, netCDF4).

    from flucs.systems.zpinch_fourier.zpinch_readers import load_diag, mean_profiles

    t, d = load_diag(["RUN/t0100/output.1d.nc", "RUN/t0200/output.1d.nc"], "zp_spectra")
    P = mean_profiles(["RUN/t0100/output.1d.nc", ...], nx=256, ny=128)
    P["x"], P["uy_x"]          # (y,z)-averaged u_y(x) = d_x Phi, shape (nt, nx)
    P["y"], P["dBx_y"]         # (x,z)-averaged dB_x(y) = -d_y Psi, shape (nt, ny)

Segments (a restart chain) are concatenated in the order given; all netCDF
groups of each file (one per run) are read in group order.
"""

from __future__ import annotations

import numpy as np

FIELD_NAMES = ("Phi", "Psi", "dBpar", "upar")


def timed_period(path):
    """save_time of a time-based output file RUN/output.<name>.nc from RUN/input.toml ([output.<name>] save_time > 0),
    else None (step-based output, or no input.toml next to it)."""
    import os
    import re

    m = re.fullmatch(r"output\.(.+)\.nc", os.path.basename(path))
    inp = os.path.join(os.path.dirname(path), "input.toml")
    if not m or not os.path.exists(inp):
        return None
    sec = re.search(rf"^\[output\.{re.escape(m.group(1))}\]\s*$(.*?)(?=^\[|\Z)", open(inp).read(), re.M | re.S)
    v = re.search(r"^\s*save_time\s*=\s*([0-9.eE+-]+)", sec.group(1), re.M) if sec else None
    return float(v.group(1)) if v and float(v.group(1)) > 0 else None


def duplicate_mask(t, period):
    """
    True for the records to keep. Time-based outputs (record grid t = k x period) written before the 2026-11-16
    trigger fix can hold a duplicate record (a step within 1e-9 t BELOW a grid point recorded it, the next step
    recorded it again): a record with the same grid index rint(t / period) as the previous kept one is dropped (the
    first, on-time, one is kept). period None (step-based output): nothing is dropped.
    """
    t = np.asarray(t, float)
    keep = np.ones(len(t), bool)
    if period is None or len(t) < 2:
        return keep
    k = np.rint(t / period)
    last = k[0]
    for i in range(1, len(t)):
        if k[i] == last:
            keep[i] = False
        else:
            last = k[i]
    return keep


def load_diag(paths, diag, names=None, dedup=True):
    """
    Concatenate the time-dependent variables of one diagnostic over files.
    Returns (time, {name: array (nt, ...)}). Complex variables (stored as
    <name>_real / <name>_imag) are returned complex under <name>.
    Time-independent variables and the dimension variables are returned
    from the first file under their own names (no time axis).
    dedup: drop duplicate records of time-based outputs (see duplicate_mask; period from timed_period of the first
    file; step-based outputs are never changed).
    """
    from netCDF4 import Dataset

    if isinstance(paths, str):
        paths = [paths]
    ts, data, static = [], {}, {}
    for p in paths:
        with Dataset(p) as ds:
            for gname in sorted(ds.groups, key=int):
                grp = ds.groups[gname]
                if diag not in grp.groups:
                    continue
                g = grp.groups[diag]
                nt = len(grp["time"])
                if nt == 0:
                    continue
                ts.append(np.asarray(grp["time"][:]))
                for v in g.variables:
                    var = g.variables[v]
                    if var.dimensions[:1] != ("time",):
                        static.setdefault(v, np.asarray(var[:]))
                        continue
                    base, part = v, None
                    if v.endswith("_real"):
                        base, part = v[:-5], "re"
                    elif v.endswith("_imag"):
                        base, part = v[:-5], "im"
                    if names is not None and base not in names:
                        continue
                    a = np.asarray(var[:])
                    data.setdefault((base, part), []).append(a)
    t = np.concatenate(ts) if ts else np.zeros(0)
    keep = duplicate_mask(t, timed_period(paths[0]) if paths else None) if dedup else np.ones(len(t), bool)
    t = t[keep]
    out = {}
    for (base, part), arrs in data.items():
        a = np.concatenate(arrs)[keep]
        if part is None:
            out[base] = a
        elif part == "re":
            out[base] = a + 1j * np.concatenate(data[(base, "im")])[keep]
    out.update({k: v for k, v in static.items() if k not in out})
    return t, out


def line_to_real(coeffs, k, L, n, deriv=0, half=False):
    """
    Real-space values on x_i = i L / n of a 1D Fourier line
    f(x) = sum_k X_k e^{i k x}, differentiated deriv times.
    coeffs: (..., nk) complex; k: (nk,) wavenumbers. half = True: only
    k >= 0 is stored (k_y line), the k < 0 part is the complex conjugate.
    """
    x = np.arange(n) * L / n
    X = np.asarray(coeffs) * (1j * np.asarray(k)) ** deriv
    ph = np.exp(1j * np.outer(np.asarray(k), x))          # (nk, n)
    if half:
        w = np.where(np.asarray(k) > 0, 2.0, 1.0)
        return x, np.real((X * w) @ ph)
    return x, np.real(X @ ph)


def mean_profiles(paths, nx=256, ny=128, Lx=4 * np.pi, Ly=2 * np.pi):
    """
    Mean-field profiles from the zp_profiles diagnostic (spectral lines).
    Returns a dict with t, x, y, kx_line, ky_line, the raw coefficients
    line_x (nt, 4, nkx), line_y (nt, 4, nky), and on the real grids:
      <field>_x  (nt, nx): (y,z)-average of Phi, Psi, dBpar, upar vs x
      <field>_y  (nt, ny): (x,z)-average vs y
      uy_x  = d_x Phi,   dBy_x = d_x Psi,   shear_x = d_x u_y = d_x^2 Phi,
      jz_x  = d_x dB_y = d_x^2 Psi,        curv_x = d_x^2 u_y = d_x^3 Phi
      ux_y  = -d_y Phi,  dBx_y = -d_y Psi,  shear_y = d_y u_x = -d_y^2 Phi,
      jz_y  = -d_y dB_x = d_y^2 Psi,        curv_y = d_y^2 u_x = -d_y^3 Phi
    (u = z x grad Phi: u_x = -d_y Phi, u_y = d_x Phi; dB_perp likewise
    with Psi; the k_x = k_y = 0 mode is zero in the solver.)
    """
    t, d = load_diag(paths, "zp_profiles")
    kx, ky = d["kx_line"], d["ky_line"]
    LX, LY = d["line_x"], d["line_y"]
    out = {"t": t, "kx_line": kx, "ky_line": ky, "line_x": LX, "line_y": LY}
    for i, f in enumerate(FIELD_NAMES):
        out["x"], out[f + "_x"] = line_to_real(LX[:, i], kx, Lx, nx)
        out["y"], out[f + "_y"] = line_to_real(LY[:, i], ky, Ly, ny, half=True)
    out["uy_x"] = line_to_real(LX[:, 0], kx, Lx, nx, 1)[1]
    out["dBy_x"] = line_to_real(LX[:, 1], kx, Lx, nx, 1)[1]
    out["shear_x"] = line_to_real(LX[:, 0], kx, Lx, nx, 2)[1]
    out["jz_x"] = line_to_real(LX[:, 1], kx, Lx, nx, 2)[1]
    out["curv_x"] = line_to_real(LX[:, 0], kx, Lx, nx, 3)[1]
    out["ux_y"] = -line_to_real(LY[:, 0], ky, Ly, ny, 1, half=True)[1]
    out["dBx_y"] = -line_to_real(LY[:, 1], ky, Ly, ny, 1, half=True)[1]
    out["shear_y"] = -line_to_real(LY[:, 0], ky, Ly, ny, 2, half=True)[1]
    out["jz_y"] = line_to_real(LY[:, 1], ky, Ly, ny, 2, half=True)[1]
    out["curv_y"] = -line_to_real(LY[:, 0], ky, Ly, ny, 3, half=True)[1]
    return out


def kpar_table(paths, t0=-np.inf, t1=np.inf, mask=None):
    """
    k_par(k_perp) as kpar_spectrum.py from zp_kpar records with t0 <= t <= t1
    (optionally only records where mask(t) is True, e.g. one two-state
    state). Returns dict: shells (lo, hi), kc, Efrac, kz_rms, k_par,
    k_par_large, k_par_wrong (energy-weighted over fields), chi (Phi),
    b_rms, n (records used).
    """
    t, d = load_diag(paths, "zp_kpar")
    sel = (t >= t0) & (t <= t1)
    if mask is not None:
        sel &= np.asarray(mask(t), bool)
    acc = {k: d[k][sel].sum(axis=0) for k in ("den", "lab", "full", "wrong", "large", "E", "vort2")}
    n = int(sel.sum())
    lo, hi = d["shell"], d["shell_hi"]
    with np.errstate(invalid="ignore", divide="ignore"):
        k2f = {k: acc[k] / acc["den"] for k in ("lab", "full", "wrong", "large")}
        Ew = acc["E"]
        comb = {k: np.sqrt(np.nansum(Ew * k2f[k], axis=0) / Ew.sum(axis=0)) for k in k2f}
        chi = np.sqrt(acc["vort2"][0] / n) / np.sqrt(k2f["full"][0])
        Efrac = Ew.sum(axis=0) / Ew.sum()
    return {"shells": np.stack((lo, hi), 1), "kc": np.sqrt(lo * hi), "Efrac": Efrac,
            "kz_rms": comb["lab"], "k_par": comb["full"], "k_par_large": comb["large"],
            "k_par_wrong": comb["wrong"], "chi": chi,
            "b_rms": np.sqrt(d["b2"][sel].mean()) if n else np.nan, "n": n}
