#!/usr/bin/env python3
"""
zp_shells_export.py -- write the FLUCS run-time shell-to-shell transfers (zp_shells, output.shells.nc) in the file
format of reference/curvy3d/hydra_shell_plotter_{elsasser,magnetic_kinetic}_gpu.py, so Will's plotting of those
CSV/JSON files works unchanged:
    <shell_type>_<name>_transfer_data.csv   time mean over the selected records / L_y^2 (as the plotters)
    <shell_type>_transfer_metadata_elsasser.json, <shell_type>_transfer_metadata_magnetic_kinetic.json
plus the sums uperp, deltaBperp, deltaBpar, upar of the kinetic/magnetic set.

    python zp_shells_export.py RUN_OR_FILES... --out DIR [--tmin T] [--tmax T] [--Ly 6.283...]
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.expanduser("~/flucs_work/flucs/src"))
from flucs.systems.zpinch_fourier.zpinch_readers import load_diag  # noqa: E402

ELS = ("zplus", "zminus", "zparplus", "zparminus")
MK = ("Tuu", "TBu", "TBB", "TuB", "TBparBpar", "TuparBpar", "Tuparupar", "TBparupar")
SUMS = {"uperp": ("Tuu", "TBu"), "deltaBperp": ("TBB", "TuB"), "deltaBpar": ("TBparBpar", "TuparBpar"),
        "upar": ("Tuparupar", "TBparupar")}


def ticks(K):
    t = [r"" if i % 2 == 0 else r"${0:.1f}$".format(v) for i, v in enumerate(np.round(K, 2))]
    return t


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="run folders (segments t????/) or output.shells.nc files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tmin", type=float, default=-np.inf)
    ap.add_argument("--tmax", type=float, default=np.inf)
    ap.add_argument("--Ly", type=float, default=2 * np.pi)
    a = ap.parse_args()
    files = []
    for p in a.paths:
        if os.path.isdir(p):
            files += sorted(glob.glob(os.path.join(p, "t[0-9][0-9][0-9][0-9]", "output.shells.nc"))) or \
                [os.path.join(p, "output.shells.nc")]
        else:
            files.append(p)
    t, d = load_diag(files, "zp_shells")
    sel = (t >= a.tmin) & (t <= a.tmax)
    if not sel.any():
        raise SystemExit("no records in the time window")
    os.makedirs(a.out, exist_ok=True)
    for st in ("kperp", "kz"):
        if f"{st}_K_boundaries" not in d:
            continue
        K = d[f"{st}_K_boundaries"]
        mean = {}
        for nm in ELS + MK:
            if f"{st}_{nm}" in d:
                mean[nm] = d[f"{st}_{nm}"][sel].mean(axis=0) / a.Ly ** 2
        for s, (x, y) in SUMS.items():
            if x in mean:
                mean[s] = mean[x] + mean[y]
        for nm, m in mean.items():
            np.savetxt(os.path.join(a.out, f"{st}_{nm}_transfer_data.csv"), m, delimiter=",")
        if st == "kperp":
            fr, to = r'$k_\bot^{\mathrm{from}}L_{y}/2\pi$', r'$k_\bot^{\mathrm{to}}L_{y}/2\pi$'
            labels = {"elsasser": (fr, to), "magnetic_kinetic": (fr, to)}
        else:
            labels = {"elsasser": (r'$k_z^{\mathrm{from}}R/2\pi$', r'$k_z^{\mathrm{to}}R/2\pi$'),
                      "magnetic_kinetic": (r'$k_z^{\mathrm{from}}v_{\mathrm{A}}/N$', r'$k_z^{\mathrm{to}}v_{\mathrm{A}}/N$')}
        for kind, (fr, to) in labels.items():
            meta = {"from_shell_label": fr, "to_shell_label": to, "klabel_ticks": ticks(K),
                    "n_shells": int(len(K) - 1), "shell_type": st}
            with open(os.path.join(a.out, f"{st}_transfer_metadata_{kind}.json"), "w") as fh:
                json.dump(meta, fh, indent=4)
    print(f"wrote {a.out}: {int(sel.sum())} records, t {t[sel][0]:.2f}-{t[sel][-1]:.2f}")


if __name__ == "__main__":
    main()
