#!/usr/bin/env python3
"""Stage 3e E1: zonal vs streamer content per snapshot and per state (high/low) of A3a.
Per snapshot (energy density as reference/curvy3d/dissipation_map.py, box averages):
  E, E_kx0/E (streamers: k_x = 0), E_ky0/E (zonal: k_y = 0, all k_z), E_ky0kz0/E (k_y = k_z = 0),
  S_zon0 = zonal-shear rms sqrt(<(d_x^2 Phi_zon)^2>), Phi_zon = k_y = k_z = 0 part; S_zon,all = same with all k_z;
  W_rms = vorticity rms sqrt(<(nabla_perp^2 Phi)^2>) for scale; b_rms.
State of each snapshot = two-state classifier label at that time (runs/stage3d/two_state_series.npz: 1 H, -1 L, 0 -).
Runs: curvy3d A3a-Y (0-500) + A3a-Yc2 (500-1500); FLUCS D2 A3a-Z-k8-C300 (its 14 restart.nc, t 350-1000).
Usage: python e1_zonal.py  (curvy3d interpreter; CPU, a few minutes)"""
import glob
import sys
import numpy as np
from netCDF4 import Dataset

sys.path.insert(0, "/mnt/users/clarkew/flucs_work/reference/curvy3d")
import zpinch_runtools as zr

DATA = "/mnt/users/clarkew/data/rmhd_zpinch/output/3D/BP2_anchors_v2"
W = "/mnt/users/clarkew/flucs_work/runs"
SER = np.load(f"{W}/stage3d/two_state_series.npz")
RUNS = [("C A3a-Y", glob.glob(f"{DATA}/*_A3a-Y.nc")[0], 0.0, "C:A3a-Y+Yc2"),
        ("C A3a-Yc2", glob.glob(f"{DATA}/*_A3a-Yc2.nc")[0], 500.0, "C:A3a-Y+Yc2"),
        ("F A3a-Z D2", sorted(glob.glob(f"{W}/stage3d/A3a-Z-k8-C300/t*/restart.nc")), 300.0, "F:A3a-Z-k8-C300")]
NAMES = ("Phi", "Psi", "deltaBpar", "upar")   # FLUCS restart field order (padded layout, same normalisation)
C = 1.0 + 1.0 / 1.13137085                     # 1 + 1/kb (A3a)


def label(key, t):
    ts, ls = SER[key + "|t"], SER[key + "|l"]
    if t < ts[0] - 1e-6 or t > ts[-1] + 1e-6:
        return 0
    return int(ls[np.argmin(np.abs(ts - t))])


def snapshots(f, toff):
    """yield (t, fields, Lx, Ly, Lz) for a curvy3d .nc (all saved snapshots) or a list of FLUCS restart.nc"""
    if isinstance(f, list):
        for g in f:
            with Dataset(g) as ds:
                a = np.array(ds["fields_real"][:]) + 1j * np.array(ds["fields_imag"][:])
                yield float(ds["current_time"][...]) + toff, dict(zip(NAMES, a)), 4 * np.pi, 2 * np.pi, 2 * np.pi
        return
    with Dataset(f) as ds:
        P = zr.params_from_dataset(ds)
        for it in range(1, zr.last_saved_snapshot(ds) + 1):
            if not np.any(ds["Phi_rfft_r"][it]):
                continue
            yield (it * P.save_time + toff,
                   {n: np.array(ds[n + "_rfft_r"][it], float) + 1j * np.array(ds[n + "_rfft_i"][it], float)
                    for n in NAMES}, P.Lx, P.Ly, P.Lz)


rows = []
for name, f, toff, key in RUNS:
    shape = None
    for t, F, Lx, Ly, Lz in snapshots(f, toff):
        if shape != F["Phi"].shape:
            shape = F["Phi"].shape
            nz, nx, hy = shape
            kx = np.fft.fftfreq(nx, d=1.0 / nx) * 2 * np.pi / Lx
            ky = np.arange(hy) * 2 * np.pi / Ly
            kz = np.fft.fftfreq(nz, d=1.0 / nz) * 2 * np.pi / Lz
            KZ, KX, KY = np.meshgrid(kz, kx, ky, indexing="ij")
            k2 = KX ** 2 + KY ** 2
            w = np.where(KY > 0, 2.0, 1.0)     # full k_y = 0 plane stored
            zon, zon0 = KY == 0, (KY == 0) & (KZ == 0)
        p2 = np.abs(F["Phi"]) ** 2 * w
        eb = 0.5 * k2 * np.abs(F["Psi"]) ** 2 * w
        e = 0.5 * k2 * p2 + eb + 0.5 * (np.abs(F["upar"]) ** 2 + C * np.abs(F["deltaBpar"]) ** 2) * w
        E = e.sum()
        rows.append(dict(run=name, t=t, l=label(key, t), E=E, b=np.sqrt(2 * eb.sum()),
                         kx0=e[KX == 0].sum() / E, ky0=e[zon].sum() / E, ky0kz0=e[zon0].sum() / E,
                         S0=np.sqrt((k2 ** 2 * p2)[zon0].sum()), Sall=np.sqrt((k2 ** 2 * p2)[zon].sum()),
                         Wr=np.sqrt((k2 ** 2 * p2).sum())))

ST = {1: "H", -1: "L", 0: "-"}
print("# per snapshot: run t state E b_rms E_kx0/E E_ky0/E E_ky0kz0/E S_zon0 S_zon,all W_rms")
for r in rows:
    print(f"{r['run']:11s} {r['t']:6.0f} {ST[r['l']]} {r['E']:7.0f} {r['b']:5.2f} {r['kx0']:.3f} {r['ky0']:.3f} "
          f"{r['ky0kz0']:.3f} {r['S0']:7.3f} {r['Sall']:7.3f} {r['Wr']:7.2f}")
print("\n# state means (snapshots t >= 100; mean +- std over snapshots)")
print(f"{'run':11s} st  n {'E':>6s} {'b_rms':>5s} {'kx0/E':>11s} {'ky0/E':>11s} {'ky0kz0/E':>11s} "
      f"{'S_zon0':>11s} {'S_zon,all':>11s} {'W_rms':>6s}")
for name, pre in (("C A3a-Y+Yc2", "C A3a-Y"), ("F A3a-Z D2", "F A3a-Z")):
    for st in (1, -1):
        R = [r for r in rows if r["run"].startswith(pre) and r["t"] >= 100 and r["l"] == st]
        if not R:
            continue
        m = lambda k: f"{np.mean([r[k] for r in R]):.3f}+-{np.std([r[k] for r in R]):.3f}"
        print(f"{name:11s} {ST[st]}  {len(R):2d} {np.mean([r['E'] for r in R]):6.0f} {np.mean([r['b'] for r in R]):5.2f} "
              f"{m('kx0')} {m('ky0')} {m('ky0kz0')} {m('S0')} {m('Sall')} {np.mean([r['Wr'] for r in R]):6.2f}")
