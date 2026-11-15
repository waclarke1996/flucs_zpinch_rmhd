#!/usr/bin/env python3
"""
Stage 4a STEP 5: production-scan campaign tooling (DRY RUN: writes CSV / plan files only, submits nothing).

    python campaign.py points                -> campaign_points.csv (+ printed summary)
    python campaign.py plan [--speed FILE]   -> campaign_plan.txt (job list, GPU-h per GPU type, wall days, disk)

Scan grid: kappaP = 0.4 x 2^(n/4), n = 0..10;  kappaBeta = (0.05/sqrt 2) x 2^(m/2), m = 0..11.
Production = linearly unstable points:
    interchange   kappaP (1 + kappaBeta) > 2
    m = 1         2 kappaBeta kappaP > 1      ("supercritical")
regime: 'supercritical' (m=1 unstable; tier T2), 'interchange' (only interchange; tier T1).
Growth rates from the SPEC dispersion relation on the box's allowed modes (kx = 0.5 i, ky = j >= 1, kz = k,
|i| <= Kx, j <= Ky, |k| <= Kz of the tier's grid):  c w^4 + w^2 [kap c N^2 - (1+c) kz^2] + kz^2 (kz^2 - kap N0^2) = 0,
kap = ky^2/kperp^2, c = 1 + 1/kb, N^2 = 2 kb/(1+kb) (kP(1+kb) - 2), N0^2 = 2 kb kP.  gamma_ideal = max Im w;
gamma_damped = max (Im w - nu_perp kperp^8 - nu_z kz^8) (the same hyperdissipation acts on all four fields, so it
shifts every eigenvalue of a mode by exactly its rate). Points with gamma_damped < 0.05 are flagged 'slow'.
"""
import argparse
import csv
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TIERS = {
    "T1": dict(nx=256, ny=128, nz=256, nu_z=7.5e-14),
    "T2": dict(nx=256, ny=128, nz=512, nu_z=5.9e-16),
    "T3": dict(nx=256, ny=128, nz=768, nu_z=3.4e-17),
}
NU_PERP = 4e-11
T_INIT = 500.0
SEG = 100.0


def grid_points():
    for n in range(11):
        for m in range(12):
            kP = 0.4 * 2 ** (n / 4)
            kb = 0.05 / np.sqrt(2) * 2 ** (m / 2)
            yield n, m, kP, kb


def growth(kP, kb, tier):
    T = TIERS[tier]
    Kx, Ky, Kz = ((T[k] - 1) // 3 for k in ("nx", "ny", "nz"))
    kx = 0.5 * np.arange(-Kx, Kx + 1)
    ky = np.arange(1, Ky + 1, dtype=float)
    kz = np.arange(0, Kz + 1, dtype=float)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
    k2 = KX**2 + KY**2
    kap = KY**2 / k2
    c = 1 + 1 / kb
    N2 = 2 * kb / (1 + kb) * (kP * (1 + kb) - 2)
    N02 = 2 * kb * kP
    a, b, cc = c, kap * c * N2 - (1 + c) * KZ**2, KZ**2 * (KZ**2 - kap * N02)
    disc = np.sqrt((b * b - 4 * a * cc).astype(complex))
    gam = np.zeros(k2.shape)
    for s in (+1, -1):
        w = np.sqrt((-b + s * disc) / (2 * a))
        gam = np.maximum(gam, np.abs(w.imag))
    damp = NU_PERP * k2**4 + T["nu_z"] * KZ**8
    i = np.unravel_index(np.argmax(gam), gam.shape)
    j = np.unravel_index(np.argmax(gam - damp), gam.shape)
    return (float(gam[i]), (KX[i], KY[i], KZ[i]), float((gam - damp)[j]), (KX[j], KY[j], KZ[j]))


def points(write=True):
    rows = []
    for n, m, kP, kb in grid_points():
        inter = kP * (1 + kb) > 2
        m1 = 2 * kb * kP > 1
        if not (inter or m1):
            continue
        regime = "supercritical" if m1 else "interchange"
        tier = "T2" if m1 else "T1"
        g, mode, gd, moded = growth(kP, kb, tier)
        rows.append(dict(run_id=f"P{n:02d}B{m:02d}-{tier}-k8", n=n, m=m, kappaP=round(kP, 6),
                         kappaBeta=round(kb, 6), regime=regime, tier=tier, interchange_unstable=int(inter),
                         m1_unstable=int(m1), N0sq=round(2 * kb * kP, 5), gamma_ideal=round(g, 4),
                         mode_ideal="%g/%g/%g" % mode, gamma_damped=round(gd, 4), mode_damped="%g/%g/%g" % moded,
                         slow=int(gd < 0.05), t_init=T_INIT))
    if write:
        with open(os.path.join(HERE, "campaign_points.csv"), "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["points", "plan"])
    a = ap.parse_args()
    rows = points()
    ns = sum(r["regime"] == "supercritical" for r in rows)
    print(f"{len(rows)} unstable points: {ns} supercritical (T2), {len(rows) - ns} interchange-only (T1); "
          f"slow (gamma_damped < 0.05): {[r['run_id'] for r in rows if r['slow']]}")
    for r in rows:
        print(f"{r['run_id']:16s} kP {r['kappaP']:7.4f} kb {r['kappaBeta']:7.4f} {r['regime']:13s} "
              f"N0^2 {r['N0sq']:6.3f} gamma {r['gamma_ideal']:.3f} ({r['mode_ideal']}) "
              f"damped {r['gamma_damped']:.3f} ({r['mode_damped']}){'  SLOW' if r['slow'] else ''}")


if __name__ == "__main__":
    main()
