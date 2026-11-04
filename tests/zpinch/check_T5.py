"""T5 invariant subspace k_x = 0: A2, nu = 0, kx0_noise (E ~ 1), nonlinear vs linear run,
same IC and same fixed dt (max_cfl huge). Usage: python check_T5.py <scratch dir>
Checks: (1) energy in k_x != 0 relative to E; (2) k_x = 0 plane of the nonlinear run equals
the linear run (max |diff| / max |f|); (3) complex rates of the (0,1,0) and (0,1,1) modes
in the nonlinear run (least-squares one-step propagator of the 4-field series) vs theory."""
import sys

import numpy as np
from zp_common import POINTS, omega2_roots, read0d, read_modes, run, write_input

scratch = sys.argv[1]
kb, kP = POINTS["A2"]
c = 1 + 1 / kb
LOCS = ["0:4, :, 1:, :", "0:4, :, 0, :", "0:4, 0, 0, 1", "0:4, 1, 0, 1"]
res = {}
for lin in (False, True):
    d = write_input(f"{scratch}/T5_{'linear' if lin else 'nonlinear'}", kb=kb, kP=kP,
                    linear=lin, dt_max=0.01, tfinal=10.0, max_cfl=1e6,
                    init={"method": "kx0_noise", "amplitude": 1e-2, "rand_seed": 257},
                    save_steps=10, locations=LOCS, loc_steps=10)
    if "--analyse" not in sys.argv:
        run(d)
    res[lin] = (read0d(d), *read_modes(d))

v, t, m = res[False]
vl, tl, ml = res[True]
key = {s: "location_" + s.replace(" ", "") for s in LOCS}
print(f"T5: E(0) {v['E'][0]:.3e}, E(T) {v['E'][-1]:.3e}, t_end {t[-1]:.2f}; "
      f"dt nonlinear [{np.diff(v['t']).min():.4g}, {np.diff(v['t']).max():.4g}]")
# (1) energy (unweighted |f|^2 sum and E-weighted) in kx != 0
off = m[key[LOCS[0]]]
kperp_w = None
amp_off = np.max(np.abs(off).reshape(len(t), -1), axis=1)
amp_on = np.max(np.abs(m[key[LOCS[1]]]).reshape(len(t), -1), axis=1)
print("   max|f(kx!=0)| / max|f(kx=0)| at saves: " + f"max over {len(t)} saves {np.max(amp_off / amp_on):.1e}")
# (2) nonlinear vs linear in kx = 0 plane
on, onl = m[key[LOCS[1]]], ml[key[LOCS[1]]]
assert np.allclose(t, tl, rtol=0, atol=1e-12), "time grids differ"
diff = np.max(np.abs(on - onl).reshape(len(t), -1), axis=1) / np.max(np.abs(onl).reshape(len(t), -1), axis=1)
print("   kx=0 plane max|NL run - linear run| / max|f| at saves: " + f"max {diff.max():.1e} (t_end {diff[-1]:.1e})")
# (3) rates
worst = 0.0
for s, mode in ((LOCS[2], (0, 1, 0)), (LOCS[3], (0, 1, 1))):
    # uniformly spaced saves only (drop the forced final save)
    h = np.median(np.diff(t))
    u = np.concatenate([[True], np.abs(np.diff(t) - h) < 1e-9 * h])
    u &= np.cumprod(u).astype(bool)
    X = m[key[s]][u, :, 0, 0, 0].T
    P = X[:, 1:] @ np.linalg.pinv(X[:, :-1])
    lam = np.log(np.linalg.eigvals(P).astype(complex)) / h
    r = omega2_roots(kb, kP, *mode)
    g = np.sqrt(-r[0])
    err = abs(lam.real.max() - g) / g
    worst = max(worst, err)
    print(f"   {mode}: gamma th {g:.6f} num {lam.real.max():.6f} rel {err:.1e}; all lambda "
          + " ".join(f"{z.real:+.4f}{z.imag:+.4f}i" for z in lam))
print(f"T5 {'PASS' if diff.max() < 1e-10 and worst < 1e-3 else 'FAIL'}: "
      f"max NL-vs-linear {diff.max():.1e}, rate rel err {worst:.1e}, "
      f"max kx!=0 / kx=0 amplitude {np.max(amp_off / amp_on):.1e}")
