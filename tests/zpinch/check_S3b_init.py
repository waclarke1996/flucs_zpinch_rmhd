"""Stage 3b (1)+(2) checks at 32^3 (run ON the GPU node: gpu.sh python check_S3b_init.py <out dir>).

(1) plugin white noise: reality error, |X| = amplitude, E(0) = numpy box energy, first-step dt
    vs the old core noise ("white_noise_core");
(2) kx split: E_kx0 + E_kxn0 = E, sum of kx0 parts = E_kx0, kx0_noise keeps E_kxn0 = 0;
    streamer init: E(0) = E_s + noise, kx0 fractions for C1/C3/C5.
"""
import re
import sys
import pathlib as pl

import numpy as np

from zp_common import POINTS, write_input, run, read0d

out = pl.Path(sys.argv[1])
NU4 = (2e-4, 8e-6)
NU8 = (4.44089e-12, 1.38778e-14)
ok = True


def check(name, val, tol):
    global ok
    good = abs(val) < tol
    ok &= good
    print(f"  {name:45s} {val: .3e}  (< {tol:g}) {'PASS' if good else 'FAIL'}")


def log_value(d, pattern):
    m = re.search(pattern, (d / "run.log").read_text())
    return float(m.group(1)) if m else np.nan


kb, kP = POINTS["A3a"]
print("(1) white noise, A3a, 32^3, k^4, 20 steps")
for meth in ("white_noise", "white_noise_core"):
    d = write_input(out / meth, kb=kb, kP=kP, nu=NU4, power=2, tfinal=0.2,
                    init={"method": meth, "amplitude": 1e-5, "rand_seed": 257})
    run(d)
    v = read0d(d)
    err = log_value(d, r"reality error:\s*([0-9.eE+-]+)")
    print(f" {meth}: reality error {err:.2e}, E(0) {v['E'][0]:.6e}, "
          f"dt first/min/last {v['dt'][1]:.2e}/{v['dt'][1:].min():.2e}/{v['dt'][-1]:.2e}")
    if meth == "white_noise":
        check("reality error", err, 1e-30)
        check("E_kx0 + E_kxn0 - E (rel)", (v["E_kx0"][0] + v["E_kxn0"][0]) / v["E"][0] - 1, 1e-13)
        parts = sum(v[f"E_kx0_{p}"] for p in ("uperp", "bperp", "dbpar", "upar"))
        check("sum kx0 parts - E_kx0 (rel, all t)", np.max(np.abs(parts / v["E_kx0"] - 1)), 1e-13)

print("(1b) k^8 on 256x128x256, plugin white noise, 10 steps (failed in S3.3b with core noise)")
d = write_input(out / "k8_Y", kb=kb, kP=kP, nu=NU8, power=4, tfinal=1.0, dims=(256, 128, 256),
                save_steps=1, init={"method": "white_noise", "amplitude": 1e-5, "rand_seed": 257},
                energy_opts={"stop_dt_min": 1e-3})
run(d)
v = read0d(d)
print(f"  t_end {v['t'][-1]:.3f}, dt min {v['dt'][1:].min():.3e}, E(0) {v['E'][0]:.4e}")
check("reached t = 1 (1 - t_end, <= 0 ok)", max(1.0 - v["t"][-1], 0.0), 1e-9)
check("dt min >= 1e-3 (1e-3 / dt_min)", 1e-3 / v["dt"][1:].min(), 1.0)

print("(2) kx0_noise keeps E_kxn0 = 0 (A2, nu = 0, t = 0..2)")
kb2, kP2 = POINTS["A2"]
d = write_input(out / "kx0", kb=kb2, kP=kP2, tfinal=2.0,
                init={"method": "kx0_noise", "amplitude": 1e-2, "rand_seed": 257})
run(d)
v = read0d(d)
check("max E_kxn0 / E", np.max(v["E_kxn0"] / v["E"]), 1e-28)
check("E_kx0 / E - 1", np.max(np.abs(v["E_kx0"] / v["E"] - 1)), 1e-13)

print("(3) streamer init t = 0 (E_s = 3, noise 3e-6), kx0 fractions uperp/bperp/upar/dbpar")
REF = {"C1": ("A2", (0, 1, 0), (0.69, 0, 0, 0.31)),
       "C2": ("A3a", (0, 1, 0), None),
       "C3": ("A3a", (0, 1, 1), (0.17, 0.32, 0.17, 0.35)),
       "C5": ("A4", (0, 1, 1), (0.03, 0.53, 0.10, 0.34))}
for case, (pt, mode, ref) in REF.items():
    kb_, kP_ = POINTS[pt]
    d = write_input(out / f"streamer_{case}", kb=kb_, kP=kP_, nu=NU4, power=2, tfinal=0.01,
                    init={"method": "streamer", "modes": [list(mode)], "energy": 3.0,
                          "noise_energy": 3e-6, "rand_seed": 257})
    run(d)
    v = read0d(d)
    f = [v[f"f_kx0_{p}"][0] for p in ("uperp", "bperp", "upar", "dbpar")]
    print(f" {case} {pt} {mode}: E(0) {v['E'][0]:.8f} E_kxn0 {v['E_kxn0'][0]:.3e} "
          f"f = {' '.join(f'{x:.3f}' for x in f)}  ref {ref}")
    e_box = log_value(d, r"E_box = ([0-9.eE+-]+)")
    check(f"{case} E(0) GPU vs numpy box energy (rel)", v["E"][0] / e_box - 1, 1e-13)
    check(f"{case} E(0) - E_s - noise (rel, cross term)", (v["E"][0] - 3.000003) / 3, 1e-3)
    if ref is not None:
        check(f"{case} max |f - ref| (2-digit ref)", np.max(np.abs(np.array(f) - ref)), 0.006)

print("ALL PASS" if ok else "SOME FAIL")
