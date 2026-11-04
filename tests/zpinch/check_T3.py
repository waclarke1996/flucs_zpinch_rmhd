"""T3 energy conservation (kP = 0, nu = 0, nonlinear, fixed dt) and T3b at kP = 0.
Usage (on the GPU node): python check_T3.py <scratch dir>
E error |E(T) - E(0)|/E(0) at T = 2 for dt = 4e-3 ... 5e-4, ssprk3e and rk4; order p.
T3b: per-invariant NL transfer (from the NL RHS alone) relative to the gross transfer
sum_k |per-mode transfer| and to the invariant's linear exchange (injection = 0 at kP = 0)."""
import sys

import numpy as np
from zp_common import POINTS, read0d, run, write_input

scratch = sys.argv[1]
kb = POINTS["A2"][0]
INV = ("zplus", "zminus", "Zpplus", "Zpminus")
DTS = (4e-3, 2e-3, 1e-3, 5e-4)
ok = True
for method in ("ssprk3e", "rk4"):
    errs = []
    for dt in DTS:
        d = write_input(f"{scratch}/T3_{method}_{dt:g}", kb=kb, kP=0.0, dt_max=dt,
                        tfinal=2.0, max_cfl=1e6, method=method,
                        init={"method": "white_noise", "amplitude": 5e-4, "rand_seed": 257},
                        save_steps=1, energy_opts={"save_nonlinear": True, "invariants": True})
        run(d)
        v = read0d(d)
        dts = np.diff(v["t"])
        E = v["E"]
        err = abs(E[-1] - E[0]) / E[0]
        errs.append(err)
        nl_E = np.max(np.abs(v["dEdt_nonlinear"])) / np.max([np.max(np.abs(v[f"{n}_nl_gross"])) for n in INV])
        line = (f"T3 {method} dt {dt:g}: steps {len(dts)}, dt range [{dts.min():.3e}, "
                f"{dts.max():.3e}], t_end {v['t'][-1]:.4f}, E0 {E[0]:.4e}, "
                f"|dE|/E0 {err:.2e}, max|E-E0|/E0 {np.max(abs(E - E[0])) / E[0]:.2e}")
        print(line, flush=True)
        if method == "ssprk3e" and dt == DTS[-1]:
            print(f"T3b kP=0 ({method}, dt {dt:g}): max_t|dE/dt_NL| / gross {nl_E:.1e}")
            for n in INV:
                r_g = np.max(np.abs(v[f"{n}_nl"]) / v[f"{n}_nl_gross"])
                r_x = np.max(np.abs(v[f"{n}_nl"]) / np.abs(v[f"{n}_exch"]))
                inj = np.max(np.abs(v[f"{n}_inj"]))
                ok &= r_g < 1e-12
                print(f"   {n}: W0 {v[n][0]:.3e}, max|nl|/gross {r_g:.1e}, "
                      f"max|nl|/|exch| {r_x:.1e}, max|inj| {inj:.1e}, "
                      f"W(T)/W(0)-1 {v[n][-1] / v[n][0] - 1:+.2e}")
    p = np.log2(np.array(errs[:-1]) / np.array(errs[1:]))
    print(f"T3 {method}: errors {' '.join(f'{e:.2e}' for e in errs)}; observed orders "
          f"{' '.join(f'{x:.2f}' for x in p)}", flush=True)
print(f"T3b(kP=0) {'PASS' if ok else 'FAIL'}")
