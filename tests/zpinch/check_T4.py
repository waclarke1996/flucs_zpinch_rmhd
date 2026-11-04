"""T4 energy and per-invariant budgets in a turbulent A2 run, and T3b (turbulent part).
Usage (on the GPU node): python check_T4.py <scratch dir> [spinup|window|analyse ...]
Spin-up: 64^3, A2, k^4, nu_perp = 5e-3, nu_z = 1e-3, white noise 1e-5, t = 0..80 (restart).
Windows: from the restart, every-step diagnostics (save_nonlinear, invariants), max_cfl 0.5
(t = 80..83) and 0.125 (t = 80..81). Budget check: 5-point non-uniform FD of E (and of each
invariant W) vs the instantaneous RHS kP Qhat - D (+ NL transfer) [lin - diss (+ nl)]."""
import pathlib as pl
import sys

import numpy as np
from zp_common import POINTS, read0d, run, write_input

scratch = pl.Path(sys.argv[1])
args = [a for a in sys.argv[2:] if not a.startswith("--tspin=")]
TSPIN = float(next((a.split("=")[1] for a in sys.argv[2:] if a.startswith("--tspin=")), 80))
what = args or ["spinup", "window", "analyse"]
kb, kP = POINTS["A2"]
NU = (5e-3, 1e-3)
INV = ("zplus", "zminus", "Zpplus", "Zpminus")
# (max_cfl, window length); after a restart FLUCS's time.tfinal is a DURATION
WINDOWS = {"cfl0.5": (0.5, 3.0), "cfl0.125": (0.125, 1.0)}
tag = "" if TSPIN == 80 else f"t{TSPIN:g}_"
spin = scratch / ("T4_spinup" if TSPIN == 80 else f"T4_spinup_t{TSPIN:g}")

if "spinup" in what:
    (spin / "restart.nc").unlink(missing_ok=True)
    write_input(spin, kb=kb, kP=kP, n=64, nu=NU, power=2, dt_max=0.01, tfinal=TSPIN,
                init={"method": "white_noise", "amplitude": 1e-5, "rand_seed": 257},
                save_steps=20, energy_opts={"invariants": True},
                restart={"write_restart_file": True, "write_steps": 1000000})
    run(spin)
    v = read0d(spin)
    print(f"spin-up: steps ~{20 * len(v['t'])}, t_end {v['t'][-1]:.2f}, E {v['E'][-1]:.3e}, "
          f"Q/Q_MHD(t>40) mean {np.mean(v['Q_over_QMHD'][v['t'] > 40]):.3f}", flush=True)

if "window" in what:
    for name, (cfl, tf) in WINDOWS.items():
        d = write_input(scratch / f"T4_{tag}{name}", kb=kb, kP=kP, n=64, nu=NU, power=2,
                        dt_max=0.01, tfinal=tf, max_cfl=cfl, save_steps=1,
                        energy_opts={"save_nonlinear": True, "invariants": True},
                        restart={"write_restart_file": False,
                                 "restart_from": str(spin / "restart.nc")})
        run(d)
        print(f"window {name} done", flush=True)


def fd5(t, y):
    """5-point derivative at interior points of a non-uniform grid."""
    out = np.full_like(y, np.nan)
    for i in range(2, len(t) - 2):
        h = t[i - 2:i + 3] - t[i]
        V = np.vander(h, 5, increasing=True).T  # V[k, j] = h_j^k
        w = np.linalg.solve(V, np.array([0, 1, 0, 0, 0.0]))
        out[i] = w @ y[i - 2:i + 3]
    return out


def fd2(t, y):
    out = np.full_like(y, np.nan)
    out[1:-1] = (y[2:] - y[:-2]) / (t[2:] - t[:-2])
    return out


if "analyse" in what:
    ok = True
    for name in WINDOWS:
        v = read0d(scratch / f"T4_{tag}{name}")
        keep = v["t"] <= TSPIN + WINDOWS[name][1] + 1e-9
        v = {k: a[keep] for k, a in v.items()}
        t = v["t"]
        s = slice(5, -2)  # skip the first steps after restart and the FD edge
        dt = np.diff(t)
        inj = v["injection"]
        print(f"T4 {name}: steps {len(t)}, t {t[0]:.2f}-{t[-1]:.2f}, dt mean {dt.mean():.3e} "
              f"[{dt.min():.2e}, {dt.max():.2e}], E {v['E'][-1]:.3e}, Q/Q_MHD mean "
              f"{v['Q_over_QMHD'].mean():.3f}, D/inj mean {np.mean(v['D'] / inj):.3f}")
        rhs = inj - v["D"] + v["dEdt_nonlinear"]
        for lab, fd in (("FD5", fd5), ("FD2", fd2)):
            r = np.abs(fd(t, v["E"]) - rhs)[s] / np.abs(inj[s])
            r2 = np.abs(fd(t, v["E"]) - rhs)[s] / np.maximum(np.abs(inj), v["D"])[s]
            print(f"   E budget {lab}: |dE/dt - (kP Qhat - D)| / |kP Qhat| median {np.median(r):.1e} "
                  f"max {r.max():.1e}; / max(|kP Qhat|, D) max {r2.max():.1e}")
            if lab == "FD5" and name == "cfl0.5":
                ok_E = r.max() < 1e-4
        nlE = np.max(np.abs(v["dEdt_nonlinear"] / inj))
        print(f"   instantaneous: max |dE/dt_NL| / |kP Qhat| {nlE:.1e}")
        for n in INV:
            rhs_n = v[f"{n}_lin"] - v[f"{n}_diss"] + v[f"{n}_nl"]
            own = np.maximum(np.abs(v[f"{n}_lin"]), np.abs(v[f"{n}_diss"]))
            r = np.abs(fd5(t, v[n]) - rhs_n)[s]
            nl_inj = np.max(np.abs(v[f"{n}_nl"] / inj))
            nl_g = np.max(np.abs(v[f"{n}_nl"] / v[f"{n}_nl_gross"]))
            ok &= nl_inj < 1e-12
            print(f"   {n}: W {v[n][-1]:.3e}, <inj> {v[f'{n}_inj'].mean():+.3e} <exch> "
                  f"{v[f'{n}_exch'].mean():+.3e} <diss> {v[f'{n}_diss'].mean():.3e}; budget FD5 "
                  f"resid/|kP Qhat| max {np.max(r / np.abs(inj[s])):.1e}, /own max "
                  f"{np.max(r / own[s]):.1e}; T3b max|nl|/|kP Qhat| {nl_inj:.1e}, /gross {nl_g:.1e}")
        if name == "cfl0.5":
            ok_T4 = ok_E
    print(f"T4 {'PASS' if ok_T4 else 'FAIL'} (FD5, max_cfl 0.5); T3b(turbulent) {'PASS' if ok else 'FAIL'}")
