"""T1 (linear growth / frequencies) and T2 (dissipation), 32^3, linear runs.
Usage (on the GPU node): python check_T1T2.py <scratch dir>
Unstable modes: eigenvector init, fit of log|Phi_k(t)| over >= 5 e-folds.
Stable modes: all fields = 1 at the mode (all 4 eigenmodes excited); the one-step
propagator P is fitted by least squares from the 4-field time series and
lambda = log(eig P)/dt_save is compared with -i*(+-omega) - damping."""
import sys

import numpy as np
from zp_common import POINTS, omega2_roots, read_modes, run, write_input

scratch = sys.argv[1]
DISS = {"none": (None, 2), "p2": ((1e-3, 2e-2), 2), "p4": ((1e-5, 2e-2), 4)}
UNSTABLE = [(0, 1, 1), (0, 1, 0), (0.5, 1, 1), (0, 3, 1)]
DT = 0.02


def loc(kx, ky, kz):
    return f"0:4, {int(round(kz)) % 32}, {int(round(2 * kx)) % 32}, {int(round(ky))}"


def damping(nu, p, kx, ky, kz):
    if nu is None:
        return 0.0
    return nu[0] * (kx * kx + ky * ky) ** p + nu[1] * (kz * kz) ** p


def gam(kb, kP, kx, ky, kz):
    r = omega2_roots(kb, kP, kx, ky, kz)
    return np.sqrt(-r[0]) if r[0] < 0 else None


worst = {"T1": 0.0, "T2": 0.0}
for dname, (nu, p) in DISS.items():
    test = "T1" if dname == "none" else "T2"
    for pt in ("A2", "A3a", "A3b", "A4"):
        kb, kP = POINTS[pt]
        modes = [m for m in UNSTABLE if gam(kb, kP, *m) is not None]
        rates = {m: gam(kb, kP, *m) - damping(nu, p, *m) for m in modes}
        T = min(6.0 / min(abs(r) for r in rates.values()), 150.0)
        d = write_input(f"{scratch}/T12_{pt}_{dname}", kb=kb, kP=kP, linear=True,
                        nu=nu, power=p, dt_max=DT, tfinal=T,
                        init={"method": "single_mode", "modes": [[float(x) for x in m] for m in modes],
                              "eigenvector": True, "amplitude": 1e-6},
                        save_steps=100, locations=[loc(*m) for m in modes], loc_steps=5)
        run(d)
        t, data = read_modes(d)
        out = []
        for m in modes:
            key = "location_" + loc(*m).replace(" ", "")
            a = np.abs(data[key][:, 0, 0, 0, 0])
            efolds = abs(rates[m]) * (t[-1] - t[0])
            g_fit = np.polyfit(t, np.log(a), 1)[0]
            err = abs(g_fit - rates[m]) / abs(rates[m])
            worst[test] = max(worst[test], err)
            out.append(f"{m}: th {rates[m]:.6f} fit {g_fit:.6f} rel {err:.1e} ({efolds:.1f} e-f)")
        print(f"{test} {pt} {dname}: " + "; ".join(out), flush=True)

    # Stable modes: S (0,1,1) and A2 (2,1,1)
    for pt, m in (("S", (0, 1, 1)), ("A2", (2, 1, 1))):
        kb, kP = POINTS[pt]
        dmp = damping(nu, p, *m)
        d = write_input(f"{scratch}/T12_{pt}_stable_{dname}", kb=kb, kP=kP, linear=True,
                        nu=nu, power=p, dt_max=DT, tfinal=40.0,
                        init={"method": "single_mode", "modes": [[float(x) for x in m]],
                              "eigenvector": False, "amplitude": 1e-3},
                        save_steps=100, locations=[loc(*m)], loc_steps=5)
        run(d)
        t, data = read_modes(d)
        X = data["location_" + loc(*m).replace(" ", "")][:, :, 0, 0, 0].T  # (4, nt)
        dts = np.diff(t)
        assert np.ptp(dts) < 1e-12 * dts[0], "save interval not constant"
        P = X[:, 1:] @ np.linalg.pinv(X[:, :-1])
        lam = np.log(np.linalg.eigvals(P).astype(complex)) / dts[0]
        w_num = np.sort(np.abs(lam.imag))
        w_th = np.sort(np.repeat(np.sqrt(omega2_roots(kb, kP, *m)), 2))
        err_w = np.max(np.abs(w_num - w_th) / w_th)
        err_d = np.max(np.abs(-lam.real - dmp)) / (dmp if dmp > 0 else w_th.max())
        worst[test] = max(worst[test], err_w, err_d if dmp > 0 else 0.0)
        print(f"{test} {pt} {m} stable {dname}: omega th {w_th[::2].round(6)} "
              f"num {w_num[::2].round(6)} rel {err_w:.1e}; damping th {dmp:.4e} "
              f"num {np.mean(-lam.real):.4e} ({'rel' if dmp > 0 else 'abs/omega'} {err_d:.1e})",
              flush=True)

# White-noise variant (T1): A2, fit after transients
kb, kP = POINTS["A2"]
d = write_input(f"{scratch}/T12_A2_whitenoise", kb=kb, kP=kP, linear=True, dt_max=DT,
                tfinal=60.0, init={"method": "white_noise", "amplitude": 1e-5,
                                   "rand_seed": 257},
                save_steps=100, locations=[loc(0, 1, 0), loc(0, 1, 1)], loc_steps=5)
run(d)
t, data = read_modes(d)
for m, (t0, t1) in (((0, 1, 0), (15, 30)), ((0, 1, 1), (30, 60))):
    a = np.abs(data["location_" + loc(*m).replace(" ", "")][:, 0, 0, 0, 0])
    s = (t >= t0) & (t <= t1)
    g_fit = np.polyfit(t[s], np.log(a[s]), 1)[0]
    g_th = gam(kb, kP, *m)
    err = abs(g_fit - g_th) / g_th
    worst["T1"] = max(worst["T1"], err)
    print(f"T1 A2 white noise {m}: th {g_th:.6f} fit {g_fit:.6f} rel {err:.1e} (t {t0}-{t1})")
for k, v in worst.items():
    print(f"{k} {'PASS' if v < 1e-3 else 'FAIL'}: worst rel err {v:.1e}")
