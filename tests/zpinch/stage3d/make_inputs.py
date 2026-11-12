#!/usr/bin/env python3
"""Stage 3d part 1: write the input.toml files of
  D1  A2 square-box turbulence (L_x = L_y = 2 pi, padded 128 x 128 x 256 = 85 x 85 x 171 modes), plugin white noise
      1e-5 (seed 257), t = 0-500 as 10 chained 50-R/vA segments (restart.nc of each = 3D state at t = 50, ..., 500),
      runs A2-SQ-k8 (curvy3d A2-Y dissipation) and A2-SQ-k4 (curvy3d A2-R0 dissipation).
      Runaway stop: |Q/Q_MHD| > 1e3 or dt < 1e-5, either holding continuously for 5 R/vA.
  D2  A3a-Z-k8-C300: FLUCS continuation of curvy3d A3a-Z (padded 256 x 128 x 512 = 171 x 85 x 341 modes, k^8,
      nu_perp 4.44089e-12, nu_z 8.7e-16) from its snapshot index 6 (t = 300, high state) to t = 1000, 14 segments of 50
      (FLUCS time = t - 300). No Q stop; dt < 1e-5 for 5 R/vA.
Run once; refuses to overwrite existing folders."""
import glob
import os

S = os.path.dirname(os.path.abspath(__file__))
DATA = "/mnt/users/clarkew/data/rmhd_zpinch/output/3D/BP2_anchors_v2"
Z = glob.glob(f"{DATA}/*_RKARS343_A3a-Z.nc")
assert len(Z) == 1, Z
Z = Z[0]
TWO_PI = "6.283185307179586"
SEG = 50


def write(folder, text):
    d = os.path.join(S, folder)
    os.makedirs(d, exist_ok=False)
    with open(os.path.join(d, "input.toml"), "w") as f:
        f.write(text)
    print("wrote", d)


def common(nx, ny, nz, lx, kb, kp, nup, p, nuz, q, init, restart, qstop, head):
    box = f"Lx = {lx}\nLy = {TWO_PI}\nLz = {TWO_PI}\n" if lx else ""
    stop = f"stop_q_over_qmhd = 1000.0, " if qstop else ""
    return f"""{head}
[setup]
system = "ZPinchFourier"
solver = "FourierSolver"
precision = "double"
linear = false

[dimensions]
nx = {nx}
ny = {ny}
nz = {nz}
{box}
[parameters]
kb = {kb}
kP = {kp}

[timestepping]
method = "ssprk3e"

[time]
dt_max = 0.01
tfinal = {float(SEG)}  # duration of this segment
max_cfl = 0.5

[hyperdissipation]
kperp = {nup!r}
kperp_power = {p}
kz = {nuz!r}
kz_power = {q}

[init]
{init}
[output]
write_steps = 500

[output.0d]
save_steps = 10
diags = [{{name = "energy", options = {{invariants = true, {stop}stop_dt_min = 1e-05, stop_persist_time = 5.0}}}}]

[output.1d]
save_steps = 500
diags = [{{name = "energy_1d"}}]

[restart]
{restart}write_restart_file = true
write_steps = 20000
"""


# ---------------- D1: A2 square box, white noise, two dissipations ----------------
DISS = {"k8": (4.44089e-12, 4, 1.38778e-14, 4),   # curvy3d A2-Y
        "k4": (0.0002, 2, 8e-06, 2)}              # curvy3d A2-R0
for tag, (nup, p, nuz, q) in DISS.items():
    run, prev = f"A2-SQ-{tag}", None
    for tend in range(SEG, 501, SEG):
        seg = f"t{tend:04d}"
        if prev is None:
            init = 'method = "white_noise"\namplitude = 1e-05\nrand_seed = 257\n'
            restart = ""
        else:
            init = 'method = "white_noise"  # ignored: restart\namplitude = 1e-05\nrand_seed = 257\n'
            restart = f'restart_from = "../{prev}/restart.nc"\n'
        write(f"{run}/{seg}", common(
            128, 128, 256, TWO_PI, 0.28284271, 2.2627417, nup, p, nuz, q, init, restart, True,
            f"# Stage 3d D1: {run} (A2 in a square box L_x = L_y = 2 pi, {tag}), segment to t = {tend}.\n"
            f"# Padded 128 x 128 x 256 (85 x 85 x 171 modes = Y-grid resolution). Stop |Q/Q_MHD| > 1e3 or dt < 1e-5 for 5 R/vA."))
        prev = seg

# ---------------- D2: A3a-Z continuation from curvy3d t = 300 ----------------
T0, TEND = 300, 1000
run, prev = "A3a-Z-k8-C300", None
for tend in range(T0 + SEG, TEND + 1, SEG):
    seg = f"t{tend:04d}"
    init = f'method = "curvy3d_snapshot"\nfile = "{Z}"\nindex = 6\n'
    restart = f'restart_from = "../{prev}/restart.nc"\n' if prev else ""
    write(f"{run}/{seg}", common(
        256, 128, 512, None, 1.13137085, 1.13137085, 4.44089e-12, 4, 8.7e-16, 4, init, restart, False,
        f"# Stage 3d D2: {run}, segment to curvy3d time t = {tend} (FLUCS time t - {T0}); from curvy3d A3a-Z t = {T0}\n"
        f"# (snapshot index 6, high state). Padded 256 x 128 x 512 = 171 x 85 x 341 modes. No Q stop; dt < 1e-5 for 5 R/vA."))
    prev = seg
