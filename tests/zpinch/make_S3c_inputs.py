#!/usr/bin/env python3
"""Stage 3c part 1: write the input.toml of run B (C6 streamer, square box) and A2 (A3a-Y-k8 from curvy3d's t = 300
snapshot, chained 50-R/vA restart segments so that every segment's restart.nc is a full 3D state at t = 350, ..., 600).
Run once; refuses to overwrite existing folders."""
import glob
import os

S = os.path.dirname(os.path.abspath(__file__))
DATA = "/mnt/users/clarkew/data/rmhd_zpinch/output/3D/BP2_anchors_v2"
Y = glob.glob(f"{DATA}/*_RKARS343_A3a-Y.nc")[0]
TWO_PI = "6.283185307179586"


def write(folder, text):
    d = os.path.join(S, folder)
    os.makedirs(d, exist_ok=False)
    with open(os.path.join(d, "input.toml"), "w") as f:
        f.write(text)
    print("wrote", d)


# ---------------- B: C6 = C1 (kb 0.28284271, kP 2.2627417, streamer (0,1,0), E_s 30) in L_x = L_y = 2 pi ----------
DISS = {"k4": (0.0002, 2, 8e-06, 2),                     # as C1
        "k8": (4.44089e-12, 4, 1.38778e-14, 4)}          # Kawazura's coefficients (SPEC.md, Stage 3b C4)
for tag, (nup, p, nuz, q) in DISS.items():
    for lin in (False, True):
        write(f"C6-S30-{tag}{'-lin' if lin else ''}", f"""# Stage 3c B: streamer C6 = C1-S30 but square box L_x = L_y = 2 pi, {tag} dissipation{', LINEAR control' if lin else ''}
[setup]
system = "ZPinchFourier"
solver = "FourierSolver"
precision = "double"
linear = {'true' if lin else 'false'}

[dimensions]
nx = 216
ny = 216
nz = 256
Lx = {TWO_PI}
Ly = {TWO_PI}
Lz = {TWO_PI}

[parameters]
kb = 0.28284271
kP = 2.2627417

[timestepping]
method = "ssprk3e"

[time]
dt_max = 0.01
tfinal = 30.0
max_cfl = 0.5

[hyperdissipation]
kperp = {nup!r}
kperp_power = {p}
kz = {nuz!r}
kz_power = {q}

[init]
method = "streamer"
modes = [[0, 1, 0]]
energy = 30.0
noise_energy = 2.9999999999999997e-05
rand_seed = 257

[output]
write_steps = 50

[output.0d]
save_steps = 5
diags = [{{name = "energy", options = {{stop_kxn0_over_kx0 = 1.0, stop_dt_min = 0.0001}}}}]

[restart]
write_restart_file = false
""")

# ---------------- A2: A3a-Y-k8 from curvy3d A3a-Y snapshot index 6 (t = 300), segments of 50 to t = 600 ----------
T0, SEG, TEND = 300, 50, 600
for run, cfl in (("A3a-Y-k8-C300", 0.5), ("A3a-Y-k8-C300-cfl025", 0.25)):
    prev = None
    for tend in range(T0 + SEG, TEND + 1, SEG):
        seg = f"t{tend:04d}"
        restart = (f'restart_from = "../{prev}/restart.nc"\n' if prev else "")
        write(f"{run}/{seg}", f"""# Stage 3c A2: {run}, segment to curvy3d time t = {tend} (FLUCS time t - {T0}); from curvy3d A3a-Y t = {T0}
# (snapshot index 6, save_time 50). Same physics/grid/diagnostics as A3a-Y-k8-S30c, no Q stop, safety dt < 1e-5.
[setup]
system = "ZPinchFourier"
solver = "FourierSolver"
precision = "double"
linear = false

[dimensions]
nx = 256
ny = 128
nz = 256

[parameters]
kb = 1.13137085
kP = 1.13137085

[timestepping]
method = "ssprk3e"

[time]
dt_max = 0.01
tfinal = {float(SEG)}  # duration of this segment
max_cfl = {cfl}

[hyperdissipation]
kperp = 4.44089e-12
kperp_power = 4
kz = 1.38778e-14
kz_power = 4

[init]
method = "curvy3d_snapshot"
file = "{Y}"
index = 6

[output]
write_steps = 500

[output.0d]
save_steps = 10
diags = [{{name = "energy", options = {{invariants = true, stop_dt_min = 1e-05, stop_persist_time = 5.0}}}}]

[output.1d]
save_steps = 500
diags = [{{name = "energy_1d"}}]

[restart]
{restart}write_restart_file = true
write_steps = 20000
""")
        prev = seg
