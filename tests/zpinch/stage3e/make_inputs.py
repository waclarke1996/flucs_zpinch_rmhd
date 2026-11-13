#!/usr/bin/env python3
"""Stage 3e part 1: write the input.toml files of the first tier runs (k^8, power 4 in k_perp and k_z, box 4pi x 2pi x 2pi).
  T2: padded 256 x 128 x 512 (171 x 85 x 341 modes), nu_perp 4e-11 (edge rate nu k^8 = 387 at k_perp 42),
      nu_z 5.9e-16 (412 = 2.4 omega_A at k_z 170).
  T1: padded 256 x 128 x 256 (171 x 85 x 171 modes), nu_perp 4e-11, nu_z 7.5e-14 (204 = 2.4 omega_A at k_z 85).
  E2 A3a-T2-k8 t = 0-1200; E3 A3b-T2-k8, A4-T2-k8 t = 0-500; E4 A2-T1-k8, A2-T2-k8 t = 0-500.
  All: plugin white noise 1e-5 (seed 257), 50-R/vA restart segments (restart.nc = 3D state every 50), no Q stop,
  stop if dt < 1e-5 for 5 R/vA (run.sh then stops the chain). Parameters from reference/curvy3d/campaign_BP2_anchors_v2.csv.
Run once; refuses to overwrite existing folders."""
import os
import shutil

S = os.path.dirname(os.path.abspath(__file__))
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



TIERS = {"T1": (256, 128, 256, 4e-11, 7.5e-14), "T2": (256, 128, 512, 4e-11, 5.9e-16)}
POINTS = {"A2": (0.28284271, 2.2627417), "A3a": (1.13137085, 1.13137085), "A3b": (1.6, 0.56568542), "A4": (0.8, 0.8)}
RUNS = [("A3a", "T2", 1200), ("A3b", "T2", 500), ("A4", "T2", 500), ("A2", "T1", 500), ("A2", "T2", 500)]
for pt, tier, tfin in RUNS:
    nx, ny, nz, nup, nuz = TIERS[tier]
    kb, kp = POINTS[pt]
    run, prev = f"{pt}-{tier}-k8", None
    for tend in range(SEG, tfin + 1, SEG):
        seg = f"t{tend:04d}"
        init = ('method = "white_noise"' + ("  # ignored: restart" if prev else "") + "\namplitude = 1e-05\nrand_seed = 257\n")
        restart = f'restart_from = "../{prev}/restart.nc"\n' if prev else ""
        write(f"{run}/{seg}", common(
            nx, ny, nz, None, kb, kp, nup, 4, nuz, 4, init, restart, False,
            f"# Stage 3e: {run} ({pt}: kb {kb}, kP {kp}; tier {tier}), segment to t = {tend}. Box 4pi x 2pi x 2pi (defaults).\n"
            f"# Padded {nx} x {ny} x {nz} = {(nx-1)//3*2+1} x {ny//3*2+1} x {(nz-1)//3*2+1} modes. k^8: nu_perp {nup}, nu_z {nuz}. "
            f"No Q stop; dt < 1e-5 for 5 R/vA."))
        prev = seg
    shutil.copy(os.path.join(S, "../stage3d/A3a-Z-k8-C300/run.sh"), os.path.join(S, run, "run.sh"))
