#!/usr/bin/env python3
"""
Stage 4a input generator. Production-style input blocks (tiers T1/T2/T3, run-time diagnostics a-f, optional g/h)
and the overhead test runs. Refuses to overwrite existing folders.
    python make_inputs.py overhead        -> overhead/{base,af,ah}/input.toml   (T2, A3a seed t=1200, 10 R/vA)
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TIERS = {   # padded grid, nu_z (k^8 both directions, nu_perp 4e-11)
    "T1": dict(nx=256, ny=128, nz=256, nu_z=7.5e-14),
    "T2": dict(nx=256, ny=128, nz=512, nu_z=5.9e-16),
    "T3": dict(nx=256, ny=128, nz=768, nu_z=3.4e-17),
}
NU_PERP = 4e-11

ENERGY = '{name = "energy", options = {invariants = true, stop_dt_min = 1e-05, stop_persist_time = 5.0}}'
DIAG_AF = f"""
[output.0d]
save_time = 0.1
diags = [{ENERGY}, {{name = "zp_scalars"}}]

[output.1d]
save_time = 1.0
diags = [{{name = "zp_spectra"}}, {{name = "zp_profiles"}}]

[output.maps]
save_time = 5.0
sample_time = 0.5
diags = [{{name = "zp_maps"}}]

[output.kpar]
save_time = 5.0
diags = [{{name = "zp_kpar"}}]

[output.slices]
save_time = 10.0
diags = [{{name = "zp_slices"}}]
"""
DIAG_GH = """
[output.kyt]
save_time = 1.0
diags = [{name = "zp_kytransfer"}]

[output.kytavg]
save_time = 25.0
sample_time = 1.0
diags = [{name = "zp_streamer_kzkx"}]

[output.shells]
save_time = 5.0
diags = [{name = "zp_shells"}]
"""
DIAG_BASE = f"""
[output.0d]
save_time = 0.1
diags = [{ENERGY}]
"""


def toml(comment, tier, kb, kP, tfinal, diags, init, restart):
    T = TIERS[tier]
    return f"""# {comment}
[setup]
system = "ZPinchFourier"
solver = "FourierSolver"
precision = "double"
linear = false

[dimensions]
nx = {T['nx']}
ny = {T['ny']}
nz = {T['nz']}

[parameters]
kb = {kb!r}
kP = {kP!r}

[timestepping]
method = "ssprk3e"

[time]
dt_max = 0.01
tfinal = {tfinal!r}  # duration of this run/segment
max_cfl = 0.5

[hyperdissipation]
kperp = {NU_PERP!r}
kperp_power = 4
kz = {T['nu_z']!r}
kz_power = 4

[init]
{init}
[output]
write_steps = 500
{diags}
[restart]
{restart}"""


def write(path, text):
    if os.path.exists(path):
        sys.exit(f"{path} exists: refusing to overwrite")
    os.makedirs(os.path.dirname(path))
    with open(path, "w") as fh:
        fh.write(text)
    print("wrote", path)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else ""
    if what == "overhead":
        seed = "../../../stage3e/A3a-T2-k8/t1200/restart.nc"
        init = 'method = "white_noise"  # ignored: restart\namplitude = 1e-05\nrand_seed = 257\n'
        rst = f'restart_from = "{seed}"\nwrite_restart_file = false\n'
        for name, diags in (("base", DIAG_BASE), ("af", DIAG_AF), ("ah", DIAG_AF + DIAG_GH)):
            write(os.path.join(HERE, "overhead", name, "input.toml"),
                  toml(f"Stage 4a overhead test '{name}': A3a T2 from t=1200, 10 R/vA", "T2",
                       1.13137085, 1.13137085, 10.0, diags, init, rst))
    elif what == "speed":
        # STEP 4: ~2000 steps each, production diagnostics a-f, restart from Stage 3e states (no restart written).
        init = 'method = "white_noise"  # ignored: restart\namplitude = 1e-05\nrand_seed = 257\n'
        seeds = {"T1": ("../../../../stage3e/A2-T1-k8/t0500/restart.nc", 0.28284271, 2.2627417, 1.5),
                 "T2": ("../../../../stage3e/A3a-T2-k8/t1200/restart.nc", 1.13137085, 1.13137085, 0.7)}

        def mk(path, tier):
            seed, kb, kP, tf = seeds[tier]
            write(path, toml(f"Stage 4a speed test {tier} (~2000 steps)", tier, kb, kP, tf, DIAG_AF, init,
                             f'restart_from = "{seed}"\nwrite_restart_file = false\n'))

        for gpu in ("h200", "a100", "rtx"):
            for tier in ("T1", "T2"):
                mk(os.path.join(HERE, "speed", gpu, f"single_{tier}", "input.toml"), tier)
        for gpu in ("h200", "a100"):
            for tier in ("T1", "T2"):
                for c in "ab":
                    mk(os.path.join(HERE, "speed", gpu, "pair", f"{tier}{c}", "input.toml"), tier)
            with open(os.path.join(HERE, "speed", gpu, "pair", "run.sh"), "w") as fh:
                fh.write("#!/bin/bash\n# two FLUCS runs at once on one GPU: T1 pair, then T2 pair\n"
                         "FLUCS=~/flucs_work/.venv/bin/flucs\nhere=$(pwd)\n"
                         "for tier in T1 T2; do\n"
                         "  for c in a b; do (cd $here/$tier$c && $FLUCS --run --io_path . > run.log 2>&1) & done\n"
                         "  wait; echo \"== $tier pair done $(date)\"\n"
                         "  for c in a b; do echo \"   $tier$c $(grep -o 'given in.*' $here/$tier$c/run.log)\"; done\n"
                         "done\n")
    else:
        sys.exit(__doc__)
