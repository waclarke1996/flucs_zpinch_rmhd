"""Shared helpers for the ZPinchFourier Stage 2 tests (run on the GPU node)."""
import os
import pathlib as pl
import subprocess
import sys
import warnings

warnings.filterwarnings("ignore")
import netCDF4  # noqa: E402
import numpy as np  # noqa: E402

FLUCS = pl.Path(sys.executable).parent / "flucs"

POINTS = {  # kb, kP
    "A1": (0.2, 2.2627417),
    "A2": (0.28284271, 2.2627417),
    "A3a": (1.13137085, 1.13137085),
    "A3b": (1.6, 0.56568542),
    "A4": (0.8, 0.8),
    "S": (0.4, 0.4),
}
SPEC_MODES = [(0, 1, 1), (0, 1, 0), (0.5, 1, 1), (1, 1, 2), (0, 3, 1), (2, 1, 1)]
SPEC_TABLE = {  # gamma, None = stable
    "A1": [None, 0.48829, None, None, None, None],
    "A2": [0.26313, 0.63093, 0.07635, None, 0.26313, None],
    "A3a": [0.71744, 0.66085, 0.60060, None, 0.71744, None],
    "A3b": [0.44940, None, 0.34944, None, 0.44940, None],
    "A4": [0.24918, None, 0.07596, None, 0.24918, None],
    "S": [None] * 6,
}


def omega2_roots(kb, kP, kx, ky, kz):
    """SPEC quartic: roots in omega^2 (sorted, real)."""
    c = 1 + 1 / kb
    kap = ky**2 / (kx**2 + ky**2)
    N2 = 2 * kb / (1 + kb) * (kP * (1 + kb) - 2)
    N02 = 2 * kb * kP
    a, b, cc = c, kap * c * N2 - (1 + c) * kz**2, kz**2 * (kz**2 - kap * N02)
    disc = np.sqrt(b * b - 4 * a * cc + 0j)
    return np.sort(np.real([(-b - disc) / (2 * a), (-b + disc) / (2 * a)]))


def gamma_theory(kb, kP, kx, ky, kz):
    r = omega2_roots(kb, kP, kx, ky, kz)
    return np.sqrt(-r[0]) if r[0] < 0 else 0.0


def write_input(run_dir, *, kb, kP, n=32, linear=False, nu=None, power=2,
                dt_max=0.01, tfinal=1.0, max_cfl=0.5, method="ssprk3e",
                init=None, save_steps=1, energy_opts=None, locations=None,
                loc_steps=1, dt_mult_increase=None,
                restart=None):
    run_dir = pl.Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    init = init or {"method": "white_noise", "amplitude": 1e-2, "rand_seed": 257}
    lines = [
        "[setup]", 'system = "ZPinchFourier"', 'solver = "FourierSolver"',
        'precision = "double"', f"linear = {str(linear).lower()}", "",
        "[dimensions]", f"nx = {n}", f"ny = {n}", f"nz = {n}", "",
        "[parameters]", f"kb = {kb!r}", f"kP = {kP!r}", "",
        "[timestepping]", f'method = "{method}"', "",
        "[time]", f"dt_max = {dt_max!r}", f"tfinal = {tfinal!r}",
        f"max_cfl = {max_cfl!r}",
    ]
    if dt_mult_increase is not None:
        lines.append(f"dt_mult_increase = {dt_mult_increase!r}")
    if nu is not None:
        lines += ["", "[hyperdissipation]", f"kperp = {nu[0]!r}",
                  f"kperp_power = {power}", f"kz = {nu[1]!r}", f"kz_power = {power}"]
    lines += ["", "[init]"]
    for k, v in init.items():
        if isinstance(v, bool):
            v = str(v).lower()
            lines.append(f"{k} = {v}")
        else:
            lines.append(f"{k} = {v!r}" if not isinstance(v, str) else f'{k} = "{v}"')
    opts = energy_opts or {}
    opt_s = ", ".join(f"{k} = {str(v).lower()}" for k, v in opts.items())
    lines += ["", "[output]", "write_steps = 50", "", "[output.0d]",
              f"save_steps = {save_steps}",
              f'diags = [{{name = "energy", options = {{{opt_s}}}}}]']
    if locations:
        locs = ", ".join(f'"{s}"' for s in locations)
        lines += ["", "[output.3d]", f"save_steps = {loc_steps}",
                  f'diags = [{{name = "fourier_data", options = {{locations = [{locs}]}}}}]']
    restart = restart or {"write_restart_file": False}
    lines += ["", "[restart]"]
    for k, v in restart.items():
        lines.append(f"{k} = {str(v).lower()}" if isinstance(v, bool) else f'{k} = "{v}"'
                     if isinstance(v, str) else f"{k} = {v!r}")
    lines.append("")
    (run_dir / "input.toml").write_text("\n".join(lines).replace("'", '"'))
    return run_dir


def run(run_dir, quiet=True):
    """Run flucs in run_dir (we are already on the GPU node)."""
    run_dir = pl.Path(run_dir)
    for f in run_dir.glob("output*"):
        f.unlink()
    with open(run_dir / "run.log", "w") as log:
        p = subprocess.run([sys.executable, "-W", "ignore", str(FLUCS), "--run",
                            "--io_path", "."], cwd=run_dir, stdout=log,
                           stderr=subprocess.STDOUT, env=os.environ)
    if p.returncode != 0:
        raise RuntimeError(f"flucs failed in {run_dir}, see run.log")


def read0d(run_dir):
    g = netCDF4.Dataset(f"{run_dir}/output.0d.nc")["0"]
    e = g["energy"]
    v = {k: np.asarray(e[k][:], dtype=float) for k in e.variables}
    v["t"] = np.asarray(g["time"][:], dtype=float)
    return v


def read_modes(run_dir):
    """Returns t and dict loc_name -> complex array (nt, field, kz, kx, ky)."""
    ds = netCDF4.Dataset(f"{run_dir}/output.3d.nc")
    g = ds["0"] if "0" in ds.groups else ds
    t = np.asarray(g["time"][:], dtype=float)
    out = {}
    fd = g["fourier_data"]
    for name, grp in fd.groups.items():
        out[name] = (np.asarray(grp["data_real"][:], dtype=float)
                     + 1j * np.asarray(grp["data_imag"][:], dtype=float))
    return t, out
