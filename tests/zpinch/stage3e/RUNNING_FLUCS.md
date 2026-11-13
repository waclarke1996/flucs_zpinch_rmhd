# Running the Z-pinch system in FLUCS (quick guide)

Everything below runs from `~/flucs_work`. FLUCS code: `flucs/` (branch `zpinch`); the system is
`flucs/src/flucs/systems/zpinch_fourier/` (defaults in `zpinch_fourier.toml`, generic ones in
`flucs/src/flucs/systems/flucs_system.toml`). Python for analysis: `PY=~/flucs_work/.venv/bin/python` (has tomli, netCDF4, matplotlib, cupy).

## 1. input.toml

Minimal production example (tier T2 at A3a, white noise, one 50-R/vA segment). Comments list the alternatives used so far.

```toml
[setup]
system = "ZPinchFourier"
solver = "FourierSolver"
precision = "double"
linear = false                 # true = linearised system (Stage 2 tests)

[dimensions]
nx = 256                       # PADDED grid (curvy3d convention): modes = 2*((n-1)//3)+1 in x, z; ny//3+1 in k_y >= 0
ny = 128                       #   256 x 128 x 512 -> 171 x 85 x 341 modes (T2);  256 x 128 x 256 -> 171 x 85 x 171 (T1)
nz = 512
# Lx = 12.566370614359172      # defaults: Lx = 4 pi, Ly = Lz = 2 pi (anchor box). Square box: Lx = 6.283185307179586

[parameters]
kb = 1.13137085                # A2 0.28284271 / 2.2627417;  A3b 1.6 / 0.56568542;  A4 0.8 / 0.8
kP = 1.13137085

[timestepping]
method = "ssprk3e"

[time]
dt_max = 0.01
tfinal = 50.0                  # = DURATION of this run/segment (time continues from a restart)
max_cfl = 0.5

[hyperdissipation]             # rate nu * k^(2 power) on each field
kperp = 4e-11                  # k^8: T1/T2 4e-11; curvy3d Y grid 4.44089e-12;  k^4 (power 2): 2e-4 (= R0)
kperp_power = 4
kz = 5.9e-16                   # T2 5.9e-16, T1 7.5e-14;  k^4 (power 2): 8e-6 (= R0)
kz_power = 4

[init]
method = "white_noise"         # "plugin" noise, as curvy3d; amplitude = per-mode amplitude
amplitude = 1e-05
rand_seed = 257
# method = "single_mode"; modes = [[kx, ky, kz], ...] (ky >= 0); eigenvector = true      (linear tests)
# method = "streamer"; energy = 1.0; noise_energy = 1e-6                                   (streamer eigenmode + noise)
# method = "curvy3d_snapshot"; file = "/path/to/curvy3d.nc"; index = 6                     (start from a curvy3d snapshot;
#                                                           index = snapshot number, t = index * save_time; grids must match)

[output]
write_steps = 500              # steps between flushes to disk

[output.0d]                    # -> output.0d.nc, every 10 steps
save_steps = 10
diags = [{name = "energy", options = {invariants = true, stop_dt_min = 1e-05, stop_persist_time = 5.0}}]
# energy options: invariants (W_zeta+-, W_Z+- budgets), save_nonlinear (NL transfer; one extra NL evaluation),
#   stop_q_over_qmhd = 1000.0 (runaway stop), stop_dt_min = 1e-5, stop_persist_time = 5.0 (criterion must hold 5 R/vA),
#   kx_split = true (E_kx0, E_kxn0, E_kz0 and kx=0 parts; default on), stop_kxn0_over_kx0 = <ratio> (secondary-instability tests)

[output.1d]                    # -> output.1d.nc: shell spectra of E, its parts, Q and D
save_steps = 500
diags = [{name = "energy_1d"}]  # options = {spectra = ["kperp", "kz", "kx", "ky"]}  (default ["kperp"])

[restart]
# restart_from = "../t0050/restart.nc"   # continue from another run's final state (3D state, also converter input)
write_restart_file = true
write_steps = 20000
```

Long runs = a chain of 50-R/vA segments `RUN/t0050, RUN/t0100, ...`, each restarting from the previous one, plus a
`RUN/run.sh` that runs them in order and stops the chain after an "Early stop". Generate them with a script; template:
`runs/stage3e/make_inputs.py` (edit RUNS/TIERS/POINTS, run once; it refuses to overwrite folders).

## 2. Run it on the queue

```bash
~/flucs_work/queue.sh submit runs/myrun                 # folder with input.toml (single run) or run.sh (chain / any GPU script)
~/flucs_work/queue.sh submit runs/a runs/b              # one job, a then b
~/flucs_work/queue.sh status                            # state, progress lines, last line of each run.log (max 3 active jobs)
tail -n 3 runs/myrun/t0050/run.log                      # time, dt, wall-time estimate; ends with "flucs given in ... (N steps)"
```
Direct (GPU node): `cd RUN && ~/flucs_work/.venv/bin/flucs --run --io_path . > run.log 2>&1`.

## 3. Read output.0d.nc

Variables are in group `0/energy` (time in group `0`). Box averages; Q_over_QMHD = Qhat/L_y^2/Q_MHD.
```python
from netCDF4 import Dataset
d = Dataset("runs/myrun/t0050/output.0d.nc"); g = d["0"]; e = g["energy"]
t, q, E, kx0 = g["time"][:], e["Q_over_QMHD"][:], e["E"][:], e["E_kx0"][:]
# also: E_uperp E_bperp E_dbpar E_upar Qhat Qhat_es Qhat_em injection D D_kperp D_kz dEdt budget_error dt (group 0)
#       dbpar2 dbperp2 dby2, E_kxn0 E_kz0 f_kx0_<part>, zplus/zminus/Zpplus/Zpminus (+ _inj _exch _diss _budget_error)
```
For a chain, concatenate the segments' series (each segment's time continues from the previous one).

## 4. Convert FLUCS states to curvy3d format

```bash
$PY runs/tools/flucs_to_curvy3d.py runs/myrun/t0[1-5]*0/restart.nc --out runs/myrun_state/myrun.nc --run-id myrun \
    --series $(ls runs/myrun/t*/output.0d.nc | paste -sd,) [--t-offset 300]    # offset if FLUCS t=0 was curvy3d t=300
```
Snapshot of time t goes to index t/save_time (default 50). Grid, box, kP, kb and hyperviscosity are taken from the
states and written into the file's `input` string, so the curvy3d diagnostics read it unchanged. ~150 MB per T2 state.

## 5. Diagnostics

```bash
C=reference/curvy3d
$PY $C/dissipation_map.py runs/myrun_state/myrun.nc --tmin 300 --tmax 500 --out runs/myrun_state/dmap_300-500.png \
    > runs/myrun_state/dmap_300-500.txt          # E, b_rms, E_kx0/E per snapshot; where D sits in (k_perp, k_z); cut-offs
$PY $C/kpar_spectrum.py runs/myrun_state/myrun.nc --tmin 300 --tmax 500 [--gpu] --out runs/myrun_state/kpar_300-500.png \
    > runs/myrun_state/kpar_300-500.txt          # per-shell kz_rms, k_par, k_par,wr, chi; --gpu needs a GPU (queue run.sh)
$PY runs/tools/two_state.py stage3e              # two-state (H/L) table + plot into runs/stage3e/; add the run to the
                                                 # RUNS dict at the top first: "F:name": [("f", path/to/output.0d.nc, t_offset), ...]
```
Average dissipation_map / kpar_spectrum only over snapshots of ONE state (H or L): mixing states smears the cut-offs.
Zonal/streamer fractions per snapshot (E_kx0/E, E_ky0/E, zonal shear): `runs/stage3e/e1_zonal.py` (edit RUNS).

All of the above for one segment-chain run, with a pass/fail table per state (runaway, kz_rms at k_d, outer-third
energy / dissipation and pile-up in k_perp and k_z, b_rms, the two cut-offs, two-state stats), written to RUN/analysis.txt:
```bash
$PY runs/stage3e/e_analyse.py runs/stage3e/A3a-T2-k8 [--t0 100] [--t-offset 0] [--no-dmap] [--keep-nc]
```
