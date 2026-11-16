# NOTES: how the Z-pinch system maps onto FLUCS (Stage 0, 2026-11-04)

## How FLUCS defines a system (read from flucs/ @879f8a2 and flucs_krehm/)
- Systems are plugins: entry point group `flucs.systems` in pyproject.toml (KREHM: `flucs_krehm/pyproject.toml`).
  Base flucs ships none. Branch `zpinch` exists, identical to main.
- A Fourier system = subclass of `FourierSystem` (`solvers/fourier/fourier_system.py`) + same-stem `.cu` (kernels) +
  same-stem `.toml` (defaults). `-I src/` is added, so the .cu includes `flucs/solvers/fourier/fourier_system.cuh`.
- The .cu supplies `get_linear_matrix` (per mode, d_t f = L f + NL), `add_nonlinear_terms` (from dealiased "bits",
  times `DFT_FULLSIZE_FACTOR`), and system kernels `find_derivatives` (Fourier, n_in arrays) and `find_nonlinear_bits`
  (real-space products, n_out arrays, plus CFL via `update_cfl`). Python wires them with
  `create_dealiased_operation(...)` (copy KREHM `register_kernels`). `compute_linear_matrix_reference` (numpy) is
  checked against the GPU matrix at start-up; `LinearEigensystemDiag` writes eigenvalues of L for all modes.
- Time stepping: linear part by [N,N] Padé exponential (`linear_pade_degree`=4); NL explicit: ab3, rk4, ssprk3,
  ssprk3e (recommended). NO IMEX ARS(3,4,3) (difference from curvy3d). Adaptive dt from CFL (`time.max_cfl`, `dt_max`).
- Hyperdissipation built in: `[hyperdissipation] kperp, kperp_power, kz, kz_power` (hyper = nu k^(2*power)),
  applied to ALL fields as exp(-dt*hyper) -> gamma - nu_perp k_perp^(2n) - nu_z k_z^(2m) exactly.
- Diagnostics: `FlucsDiagnostic` + `FourierReductions.get_reduction(functor=...)` on device functors in the .cu.
  Reductions sum the half plane with x2 for k_y>0 = full Parseval sum = BOX AVERAGE (no curvy3d factor 1/2).
- Run: `flucs --run --io_path <dir>` (dir holds input.toml); `flucs --init <System>` writes defaults.

## Files to add (all in flucs/, branch zpinch)
`src/flucs/systems/zpinch_fourier/{__init__.py, zpinch_fourier.py, zpinch_fourier.cu, zpinch_fourier.toml,
zpinch_fourier_diagnostics.py}`; entry point in `flucs/pyproject.toml`:
`ZPinchFourier = "flucs.systems.zpinch_fourier.zpinch_fourier:ZPinchFourier"`; `tests/zpinch/` (inputs + checks).
No solver-core changes expected.

## Term mapping (fields 0 Phi, 1 Psi, 2 dBpar, 3 upar; c = 1+1/kb, G = kP(1+kb)-2; d -> i k)
- Linear matrix rows: Phi: [0, i kz, -2 i ky/kperp^2 (0 at kperp=0), 0]; Psi: [i kz, 0, 0, 0];
  dBpar: [i ky G/c, 0, 0, i kz/c]; upar: [0, -i ky kb kP, i kz, 0]. Phi evolved directly (vorticity eq / -kperp^2).
- Nonlinear: 8 derivatives in (dxPhi, dyPhi, dxPsi, dyPsi, w=lap Phi, j=lap Psi, dBpar, upar), 7 bits out, with
  {f,g} = dy(dxf g) - dx(dyf g):  P = dxPsi dyPhi - dyPsi dxPhi -> NL_Psi = {Psi,Phi};
  A_{x,y} = d_{x,y}Phi w - d_{x,y}Psi j -> NL_Phi = -(i ky A_x - i kx A_y)/(-kperp^2);
  B_{x,y} = d_{x,y}Phi dB - d_{x,y}Psi upar/c -> NL_dB = -(i ky B_x - i kx B_y);
  C_{x,y} = d_{x,y}Phi upar - d_{x,y}Psi dB -> NL_u = -(i ky C_x - i kx C_y).  15 FFTs per evaluation.
- CFL: |u_perp| + |dB_perp| (as KREHM; matches curvy3d field-line advection).
- Dissipation: built-in section above (kperp = nu_perp, kz = nu_z, power 2 or 4).
- Parameters: `[parameters] kb, kP`; `[dimensions] Lx = 4 pi, Ly = Lz = 2 pi`.
- Diagnostic `energy` (time series): E and 4 parts, kP*Qhat, D per component, dE/dt (FD), budget residual,
  instantaneous NL energy transfer (~roundoff), Q/Q_MHD = Qhat/Ly^2/(2 kb kP)^1.5, <dBpar^2>, <|dB_perp|^2>; dt is
  already a column. 1D diag: E(kperp) shells (`_compute_kperp_shells`).
- Initial conditions: `single_mode` (list of (kx,ky,kz), optional exact eigenvector), `kx0_noise`, else solver
  `white_noise` (curvy3d: amplitude 1e-5).

## Stage 1 findings (2026-11-04)
- SIGN CONVENTION (corrects the term mapping above): FLUCS integrates d_t f = -M f - N, where M = get_linear_matrix
  (Pade of exp(-M dt): lhs = I + A/2, rhs = I - A/2) and N = add_nonlinear_terms (stages use u - dt*E). So M = -L.
  The NL bits above were already in this convention; the linear rows above are L and are negated in the .cu.
  The start-up eigenvalue check cannot see this (spectrum of L is +-symmetric); the energy budget did (nu=0 run gave
  dE/dt = -kP Qhat before the fix). `compute_linear_matrix_reference` returns M; `compute_physical_linear_operator` L.
- Grid: padded nx, ny, nz = 216, 108, 256 -> 143 x 71 x 171 modes (FLUCS: x = n//3, minus 1 if 3|n; n_unpadded = 2x+1).
- Fourier normalisation: both codes store X_k with f(x) = sum_k X_k exp(+i k.x) (unnormalised c2r; r2c / full padded
  size). FLUCS stores the padded layout (NZ, NX, NY/2+1), curvy3d the compact unpadded one (171, 143, 36).
- GPU of allocation 13099820 (gpu08, device 0): NVIDIA RTX 6000 Ada (48 GB), not H200/A100; curvy3d A2-R0 ran on an
  A100 80GB of the same node. Stage 3 speed comparison must account for this.

## (a) k_perp = 0 modes (k_x = k_y = 0, all k_z)
- curvy3d: `wavenumber_mask` (solver l.406-410) is NOT a dynamical mask: it marks the independent half of Fourier space
  for sums (zeroes kx<0 at ky=0, kz<0 at kx=ky=0, and k=0). At kperp=0 it sets eta = 1 instead of -kperp^2 (l.638-644,
  "to avoid NaNs"), so its Phi row becomes the spurious d_t Phi = i kz Psi (plus dissipation). All four fields are
  zeroed at [:,0,0] in the white-noise init (l.1133-1138, 1158-1161), and every NL term in combine_NL_kernel is
  dy*X - dx*Y (l.1449-1510), i.e. identically 0 at kperp=0. => all kperp=0 modes stay exactly 0 for all time.
  Confirmed on A2-R0 t=0 snapshot: max |X[:,0,0]| = 0 for all fields.
- ZPinchFourier: same outcome. Init zeroes fields[:, :, 0, 0] (all methods except restart); add_nonlinear_terms
  returns early at kperp=0; Phi row of M is 0 there (vorticity eq. is 0=0; no division). Psi row (i kz Phi) and the
  dBpar-upar sound-wave coupling are kept but act on zeros. Dissipation acts on them too (no effect).

## (b) CFL definition
- curvy3d (solver l.1297-1307, 2306-2307): cfl_rate = max_x [ (|dxPhi| + |dxPsi|) ny/Ly + (|dyPhi| + |dyPsi|) nx/Lx ],
  nx, ny = unpadded mode counts (143, 71), evaluated on the padded real-space grid in the first NL evaluation;
  dt = min(dt_max, maxcfl/cfl_rate, 0.1*dt_cfl + 0.9*dt_prev) (time_alpha = 0.1 smoothing on increases).
- ZPinchFourier find_nonlinear_bits: identical expression (NX_UNPADDED/LX, NY_UNPADDED/LY), same padded grid;
  dt = min(dt_max, max_cfl/cfl_rate, 1.10*dt_prev) (`time.dt_mult_increase`).
- => FACTOR = 1: use max_cfl = 0.5 to match curvy3d maxcfl = 0.5. Remaining differences: dt growth limiter (1.10x
  per step vs exponential smoothing), and stability of ssprk3e vs ARS(3,4,3) at the same CFL number.

## Environment / open points
- Will's python3.10: cupy 13.0, numpy 1.26, netCDF4, scipy, matplotlib; MISSING toml, tornado -> Stage 1: `.venv`
  with --system-site-packages, `pip install toml tornado`, `pip install -e flucs --no-deps`.
  DONE (Stage 1): Will's pyenv is itself a venv, so ~/flucs_work/.venv gets his site-packages via
  `.venv/lib/python3.10/site-packages/zz_will_pyenv.pth`; installed toml, tornado, nvmath-python[cu12] 1.0.0
  (cuda-bindings 12.9; nvmath is a hard import of FLUCS); flucs via `pip install --no-deps --no-build-isolation -e`.
  Run: `~/flucs_work/gpu.sh ~/flucs_work/.venv/bin/python -W ignore ~/flucs_work/.venv/bin/flucs --run --io_path .`
  (CuPy "multiple packages" warning comes from Will's env; harmless). Output files appear on the login node with
  a few seconds' NFS delay.
- To check in Stage 1: FLUCS nx/ny/nz convention (padded vs modes) for 143x71x171 modes
  (`_setup_two_thirds_dealiasing`); sign of `dx_from_ikx` (+i kx); GPU type on gpu08.

## For Toby Adkins: core `white_noise` initial condition draws phases on [0, pi) only (Stage 3b, 2026-11-06)
- Where: `flucs/src/flucs/solvers/fourier/fourier_system.py`, `FourierSystem._set_initial_conditions`, case
  `"white_noise"` (flucs @879f8a2, l. ~2076): `solved_fields = np.exp(1j * np.pi * np.random.random(...))`.
- Effect: the phases are uniform on [0, pi), not [0, 2 pi), so <exp(i phase)> = 2i/pi != 0: every mode has a common
  imaginary part. In real space this is a coherent spike at the origin (32^3 test: peak/rms = 45 at x ~ 0 vs 3.9 for
  [0, 2 pi) phases; |<X/|X|>| = 0.58 vs 0.007 ~ 1/sqrt(N)). At 256x128x256 with 1e-5 per mode the spike gives a CFL
  rate ~ 9e3 (dt 5.6e-5 at step 10); k^4 dissipation removes it by t ~ 0.01, k^8 cannot (Z-pinch run S3.3b failed).
- Second, smaller issue: `_check_initial_conditions` then replaces the ky = 0 plane by 0.5 (f + conj f(-k)), which for
  independent random phases halves the mean energy there and leaves |X| != amplitude (not a bug, but the noise is
  not "amplitude per mode" on ky = 0).
- Suggested fix: `np.exp(2j * np.pi * ...)`; optionally draw one phase per (k, -k) pair on ky = 0 and set the partner to
  its conjugate (then the projection is a no-op).
- Workaround used here (no core change): `ZPinchFourier._white_noise` (plugin) does exactly that; the old behaviour is
  kept as `init.method = "white_noise_core"` to reproduce runs S3.1 / S3.3a / S3.3b (which used the core noise).
