# PROGRESS (append-only)


## Stage 0 - read and plan (2026-11-04)
Done: read CLAUDE.md, SPEC.md, PLAN.md; studied flucs/ (FourierSystem, fourier_system.cuh/.toml, Padé + timesteppers,
hyperdissipation, reductions, diagnostics, CLI) and flucs_krehm/ (KREHMFourier .py/.cu/.toml/diagnostics) as example.
Wrote NOTES_flucs.md (design + term mapping) and TESTS_stage2-3.md (test list, commands, pass criteria). No code
changes; nothing to commit (flucs on branch zpinch = main @879f8a2).
Key facts:
- Systems are plugins (entry point `flucs.systems`); plan: add `flucs/src/flucs/systems/zpinch_fourier/` + entry
  point in flucs/pyproject.toml. Needs per-mode linear matrix, find_derivatives / find_nonlinear_bits kernels,
  add_nonlinear_terms (8 derivatives in, 7 bits out, conservative-form brackets).
- FLUCS steps linear terms with a Padé [4,4] exponential and nonlinear terms explicitly (ssprk3e/rk4/ab3), not
  IMEX ARS(3,4,3) as in curvy3d.
- Dissipation reuses built-in [hyperdissipation] (kperp, kz, power 2/4), applied as exp(-dt*nu k^2p) to all fields
  -> exact gamma_diss.
- FLUCS reductions give box averages directly (no curvy3d factor 1/2).
Environment: Will's python3.10 has cupy 13.0, numpy 1.26, netCDF4, scipy, matplotlib but lacks toml and tornado ->
Stage 1 creates ~/flucs_work/.venv (--system-site-packages) and pip-installs them (record if network blocks it).
GPU allocation 13099820 RUNNING on gpu08 (type not yet checked).
Note: plan mode's plan file (~/.claude/plans/) is blocked by the guard hook; plan was given in chat instead.
Open: FLUCS nx/ny/nz convention for 143x71x171 modes; sign of dx_from_ikx.
Next: Stage 1 (venv, implement ZPinchFourier, diagnostics, smoke test 32^3).

## Stage 1 - implement (2026-11-04), flucs zpinch @ f9eafb7
Done: `flucs/src/flucs/systems/zpinch_fourier/{zpinch_fourier.py,.cu,.toml,_diagnostics.py}` + entry point
(pyproject). Fields 0 Phi, 1 Psi, 2 dBpar, 3 upar; [parameters] kb, kP; [dimensions] Lx=4pi, Ly=Lz=2pi default;
dissipation through built-in [hyperdissipation] (kperp/kz, power 2 = k^4, 4 = k^8). 8 derivatives in, 7 conservative
bracket products out (15 FFTs/eval). Diagnostics `energy` (0d: E + 4 parts, Qhat (+es/em), injection kP*Qhat,
Q_over_QMHD, D + per component, dEdt (1-step FD), budget_error, dbpar2, dbperp2, dby2, dt; option save_nonlinear ->
dEdt_nonlinear) and `energy_1d` (kperp spectra of E, parts, Qhat, D). Init: white_noise (solver), single_mode
(eigenvector of fastest-growing root), kx0_noise; kperp=0 modes always zeroed.
Bug found and fixed in the smoke test: FLUCS convention is d_t f = -M f - N (M = get_linear_matrix), so M = -L.
Start-up eigen-check cannot detect this (+-symmetric spectrum); the nu=0 budget showed dE/dt = -kP Qhat. Recorded
in NOTES_flucs.md.
Smoke tests (32^3 padded = 21^3 modes, A2, ssprk3e, max_cfl 0.5, GPU, runs/smoke32*):
- GPU linear matrix == numpy reference; max growth 0.6309 (= SPEC A2 (0,1,0) 0.63093); Pade err rel 4e-12.
- NL energy transfer / injection: 7e-14 (nu = 2e-4/8e-6) and 1e-13 (nu = 0): brackets conserve E.
- Budget dE/dt = kP Qhat - D, central FD of saved E: median rel residual 2e-5 (NL + nu, amplitude 1e-2 noise,
  t=0..0.5, 720 steps), 6e-4 (nu=0), 3e-4 (linear, dt=0.01) - consistent with FD truncation; proper T4 in Stage 2.
- Normalisation: E parts are box averages (E_upar(0) = 0.4476 for 1e-2 noise on 21^3 modes = full-space sum).
- 1.4 ms/step at 32^3 incl. every-step diagnostics.
Tasks (a)-(c) recorded: NOTES_flucs.md (a) kperp=0: curvy3d zeroes all 4 fields at init and its NL terms vanish
there identically -> they stay 0; ZPinchFourier does the same. (b) CFL expressions identical -> factor 1, use
max_cfl = 0.5; only the dt growth limiter differs. TESTS_stage2-3.md: S3.0 with index/field-order/normalisation
mapping (factor 1); the t=0 snapshot is real-valued 1e-5 noise, not Hermitian on ky=0 (FLUCS will project it).
Environment: .venv gets Will's site-packages via a .pth (his interpreter is itself a venv); added toml, tornado,
nvmath-python[cu12] 1.0.0 (hard import of FLUCS). pip network worked.
GPU: the allocation's device is an NVIDIA RTX 6000 Ada 48 GB (gpu08 also has A100s; A2-R0 ran on an A100) ->
Stage 3 speed will not be directly comparable to 11 ms (H200) / 26 ms (A100).
Guard-hook notes: commands containing `cp.`/`install` with a protected path, `<...>` in a commit message, or a shell
cwd under the read-only tree are blocked; use scripts in runs/ with path arguments and `git commit -F`.
Open: implement init "curvy3d_snapshot" (Stage 3, S3.0); the diag's dEdt is 1st order (use offline FD in T4).
Next: Stage 2 (T0-T5).

## Stage 2 - verify against theory (2026-11-04), flucs zpinch @ 5921247
Code: `energy` diag option `invariants = true` -> per invariant zplus, zminus (W_zeta+-), Zpplus, Zpminus (W_Z+-):
W, _inj (prop. kP), _exch (curvature, +-2 terms), _lin = inj+exch (= 2 x curvy3d saved _inj, solver l.3120-3153),
_diss (= 2 x curvy3d _diss), _dWdt, _budget_error, and with save_nonlinear _nl, _nl_gross (sum_k|transfer|).
Identities hold to round-off: E = sum W/2 (4e-16), sum inj/2 = kP Qhat (2e-16), sum exch = 0, sum diss/2 = D.
Scripts `flucs/tests/zpinch/check_T*.py` (zp_common.py); outputs/logs in runs/stage2/. 32^3 unless stated.
- T0 PASS: 64^3 grid (40635 modes/pt, A1-A4, S): char. poly of L vs quartic max rel 5e-15; gamma = max Re eig
  vs quartic max rel 1.3e-12; SPEC table reproduced to <= 5e-6 abs (5-digit rounding).
- T1 PASS (crit 1e-3): eigenvector init, fit log|Phi_k| over 6-50 e-folds, A2/A3a/A3b/A4 at (0,1,1),(0,1,0),
  (0.5,1,1),(0,3,1): max rel err 9e-14 (e.g. A2 0.630932, 0.263129, 0.076346). Stable S(0,1,1), A2(2,1,1): both
  omega roots (0.310746/1.418453, 0.410782/0.985964) from fitted one-step propagator, rel 2e-14. White-noise
  A2: (0,1,0) rel 1.6e-10, (0,1,1) rel 5.9e-5 (fit t=30-60, residual subdominant mode).
- T2 PASS: same modes, k^4 (nu 1e-3/2e-2) and k^8 (1e-5/2e-2): gamma - nu_perp kperp^2n - nu_z kz^2n, max rel
  9e-14; purely damped stable modes: damping rel <= 1e-12, omega rel 3e-14.
- T3 PASS: kP=0, nu=0, A2 kb, white noise E0 = 0.112, t=0..2 fixed dt 4e-3..5e-4: |dE|/E0 ssprk3e 1.5e-7,
  1.9e-8, 2.4e-9, 3.0e-10 (orders 2.99, 3.00, 3.00); rk4 4.7e-10 ... 1.1e-13 (orders 4.47, 4.28, 3.35 = round-off).
- T3b PASS (crit 1e-12): kP=0 run (injection = 0, so normalised by gross transfer): max|nl|/gross <= 4e-16 for
  all four (/|exch| <= 1.3e-12) while each W changes by -8%..+218% via exchange. Turbulent A2 (T4 windows):
  max|nl|/|kP Qhat| <= 1.1e-14 (burst t=20-23), <= 4.5e-14 (t=80-83); /gross <= 1.6e-15.
- T4 FAIL as literally specified, PASS at smaller dt. A2 64^3, k^4 nu 5e-3/1e-3, every-step diag, 5-pt FD of E:
  burst window t=20-23 (Q/Q_MHD ~0.03, D/inj ~1): max_cfl 0.5 (dt 5.6e-3) resid/|kP Qhat| median 2.7e-4, max
  1.7e-3; max_cfl 0.125 (dt 1.4e-3) median 3.5e-6, max 6.6e-6. FD5 = FD2 and resid ~ dt^3-4 => it is the
  ssprk3e truncation (discrete trajectory vs instantaneous RHS), not a missing term; NL part 8e-15. Invariant
  budgets (dW/dt = lin - diss + nl) same behaviour: /own max 6e-4 (cfl 0.5), 5e-6 (cfl 0.125).
  Late window t=80-83 is not turbulent (see below): /|kP Qhat| 5e-3 (cfl 0.5), 1e-4 (0.125); /D 2e-5, 4e-7.
- T5 PASS: A2, nu=0, kx0_noise E0 = 1.73 -> 4.0e3 at t=10: k_x != 0 modes exactly 0 at all 102 saves; k_x = 0
  plane = linear run to 6e-14; propagator-fit gamma(0,1,0) rel 3e-13, (0,1,1) 3e-14.
Findings/notes:
- 64^3 A2 with nu 5e-3/1e-3: after the first burst (Q/Q_MHD 0.08 at t=20) the run settles into a u_perp-
  dominated (zonal-flow-like) state, Q/Q_MHD ~1e-6 for t=40-160, E ~ 10. Physics of the heavy-nu small box, not
  checked further (Stage 3 uses curvy3d's nu).
- FLUCS restart: time.tfinal is a DURATION after restart (restart.py l.131). zplus = zminus, Zpplus = Zpminus to
  4 digits in the turbulent runs (small cross-helicity from noise IC); the 1e-2 smoke run shows them distinct.
- Test-script bug fixed once: T5 fit included the forced final save (non-uniform spacing); re-analysed same data.
Open for Will: T4 criterion 1e-4 at production CFL 0.5 is not met (1.7e-3 max) because of time-step truncation;
accept the cfl-0.125 pass / the dt-scaling, or state another criterion? Per rule 5 I stopped here.
Next: Stage 3 (after Will's decision on T4).

## Stage 2 addendum - T4 decision (2026-11-05)
Will accepted T4: criterion is "residual -> 0 at the scheme's order" (dt^3-dt^4, NL part at round-off).
T4 PASS (cfl 0.5: max 1.7e-3, cfl 0.125: max 6.6e-6 of |kP Qhat|; NL part 8e-15). Stage 2 complete.
Stage 3 GPU: allocation 13102476 on sondhigpu, NVIDIA H200 NVL (CUDA_VISIBLE_DEVICES=1).

## Stage 3 - compare with curvy3d (2026-11-05), flucs zpinch @ cf48a07 (runs), ab7f902 (scripts)
GPU: H200 NVL (alloc 13102476). G1 = padded 216x108x256 (143x71x171 modes), ssprk3e, max_cfl 0.5, dt_max 0.01.
Code: init.method "curvy3d_snapshot" (init.file/index); energy diag options stop_q_over_qmhd, stop_dt_min (tested
32^3). Scripts flucs/tests/zpinch/{curvy_compare,check_S30,check_S31,check_S33a}.py use zpinch_runtools (x2).
Runs/plots: runs/stage3/<run>/, S30.png, S31.png, S33a.png, S33a_S30.png, S33b.png.
- S3.0 PASS (A2-G1-k4-S30, A2-R0 t=0 IC, t=0-60, 14597 steps, 126 s): t=0 E and all 4 invariants = curvy3d to
  2e-16 (ky=0 projection removes nothing; Qhat(0) = 0 in both: real-valued IC). Linear phase t=0.5-17.2:
  max |Q_F/Q_C - 1| = 1.3e-3, E and invariants 1.2-1.4e-3. Q/Q_MHD crosses 1e-2 at t=17.30 in both; first peak
  0.0502 @22.6 (F) vs 0.0512 @22.4 (C); |ln Q_F/Q_C| > 0.1 first at t=22.1 (chaotic decorrelation).
  t=40-60: Q/Q_MHD 0.0117+-0.0028 (F) vs 0.0085+-0.0030 (C). Mean dt t=0-20/20-40/40-60: F 9.2e-3/3.3e-3/3.1e-3,
  C 9.5e-3/3.4e-3/3.7e-3.
- S3.1 PASS (A2-G1-k4, white noise 1e-5 seed 257, t=0-300): <Q/Q_MHD>_60-300 = 0.0777 +- 0.0041 (tau_int 1.85)
  vs 0.078 +- 0.004 (my recomputation from A2-R0: 0.0784 +- 0.0044). <E> 26.5 vs 25.2. E_A(kperp) (unit bins as
  spectrum_check.py): slope k=2-7 mean local -1.68 (F, t=60-300) vs -1.72 (C, t>=125); log-fit -1.42 vs -1.49;
  E_F/E_C = 0.9-1.3 for k=1-12, curves overlay over 12 decades to the dissipation range.
- S3.2: FLUCS 7.05 ms/step (--timing 3000, no output), 7.80 ms/step in the production run (energy diag every 10
  steps): 207371 steps, 26 min 57 s. curvy3d: 11.8 ms (bundle), 10.98 ms (A3a-R0 log, same G1, H200).
  Mean dt t=0-300 (t/steps): F 1.447e-3, C 1.481e-3 (harmonic mean of dt_series); t=60-300 F 1.29e-3, C 1.27e-3.
  => wall time per unit t: F 5.4 s (4.9 s w/o diag) vs C 8.0 s: FLUCS ~1.5-1.6x faster.
- S3.3a (A3a-G1-k4, t<=60, stop Q/Q_MHD>1e3 or dt<1e-4): white-noise run reaches t=60 with NO runaway (112779
  steps, 14.5 min): first saturation ~1.3 @t=23, <Q/Q_MHD> 1.06+-0.27 (t=20-40), 2.99+-0.54 (40-60), max 6.9 @51.8,
  dt >= 1.8e-4; E still rising (900). curvy3d A3a-R0 (.nc empty - never saved; values from its .log): 3.2 @27.2,
  22.5 @31.6, 90 @33.85, 124 @36.4, dt 8.2e-5 @33.85. Same-IC replay (A3a-G1-k4-S30, A3a-R0 t=0 snapshot):
  2.15 @27.2 (C 3.19), dt 3.6e-4 (C 3.9e-4), then decays to 0.34 @31-38 while curvy3d runs away; FLUCS runs away
  later: Q/Q_MHD > 5 @53.0, > 20 @54.8, stopped by dt < 1e-4 at t=55.62 (Q/Q_MHD 44, E 7.2e3, D/inj 0.15),
  budget residual (1-step FD) median 6e-5 of inj. => FLUCS also runs away; onset after the first saturation is
  chaotic (t ~ 32 C vs ~53 F from the same IC; none by t=60 from another IC).
- S3.3b (A3a-Y-k8: padded 256x128x256, k^8, nu 4.44089e-12/1.38778e-14): FAILED at step 10 from FLUCS white
  noise: dt 5.6e-5 < 1e-4. Cause: FLUCS core white_noise uses phases exp(i pi U[0,1)) (fourier_system.py:2076),
  i.e. [0,pi) not [0,2pi) -> nonzero mean phase -> coherent spike at x=0 (E(0) 0.30, CFL rate ~9e3). k^4 damps it
  by t~0.01 (S3.1/S3.3a dt dips to 1.7e-4 at t<0.01, then 0.01); k^8 cannot. Not fixed (core code; Will to decide).
  Rerun from curvy3d A3a-Y's own t=0 snapshot (A3a-Y-k8-S30): saturates, Q/Q_MHD 0.2-5 for t=20-167, then the
  dt rule stopped it at t=167.0 (dt 9.9e-5) during a burst (Q/Q_MHD 5.1 @166.1). curvy3d A3a-Y has a similar
  burst (8.5 @122.3, dt min 1.017e-4 @123.3) and recovers -> this stop is a burst, not evidence of runaway.
  440k steps, 1 h 17 min (10.5 ms/step). Like-for-like <Q/Q_MHD>: t=100-167 F 1.68+-0.58 vs C 2.58+-0.54;
  t=50-100 0.51+-0.09 vs 1.34+-0.55; curvy3d t=191-500 0.633+-0.109 (= Will's 0.63+-0.11), t=100-200 2.37+-0.43.
  Spectrum slope k=2-7 (F 100-167, C 100-500): mean local -1.75 vs -1.71, fit -1.62 vs -1.60; F level 1.5x (burstier
  window). Within Kawazura's 0.05-5.
Open for Will: (1) fix the [0,pi) phase in FLUCS white_noise (upstream) or keep curvy3d-snapshot ICs; (2) continue
A3a-Y-k8-S30 past t=167 from restart.nc with a looser dt stop (e.g. 5e-5) to get t=100-200 / 191-500 statistics.

## Stage 3b - plugin noise, kx split, streamer secondary test (2026-11-06), flucs zpinch @ e854375 (code), 6cce433 (scripts)
GPU: alloc 13105774 (GPU_JOBID was stale at start; Will updated it), H200 NVL. Runs/plots: runs/stage3b/ (S3b_table.txt,
S3b.png, init_tests/), runs/stage3/A3a-Y-k8-S30c/, S33c.png. Scripts tests/zpinch/{check_S3b_init,make_S3b,check_S3b,check_S33c}.py.
(1) init.method "white_noise" now = plugin version: |X| = 1e-5, phase U[0,2pi), ky=0 plane exactly Hermitian (one phase per
  (k,-k) pair), kperp=0 zeroed; old core noise kept as "white_noise_core". 32^3: reality err 0, |<X/|X|>| 0.007 (core 0.58),
  real-space peak/rms 3.9 (core 45, at x=0). k^8 256x128x256: min dt 8.1e-3 over t=0-1 (core noise: 5.6e-5 at step 10).
  Core bug + fix recorded for Toby Adkins in NOTES_flucs.md. Core untouched.
(2) energy diag (default kx_split): E_kx0, E_kxn0, E_kz0, E_kx0_<part>, f_kx0_<part>; stop_kxn0_over_kx0, stop_persist_time.
  E_kx0+E_kxn0 = E (1e-16), parts sum (3e-16), kx0_noise run: E_kxn0 = 0 exactly. New init "streamer" (fastest eigenmode
  at box energy E_s + plugin noise at 1e-6 E_s); E(0) GPU = numpy box energy (2e-16).
(3) Streamer test, 20 runs (NL + linear control, E_s = 3, 30), stop at E_kxn0 > E_kx0 or safety dt < 1e-4 (E_kx0 ~ 1e5).
  t=0 kx0 fractions u_perp/b_perp/u_par/dB_par: C1 .689/0/0/.311, C2 .829/0/0/.171, C3=C4 .165/.320/.170/.346,
  C5 .033/.530/.101/.336 (= Will's numbers). Streamer gamma (NL and lin) = gamma - nu terms to 5 digits in all 20.
  sigma = d ln E(kx!=0)/dt (energy rate). Linear control late-time: C1 1.178, C2/C3 1.333, C4 1.381, C5 0.388.
  | run   | t10%  | t_end (reason)      | sigma_NL peak | sigma at E_kx0=300 | sigma_pk/sqrt(E_kx0) |
  | C1-S3 | 4.21  | 4.29 (kxn0>kx0)     | 20.6          | 15.7               | 0.88 |
  | C1-S30| 2.56  | 2.64 (kxn0>kx0)     | 22.8          | 15.5               | 0.88 |
  | C2-S3 | 3.96  | 4.04 (kxn0>kx0)     | 21.7          | 17.1               | 0.97 |
  | C2-S30| 2.39  | 2.47 (kxn0>kx0)     | 24.1          | 17.0               | 0.96 |
  | C3-S3/S30 | never | 7.26/5.65 (dt)  | 1.1/1.3       | 0.2/0.9            | <=0.14 |
  | C4-S3/S30 | never | 7.01/5.40 (dt)  | 2.0/1.9       | 1.6/1.0            | <=0.08 |
  | C5-S3/S30 | never | 20.6/16.0 (dt)  | 1.2/1.4       | ~0                 | <=0.006 |
  Answer: yes. The kz=0 (electrostatic, 2D) streamer is destroyed by a secondary whose rate is ~0.9 sqrt(E_kx0)
  (proportional to streamer amplitude; same sigma at the same E_kx0 for both E_s; E_s only shifts the time by
  ~ln10/2gamma), 15-24 vs 1.2-1.3 linear; breakup when E_kx0 ~ 300-400. The m=1 streamers (C3, C4, C5) are NOT broken:
  they reach E_kx0 ~ 1e5 (1000x saturated A3a E(kperp=1) ~ 80) with E(kx!=0)/E < 1e-6; NL sigma <= linear control
  (C3: E(kx!=0) decays 3e-6 -> 2e-10). C4 vs C3: k^4 erases the high-k noise at once, k^8 keeps it (5e-8 - 1e-6)
  with weak growth (sigma <= 2); no breakup in either -> dissipation changes the noise floor, not the outcome.
  Caveat: single eigenmode = exact NL solution; only noise-seeded secondaries tested; t <= ~7 for m=1 (dt stop).
(4) A3a-Y-k8-S30c: restart from S30 restart.nc (t=167) to t=500, no stop (stop rule never met). 1.607e6 steps,
  5 h 14 min (11.7 ms/step, incl. invariants + kx split diag; mean dt 2.1e-4). 14 bursts Q/Q_MHD > 5 (max 12.7 @256.0,
  min dt 5.9e-5), all recover. <Q/Q_MHD> t=191-500: FLUCS 3.22 +- 0.44 vs curvy3d 0.633 +- 0.109 (DISAGREE, ~5x, ~6 sigma);
  191-350 3.26+-0.82, 350-500 3.18+-0.29 (C 0.62, 0.65); t=100-200 F 1.53+-0.44 vs C 2.37+-0.43. <E> 1118 vs 205;
  F: E_kx0/E = 0.58 (streamer-dominated), D/inj 1.02, Qes/Q 0.38 (C 0.39). curvy3d has 5 bursts, none after t=178.
Open for Will: (a) A3a-Y-k8 long-time state differs (FLUCS high-Q streamer state vs curvy3d 0.63): bistability or a
  scheme effect (ssprk3e+Pade vs ARS343) with near-zero k^8 dissipation? Suggest: E(kperp) spectra at t>191 both codes,
  kx-split of curvy3d (Stage 4 output), a FLUCS rerun from curvy3d's t~200 state. Not investigated (stage stop).

## Stage 3c part 1 - converter, two-state analysis, B + A2 submitted (2026-11-08), flucs zpinch @ 6cce433 (code), scripts see commit below
Files: runs/tools/{flucs_to_curvy3d,check_convert,two_state}.py (copies in flucs/tests/zpinch/), runs/stage3c/ (two_state.txt/.png,
dmap_*, kpar_*, same_state/table.txt). No code change to FLUCS this stage.
A1 converter: FLUCS restart.nc -> curvy3d-format .nc (compact 2/3 layout, input string = A3a-Y's + overrides, optional heatflux
  series from output.0d.nc). dissipation_map.py / kpar_spectrum.py / zr.heat_flux_series read it unchanged. S30c t=500: E 745.98348,
  Q/Q_MHD 2.872129 = FLUCS diag (rel 0 / 2e-16); S30 t=167 also (4e-16). Energy outside 2/3 set exactly 0. curvy3d E series
  (zplus+zminus+Zpplus+Zpminus)/2 = exactly 1/2 box E (factor 2.000000, all runs). kpar --pad 1.5: ~25 s/snapshot on CPU.
A1 two-state (t>=100; running MEDIAN of Q/Q_MHD over 5 R/vA -- ignores bursts of 1-2 R/vA, a mean does not; high >1.5, low <0.8,
  each >= 5 R/vA; same-state stretches merged across undecided gaps). E-based check (E>400/<250) agrees 98.6-100 %.
  | run              | t        | f_high | Q_H  Q_L | E_H   E_L | kx0/E H L | switches | episodes (longest)      |
  | C A3a-Y+Yc2      | 100-1500 | 0.54   | 3.89 0.47| 1120  157 |    -      | 29       | H 90-99, L 143 (190-333) |
  | C A3a-YP4        | 100-500  | 0.51   | 5.04 0.40| 1456  133 |    -      | 4        | L 171, H 128             |
  | C A3a-Z          | 100-300  | 0.18   | 2.99 0.45|  837  151 |    -      | 4        | L 95                     |
  | C A3a-YHR2       | 100-500  | 0.34   | 3.60 0.51| 1014  167 |    -      | 5        | L 156, H 107             |
  | F S30+S30c       | 100-500  | 0.83   | 3.47 0.38|  993  124 | 0.58 0.29 | 3        | L 40 (~185-225), H >266  |
  Residence times in curvy3d: 5-171 R/vA (many 40-100). NB curvy3d A3a-Y is NOT stuck low: in Yc2 (t=500-1500) it is high ~60 %.
  -> FLUCS shows BOTH states, at the curvy3d levels (Q, E, E_kx0/E). Its 266+ R/vA high stretch is longer than any curvy3d high
  episode (max ~99) but within the spread of the low ones (171); one run cannot fix f_high (few switches).
Same-state snapshots (dissipation_map per snapshot; 12 curvy3d H, 16 L): curvy3d H median E 1030, kx0/E 0.59 [0.41-0.70], f_par
  0.69 [0.61-0.82], alpha 0.32 [0.27-0.42], inj(kperp<1.5) 0.84; L: E 106, kx0/E 0.26, f_par 0.45, alpha 0.43, inj 0.66.
  FLUCS H: t=167 E 1262 kx0/E 0.50 f_par 0.70 alpha 0.30 inj 0.81; t=500 E 746, 0.59, 0.72, 0.38, 0.86 -> inside curvy3d H ranges.
  kpar/kz (shells 1.5-3, 3-6, 6-12): curvy3d H 1.10-1.18, 0.99-1.15, 1.08-1.51; FLUCS 1.24/1.24/1.58 (t167), 1.05/1.06/1.36 (t500).
  Planned A3a-Y t>=400 (low) vs S30c t500 (high): f_par 0.49 vs 0.72, alpha 0.43 vs 0.38 -- a state difference, not a code one.
Submitted (queue.sh, all started 04:50 and stepping; H200):
  13112635 B: runs/stage3c/C6-S30-{k4,k4-lin,k8,k8-lin} (216x216x256, Lx=Ly=2pi, E_s=30, stop kxn0>kx0 or dt<1e-4). <1 h.
     k4 done: dt stop at t=5.05 (no breakup), k4-lin 11 s; k8 running.
  13112636 A2 A3a-Y-k8-C300 (cfl 0.5), 13112637 A3a-Y-k8-C300-cfl025: init curvy3d A3a-Y snapshot 6 (t=300, low state), six chained
     50-R/vA restart segments runs/stage3c/<run>/t0350 ... t0600 (FLUCS time = t - 300; restart.nc of each = 3D state at t, overshoot
     < 1 dt). Diags as S30c, no Q stop, dt<1e-5 for 5 R/vA. Speed: C300 ~10 min/segment in the low state (high state ~2-3x),
     cfl025 ~1 h/segment -> C300 1-3 h, cfl025 ~6-10 h. First steps: C300 t=0.87 E 430 Q 1.50 dt 2.4e-4 (curvy3d t=300: E 355 Q 1.52).
     If a segment stops early, later segments restart from that state (each stops again after 5 R/vA at dt<1e-5).
Part 2 to do: B table vs C1-S30. A2: concatenate segment 0d files (+300), two_state.py on them (add to RUNS) and Q(t) plot vs curvy3d
  A3a-Y+Yc2, first |ln Q_F/Q_C| > 0.1; convert t0450/t0600 restart.nc (--t-offset 300) and run dissipation_map/kpar (one
  snapshot per call, kpar bug). Judge by (ii): same low/high levels = agree; code difference only if levels clearly differ or FLUCS
  never goes low while curvy3d mostly low.

## Stage 3c part 2 - B square-box streamer, A2 restart from curvy3d t=300 (2026-11-12), flucs zpinch @ ad5dd27 (code), scripts see commit
Jobs 13112635-37 all COMPLETED, every segment exit 0. Files: runs/stage3c/{b_table.py, a2_compare.py/.txt, a2_Qt.png, two_state.txt/.png},
runs/stage3c/a2_state/ (converted .nc, dmap_*/kpar_* .txt/.png, kpar_tab.py, kpar_table.txt); copies in flucs/tests/zpinch/.
NB: the curvy3d kpar_spectrum.py md5 is 6482ef9c (NOT ea3360c2) and dissipation_map.py a930735d (NOT bb5014d5): the fixed
  versions are not installed. Used one snapshot per call, saved stdout as .txt myself; chi = old def. k_perp u_s/k_par (not shell
  rms vorticity/k_par); k_z* crossings computed in kpar_tab.py with k_z* = (0.717/nu_z)^(1/8) = 51.8 (gamma as dissipation_map).
B: C6 = C1-S30 in a square box Lx = Ly = 2pi (sigma = d ln E(kx!=0)/dt; lin control late sigma 1.13 (k4) / 1.22 (k8), C1 1.19).
  | run       | sigma_NL peak (t, E_kx0)  | sigma at E_kx0=300 | t_end (reason) | E(kx!=0)/E end |
  | C1-S30    | 23.3 (2.53, 686)          | 15.5               | 2.64 kxn0>kx0  | 0.52           |
  | C6-S30-k4 | 3.9 (1.16, 130) transient | 0.8                | 5.05 dt<1e-4   | 1.5e-11        |
  | C6-S30-k8 | 3.2 (1.74, 268) transient | 3.1 (transient)    | 5.05 dt<1e-4   | 8.4e-10        |
  | C6-*-lin  | 1.13 / 1.22 (late)        | -                  | 30 (tfinal)    | 6e-17 / 1e-13  |
  C6 dt stop = the streamer growing, not a secondary: E_kx0 107 (t=1) -> 1.75e4 (t=5.05), dt 1.28e-3 -> 1.0e-4 = exactly
  (E_kx0)^-1/2 (CFL on streamer flow); kxn0/E stays 1e-11 (k4) / 1e-9 (k8), late sigma_NL 0.76/0.40 < linear 1.13/1.22.
  -> no secondary at all with k_x,min = 1: the C1 secondary (rate 0.9 sqrt(E_kx0)) needs k_x < 1 (k_x = 0.5 exists only for Lx=4pi).
A2: restart from curvy3d A3a-Y snapshot t=300 (low), k8, 6 x 50 R/vA. Wall time 10.5 ms/step (H200, both states; C300 2.8 h total,
  cfl025 5.8 h). a2_Qt.png: Q(t), E(t) of both FLUCS runs on curvy3d A3a-Y(+Yc2), t=300-600.
  First |ln Q_F/Q_C| > 0.1: C300 t=304.3, cfl025 t=302.8 (>0.5 at 305.6/305.3); C300 vs cfl025 differ by >0.1 at t=302.6 too
  -> divergence within 3-5 R/vA is chaotic (dt-level perturbation); trajectory comparison beyond ~t=303 is meaningless.
  | t=300-600        | f_high f_low | Q_H  Q_L (mean) | E_H  E_L  | sw | episodes                          |
  | C A3a-Y(+Yc2)    | 0.13   0.80  | 2.20 0.50       | 649  165  | -  | L to 333, H6, L112, H11, L75, ... |
  | F C300 (cfl .5)  | 0.27   0.63  | 2.81 0.44       | 795  144  | 6  | L101 H57 L19 H16 L48 H7 L>22      |
  | F C300-cfl025    | 0.30   0.63  | 3.72 0.40       | 1053 135  | 4  | L72 H67 L57 H25 L>60              |
  (curvy3d whole run 100-1500: Q_H 3.89 Q_L 0.47, E_H 1120 E_L 157.) cfl 0.5 vs 0.25: same f_high, same Q_L/E_L; Q_H 2.8 vs 3.7
  but each from 1-2 high episodes (curvy3d per-run Q_H 3.0-5.0) -> no time-step effect on the levels.
  Rule (ii): FLUCS goes low and high, Q_L 0.40-0.44 vs 0.47-0.50, high level within curvy3d spread -> AGREE. FLUCS is high
  more often in this window (0.27-0.30 vs 0.13), within curvy3d run-to-run spread (f_high 0.18-0.54).
States: C300 t450 H (E 573), t600 L (E 242, borderline); cfl025 t450 L (E 100), t600 L (E 176). dmap f_par/alpha/inj<1.5: C300
  t450 .67/.39/.78 (curvy3d H .69/.32/.84); FLUCS L .51-.53/.36-.48/.61-.70 (curvy3d L .45/.43/.66) -> inside curvy3d ranges.
kpar per shell (k_c 2.1, 8.5, 17, 34*): kz_rms / k_par / k_par,wr / chi (all shells: a2_state/kpar_table.txt)
  H F C300 t450    16/17/30/.50  35/40/73/.63  42/68/102/.55  45/118/144/.36 | k_par>kz* from 17; kz_rms never
  H C YP4 t400     17/20/32/.51  35/46/75/.67  41/79/108/.55  45/138/160/.26 | from 17 (YP4 t450, Yc2 t1000: from 8.5)
  L F C300 t600    10/13/18/.49  27/32/55/.70  38/54/85/.62   43/92/120/.41  | from 17
  L F cfl025 t450   8/10/14/.52  26/25/52/.59  36/40/77/.57   42/64/101/.38  | from 34 (cfl025 t600 same)
  L C A3a-Y t450    7/ 9/12/.45  24/22/47/.59  34/35/71/.62   41/56/94/.43   | from 34 (YP4 t250 same; A3a-Y t500 none)
  Same picture as Will's YP4 result. HIGH: lab kz_rms saturates ~45, k_par -> k_par,wr above k_perp ~ 8-17 (ratio .82 FLUCS vs
  .86-.92 curvy3d at 34) = alignment lost at small scales. LOW: k_par ~ kz_rms ~ k_par,wr/2 to k_perp ~ 17, chi 0.4-0.7.
  FLUCS matches curvy3d state for state (C300 t600, a borderline-low state, sits between the two).
Stage 3d proposal (not started): (1) FLUCS A3a-Y-k8 to t ~ 1500 (as Yc2) for f_high and residence times from ~30 switches;
  (2) B: C1 (Lx = 4pi) with the k_x = 0.5 modes removed, to confirm the secondary needs k_x < 1; (3) redo kpar/dmap state
  averages with the fixed scripts once installed; (4) GPU speed FLUCS vs curvy3d (ms/step, same grid, same H200) and a
  k_z-resolution test (k_z,max ~ 170) of the high-state alignment loss.

## Stage 3d part 1 - square-box A2 turbulence, A3a-Z continuation submitted, speed table (2026-11-12), flucs zpinch @ 5ccec64 (code)
md5 kpar_spectrum.py ea3360c2, dissipation_map.py bb5014d5 = the fixed versions -> use them normally in part 2 (state averages
  only within one state). Files: runs/stage3d/{make_inputs.py, d3_speed.py/.txt, <run>/run.sh}; copies in flucs/tests/zpinch/.
No FLUCS code change. Each run = folder with 50-R/vA restart segments t*/ (restart.nc = 3D state at each t, converter-readable) and
  a run.sh that runs them in order (same flucs command as queue.sh) and STOPS THE CHAIN after an "Early stop" (writes
  <run>/EARLY_STOP): otherwise every later segment would restart the runaway state and spend another 5 R/vA at dt < 1e-5.
Submitted (queue.sh, both started 14:50, H200):
  13119588 D1: runs/stage3d/A2-SQ-k8 then A2-SQ-k4. A2 (kP 2.2627417, kb 0.28284271), Lx=Ly=2pi, padded 128x128x256 (85x85x171),
    plugin white noise 1e-5 seed 257, t=0-500 (10 segments), stop |Q/Q_MHD|>1e3 or dt<1e-5 each held 5 R/vA. k8: nu_perp
    4.44089e-12, nu_z 1.38778e-14 (p 4, = A2-Y); k4: 2e-4, 8e-6 (p 2, = A2-R0). k8 t0050 done: 53903 steps, 4 m 50 s (5.4 ms/step),
    t=50: E 359, Q/Q_MHD 2.03, E_kx0/E 0.68, dt 2.8e-4 (A2-Y in the 4pi box: 0.116) -> may run away. Expect 1-4 h for both.
  13119589 D2: runs/stage3d/A3a-Z-k8-C300, curvy3d A3a-Z snapshot 6 (t=300, high: E 1180, Q/Q_MHD 4.7, high since ~281) to t=1000,
    14 segments t0350..t1000 (FLUCS time = t-300), padded 256x128x512 (171x85x341), nu_perp 4.44089e-12, nu_z 8.7e-16 (p 4),
    no Q stop, dt<1e-5 for 5 R/vA. First steps t=2.1: E 1420, Q/Q_MHD 5.3; FLUCS estimate 1 h 40 min for the first (high) segment.
    Wall-time estimate: ~21 ms/step (10.5 x curvy3d's Z/Y ratio 32.2/15.8), 2.2-2.9 M steps -> 13-17 h (up to ~23 h if mostly high).
    Disk ~150 MB per restart.nc (2.2 GB total).
D3 speed, Y grid (padded 256x128x256 = 171x85x171), k8, max_cfl 0.5, H200 NVL (runs/stage3d/d3_speed.txt). curvy3d steps =
  integral of dt_series (matches the logs' last printed step to 0.1-2 %), wall time = registry wall_hours:
  | run                     | T    | steps  | wall h | ms/step | mean dt | s per R/vA |
  | C A3a-Y   (0-500)       | 500  | 1.47 M | 6.46   | 15.8    | 3.4e-4  | 46.5       |
  | C A3a-Yc2 (500-1500)    | 1000 | 4.20 M | 18.50  | 15.9    | 2.4e-4  | 66.6       |
  | C A3a-YP4               | 500  | 2.01 M | 8.84   | 15.8    | 2.5e-4  | 63.6       |
  | C A2-Y                  | 500  | 0.70 M | 3.14   | 16.1    | 7.1e-4  | 22.6       |
  | C A3a-Z (nz 341)        | 300  | 0.92 M | 8.19   | 32.2    | 3.3e-4  | 98.3       |
  | F A3a-Y-k8-C300 (300-600)| 300 | 0.96 M | 2.79   | 10.5    | 3.1e-4  | 33.5       |
  | F C300-cfl025           | 300  | 1.98 M | 5.80   | 10.5    | 1.5e-4  | 69.6       |
  -> FLUCS 1.50x faster per step (both incl. diagnostics/IO); mean dt equal at the same CFL (it depends on the state mix:
  curvy3d 2.4-3.4e-4, FLUCS 3.1e-4) -> ~1.5x faster per R/vA. (G1 grid, Stage 3: 1.5-1.6x.)
Part 2 to do ("continue: analyse Stage 3d"): D1 Q/Q_MHD(t), E_kx0/E(t), runaway or not vs A2-Y 0.116 / A2-R0 0.078; convert last
  state(s) (flucs_to_curvy3d.py), dissipation_map + kpar_spectrum. D2: concatenate 0d (+300), add to two_state.py RUNS, compare
  with C A3a-Y+Yc2 (f_high 0.54) and C A3a-Z (0.18); per-shell kpar table (k_z,rms, k_par, k_par,wr, chi, k_z* crossings;
  k_z* with nu_z 8.7e-16) in the high state vs Y-grid high states.

## Stage 3d part 2a - D1 square-box A2 (2026-11-13), flucs zpinch @ 5ccec64 (code, unchanged)
Job 13119588 COMPLETED (k8 2 h 01, k4 1 h 24), all 10 segments exit 0, no EARLY_STOP. D2 (13119589) still RUNNING: segments
  t0350-t0750 done, in t0800 at t ~ 779 at 02:20 (t0750: 216 k steps in 1 h 15) -> not analysed.
Files: runs/stage3d/{d1_analysis.py/.txt, d1_Qt.png, d1_series.npz}, runs/stage3d/d1_state/ (converted .nc 697 MB, dmap_*/kpar_*).
Converter fix: flucs_to_curvy3d.py now overrides Ly, Lz, Lx_mult = Lx/Ly (it used to inherit A3a-Y's 4pi box); checked: L 2pi,
  snapshot Q/Q_MHD = saved series to 4 digits.
SATURATED, both runs, no runaway (k8's t=50 value Q 2.0 / kx0 0.68 was the initial overshoot). Bursty single state, no
  two-state switching: Q/Q_MHD from 0.05 to 2.5 (k8) / 5.1 (k4, one burst at t=450, E 836), 50-R/vA means 0.5-1.2 (k8), 0.2-1.5 (k4).
  Window [150,500] (transient over by t~100; 150 = >= 2 tau_int after it; [100|200|300,500] agree within 1 sigma); error =
  std*sqrt(tau_int/T), tau_int Sokal-windowed (checked against 4-block means):
  | run       | <Q/Q_MHD>     | tau_int | median | <E> | <E_kx0/E> | 4pi box (curvy3d)     |
  | A2-SQ-k8  | 0.80 +- 0.10  | 12      | 0.74   | 156 | 0.55      | A2-Y  0.116 (p4)      |
  | A2-SQ-k4  | 0.78 +- 0.20  | 18      | 0.45   | 155 | 0.56      | A2-R0 0.078 (p2)      |
  -> k8 and k4 agree; both are 7-10x the 4pi-box level (> 5 sigma for k8).
State averages (snapshots t=200-500 for k8, Q 0.37-1.0; t=200-400 for k4, Q 0.08-0.9; k4 t=450 burst alone):
  | state          | E_kx0/E | E k<=1.5 | f_par | 50/90% D below k_perp | outer-third D (kperp/kz) | k_d: kz_rms (flat 49) | cut binding |
  | k8 200-500     | 0.56    | 0.61     | 0.45  | 33 / 44               | 0.62 / 0.55              | 33: 43.3 SATURATED    | nu_z tilt 14.8 < nu_perp 24.8 |
  | k4 200-400     | 0.53    | 0.62     | 0.51  | 9.2 / 17              | 0.003 / 0.02             | 9.2: 14.9             | tilt 6.3 ~ nu_perp 7.5 |
  | k4 t=450 burst | 0.77    | 0.80     | 0.73  | 7.2 / 16.6            | 0.005 / 0.05             | 7.2: 18.2             | tilt 4.5 < 7.5 |
  kpar: 61-65 % of E at k_perp 0.75-1.5 (kz_rms 2.5, k_par 3.4-3.9); positive injection 84 % at k_perp < 1.5. k8 is grid-limited
  in k_z and k_perp (edge dissipation > 50 %: the k_z,max needed is ~144); k4 is well resolved (edge < 2 %).
Reading: the well-resolved k4 run saturates, so Paper 1's Lx ~ Ly runaway is NOT structural at this kP, kb: no runaway at
  either dissipation, and the level is insensitive to the dissipation (k4 = k8 within errors; in the 4pi box R0 vs Y differ by
  1.5x). But it does NOT saturate near the 4pi-box level: the square box gives 7-10x more transport with the same E_kx0/E
  fraction (~0.55). So the box size changes Q a lot; a resolution/dissipation effect explains the old failure only if
  Paper 1's setup was less resolved. Next (Will): the 4pi-box A2 E_kx0/E for comparison, and an Lx = 8pi run if Q(Lx) matters.

## Stage 3d part 2b - D2 A3a-Z continuation (2026-11-13), flucs zpinch @ 5ccec64 (code, unchanged)
Job 13119589 COMPLETED, 14 segments exit 0 (t 300-1000), 2.76 M steps, 16.0 h: 21.0 ms/step (curvy3d A3a-Z 32.2 -> 1.54x),
  mean dt 2.5e-4, 82.5 s per R/vA (curvy3d 98.3; smaller gain because FLUCS spent 64 % in the high state, which has smaller dt).
Files: runs/stage3d/{two_state.txt/.png, two_state_series.npz, d2_fhigh.py/.txt}, d2_state/ (converted .nc, dmap_*/kpar_*,
  d2_kpar_tab.py, d2_kpar_table.txt); tools/two_state.py has the D2 entry and an output-folder argument. Copies in flucs/tests/zpinch/.
Two-state (same classifier as Stage 3c; Q-based, E-based in brackets):
  | run                  | t        | f_high      | Q_H  | Q_L  | E_H  | E_L | kx0/E H L | switches |
  | F A3a-Z-k8-C300      | 300-1000 | 0.64 (0.69) | 3.09 | 0.52 | 869  | 171 | 0.53 0.32 | 9        |
  | C A3a-Z              | 100-300  | 0.18 (0.22) | 2.99 | 0.45 | 837  | 151 | -         | 4        |
  | C A3a-Y+Yc2 (Y grid) | 100-1500 | 0.54 (0.56) | 3.89 | 0.47 | 1120 | 157 | -         | 29       |
  f_high in 200-R/vA windows (step 100): Z-FLUCS 0.47-0.92; Y-curvy3d 0.03-0.92 -> curvy3d A3a-Z's 0.18 (one 200-R/vA window)
  is within the Y-grid spread. Complete episodes Z-FLUCS H 4 x 100, L 4 x 48 R/vA (Y: 44 / 41). NB: D2 starts from a high
  snapshot (first episode cut, 26 R/vA). => Doubling nz (and nu_z 16x lower) does NOT remove the high state; f_high 0.64 vs
  Y's 0.54 is within the window scatter; Q_H, Q_L and E_L agree with the Y grid within ~20 % (Q_H 3.1 vs 2.8-3.9 across Y runs).
State averages (dissipation_map, fixed versions; H = t 550-700 (4 snapshots) and 850-900, L = 750-800 and 1000):
  | state            | f_par | outer-third D kperp/kz | k_d: kz_rms (flat) | k_perp,tilt vs nu_perp cut | kz,max needed |
  | Z H 550-700      | 0.83  | 0.27 / 0.52            | 15.7: 59 (98)      | 8.0 < 25.2                 | 386 (170)     |
  | Z H 850-900      | 0.81  | 0.29 / 0.48            | 17.3: 59 (98)      | 9.0 < 25.2                 | 342           |
  | Z L 750-800      | 0.51  | 0.59 / 0.27            | 31.2: 57           | 19.2 < 25.2                | 160           |
  | Y H (C300 t450)  | 0.67  | 0.38 / 0.71            | 19.6: 43 SAT (49)  | 7.5                        | 288 (85)      |
  | Y H (YP4 t450)   | 0.68  | 0.30 / 0.72            | 14.6: 44 SAT (49)  | 6.4                        | 339 (85)      |
  Per-shell kz_rms (high; k_perp ~1, 2, 4, 8, 17, 34): Z 6.3, 23, 33, 47, 58, 67; Y 6.0, 21, 28, 38, 43, 46 (YP4 t450); k_par
  equal or lower on Z (shell 17: 71-81 vs 68-107). Low state Z 3.4, 7.9, 16, 31, 45, 56 vs Y 2.7, 6.5, 13, 24, 34, 41.
  -> The energy-containing scales (k_perp <= 4, 75-80 % of E) are unchanged; Y's k_z cap was only in the dissipation range
  (k_perp >= 8). The Z grid is no longer k_z-saturated at k_d, but the high state is still parallel-dissipation limited (52 %
  of D in the outer k_z third, tilt cut-off ~8 << 25): the high state is not a k_z-resolution artefact of the Y grid, and its
  Q_H on Z lies inside the Y-run spread (2.8-3.9); a fully resolved high state would need k_z,max ~ 350-390.

## Stage 3e part 1 - checkpoint (2026-11-13), flucs zpinch @ dfb6ae7 (code, unchanged)
Tiers (Will): T2 padded 256x128x512 (171x85x341), T1 256x128x256 (171x85x171); k^8 both directions, nu_perp 4e-11; nu_z
  5.9e-16 (T2) / 7.5e-14 (T1); box 4pi x 2pi x 2pi (FLUCS defaults). Inputs: runs/stage3e/make_inputs.py (run once, done).
DONE
 E1 runs/stage3e/e1_zonal.py/.txt (per snapshot + state means; labels from stage3d/two_state_series.npz; D2 read directly
   from its 14 restart.nc). State means (t >= 100):
   | run         | st | n | E   | b_rms | E_kx0/E    | E_ky0/E     | E_ky0kz0/E | S_zon0 (kz=0) | S_zon all kz |
   | C A3a-Y+Yc2 | H  | 3 | 754 | 19.2  | 0.53+-0.11 | 0.074+-0.016| 0.004      | 0.56          | 5.5          |
   | C A3a-Y+Yc2 | L  | 7 | 136 | 8.6   | 0.36+-0.12 | 0.087+-0.020| 0.013      | 0.51          | 3.4          |
   | F A3a-Z D2  | H  | 9 | 821 | 21.0  | 0.54+-0.06 | 0.055+-0.014| 0.002      | 0.58          | 5.3          |
   | F A3a-Z D2  | L  | 3 | 118 | 8.5   | 0.28+-0.07 | 0.094+-0.015| 0.012      | 0.55          | 4.0          |
   -> H = streamer-rich (E_kx0/E 0.53): yes. L is NOT a zonal condensate: E_ky0/E only 0.09 (H 0.06), k_y=k_z=0 part ~1 %,
   zonal-shear rms equal in both states (~0.55 < gamma_max 0.72). L = weaker broadband state (E_kx0/E ~0.3).
   (curvy3d Yc2 has snapshots only at t = 1000, 1500; Y every 50.)
 E5 ~/flucs_work/RUNNING_FLUCS.md (input options, queue.sh, output.0d.nc, converter, dmap/kpar/two_state/e_analyse).
 Part-2 tool runs/stage3e/e_analyse.py RUN [--t0 100] [--t-offset] [--no-dmap] [--keep-nc] [--redo] -> RUN/analysis.txt:
   series (runaway, <Q/Q_MHD> +- err, two-state via tools/two_state.py), direct state pass (E, b_rms, E_kx0/E, E_ky0/E,
   zonal shear, outer-third E in k_perp/k_z, pile-up = energy PER MODE rising in the outer third), dissipation_map per
   same-state snapshot group (converts states to RUN_state/, deletes the .nc), pass/fail table per state. Tested on D2:
   reproduces Stage 3d (f_high 0.63, Q_H 3.09, D_out 0.28/0.50, k_tilt 8-9); output runs/stage3d/A3a-Z-k8-C300/analysis.txt
   (+ _state/dmap_*). D2 lab kz_rms never reaches k_z* = 73 (max ~62) -> k(kz*) = inf. ~8 min CPU per T2 run (14 states).
   tools/two_state.py: E2 entry "F:A3a-T2-k8" added; skips missing segments / too-short runs. Copies in flucs/tests/zpinch/.
 NB: run the analysis with ~/flucs_work/.venv/bin/python (it has tomli; Will's curvy3d interpreter has no TOML parser).
QUEUE (submitted 13:26):
   13122290 RUNNING since 13:33 (H200): E2 runs/stage3e/A3a-T2-k8 (24 segments, t 0-1200); at ~14:10 t ~41 in t0050.
     Expect ~27 h (D2 grid: 82 s per R/vA), up to ~33 h if mostly high.
   13122291 PENDING (Resources/Priority): E3 runs/stage3e/A3b-T2-k8 then A4-T2-k8 (10 segments each, t 0-500), ~10-12 h each.
   13122292 PENDING: E4 runs/stage3e/A2-T1-k8 then A2-T2-k8 (t 0-500), ~2 h + ~5 h.
   All chains: plugin white noise 1e-5 seed 257, no Q stop, dt < 1e-5 for 5 R/vA -> EARLY_STOP file, chain stops.
   E3/E4 have not started, so their first time steps are NOT yet confirmed (rule 4).
TO DO (order): 1. when 13122291/92 start: tail runs/stage3e/{A3b-T2-k8,A2-T1-k8}/t0050/run.log (confirm steps).
 2. part 2 when all done ("continue: analyse Stage 3e"), for each of A2-T1-k8, A2-T2-k8, A3b-T2-k8, A4-T2-k8, A3a-T2-k8:
    ~/flucs_work/.venv/bin/python runs/stage3e/e_analyse.py runs/stage3e/<run>      then
    ~/flucs_work/.venv/bin/python runs/tools/two_state.py stage3e   (E2 vs C A3a-Y+Yc2 f_high 0.54 / Z-FLUCS 0.64).
 3. compare: E2 Q_H/Q_L/E_H/E_L/f_high vs Y and Z grids; E4 T1 vs T2 vs A2-Y 0.116 (A2-R0 0.078); E3 b_rms -> tier per
    point (T1 only if b_rms <~ 5); <= 40 lines + 3-5 lines production-scan proposal; commit; stop.
## Stage 3e part 2 - analysis (2026-11-15), flucs zpinch @ 7f68d85 (solver code = dfb6ae7; tests only since)
Jobs 13122290-92 COMPLETED; all 54 segments exit 0, no EARLY_STOP, min dt >= 9.2e-5 (no runaway in any run). Wall: T2 20.7
  ms/step, T1 10.5 ms/step. Files: runs/stage3e/<run>/analysis.txt (+ _state/dmap_*), two_state.txt, e_blocks.py/.txt
  (sub-window means, A2-Y error, f_high windows; copy in flucs/tests/zpinch/). <Q> window [100,end]: transient over by ~100,
  >= 47 tau_int; error std*sqrt(tau_int/T); time-weighted (uniform dt 0.1 grid).
| run (st,n)  | <Q/Q_MHD>   | b_rms | kx0/E ky0/E | kzrms(k_d)/lim | E_out p/z     | pile p/z  | D_out p/z | k_cut k(kz*) | pass |
| A2-T1  L 8  | 0.088+-.007 | 3.5   | 0.16 0.12   | 28/39          | 1.8e-3/1.1e-3 | 0.45/0.58 | 0.35/0.18 | 18.8 inf     | yes  |
| A2-T2  L 8  | 0.105+-.006 | 3.6   | 0.17 0.15   | 43/79          | 2.4e-3/1.4e-4 | 0.47/0.73 | 0.47/0.04 | 18.8 inf     | yes  |
| A4-T2  L 7  | 0.70+-.07   | 7.7   | 0.45 0.06   | 52/79          | 1.8e-3/5.6e-4 | 0.48/0.72 | 0.38/0.15 | 16.8 inf     | yes  |
| A4-T2  - 2  | (f_high .06)| 12.2  | 0.57 0.07   | 53/79          | 1.6e-3/9.1e-4 | 0.50/0.74 | 0.35/0.22 | 16.8 40      | yes  |
| A3b-T2 H 5  | 3.10+-.65   | 13.0  | 0.54 0.05   | 56/79          | 1.8e-3/1.1e-3 | 0.55/0.76 | 0.34/0.29 | 18.0 inf     | yes  |
| A3b-T2 L 1  | (f_high .68)| 5.4   | 0.30 0.08   | 51/79          | 2.8e-3/5.3e-4 | 0.46/0.71 | 0.44/0.10 | 18.0 inf     | yes  |
| A3a-T2 H 6  | 1.03+-.16   | 19.5  | 0.51 0.07   | 61/79          | 1.7e-3/2.8e-3 | 0.59/0.81 | 0.27/0.44*| 19.1 40      | yes  |
| A3a-T2 L 15 | (t 100-1200)| 9.1   | 0.32 0.10   | 57/79          | 2.7e-3/1.2e-3 | 0.55/0.77 | 0.42/0.22 | 19.1 inf     | yes  |
  Energy per mode falls through the outer third in both directions everywhere (pile < 1): no pile-up. *A3a H: D_out,z 0.52
  in the strongest snapshot (t=1100, b 26) and "k_z,max needed" 342 vs 170: the high state still dissipates near the k_z
  edge (as on the Z grid); every criterion still passes.
E2 A3a (two_state, Q-classified) | f_high | Q_H  Q_L  | E_H  E_L | kx0/E H L | sw | f_high in 200-R/vA windows (mean +- err)
  F T2 100-1200                  | 0.25   | 2.80 0.43 | 791  144 | 0.53 0.30 | 19 | 0.12-0.60 (0.27 +- 0.06)
  C A3a-Y+Yc2 (Y) 100-1500       | 0.54   | 3.89 0.47 | 1120 157 | -         | 29 | 0.03-0.92 (0.54 +- 0.10)
  F D2 (Z) 300-1000              | 0.64   | 3.09 0.52 | 869  171 | 0.53 0.32 | 9  | 0.47-0.92 (0.67 +- 0.13)
  -> Q_L, E_L, kx0/E unchanged (<= 15 %). Q_H 2.8 is at the bottom of the Y-run spread (2.8-3.9) and 10 % below Z; E_H 30 %
  below Y, 9 % below Z. f_high is lower: 0.27 +- 0.06 vs 0.54 +- 0.10 (2.4 sigma from window blocks, few switches), because
  high episodes are shorter (mean H 26 vs 48 R/vA on Y; L 76 vs 42). Both states survive with 9x nu_perp and the stronger
  edge damping. The levels are robust; how often the high state occurs is weakly dissipation-sensitive (suggestive, not proven).
E4 A2 (same window [100,500]): T1 0.088+-0.007, T2 0.105+-0.006, C A2-Y 0.125+-0.011 (0.116 in Stage 3d was another window).
  100-R/vA blocks: T1 0.067-0.117, T2 0.089-0.120 -> T1-T2 1.8 sigma (tau_int error) / 1.3 sigma (block error); T2 vs Y 1.6
  sigma. The tiers agree within the errors; T1 may sit ~15 % low (not significant).
E3: A4 saturated (stationary, blocks 0.56-0.83), mostly low (f_high 0.06). A3b saturated (no runaway, two-state, f_high
  0.68, 9 switches, Q_H 4.6 Q_L 0.84, E_H 502) but <Q> is NOT converged: blocks 1.06/2.84/2.86/5.67, [200,500] 3.79.
  b_rms per state: A2 3.5 | A4 L 7.7 (up to 12.6) | A3b H 13 (up to 23) L 5.4 | A3a H 19.5 L 9.1.
  Tier: A2 -> T1 (b_rms 3.5); A4, A3b, A3a -> T2 (b_rms > 5 in every state).
PRODUCTION SCAN (proposal; GPU h per 100 R/vA measured: A2-T1 0.35, A4-T2 0.64, A3b-T2 1.19, A3a-T2 1.62):
  A2 T1 t 0-500 (~1.8 h); A4 T2 t 0-800 (~5 h); A3b T2 t 0-1200 (~14 h, two-state with a rising mean, needs >= 10 switches);
  A3a T2: extend the existing E2 run to t=2000 (+13 h) for f_high. Same k^8/nu settings, noise 1e-5 seed 257, 50-R/vA
  segments. Bundle A2+A4 in one job (~7 h); A3b and A3a-ext as separate jobs (3 jobs, ~34 GPU h in total).
## Stage 4a part 1 - checkpoint (2026-11-16), flucs zpinch @ 2d3e3b0
STEP 0: du listed; deletion list runs/DELETE_LIST_stage4a.txt (145 paths, 15.1 GB on disk / 45.9 GB apparent; seeds
  kept). Will: delete NOTHING yet -> nothing deleted.
STEP 1 IMPLEMENTED + VERIFIED (flucs/src/flucs/systems/zpinch_fourier/; equations unchanged):
  zpinch_output.py: ZPTimedOutputNC - [output.<name>] save_time (records at multiples of t, restart-chain aware),
    sample_time (window averages of sample() diags), timing file zp_timing_<name>.txt. Outputs: 0d 1d maps kpar slices
    kyt kytavg shells (defaults in zpinch_fourier.toml). diagnostic.py/output.py: per-variable dtype (f4) and zlib.
  zpinch_runtime_diags.py: (a) zp_scalars (b) zp_spectra (unit bins k_perp/k_y/|k_z|) (c) zp_maps (k_perp,|k_z|) f4
    window means (d) zp_kpar (kpar_spectrum.py accumulators, pad 2 exact) (e) zp_profiles (complex k_y=k_z=0 and
    k_x=k_z=0 lines, f8) (f) zp_slices (z=0, f4) (g) zp_kytransfer + zp_streamer_kzkx (curvy3d subcrit balances; flag =
    include in diags) (h) zp_shells (Elsasser + kin/mag sets, kperp and kz shells; exact triad sums, ~25x fewer FFTs).
  zpinch_readers.py: load_diag, mean_profiles (u_y(x), dB_y(x), u_x(y), dB_x(y), shears, j_z, curvatures), kpar_table.
  Verification runs/stage4a/verify/ (run.sh, compare_stage4a.py, verify.txt; job 13132249): A3a-T2 t=1200 state, all
  diags at t=0 vs dissipation_map.py + kpar_spectrum.py --gpu (captured locals), both hydra shell plotters run
  unmodified, numpy re-implementations of e1_zonal / curvy3d subcrit+streamer formulas: 116 checks, 0 FAIL
  (f8 <= 7e-14, f4 <= 5e-8). Maps use dissipation_map's rint(k_perp) shells -> no binning difference.
  First per-call costs (H200, T2): scalars 25 ms, spectra 61, maps.sample 20, kpar 420, slices 26, kyt 100,
  streamer.sample 34, shells 1460 ms -> estimate a-f < 1 %, g+h ~0.7 % at production cadence (to be measured).
BUGS FIXED: runs/tools/flucs_to_curvy3d.py never overrode the template's explicit nx/ny/nz lines -> Curvy3DInput gave
  nz 171 for T2 files (Stage 3 results unaffected: dmap/kpar use array shapes). curvy3d solver adds dBx_profile twice
  per save (lines 2580, 2583) -> saved profile = 2x mean; ours saves the true value (Will: fix in curvy3d?).
  Toml note: mixed dict/string diag lists are silently dropped -> always use {name = "..."} entries.
DONE, UNTESTED: STEP 2 runs/stage4a/chain_run.sh (100-R/vA segments, deletes previous restart.nc, flucs_git.txt).
  STEP 3 runs/tools/flucs_summary_nc.py (curvy3d-format summary; tested on stage3e A2-T2-k8: build_registry.py and
  make_flux_table.py run unchanged, Q/Q_MHD 0.101+-0.006; 5.6 MB disk / 69 MB apparent with f4 final snapshot; not yet
  with ky-transfer outputs), runs/stage4a/f_analyse.py (analysis from run-time outputs only; NOT run yet),
  runs/tools/zp_shells_export.py (plotter CSV/JSON). STEP 5 runs/stage4a/campaign.py points -> campaign_points.csv:
  52 points (41 supercritical T2, 11 interchange T1); no slow point (min damped gamma 0.070, P09B02-T1).
JOBS (H200): 13132247 overhead/{base,af,ah} (A3a T2 t 1200-1210; wall time + zp_timing files -> overhead a-f, g+h);
  13132250 speed/h200/single_T1, single_T2 (~2000 steps, ms/step); 13132251 speed/h200/pair (2 runs at once, T1 pair
  then T2 pair: GPU sharing). A100/RTX submissions failed (sbatch "Requested node configuration is not available";
  logged as bogus job 49152) -> queue.sh now fixed by Will (-m = GB per core, robust job-id parsing).
TO DO: STEP 1: overhead from 13132247 (target < 5 % a-f). STEP 2: test chain_run.sh on a short 2-segment run; report
  disk per run T1/T2/T3 (restart + run-time outputs). STEP 3: run f_analyse.py and flucs_summary_nc.py on overhead/ah
  (or a short chain), incl. ky-transfer variables; fix; copy to tests. STEP 4: resubmit speed/a100/{single_T1,single_T2},
  speed/a100/pair, speed/rtx/{single_T1,single_T2} (new folders if outputs exist); table ms/step and pair gain.
  STEP 5: extension rule, T3 rule, campaign plan (GPU-h per type from STEP 4, wall days, disk) -> campaign_plan.txt.
  Finish: <= 40-line PROGRESS entry, commit, stop for Will's approval (no production submission). Also pending:
  Will's review checklist for the work above.
## Stage 4a part 2 - review, storage chain, speed, campaign plan (2026-11-16), flucs zpinch @ 7383e0e
STORAGE: deleted 142/145 paths of runs/DELETE_LIST_stage4a.txt (no job running; explicit paths). Kept stage3e/A3a-T2-k8
  t1050, t1100 and, for now, stage3e/A2-T1-k8/t0500/restart.nc (speed-test seed + REALTEST reference). runs/: 20 -> 5.6 GB
  disk (53 -> 8.6 GB apparent). QUEUE_LOG.tsv: FOUR bogus 49152 lines removed (2 a100, 2 rtx), not two.
SPEED ms/step (~2000 steps, diags a-f): H200 T1 9.5 T2 18.9 (same with 1 or 4 cores; REALTEST T1 with a-h 9.9); A100 21.9
  / 40.8; RTX 6000 Ada 61.8 / 125.8 (fp64). Two runs on one H200 (4 cores): 0.93x throughput -> --parallel does not pay.
  NB queue.sh --parallel still requests ONE core for the whole job; each FLUCS process uses ~1.1-1.3 cores.
  OVERHEAD (13132247, A3a T2, 23952 steps, 456 s): diags a-f 0.4 %, a-h 1.1 %.
REVIEW (runs/stage4a/review/, chaintest*/; copies in flucs/tests/zpinch/stage4a/):
 (1) shells (k_perp, k_z): self-transfers antisymmetric, TuB = -TBu^T, TBparupar = -TuparBpar^T, sums 0, per-shell
   Elsasser budget = kin/mag budget: <= 8e-14. k_y line: dE/dt = T (+I) - D vs the solver's step change 3.5e-3 (O(dt);
   flipped sign 2.0); real-space Parseval = half-plane x2 sums, turb + line = energy diag <= 2e-15; streamer maps summed
   = k_y=1 transfers 3e-8. BUG FIXED: one-step differences (dEdt, budget_error, *_dWdt, dtdBx_sq, dtupar_sq) were garbage
   at step 0 (t = 0 record; dEdt 6.8e5) -> NaN.
 (2) tiny chain (6 segments incl. extension): 8 outputs contiguous, no duplicate/missing records, window counts right.
 (3) chain_run.sh rewritten: restart.nc -> tools/check_state.py (opens, grid, finite, t in [t0+tf, t0+tf+2 dt_max]) ->
   mv state.nc -> SEG_OK -> only then previous state deleted; start state validated; reruns clean partial files.
   chaintest (13134352): SIGKILL mid-segment, stops after validate/rename/SEG_OK, resubmission, extension from the kept
   state, EARLY_STOP: chain behaved correctly in every case (inspector FAILs = my mistimed P2 + a wrong state-time
   assumption; both fixed in chaintest2 = 13134354, PENDING, not yet confirmed). FLUCS core writes one extra restart
   at step 1 (steps_until_write starts at 0): harmless, not patched.
 (4) summary .nc: zpinch_runtools.heat_flux_series on it = FLUCS Qhat/Ly^2 to 8e-13 (curvy3d half-plane prefactors).
   FIXED: ky-transfer series on curvy3d's t = i x hfmult x save_time_hf grid from 0 (was index-based: wrong times for
   chains extended from older runs); no interpolation across 0d gaps; state.nc/SEG_OK. build_registry/make_flux_table OK.
 (5) f_analyse.py had never run: syntax error; BUG: k_perp pile-up mode counts in other shells than the energy -> spurious
   pile 1.46 (fixed 0.52; e_analyse 0.55). New --t0 auto, --at-times. e_analyse.py reads state.nc too.
 (6) profiles from zp_profiles vs the full 3D state (A3a t=1200): <= 3e-14; k_y=0 plane Hermitian 2e-13.
 (8) 1e-3 white noise is NOT linear here (E(0) 0.6 at 1e-5 on T2 -> 6e3 at 1e-3). Kept 1e-5; t_sat = (12.5-17.3)/gamma
   for the 5 anchors -> t_init = ceil_100(15/gamma + 450); extension rule (campaign.py assess): stats from t_sat + 50,
   +500 while T_stat < 400, rel. err > 10 %, drift, or two-state < 10 switches; cap T_stat 1500.
PLAN runs/stage4a/campaign_plan.txt (campaign.py make/extend/assess/plan): 52 points, 316 H200-h (684 A100, 2098 RTX);
  all 8 slots 2.5 days (H200 only 3.3), +50 % extensions 3.8; disk 30 GB (46). steps/RvA = max(1100, 1270 N0^2 - 425)
  from anchors; 13 points with N0^2 2.6-7.2 extrapolated (up to 23 H200-h each). T3 not measured/costed.
JOBS: 13134353 REALTEST runs/stage4a/realtest (A2-T1 production chain 0-300 + extend to 500, KEEP_STATES; started 03:33,
  t0100 validated, ~2 h). 13134354 chaintest2 (~2 min, PENDING Resources at 04:02).
TO DO ("continue: analyse Stage 4a realtest"): chaintest2 run.log (expect 0 FAIL); f_analyse vs e_analyse on realtest
  (also --at-times 100,...,500); summary + make_flux_table vs f_analyse <Q>; check_transfers + inspect_chain on it;
  0d series / t0500 state vs stage3e/A2-T1-k8 (determinism, 50- vs 100-R/vA segments); then delete
  stage3e/A2-T1-k8/t0500/restart.nc and realtest states. Production only after Will's approval.

## Stage 4a part 3 - realtest + chaintest2 analysis, duplicate-record fix (2026-11-16), flucs zpinch @ a21eb61
chaintest2 (13134354): all 8 phases behaved as intended; 3 inspector FAILs were the inspector again (state time vs the
  nominal grid; FLUCS final_time = init_time + tfinal, so segment ends drift +<= 1 dt per segment: 5.01, 10.02 ...
  30.03; harmless, records are on the absolute grid). Inspector fixed (start-state time + tfinal): 0 FAIL.
REALTEST 13134353 (A2-T1-k8 chain 0-300 + extension to 500, 5 x 100 R/vA, 1.62 h, 9.9 ms/step): all segments validated.
 BUG (FLUCS, ZPTimedOutputNC): a step within eps = 1e-9 t BELOW a grid point recorded it and the next step recorded
   it AGAIN (next point computed from t, not t + eps): 3 duplicate 0d records, 1 in 1d/kyt, one maps/kytavg window
   with 11/26 samples instead of 10/25. Rate ~ eps/dt per record (~1 per 1600 at t = 500): every Stage 3e/4a run with
   time-based outputs may hold a few; effect on averages negligible, but the summary writer's kyt grid placement
   raised on them. FIXED a21eb61 (_grid_point_after, also at restart); unit test review/test_timed_trigger.py
   (old code reproduces the duplicate, new: 16 PASS). Readers: load_diag / series_0d drop duplicates of TIME-based
   outputs only (period from the segment's input.toml; step-based Stage 3e outputs untouched - a median-gap rule
   would have dropped real records there). Regression run chaintest3 = 13134452 (see below).
 f_analyse vs e_analyse (t0 = 100, same run): <Q/Q_MHD> 0.088 +- 0.007 (tau_int 4.7), <E> 30, E_kx0/E, b_rms, piles
   identical. E_out,z: f = ratio of means (1.60e-3), e = mean of ratios (1.38e-3; ratio of means 1.61e-3). E_out,perp
   was 0.8 x e: unit bin k_perp = 28 straddles the cut 2/3 x 42 and was dropped -> FIXED (bin weight = fraction above
   the cut): 1.83e-3 vs 1.84e-3 (t = 500 state: 1.50e-3 vs 1.52e-3). maps vs Stage 3e state dmaps: f_par 0.27/0.25,
   D_out 0.36/0.20 vs 0.35/0.18, k_d 26/25.6, kz_rms(k_d) 29.0/28.3, k_tilt 21.4/21.6; k(kz*) 51 (corner bins) vs inf.
 summary .nc (after de-dup: 5001 0d, 501 kyt on the grid) -> build_registry + make_flux_table: Q/Q_MHD 0.0882,
   tau_int 4.70 = f_analyse. Pass build_registry the summary file(s), not the run folder (raw FLUCS .nc = ERROR rows).
 check_transfers on t0500 (production grid): 0 FAIL (K1 line budgets 1.6e-2/8.3e-3 [2e-2] at dt 7.5e-4; rest <= 6e-14).
 Determinism: realtest (5 x 100) vs stage3e A2-T1 (10 x 50 R/vA): 0d E agrees to 1e-5 (= interpolation of the
   step-based Stage 3e records) for t > 50; <Q> 0.0882 and <E> 30.25 both; end states at 500.0019 vs 500.0062 differ
   by 4e-3 of max (= time offset). Restarts are lossless; segment length does not change the trajectory.
Storage: deleted stage3e/A2-T1-k8/t0500/restart.nc and realtest t0100-t0400 state.nc (KEEP_STATES test copies);
  runs/ 6.1 -> 5.7 GB. Kept realtest t0500/state.nc (latest validated state). The speed-test inputs pointed at the
  deleted A2 state: to re-run them use runs/stage4a/realtest/A2-T1-k8/t0500/state.nc.
JOBS: 13134452 chaintest3 (runs/stage4a/chaintest3: chaintest2 rerun on a21eb61, ~1.5 min; PENDING at 05:51).
TO DO ("continue: check chaintest3"): grep "FAIL count" runs/stage4a/chaintest3/run.log (expect 0 in all 8 phases, and
  no duplicate records). Then the campaign is ready; production only after Will's approval.

## Stage 4a part 4 - lanes, disk, DIAGNOSTICS.md (2026-11-16), flucs zpinch @ 06f74a4 (+ this commit)
chaintest3 rerun on a100 (chaintest3a, 13134604; 13134452 on h200 was cancelled): 0 FAIL in P1-P7, no duplicate
  records after the a21eb61 trigger fix. git hash was empty on gpu08 (no git there): chain now reads .git directly.
(1) LANES (Will's queue policy). One GPU job = one worker (runs/stage4a/lane_worker.sh) that claims runs from shared
  to-do lists ROOT/queue/todo/{T2,T1} by atomic mv (longest first), runs each chain to FINAL and claims the next; exits
  only when nothing eligible is left. Extension rule applied INSIDE the chain (chain_run.sh -> campaign.py assess
  --apply: EXTEND adds segments and continues on the same GPU; FINAL -> summary .nc without snapshot). h200/a100 take
  T2 then T1, rtx only T1. Workers 4+1 h200, 2+1 a100, 2+1 rtx = 11 jobs (queue.sh: MAX 8/4/4, MAX_ACTIVE is 16, not
  20): 3 pending at all times (one per type) until a type runs out of work. Crashed worker: its claim is adopted by
  the next worker once sacct reports the owner job ended. Two consecutive failures stop a worker. Drain:
  ROOT/queue/STOP_WORKERS. Simulation (campaign_plan.txt): 2.68 days at t_init (lower bound 2.52, all 8 GPUs busy),
  4.0 days x1.5 lengths (+ <= 6 h if non-priority jobs hold GPUs); rtx lanes end last. How-to: RUNNING_FLUCS.md 7.
  BUG found by the lane test: assess (extension rule) crashed once statistics existed (drift_z returns (z, rel));
  never exercised before. Fixed 06f74a4; assess checked on realtest (FINAL, <Q/Q_MHD> 0.089 +- 0.006 from t 70).
  lanetest2 (13134623, tiny grid, shortened rule): fake orphan adopted; two workers at once, each run claimed once;
  2-5 in-place extensions per run until FINAL; one state.nc + summary per run; records continuous (inspect 0 FAIL);
  idle worker exits at once; restart from a zlib-compressed state validated.
(2) DISK. /mnt/users is ZFS-compressed: a state.nc takes 80 MB (T1) / 159 MB (T2) on disk (apparent 273/545); outputs
  and the summary snapshot do not compress. Breakdown per output type in campaign_plan.txt. Per run on disk:
  outputs 7.9 (T1) / 11.0 (T2) MB per 100 R/vA; T1 126 MB, T2 219 MB at t_init without the summary snapshot (164/298
  with it). Adopted cut: summaries without snapshot (regenerable from state.nc). zlib on state.nc saves nothing on
  disk (ZFS already does it): tools/compress_state.py kept for copies. Optional if needed: slices every 20, kytavg 50.
  Campaign: 10.4 GB (11.8 GB with x1.5), + <= 2.5 GB transient. A T2 run at the 1500 R/vA cap: 0.34 GB.
  Quota: the message said "[FILL IN]" and `quota` is not installed: please give the free space.
(3) DIAGNOSTICS.md (~/flucs_work; copy in flucs/tests/zpinch): formulas, code location, outputs, readers, verification
  for energy and (a)-(h). Open points in its section 16. curvy3d adds dBx_profile twice only in its INITIAL save
  (solver l.2580/2583), so its value = mean + dBx(t0)/N_saves, not 2 x mean; FLUCS and the summary always store the
  true mean (no number affected; the two comments are corrected). The absolute sign of zp_shells transfers is checked
  algebraically and against the plotters, not dynamically.
STOP: production (campaign.py lanes runs/prod4 + 11 queue.sh submits) waits for Will's approval and the free quota.

## Stage 4b - Elsasser/k_x spectra, (k_x,k_y) maps, Paper 1 k_perp bins, shell sign test (2026-11-16), flucs zpinch @ 16dd8df+
NEW (zpinch_runtime_diags.py): zp_spectra adds zplus, zminus, Zpplus, Zpminus (W_zeta+-, W_Z+-, names as the 0d
  invariants; shells call Z+- zparplus/minus) in k_perp/k_y/|k_x|/|k_z|, and |k_x| spectra of E, its parts, the four W,
  Qhat(_es,_em) (dk_x = 2pi/L_x from the box, +-k_x summed). zp_kxky (maps output): (k_x, k_y) maps summed over k_z of E,
  four W, Qhat; window 5 R/vA (sample 0.5), f4. k_perp unit bins: Paper 1 rule (bin 0 = {0}, (e_j-1, e_j], e_0 =
  0.5 min(dkx,dky), e_j = j + 0.50001; edges stored, reader kperp_bins); zp_kpar octave shells same edge rule.
  zp_shells adds shell energies <st>_E_<part>. f_analyse uses the stored bins.
VERIFY runs/stage4b/verify/verify.txt (job 13136371, A100): 81 PASS / 0 FAIL. Per bin (z+ + z-)/2 = E_u+E_b and
  (Z+ + Z-)/2 = E_dbpar+E_upar <= 1.2e-14; bin sums = 0d invariants <= 1.2e-15; k_perp/k_y/k_x/k_z sums equal <= 6e-15;
  zp_kxky = numpy <= 3.9e-8 (f4) and = curvy3d X_spec_calc formula with its mask on the converted state <= 4.6e-8
  (factors: ours = k_perp^2 x curvy at k_y > 0; ours(kx,0)+ours(-kx,0) = k_perp^2 curvy(kx,0); W_Z without k_perp^2).
  L_x = 8 pi run: dk_x 0.25, edges [0, 0.125, 1.50001, ...], k_x = 0.25 in bin 1, spectra = numpy <= 1.5e-15. CPU test
  test_kperp_bins.py (L_x 2pi/4pi/8pi): 0.5 -> 1, 1.5 -> 1, 2.5 -> 2. Reference tools keep rint (differ at half-integers).
SIGN TEST (open point 9 closed): A2-T1 state, kP = 0, nu = 0, dt 1e-4, shells every step: dE_l/dt = sum_j T[j,l]
  (kin/mag) = 1/2 sum_j T (Elsasser) to 8.8e-7 (k_perp) / 4.3e-7 (k_z) of max|T|, sign right on every active shell:
  T[j,l] = into l from j. Total E constant 7e-11.
OVERHEAD (H200, all production outputs, zp_timing x production calls/RvA): T1 2.9 % (shells 1.1, 0d 1.0, kpar 0.3,
  1d 0.2, kxky 0.02), T2 1.6 % of the time-step cost (overhead_4b.py); now included in campaign_plan.txt.
LANES lanetest3 (13136371): as lanetest2 with the new outputs, identical verdicts (<Q/Q_MHD> 0.061/0.035/0.003), records
  continuous, zp_kxky in every record. BUG: f_analyse segfaulted on 16 segments (unclosed netCDF Datasets); fixed, also
  in e_analyse.
DISK per run (on disk, no snapshot): T1 153 MB, T2 246 MB (outputs 12.9 / 16.2 MB per 100 R/vA); campaign 11.8 GB
  (14.0 x1.5) of 52 GB free. Plan: 2.75 days (x1.5: 4.1), 321 H200-h. Approved by Will: no snapshot, lanes 5/3/3.
STOP for Will's go-ahead: `$PY runs/stage4a/campaign.py lanes runs/prod4` + the 11 printed queue.sh submits.

## Housekeeping - repo copies (2026-11-16)
flucs/zpinch_tools/sync.sh copies (never moves) the docs to flucs/zpinch_docs/ and the scripts/verification texts from
runs/ to flucs/zpinch_tools/<path under runs/> (47 files, text only; refuses .nc/.png/run logs/> 1 MB). Run it before
every commit (RUNNING_FLUCS.md section 8). Production not started.

## Stage 4b production - prod4 submitted (2026-11-16 14:42 UTC), flucs zpinch @ dea9d8d (+ this commit)
sinfo: sondhigpu and heraclesgpu wall time "infinite"; queue.sh sets none. `campaign.py lanes runs/prod4`: 52 runs (41 T2,
  11 T1), 11 workers W11161441-NN. Checked P10B04-T1, P00B11-T2, P10B11-T2 vs campaign_plan.txt: grid 256x128x256/512,
  nu_perp 4e-11 (p4), nu_z 7.5e-14/5.9e-16 (p4), t_init 500/600/500 (5/6/5 segments, restart_from chained), white noise
  1e-5 seed 257, production cadences 0d .1, 1d 1, maps 5 (sample .5), kpar 5, slices 10, kyt 1, kytavg 25 (1), shells 5.
JOBS h200 13137495-13137499, a100 13137500-13137502, rtx 13137503-13137505. At 14:53 RUNNING 95, 96 (h200: P10B11,
  P09B11), 00, 01 (a100: P08B11, P10B10), 03, 04 (rtx: P08B06, P09B02); all stepping (T2 t ~10, dt 2.5e-4 on P10B11; T1
  t 40, dt 0.01). PENDING 97 (Resources: 4th H200 held by another user), 98, 99, 02, 05 (spares by design).
  Expected wall time 2.75 d (x1.5: 4.1 d) + up to 6 h for lanes waiting on held GPUs.
MONITOR FIRST the 12 extrapolated-cost points (N0^2 > 2.56; the brief said 13, the plan stars 12; the four at 2.56 are
  measured): P05B11 P06B11 P07B10 P07B11 P08B10 P08B11 P09B09 P09B10 P09B11 P10B09 P10B10 P10B11. Check dt vs the
  steps/RvA estimate, edge fraction / D_out, Q/Q_MHD.
DOC (no code change): DIAGNOSTICS.md 5/15: curvy3d plot-script spectra = 2 x ours (k_perp, k_y, k_x = 0), ~1 x ours at
  k_x > 0 (k_y = 0 part exactly 2 x; mask keeps k_x > 0 on k_y = 0, no double count); k_perp > 42 bins = corner modes
  only, plot/fit k_perp <= 42; header line numbers (54dbc2d) marked approximate. Copy refreshed in flucs/tests/zpinch.

## Stage 4b monitoring 1 (2026-11-17 ~05:00 UTC, prod4 at ~14 h), flucs zpinch @ 64c890b (+ this commit)
Table: runs/monitor4b/monitor_20261117.txt (f_analyse --t0 auto on symlinked FINISHED segments + monitor4b/dt_trend.py;
  nothing written into prod4). Started extrapolated points 6/12 (P10B11 P09B11 P08B11 P10B10 P07B11 P09B10): measured
  steps/RvA (last 100) / plan = 0.92 0.91 0.74 0.98 1.07 0.89 (plan conservative); no steady dt fall; E_out <= 6.7e-3
  (< 1e-2: no flag); D_out(k_z) 0.56-0.63 "warn" in H states; <Q/Q_MHD> 0.54-1.43. P07B11 two-state -> EXTEND to 1100.
PROGRESS: 8 done (all T1), 8 running (6 T2, 2 T1), 36 todo, 0 failed; validated R/vA: T1 13200 (t_init sum 6400),
  T2 1900 of 21300. All 4 H200 now ours (95-98 running); pending 13137499 (h200), 13137502 (a100), 13137505 (rtx).
SIGN TEST dt 5e-5: runs/stage4b/verify/V3b_signtest_dt5e-5 (only dt_max changed; tfinal 8e-4 = 16 steps), job 13141680
  (h200, pending behind 13137499). Expect residuals ~4x smaller (~2.2e-7 k_perp, ~1.1e-7 k_z). Analysis next session.
FIX LATER (after the campaign, not now): (a) campaign_points.csv's t_init column is the default, not the value used
  (point.json / campaign_plan.txt). (b) assess: quiescent T1 runs (Q/Q_MHD ~1e-6-1e-4 after t~150, E const) never meet
  the relative-error rule -> all 8 ran to the 1600 cap (cheap: dt = dt_max, ~100 steps/RvA); add an absolute floor.
  (c) f_analyse: a state with 1 record gives nan -> FAIL (cosmetic). Will added stage4a/campaign_points.csv to sync.sh
  TOOLS (kept); I added monitor4b/dt_trend.py, the monitor table and the V3b input.
QUIET T1 (runs/monitor4b/quiet_t1.py -> quiet_t1.txt; all 8 FINAL T1, <Q/Q_MHD> 1e-5..1e-4): peak Q/Q_MHD 3.8e-4..1.3e-2
  at t 36-111; Q < 1e-3 for good from t 61-112 (3 runs never reach 1e-3). Final 200 R/vA: E_ky0_frac 0.90-0.99,
  E_ky0kz0_frac 0.90-0.99, E_kx0_frac <= 0.04, S_zon0 0.33-0.89; last record: top |k_x| = 0.5 (88-99 % of E),
  k_y = 0, k_z = 0; 95-100 % of E in u_perp. Reading: a 2D (k_z = 0) zonal-FLOW condensate at the box-scale k_x = 0.5
  (L_x = 4 pi), not a k_x = 0 profile; no turbulent flux phase. Paper note: these T1 Q values may be box-dependent
  zeros (an L_x = 8 pi check would show whether the condensate tracks the box).
