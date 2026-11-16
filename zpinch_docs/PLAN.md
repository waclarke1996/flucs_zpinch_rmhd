# PLAN: Z-pinch RMHD in FLUCS, verified against theory and against curvy3d

Work in stages. At the end of EVERY stage: append a short entry to `PROGRESS.md` (what was done, test results with
numbers, open problems, next step), commit in `flucs/` on branch `zpinch`, and STOP unless the stage says otherwise.
Will restarts you with "continue with stage N". Keep each stage's report under ~40 lines.

## Stage 0 - read and plan (plan mode, no code changes)
- Read FLUCS's README/docs and one or two existing physics systems end to end. Find: how a new system of equations is
  defined; how linear terms, nonlinear brackets, dissipation and time stepping are specified; how output is written;
  how runs are configured and launched on a GPU.
- Write `NOTES_flucs.md` (<= 1 page): the files you would add or change, and how each term of SPEC.md maps onto FLUCS.
- Write the concrete test list for stages 2-3 (from this file) with the commands you will use. STOP.

## Stage 1 - implement (in `flucs/`, branch `zpinch`)
- Add the four-field Z-pinch system exactly as in SPEC.md, with parameters kb, kP, L_x, L_y, nu_perp, nu_z and the
  dissipation order (k^4 or k^8). Add diagnostics: E, kP*Qhat (= injection), D, Q/Q_MHD, delta B^2, dt, as time series.
- Small CPU or small-GPU smoke test only (e.g. 32^3, a few steps). STOP.

## Stage 2 - verify against theory (small grids; GPU via the allocation described in CLAUDE.md)
Acceptance criteria (all must pass; report the numbers):
1. Linear growth rates: single-mode or low-amplitude runs reproduce the SPEC.md table to relative error < 1e-3
   (fit the exponential over >= 5 e-folds), for at least A2, A3a, A3b, A4 and one stable case (oscillation frequency).
2. Dissipation: with nu > 0, gamma_diss = gamma - nu_perp k_perp^(2n) - nu_z k_z^(2m) to < 1e-3, for n = 2 and n = 4.
3. Energy conservation: kP = 0 (no injection; the curvature exchange terms remain), nu = 0, finite-amplitude
   random initial condition, short run: E must be conserved to time-stepping accuracy (show the error falls with dt
   at the scheme's order).
4. Energy budget: in a turbulent run, dE/dt = kP*Qhat - D with relative residual < 1e-4 (instantaneous).
5. Invariant subspace: initialise ONLY k_x = 0 modes at large amplitude; the run must stay in k_x = 0 and grow at
   exactly the linear rate (all Poisson brackets vanish there). This is a sharp test of the nonlinear terms.
STOP after writing the results.

## Stage 3 - compare with curvy3d (same box, same grid, same parameters)
1. A2, k^4, nu_perp = 2e-4, nu_z = 8e-6, modes 143 x 71 x 171, t = 0..300: Q/Q_MHD over t = 60..300 must agree with
   0.078 +- 0.004 within errors; compare the time-averaged k_perp spectrum shape (slope ~ -1.7 for k = 2-7).
2. Speed: wall time per time step at that grid on the same GPU type as curvy3d (H200: ~11 ms; A100: ~26 ms), and
   the average dt. Report both.
3. Optional, only if Will says so: A3a with k^8 (Kawazura's settings): does it saturate past t ~ 100?
STOP.

## Stage 3b (done 6 Nov) - plugin noise, kx split, streamer secondary test, A3a-Y-k8 to t = 500. See PROGRESS.md.

## Stage 3c - why do FLUCS and curvy3d differ at A3a-Y-k8?  (two sessions: part 1 = prepare/submit, part 2 = analyse)
Background. FLUCS A3a-Y-k8-S30c, t = 191-500: <Q/Q_MHD> 3.22 +- 0.44, <E> 1118, E_kx0/E 0.58; curvy3d A3a-Y 0.63 +- 0.11,
<E> 205 (both codes had a high phase at t ~ 100-200; curvy3d dropped to a low state after t ~ 190 and stayed there
to t > 739). New curvy3d runs (same seed/IC as A3a-Y; dissipation_map.py averages):
  | curvy3d run | change vs A3a-Y        | window  | E   | b_rms | alpha | f_par | E(k_perp<=1)/E | outcome                |
  | A3a-Y       | -                      | 200-500 | 192 | 10.5  | 0.48  | 0.59  | 0.55           | saturated, 0.63 Q_MHD  |
  | A3a-YP4     | nu_perp x4             | 200-500 | 831 | 21.3  | 0.36  | 0.67  | 0.69           | saturated, high state  |
  | A3a-YZ4     | nu_z x4                | -       | -   | -     | -     | -     | -              | runaway, then "locks"  |
  | A3a-Z       | 341 k_z modes, nu_z/16 | 150-300 | 363 | 14.3  | 0.54  | 0.75  | 0.64           | saturated              |
So bounded high-amplitude states exist in curvy3d too, and nu_z (not nu_perp) decides runaway at a fixed grid. The
FLUCS/curvy3d difference may be bistability (long-lived low and high states) or a scheme effect.

Part 1 (one session; no interactive GPU unless Will says so):
- A1 (CPU): write runs/tools/flucs_to_curvy3d.py: FLUCS state (restart.nc or a saved 3D state) -> minimal curvy3d-format
  .nc that reference/curvy3d/dissipation_map.py and kpar_spectrum.py read unchanged (read how they and
  zpinch_runtools.params_from_dataset read a file: <field>_rfft_r/_i [nt, nz, nx, half_ny] in curvy3d's mode layout,
  the 'input' string, save_time; heatflux/heatflux_em series optional). Check: E and Q/Q_MHD printed by
  dissipation_map.py on the converted A3a-Y-k8-S30c t = 500 state equal the FLUCS energy diagnostic at that time
  (<= 1e-10 relative). Run both scripts on it and on curvy3d A3a-Y (t >= 400) with --out into runs/stage3c/
  (kpar_spectrum.py: --pad 1.5 or --every 2 on the login node; queue it as run.sh with --gpu if it takes > 10 min).
- B (queue, one job): streamer case C6 = C1 but L_x = L_y = 2 pi (square grid, padded 216 x 216 x 256), E_s = 30, once
  with k^4 (as C1) and once with k^8 (Kawazura's coefficients), NL + linear control (4 runs), stop rules as Stage 3b.
  In this box no Kelvin-Helmholtz secondary with k_x < k_y = 1 exists.
- A2 (queue, two jobs): FLUCS from curvy3d A3a-Y's t = 300 snapshot (low state) to t = 600, no Q stop (safety dt < 1e-5
  only), diagnostics as S30c, full 3D state saved every 50 (in a form the converter reads): A3a-Y-k8-C300 (ssprk3e,
  max_cfl 0.5) and A3a-Y-k8-C300-cfl025 (max_cfl 0.25). curvy3d's own continuation of that state is A3a-Y (to 500)
  plus A3a-Yc2 (500 onward) in the data folder.
- Submit, confirm the jobs started, write the plan and job ids in PROGRESS.md, commit, end the turn.
Part 2 (after the jobs finish; "continue: analyse Stage 3c"):
- B: sigma_NL peak, sigma at E_kx0 = 300, t_end and reason, for k^4 and k^8, vs C1-S30 (20.6-22.8; 15.5; breakup at
  E_kx0 ~ 300-400).
- A2: one plot of Q/Q_MHD(t) from t = 300 for both FLUCS runs and curvy3d (A3a-Y + Yc2); time when |ln Q_F/Q_C| first
  exceeds 0.1 (the codes should track each other for a few R/vA from the same state); <Q/Q_MHD>, <E>, E_kx0/E over
  300-450 and 450-600 for all three; dissipation_map.py and kpar_spectrum.py on the converted t = 450, 600 states.
  Reading: both FLUCS runs stay near the curvy3d level -> the codes agree, S30c sat in another long-lived state
  (bistability). Both climb to ~3 within tens of R/vA while curvy3d stays low -> systematic difference between the
  codes: report it and stop (don't investigate). Mixed, or switching late -> inconclusive (needs an ensemble): report.
- <= 40 lines in PROGRESS.md, commit, STOP.

## Stage 4 (later, only when asked) - output compatible with Will's diagnostics
Write netCDF output using the curvy3d variable names and conventions (half-plane storage, same attribute names) so
`reference/curvy3d/*.py` diagnostics run unchanged. Not before Stages 2-3 pass.
