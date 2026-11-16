# SPEC: Z-pinch reduced-MHD system to implement in FLUCS

Source of truth: the paper manuscript, section 2 (`reference/paper/main.tex`, lines ~72-160). This file summarises it.
Do NOT reverse-engineer the equations from the curvy3d solver; use the solver only to check conventions and to compare results.

## Equations (normalised units)

Box: x radial in [0, L_x), y in [0, L_y), z along B0 in [0, 2*pi). Triply periodic. Standard runs: L_y = 2*pi, L_x = 2 L_y = 4*pi.
So k_y is an integer, k_x a multiple of 0.5, k_z an integer (field lines close: |k_z| >= 1 for 3D modes).

Fields: Phi (stream function), Psi (flux function), dBpar (= delta B_parallel), upar (parallel flow).

    u_perp = z x grad Phi        ->  u_x = -d_y Phi,  u_y = d_x Phi
    dB_perp = z x grad Psi       ->  dB_x = -d_y Psi, dB_y = d_x Psi

    d_t Psi                          = dpar Phi
    d/dt (lap Phi)                   = dpar (lap Psi) + 2 d_y dBpar
    (1 + 1/kb) d/dt dBpar            = [kP (1 + kb) - 2] d_y Phi + dpar upar
    d/dt upar                        = dpar dBpar - kb kP d_y Psi

    d/dt  = d_t + {Phi, .}            dpar = d_z + {Psi, .}           {f, g} = d_x f d_y g - d_y f d_x g
    lap   = d_x^2 + d_y^2 (perpendicular Laplacian)

Parameters: kb = kappa_beta = beta*gamma/2 > 0, kP = kappa_P > 0 (both O(1)). Curvature factor is 2 (kappa_B = 1).

Dissipation, applied to all four fields, same form for each (add to the right-hand sides as damping):
    - nu_perp (k_perp^2)^n  - nu_z (k_z^2)^m
  with n = m = 2 (k^4, "power 2" in curvy3d) or n = m = 4 (k^8, "power 4"). Must support both.

## Linear theory (test against this exactly)

For exp(i k.r - i omega t), with c = 1 + 1/kb, kap = k_y^2/k_perp^2:

    c omega^4 + omega^2 [kap c N^2 - (1 + c) k_z^2] + k_z^2 (k_z^2 - kap N0^2) = 0
    N^2 = (2 kb/(1 + kb)) [kP (1 + kb) - 2],   N0^2 = 2 kb kP

omega^2 is always real. Growth rate gamma = sqrt(-min root) if the smaller root is negative, else 0.
With dissipation identical on all fields, gamma_diss = gamma - nu_perp k_perp^(2n) - nu_z k_z^(2m) exactly.
The growth rate depends on k_perp only through kap: it is scale-independent in |k_perp|.

Reference growth rates (no dissipation); columns are (k_x, k_y, k_z); "st" = stable:

| point | kb | kP | (0,1,1) | (0,1,0) | (0.5,1,1) | (1,1,2) | (0,3,1) | (2,1,1) |
|---|---|---|---|---|---|---|---|---|
| A1 | 0.2 | 2.2627417 | st | 0.48829 | st | st | st | st |
| A2 | 0.28284271 | 2.2627417 | 0.26313 | 0.63093 | 0.07635 | st | 0.26313 | st |
| A3a | 1.13137085 | 1.13137085 | 0.71744 | 0.66085 | 0.60060 | st | 0.71744 | st |
| A3b | 1.6 | 0.56568542 | 0.44940 | st | 0.34944 | st | 0.44940 | st |
| A4 | 0.8 | 0.8 | 0.24918 | st | 0.07596 | st | 0.24918 | st |
| S | 0.4 | 0.4 | st | st | st | st | st | st |

## Energy, heat flux, budget

    E    = (1/2) < |grad_perp Phi|^2 + |grad_perp Psi|^2 > + (1/2) < upar^2 + (1 + 1/kb) dBpar^2 >
    Qhat = -(1 + kb) < dBpar u_x > + kb < upar dB_x >          (< > = box average)
    dE/dt = kP Qhat - D                                        (curvature terms cancel in the sum)

The nonlinear terms conserve E exactly. kP*Qhat = P_AW + P_compr in Kawazura's notation.
Mixing-length normalisation used for plots: Q / Q_MHD with Q = Qhat / L_y^2 and Q_MHD = (2 kb kP)^(3/2).

## Conventions of the curvy3d solver (for comparison only)

- Stores the k_y >= 0 half of Fourier space (rfft in y); arrays (nz, nx, half_ny); grids are 2/3-dealiased:
  "padded" 216 x 108 x 256 -> 143 x 71 x 171 modes (G1); 256 x 128 x 256 -> 171 x 85 x 171 ("Y" grid).
- Its saved 'heatflux', 'heatflux_em' and energies are sums over the independent half of Fourier space = HALF the box
  average. Conversion to Q, Q_MHD and Q/Q_MHD: `reference/curvy3d/zpinch_runtools.py`, functions `heat_flux_series`,
  `q_mhd` (docstring at the top explains the factor 2).
- Time stepping: ARS(3,4,3) IMEX RK, linear terms implicit, nonlinear explicit, adaptive dt from a CFL condition with
  advection by u_perp AND by dB_perp (field-line advection), dt <= dt_max = 0.01, maxcfl = 0.5.
- Initial condition: small white noise (amplitude 1e-5 per mode) in all four fields, rand_seed 257.

## Reference results from curvy3d (Q/Q_MHD, box average, mean +- autocorrelation-corrected error)

- A2, G1, k^4, nu_perp = 2e-4, nu_z = 8e-6: Q/Q_MHD = 0.078 +- 0.004 (saturates by t ~ 40-60, tau_int ~ 2.5).
  Same nu on the 1.5x grid: 0.080 +- 0.004. nu = 5e-5/2e-6 on the 1.5x grid: 0.115 +- 0.005.
- High beta with k^4 (A3a, A3b, A4): runs away (Q/Q_MHD -> 1e2-1e4, dt collapses) after a first saturation at
  ~1-10 Q_MHD; the energy budget closes, so this is a property of the k^4-dissipated equations, not an integration error.
- Kawazura's code (k^8, nu_perp = 4.44089e-12, nu_z = 1.38778e-14, 256x128x256): A3a-like (1.13, 1.13) saturates for
  t ~ 1000 at Q/Q_MHD ~ 0.05-5 (typically ~0.5).
- Timing: curvy3d G1 on an H200: ~11 ms per time step (A100: ~26 ms).
