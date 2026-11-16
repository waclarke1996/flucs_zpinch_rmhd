# FLUCS ZPinchFourier run-time diagnostics: reference (flucs `zpinch` @ 16dd8df+ (Stage 4b), 2026-11-16)

Paths: `D/` = `flucs/src/flucs/systems/zpinch_fourier/`; `R/` = `runs/stage4a/`; `C/` = `reference/curvy3d/`;
`.cu` = `D/zpinch_fourier.cu`. Every formula below was read off the code at 54dbc2d; line numbers refer to it.

## 0. Conventions shared by all diagnostics

| item | definition (code) |
|---|---|
| fields | FLUCS index 0 Φ, 1 Ψ, 2 δB∥, 3 u∥ (`D/zpinch_fourier.py:5`). curvy3d `X[i]`: 0 Φ, 1 δB∥, 2 Ψ, 3 u∥. kpar_spectrum.py: (Φ, Ψ, u∥, δB∥) |
| Fourier | f(x) = Σ_k f_k e^{ik·x}, no 1/N (FLUCS = curvy3d, factor 1). Stored half space k_y ≥ 0, padded layout (n_z, n_x, n_y/2+1) |
| box average | ⟨fg⟩ = Σ_{all k} Re f_k g_k* = Σ_{k_y≥0} w_k Re f_k g_k*, **w = 2 for k_y > 0, w = 1 on k_y = 0**. Both members of every (k, −k) pair of the k_y = 0 plane are stored (plane Hermitian), so the w = 1 sum over that plane is complete. All FLUCS quantities are box averages unless stated (`D/zpinch_runtime_diags.py:5-8`; `.cu` reductions; core `reductions.cuh:186` weight 2 at i_y > 0) |
| solved set | 2/3 rule: \|m_x\| ≤ K_x, 0 ≤ m_y ≤ K_y, \|m_z\| ≤ K_z, K = (n_pad − 1)//3 (`ZPGrid.solved_mask` l.157-167, checked against the solver mask at `ready()`, l.169-175). k⊥ = 0 modes are zero. Production T1: 256×128×256 → 171×85×171 modes, K = (85, 42, 85), k_x ≤ 42.5, k_y ≤ 42, \|k_z\| ≤ 85 |
| m, Δk | m = rint(k/Δk), Δk_x = 2π/L_x = 0.5, Δk_y = 2π/L_y = 1, Δk_z = 2π/L_z = 1 (l.102-110) |
| c, s | c = 1 + 1/κ_β (`c_dbpar`), s = √c; Q_MHD = (2κ_βκ_P)^{3/2} (`D/zpinch_fourier.py:117-119`) |
| per-mode energy | e_k = ½ w [k⊥²\|Φ_k\|² + k⊥²\|Ψ_k\|² + c\|δB∥_k\|² + \|u∥_k\|²]; parts (u⊥, b⊥, δB∥, u∥) in this order (`ZPGrid.mode_energy_parts` l.181-185; `.cu` ZPEnergy_Functor l.237-256). Σ_k e_k = E of main.tex eq:energy |
| per-mode flux | q_k = w[−(1+κ_β) Re(δB∥_k u_{x,k}*) + κ_β Re(u∥_k δB_{x,k}*)], u_{x,k} = −ik_yΦ_k, δB_{x,k} = −ik_yΨ_k (l.187-193; `.cu` l.286-306). Σ q_k = Q̂ of eq:Qhat; es = first term, em = second |
| per-mode injection | i_k = w[κ_P(1+κ_β) Re(ik_yΦ_k δB∥_k*) − κ_Pκ_β Re(ik_yΨ_k u∥_k*)] (l.195-199) = κ_P q_k **identically, mode by mode** |
| dissipation | rate r_k = ν⊥(k⊥²)^{p⊥} + ν_z(k_z²)^{p_z} (p = `*_power`: 2 → k⁴, 4 → k⁸), same on all four fields; D_k = 2 r_k e_k (l.129-149; `.cu` l.309-335; core `hyperdissipation.cuh:80-112`). The zp_* diags raise NotImplementedError for adaptive/normalised dissipation or a kx/ky/kmod component with ν > 0 |
| unit bins | **k⊥ (Paper 1 rule, from 2026-11-16 / Stage 4b):** width Δk = Δk_y = 2π/L_y = 1, bin j centred at j; bin 0 = {k⊥ ≤ e_0} with e_0 = ½ min(Δk_x, Δk_y) (only k⊥ = 0), bin j ≥ 1 = (e_{j−1}, e_j], e_j = (j + 0.50001)Δk: exact half-integers go to the LOWER bin (k⊥ = 0.5 → 1, 1.5 → 1, 2.5 → 2). `kperp_bin_edges`, `kperp_bin_index` (`D/zpinch_runtime_diags.py`); edges stored as `kperp_bin_edges` (dim `kperp_edge`, n+1 values [0, e_0, e_1, …]) in zp_spectra and zp_maps. Before Stage 4b: `np.rint(k⊥/Δk_y)` (half to even: 0.5 → 0, 1.5 → 2, 2.5 → 2), which the reference tools (`C/dissipation_map.py:183`, kpar_spectrum.py) still use: their shells differ at exact half-integer k⊥ (expected). k_y: i_y = m_y; \|k_x\|: i_x = \|m_x\|, one bin per allowed value, Δk_x = 2π/L_x **from the run's box** (0.5 for L_x = 4π); \|k_z\|: \|m_z\|. Dims store centres i·Δk |
| units | normalised as main.tex §2 (Φ, Ψ in ρ⊥²v_A/R, δB∥/B₀ in ρ⊥/R, u∥ in v_Aρ⊥/R, t in R/v_A). Q/Q_MHD = Q̂/(L_y² Q_MHD) |

## 1. Time-based output mechanism (`D/zpinch_output.py`)

An `[output.<name>]` with `save_time > 0` becomes `ZPTimedOutputNC` (`build_outputs` l.146-158), else the core
step-based output. Each output → `output.<name>.nc`, group `"0"` (one group per run in the folder) holding `time`
(f8), `dt` (f8, dt of the step just taken), `input_file`, and one subgroup per diagnostic. Records are buffered and
flushed every `[output] write_steps` steps (production 500) and at run end.

| rule | code |
|---|---|
| time check | every `save_steps` steps; unset/≤0 → 1 (l.72-73). Production: unset → every step |
| record trigger | first step with t + ε ≥ t_grid, ε = 1e-9·max(1,\|t\|) (l.123-131). Record time = actual t ∈ [t_grid − ε, t_grid + dt), not t_grid |
| grid | absolute multiples k·save_time (and k·sample_time); `_grid_point_after(t)` = smallest multiple > t + ε (l.40-50) |
| fresh run (t = 0) | records and samples at t = 0 (l.110-111): the initial state is record 0 (one-sample window for windowed diags) |
| restart segment | first record/sample = first grid point > t_start + ε (previous segment recorded the one at its end) (l.107-114). FLUCS ends a segment at the first step ≥ t_init + tfinal (drift ≤ 1 dt/segment), so the end grid point is always recorded by its own segment |
| windows (`sample_time > 0`, save_time must be an integer multiple, l.66-71) | diags with `sample()` (zp_maps, zp_streamer_kzkx) are sampled at the first step past each sample grid point; at a record, `execute()` writes the mean over the samples since the previous record and resets (l.124-141). Sample-before-record in the same step, so window k contains the samples at grid points in (t_{k−1}, t_k]: n = save/sample (10 for maps, 25 for kytavg; 1 for the t = 0 record). `n_samples` is stored |
| incomplete window | dropped at run end (no forced final record: core `execute_diagnostics(force=True)` skips outputs with save_steps = 1) |
| window without samples | no record written (l.133-136) |
| timing | `zp_timing_<name>.txt`: calls, total s, ms/call per diag method (GPU-synchronised, l.79-97) |

**Duplicate-record bug (fixed 2026-11-16, commit a21eb61).** Before the fix the next grid point was computed from t,
not t + ε: a step landing within ε *below* a grid point recorded it and the next step recorded it again (also for
samples). Rate ≈ ε/dt per record (~1 per 1600 records at t = 500). Seen in REALTEST (pre-fix): 3 duplicate 0d records,
1 in 1d/kyt, one maps/kytavg window with 11/26 samples instead of 10/25 (PROGRESS Stage 4a part 3). Example:
`R/realtest/A2-T1-k8/t0500/output.0d.nc` records at t = 481.3999998 and 481.40075.
Readers: `duplicate_mask(t, period)` (`D/zpinch_readers.py:37-55`) keeps the first record of each grid index
rint(t/period) and drops later ones; `load_diag(..., dedup=True)` applies it with `period = timed_period(paths[0])`
(save_time parsed from the `input.toml` next to the first file, l.22-34); step-based outputs (no save_time) untouched.
Not corrected: a duplicated *sample* inside a pre-fix window (visible as n_samples ≠ save/sample).
Verification: `R/review/test_timed_trigger.py` (old code reproduces the duplicate; new: 16 PASS);
`R/chaintest3a/run.log` (chain test rerun on a21eb61, not yet entered in PROGRESS.md): phases P1-P7 "FAIL count 0"
(inspector checks strictly increasing times, unique grid indices, window counts).

**One-step finite differences.** `dEdt`, `budget_error`, `<inv>_dWdt`, `<inv>_budget_error` (energy) and `dtdBx_sq`,
`dtupar_sq` (zp_kytransfer) are backward differences (X(t_n) − X(t_{n−1}))/dt_n using the solver's previous-step
fields (`get_fields(1)`). They are **NaN at step 0** (t = 0 record of a fresh run), since there is no previous step
(`zpinch_fourier_diagnostics.py:185-187, 286-287`; `zpinch_runtime_diags.py:674-675`). Before 7383e0e they were garbage
there (dEdt 6.8e5). They are first order: `budget_error` = O(dt·d²E/dt²), not a round-off check.

## 2. Production files and cadences (`R/campaign.py:53-60`, CADENCE["production"])

| file | diagnostics | save_time | sample_time | records per 100 R/v_A |
|---|---|---|---|---|
| output.0d.nc | `energy` {invariants = true, stop_dt_min = 1e-5, stop_persist_time = 5}, `zp_scalars` | 0.1 | – | 1000 |
| output.1d.nc | `zp_spectra`, `zp_profiles` | 1.0 | – | 100 |
| output.maps.nc | `zp_maps`, `zp_kxky` | 5.0 | 0.5 | 20 (10 samples each) |
| output.kpar.nc | `zp_kpar` | 5.0 | – | 20 |
| output.slices.nc | `zp_slices` | 10.0 | – | 10 |
| output.kyt.nc | `zp_kytransfer` | 1.0 | – | 100 |
| output.kytavg.nc | `zp_streamer_kzkx` | 25.0 | 1.0 | 4 (25 samples each) |
| output.shells.nc | `zp_shells` | 5.0 | – | 20 |

"test" cadence (campaign.py:61-65): maps/kpar/shells 2.5, slices 5, kytavg 5 (sample 1). Example input:
`R/realtest/A2-T1-k8/t0100/input.toml`. Shapes below are for T1 (verified with netCDF4 on
`R/realtest/A2-T1-k8/t0500/output.*.nc`). Overhead (A3a T2, H200): diags a-f 0.4 %, a-h 1.1 % of wall time.
Stage 4b (all production outputs incl. the new spectra and zp_kxky, H200, `runs/stage4b/overhead_4b.py`): see
PROGRESS Stage 4b (T1 2.9 % of the time-step cost; zp_shells 1.1 %, 0d energy + scalars 1.0 %).

## 3. `energy` (0d), `ZPinchEnergyDiag`, `D/zpinch_fourier_diagnostics.py:29-314`

GPU reductions over the padded half grid with weight w (box averages); functors in `.cu`. All variables `(time,)` f8,
instantaneous.

| variable | definition |
|---|---|
| E, E_uperp, E_bperp, E_dbpar, E_upar | Σ e_k and its parts: ½⟨\|∇⊥Φ\|²⟩, ½⟨\|∇⊥Ψ\|²⟩, (c/2)⟨δB∥²⟩, ½⟨u∥²⟩ (`.cu` l.237-256) |
| Qhat, Qhat_es, Qhat_em | Q̂ = −(1+κ_β)⟨δB∥ u_x⟩ + κ_β⟨u∥ δB_x⟩; es/em = the two terms (`.cu` l.286-306) |
| injection | κ_P Q̂ |
| Q_over_QMHD | Q̂/(L_y² Q_MHD) (l.274); NaN if κ_P = 0 |
| D_<comp>, D | D_c = Σ 2 r_{c,k} e_k per hyperdissipation component (kz, kx, ky, kperp, kmod; all written, zero if ν_c = 0); D = Σ_c D_c (`.cu` l.309-321) |
| dEdt, budget_error | (E_n − E_{n−1})/dt_n; dEdt − (κ_P Q̂ − D). NaN at step 0 (§1) |
| dbpar2, dbperp2, dby2 | ⟨δB∥²⟩ (no c), ⟨\|δB⊥\|²⟩ = ⟨\|∇⊥Ψ\|²⟩, ⟨δB_y²⟩ = ⟨(∂_xΨ)²⟩ (`.cu` l.484-498) |
| E_kx0, E_kxn0, E_kz0 (kx_split, default on) | E restricted to k_x = 0, k_x ≠ 0, k_z = 0 (`.cu` l.258-275) |
| E_kx0_<p>, f_kx0_<p> | part p ∈ {uperp, bperp, dbpar, upar} of E_kx0, and its fraction of E_kx0 (NaN if E_kx0 = 0) |
| dEdt_nonlinear (save_nonlinear) | −Σ w Re[k⊥²Φ*N_Φ + k⊥²Ψ*N_Ψ + cδB∥*N_B + u∥*N_u] (d_t f = −N), one extra NL evaluation; should be round-off (`.cu` l.338-366) |

Invariants (option `invariants`, on in production), `.cu` l.367-448, ζ± = Φ ± Ψ, Z± = u∥ ± sδB∥, order zplus, zminus,
Zpplus, Zpminus = W_ζ+, W_ζ−, W_Z+, W_Z−:

| variable | definition |
|---|---|
| `<n>` | W_ζ± = ½⟨\|∇⊥ζ±\|²⟩, W_Z± = ½⟨Z±²⟩; E = (W_ζ+ + W_ζ− + W_Z+ + W_Z−)/2 |
| `<n>_inj` | ζ±: 0. Z±: −κ_βκ_P⟨Z± ∂_yΨ⟩ ± (κ_P(1+κ_β)/s)⟨Z± ∂_yΦ⟩. Σ_n inj/2 = κ_P Q̂ |
| `<n>_exch` | ζ±: 2⟨δB∥ ∂_yζ±⟩; Z±: ∓(2/s)⟨Z± ∂_yΦ⟩ (curvature exchange); Σ_n exch = 0 |
| `<n>_lin`, `<n>_diss` | inj + exch; Σ_c Σ_k 2 r_{c,k} W_k |
| `<n>_dWdt`, `<n>_budget_error` | one-step FD of W; dWdt − (inj + exch − diss). NaN at step 0 |
| `<n>_nl`, `<n>_nl_gross` (save_nonlinear) | NL transfer into W (should vanish) and Σ_k \|per-mode transfer\| |

Early stop (l.206-250): \|Q/Q_MHD\| > stop_q_over_qmhd, or dt < stop_dt_min (step > 0), held continuously >
stop_persist_time → final_time := now; or E_kxn0 > stop_kxn0_over_kx0·E_kx0 (immediate).
**Reader:** `t, d = load_diag(segs_0d, "energy", names=["E", "Q_over_QMHD", "dEdt"])`.
**Verification:** Stage 1 smoke: NL transfer/injection 7e-14; E parts are full-space sums. Stage 2 (PROGRESS l.57-96):
E = ΣW/2 (4e-16), Σinj/2 = κ_PQ̂ (2e-16), Σexch = 0, Σdiss/2 = D; T3 (κ_P = ν = 0) \|ΔE\|/E₀ converges at order 3.00
(ssprk3e); T3b NL transfer/gross ≤ 4e-16; T4 budget dE/dt = κ_PQ̂ − D with a 5-point FD: residual/\|κ_PQ̂\| max 1.7e-3
(cfl 0.5) → 6.6e-6 (cfl 0.125), scaling dt³-dt⁴, NL part 8e-15 (accepted by Will 2026-11-05). kx split: E_kx0 + E_kxn0
= E 1e-16 (PROGRESS l.147-148). Stage 4a verify: E vs dissipation_map.py 2e-16. The one-step `dEdt` itself is O(dt).
**curvy3d:** `heatflux` = (κ_P(1+κ_β) − 2)·Σ_half Re(ik_yΦδB∥*), `heatflux_em` = −κ_Pκ_β Σ_half Re(ik_yΨu∥*), saved
every save_time_hf at t ≈ i·save_time_hf (solver l.3046-3047: first step with t ≥ t_grid − dt/2; no time variable).
Q̂_es = 2·heatflux·(1+κ_β)/(κ_P(1+κ_β) − 2), Q̂_em = 2·heatflux_em/κ_P (`C/zpinch_runtools.py:8-23`).
Invariants: curvy3d `zplus` etc. = W/2 (half-space sums), FLUCS `<n>_lin` = 2 × curvy3d `<n>_inj`, `<n>_diss` = 2 ×
curvy3d `<n>_diss` (PROGRESS l.58-59; curvy3d l.2515-2547). FLUCS records carry their actual time.

## 4. (a) `zp_scalars` (0d), `ZPScalarsDiag`, `D/zpinch_runtime_diags.py:244-300`

All `(time,)` f8, instantaneous; sums over solved modes with w.

| variable | definition |
|---|---|
| E_kx0_frac, E_ky0_frac, E_ky0kz0_frac | E(k_x = 0)/E, E(k_y = 0)/E, E(k_y = k_z = 0)/E |
| S_zon0 | [Σ_{k_y=k_z=0} k_x⁴\|Φ\|²]^{1/2} = rms of the zonal shear ∂_x ū_y of the (y,z)-averaged flow |
| S_zon_all | [Σ_{k_y=0} k⊥⁴\|Φ\|²]^{1/2}: same for the y-averaged (z-dependent) flow |
| W_rms | ⟨(∇⊥²Φ)²⟩^{1/2} (rms vorticity) |
| b_rms, u_rms | (2E_bperp)^{1/2} = ⟨\|δB⊥\|²⟩^{1/2}; (2E_uperp)^{1/2} |
| upar_rms, dbpar_rms | ⟨u∥²⟩^{1/2}, ⟨δB∥²⟩^{1/2} (no c) |
| Q_es_over_QMHD, Q_em_over_QMHD | Q̂_es/(L_y²Q_MHD), Q̂_em/(L_y²Q_MHD) |
| kzrms_<lo>_<hi> | [Σ k_z² e_k / Σ e_k]^{1/2} over lo ≤ k⊥ < hi; edges option `kz_bands` (default 1,2,4,8,16,32 → 5 bands; these are **k⊥** edges); NaN if empty |
| cfl | CFL number of the last step = rate·dt, rate = max_x[(\|u_y\|+\|δB_y\|)n_y/L_y + (\|u_x\|+\|δB_x\|)n_x/L_x], n = solved mode counts (= curvy3d, `.cu` l.148-155) |
| cfl_max, dt_min | max CFL / min dt over all steps since the previous zp_scalars record (system state updated in `finish_time_step`, `D/zpinch_fourier.py:390-398`, reset at each record, l.297-300) |

**Reader:** `t, s = load_diag(segs_0d, "zp_scalars")`.
**Verification:** `R/verify/verify.txt` (A3a-T2 t = 1200 state vs numpy re-implementation / dissipation_map.py):
all ≤ 1e-14 rel (bands ≤ 9.8e-15); Q_es + Q_em = energy Q/Q_MHD 2.7e-16. cfl/cfl_max/dt_min: not independently
checked (t = 0 values only).

## 5. (b) `zp_spectra` (1d), `ZPSpectraDiag`

Instantaneous; f8; Σ over the solved modes in each bin (box-average contributions, w included). Dims `kperp` (Paper 1
unit bins, §0; T1: 61 bins), `ky` (K_y+1 = 43), `kx` (\|k_x\|, K_x+1 = 86 for L_x = 4π: centres 0, 0.5, …, 42.5),
`kz` (\|k_z\|, K_z+1 = 86 on T1); time-independent `kperp_bin_edges(kperp_edge)`.

Elsasser energies per mode (`ZPGrid.mode_elsasser`), names as the 0d invariants of `energy` (zplus, zminus, Zpplus,
Zpminus; the zp_shells set calls the last two zparplus/zparminus): ζ± = Φ ± Ψ, Z± = u∥ ± s δB∥, s = √(1 + 1/κ_β)
(curvy3d `zetaplus`, `zparplus`, solver l.2505-2507):
  zplus_k = ½ w k⊥² \|ζ+_k\|², zminus_k = ½ w k⊥² \|ζ−_k\|², Zpplus_k = ½ w \|Z+_k\|², Zpminus_k = ½ w \|Z−_k\|²,
so per mode (and per bin) (zplus + zminus)/2 = E_uperp + E_bperp and (Zpplus + Zpminus)/2 = E_dbpar + E_upar.

| quantity | `_kperp` | `_ky` | `_kx` | `_kz` |
|---|---|---|---|---|
| E, E_uperp, E_bperp, E_dbpar, E_upar | ✓ | ✓ | ✓ | E only |
| zplus, zminus, Zpplus, Zpminus | ✓ | ✓ | ✓ | ✓ |
| Qhat, Qhat_es, Qhat_em (Σ q_k) | ✓ | ✓ | ✓ | – |
| inj (Σ i_k = κ_P·Qhat exactly) | ✓ | – | – | ✓ |
| D_perp, D_par (Σ 2ν⊥(k⊥²)^{p⊥} e_k, Σ 2ν_z(k_z²)^{p_z} e_k) | ✓ | – | – | ✓ |
| kz2E (Σ k_z² e_k; k_z,rms(k⊥) = (kz2E/E)^{1/2}) | ✓ | – | – | – |

Names `<n>_kperp`, `<n>_ky`, `<n>_kx`, `<n>_kz`. The \|k_x\| spectra sum +k_x and −k_x (box average): **curvy3d's plot
scripts plot only the +k_x half of their (k_x, k_y) spectra, so their k_x > 0 points are about half of ours** (exactly
half on k_y = 0, where ours is ±k_x symmetric). **Reader:** `t, S = load_diag(segs_1d, "zp_spectra"); S["kx"],
S["zplus_kx"]`; bins: `c, e = kperp_bins(file)` (`D/zpinch_readers.py`; older rint files return rint-equivalent edges).
**Verification:** Stage 4a verify.txt (b) (rint bins): vs dissipation_map.py ≤ 6.7e-15. Stage 4b: see §5a.
**curvy3d:** dissipation_map.py uses rint(k⊥) shells (differ at exact half-integer k⊥); curvy3d's own `*_spec_calc`
are half-space time averages without ½, w or k⊥² (§6a).

## 5a. Stage 4b verification of the new spectra, maps and bins (`runs/stage4b/verify/verify.txt`, 81 PASS, 0 FAIL)

V1 (A2-T1 turbulent state, reset to t = 0; records 0, 0.1, 0.2): per bin (zplus+zminus)/2 = E_uperp+E_bperp and
(Zpplus+Zpminus)/2 = E_dbpar+E_upar ≤ 1.2e-14 (k⊥, k_y, \|k_x\|; k_z: Σ W/2 = E 5.6e-15); bin sums = 0d invariants
≤ 1.2e-15; sums over k⊥, k_y, \|k_x\|, \|k_z\| equal ≤ 6.2e-15 (E, W, Qhat, Qhat_es, Qhat_em); zp_kxky (t = 0, n = 1):
Σ_ky (±k_x folded) = `_kx`, Σ_kx = `_ky` ≤ 2.7e-8 (float32); = independent numpy per-mode formula on the seed ≤ 3.9e-8;
= curvy3d `X_spec_calc` formula with its mask on the seed converted by flucs_to_curvy3d.py (§6a relations) ≤ 4.6e-8,
curvy3d's −k_x entries on k_y = 0 exactly 0. V2 (48×24×48, L_x = 8π): Δk_x = 0.25, edges [0, 0.125, 1.50001, …] =
`kperp_bin_edges(0.25, 1)`, k_x = 0.25 (k_y = 0) in bin 1; E, zplus, Zpminus, Qhat in k⊥ and \|k_x\| bins = numpy from
the final state ≤ 1.5e-15. CPU unit test `runs/stage4b/test_kperp_bins.py` (L_x = 2π, 4π, 8π): k⊥ = 0 → 0, k_x = Δk_x
→ 1, 1.5 → 1, 2.5 → 2, integers j → j, every box k⊥ in (e_{j−1}, e_j], all bins used.

## 6. (c) `zp_maps` (maps), `ZPMapsDiag`

Window averages (needs sample_time). 2D maps (time, kperp, kz), **f4, zlib**: `E_map` (Σ e_k), `inj_map` (Σ i_k),
`D_perp_map`, `D_par_map` per (k⊥ bin, \|m_z\|), averaged over the n_samples samples of the window; `n_samples`
(time,) f8; `kperp_bin_edges` as zp_spectra. **Reader:** `t, M = load_diag(segs_maps, "zp_maps")` (window = (t −
save_time, t]). **Verification:** Stage 4a verify.txt (c): f4 rel ≤ 5e-8 vs dissipation_map.py (rint bins); Σ_kz
E_map = E_kperp 3.7e-9; window counts in chain tests. Consumers: `R/f_analyse.py` (dissipation_map quantities per
two-state state; outer bin weighted by its fraction above the cut, `above()`; pile-up mode counts in the file's bins).

## 6a. (c') `zp_kxky` (maps), `ZPKxKyMapDiag` — (k_x, k_y) maps summed over k_z

Window averages like zp_maps (same output, sample 0.5, save 5 R/vA in production), **f4, zlib**. Dims `kx` (signed,
−K_xΔk_x … K_xΔk_x, 2K_x+1 = 171 on T1/T2), `ky` (0 … K_y). Variables `<n>_kxky` for n = E, zplus, zminus, Zpplus,
Zpminus, Qhat: Σ_{k_z} of the per-mode quantity (w included), plus `n_samples`. Box-average convention: the k_y = 0
row holds both members of every conjugate pair, so it is symmetric in k_x. **curvy3d** `X_spec_calc[k_x, k_y]` = time
average of Σ_kz \|mask·X\|² (solver l.2650-2658) with `wavenumber_mask` (l.406-409: on k_y = 0 only k_x > 0, and at
k_x = 0 only k_z > 0), no ½, no w, no k⊥². Hence zplus_kxky = k⊥²·zplus_spec_calc at k_y > 0 (Zp±: = Zp±_spec_calc),
and zplus_kxky(k_x, 0) + zplus_kxky(−k_x, 0) = k⊥²·zplus_spec_calc(k_x, 0) for k_x > 0 (curvy3d's −k_x entries on
k_y = 0 are 0). E_kxky = k⊥²(Phi_spec_calc + Psi_spec_calc) + c·deltaBpar_spec_calc + upar_spec_calc likewise.
Sizes: 6 × 171 × 43 × 4 B = 176 kB per record (3.5 MB per 100 R/vA). **Reader:** `t, M = load_diag(segs_maps,
"zp_kxky"); M["kx"], M["ky"], M["zplus_kxky"]`. **Verification:** §5a.

## 7. (d) `zp_kpar` (kpar), `ZPKparDiag`, l.382-499 — per-record accumulators of `C/kpar_spectrum.py`

Octave k⊥ shells, nominal edges 0.25, 0.75, 1.5, 3, 6, … (shells with lo < min(K_xΔk_x, K_yΔk_y); T1: 7 shells,
lo 0.25 … 24, hi 0.75 … 48). **From Stage 4b the boundary rule of §0:** shell = (lo', hi'] with lo' = lo + 1e-5Δk,
hi' = hi + 1e-5Δk (an exact edge value goes to the LOWER shell: k⊥ = 1.5 now in (0.75, 1.5], k⊥ = 3 in (1.5, 3]),
and the first lower edge = ½ min(Δk_x, Δk_y) (0.25 for L_x = 4π). kpar_spectrum.py keeps [lo, hi): its shells differ
only for k⊥ exactly on an edge. The k_y = 0 plane of a copy is projected Hermitian, f → ½(f + f*_{−k}) (l.456-458).
Products on an (N_z, N_x, N_y) grid, N = next_fast_len(⌈2·pad·K⌉+1), pad = 2 → exact. f_s = f restricted to shell s,
Ψ_< = Ψ restricted to k⊥ ≤ lo' (all lower shells). Dims (time, field, shell), field order **Φ, Ψ, δB∥, u∥**, f8:

| var | definition | var | definition |
|---|---|---|---|
| den | Σ w\|f_s\|² | lab | Σ w k_z²\|f_s\|² |
| full | ⟨(∂_z f_s + {Ψ, f_s})²⟩ | wrong | ⟨(∂_z f_s − {Ψ, f_s})²⟩ |
| large | ⟨(∂_z f_s + {Ψ_<, f_s})²⟩ (= lab if Ψ_< = 0) | E | energy of f_s (weights ½k⊥², ½k⊥², ½c, ½) |
| vort2 | Σ w k⊥⁴\|f_s\|² | b2 (time) | Σ w k⊥²\|Ψ\|² = ⟨\|δB⊥\|²⟩ |
| norm_err (time) | max_s,f \|⟨(∂_z f_s)²⟩_real / lab − 1\| | shell (dim), shell_hi | lo, hi (time-independent) |

**Reader:** `kpar_table(paths, t0, t1, mask)` (`D/zpinch_readers.py:162-186`): sums accumulators over the selected
records, k²_X = ΣX/Σden per field; combined (energy-weighted over fields) k_z,rms, k_par, k_par,<, k_par,wrong =
[Σ_f E_f k²_{X,f}/Σ_f E_f]^{1/2}; chi = [Σvort2_Φ/n]^{1/2}/k_par,Φ; Efrac; b_rms = ⟨b2⟩^{1/2}.
`T = kpar_table(segs_kpar, t0=150); T["k_par"], T["chi"]`. **Verification:** verify.txt (d): all accumulators vs
kpar_spectrum.py --gpu captured locals ≤ 1.3e-15; norm_err 4.4e-16. **curvy3d tool:** identical definitions and
ratio-of-sums averaging; per-field index order differs (tool: Φ, Ψ, u∥, δB∥).

## 8. (e) `zp_profiles` (1d), `ZPProfilesDiag`, l.505-534

Complex Fourier coefficients, f8 (stored `_real`/`_imag`), instantaneous, field order Φ, Ψ, δB∥, u∥:
`line_x` (time, field, kx_line): X_f(k_x, 0, 0), k_x ascending, both signs (171 for T1), X(−k_x) = X(k_x)* →
f̄^{yz}(x) = Σ_kx X e^{ik_x x}. `line_y` (time, field, ky_line): X_f(0, k_y, 0), k_y = 0…K_y →
f̄^{xz}(y) = X(0) + 2 Re Σ_{k_y>0} X e^{ik_y y}. No weights (raw coefficients).
**Reader:** `mean_profiles(paths, nx=256, ny=128, Lx=4π, Ly=2π)` (l.128-159; L_x, L_y are arguments, not read
from the file) returns `x`, `y`, `<field>_x`, `<field>_y` and u_y(x) = ∂_xΦ̄, dB_y(x) = ∂_xΨ̄, shear_x = ∂_x²Φ̄,
jz_x = ∂_x²Ψ̄, curv_x = ∂_x³Φ̄; u_x(y) = −∂_yΦ̄, dB_x(y) = −∂_yΨ̄, shear_y = −∂_y²Φ̄, jz_y = ∂_y²Ψ̄, curv_y = −∂_y³Φ̄,
each (nt, n), via `line_to_real(coeffs, k, L, n, deriv, half)` (l.112-125).
`P = mean_profiles(segs_1d); plt.plot(P["x"], P["shear_x"].mean(0))`.
**Verification:** verify.txt (e): lines exact (0), reader profiles ≤ 1.8e-14; `R/review/check_profiles.py` vs real
space from the full 3D state (A3a t = 1200) ≤ 3e-14, k_y = 0 plane Hermitian 2e-13; K2 (§10) line_y = kyt profiles 1e-12.

## 9. (f) `zp_slices` (slices), `ZPSlicesDiag`, l.540-565

`Phi_z0`, `dBpar_z0` (time, x, y), **f4, zlib**, instantaneous: real-space field at z = 0, x_i = iL_x/n_x,c,
y_j = jL_y/n_y,c, n_x,c = 2K_x+1 = 171, n_y,c = 2K_y+1 = 85 (exact trigonometric evaluation of the solved modes,
Σ over k_z then 2D inverse FFT). Option `fields` (default Phi, dBpar). **Reader:** `load_diag(segs, "zp_slices")`.
**Verification:** verify.txt (f): f4 rel ≤ 4.8e-8 vs numpy.

## 10. (g) `zp_kytransfer` (kyt), `ZPKyTransferDiag`, l.583-714 — curvy3d `subcrit_mode_balances`

Modes on the line k = (0, k_y, 0), k_y = 1…n (n = `n_subcrit_modes`, −1 → K_y). **Per-mode values, no w** (as curvy3d):
the box-average contribution of the ±k_y pair is 2× each. Products are (x,z)-averages on the 3/2 padded grid (exact),
P[k_y] = Fourier coefficient in y: N_Ψ = ik_y(Φ_xΨ)_ky, N_u^R = ik_y(Φ_x u∥)_ky, N_u^M = −ik_y(Ψ_x δB∥)_ky,
d_tf|_NL = −N (the line has k_z = 0: no ∂_z terms, no ν_z). Dims (time, ky_mode) f8:

| variable | definition |
|---|---|
| dBx_sq | ½k_y²\|Ψ_k\|² = ½\|δB_{x,k}\|² |
| dBx_transfer | −k_y² Re(Ψ_k* N_Ψ): NL rate of change of dBx_sq (received by the mode) |
| dBx_diss | ν⊥ k_y^{2p⊥+2}\|Ψ_k\|² (= 2ν⊥k_y^{2p⊥}·dBx_sq) |
| upar_sq | ½\|u∥_k\|² |
| upar_transfer = reynolds + maxwell | −Re(u∥_k*(N_u^R + N_u^M)); reynolds = −Re(u∥*N_u^R) (advection by u⊥), maxwell = −Re(u∥*N_u^M) ({Ψ, δB∥} part of ∇∥δB∥) |
| upar_inj | Re(u∥_k* κ_βκ_P(−ik_y)Ψ_k) (the −κ_βκ_P∂_yΨ source) |
| upar_diss | ν⊥ k_y^{2p⊥}\|u∥_k\|² |
| dtdBx_sq, dtupar_sq | backward one-step FD of dBx_sq, upar_sq; NaN at step 0 |

Budgets: dtdBx_sq ≈ dBx_transfer − dBx_diss; dtupar_sq ≈ upar_transfer + upar_inj − upar_diss (to O(dt)).
Profiles (time, ky_profile = 0…K_y), complex f8, instantaneous: `dBx_profile` = −ik_yΨ(0,k_y,0) (coefficients of the
(x,z)-averaged δB_x(y)), `upar_profile` = u∥(0,k_y,0); `turb_uperp_sq_profile`, `turb_dBpar_sq_profile`,
`turb_dBperp_sq_profile`, `turb_upar_sq_profile` = y-Fourier coefficients of the (x,z)-averages of \|u⊥\|², δB∥², \|δB⊥\|²,
u∥² computed from the fields **without the k_x = k_z = 0 line** ([0] = box average of the turbulent part).
`turb_char_wavenumbers` (time, 12): energy-weighted ⟨k_x²⟩, ⟨k_y²⟩, ⟨k⊥²⟩ (mean squares, no sqrt) of the turbulent
spectra k⊥²\|Φ\|², \|δB∥\|², k⊥²\|Ψ\|², \|u∥\|², in this order (u⊥ 0-2, δB∥ 3-5, δB⊥ 6-8, u∥ 9-11).
`turb_alfvenic_comps` (time, 6): ⟨\|u⊥\|²⟩, ⟨u_x²⟩, ⟨u_y²⟩, ⟨\|δB⊥\|²⟩, ⟨δB_x²⟩, ⟨δB_y²⟩ of the turbulent part (box averages;
weight 2 on one member of each conjugate pair, as curvy3d's wavenumber_mask × turb_mask, l.628-633).
**Reader:** `t, K = load_diag(segs_kyt, "zp_kytransfer"); K["dBx_transfer"][:, 0]` (k_y = 1).
**Verification:** verify.txt (g): all vs numpy re-implementation of curvy3d's formulas ≤ 1.4e-15 (dt* only finite).
`R/review/check_transfers.py` on REALTEST t0500 (production grid, PROGRESS part 3): K1 line budgets vs the solver's
own step change: 1.6e-2 (dBx) / 8.3e-3 (u∥) of max\|T\| [tol 2e-2] at dt 7.5e-4 (O(dt); flipped transfer sign gives
O(1), PROGRESS part 2: 2.0); reynolds + maxwell = transfer [1e-12]; K2 real-space Parseval of turb profiles vs
alfvenic_comps, turbulent + line parts vs the energy diag parts, line_y vs kyt profiles, k_x line Hermitian [1e-10 to
1e-12]: ≤ 6e-14 (part 2: ≤ 2e-15).
**curvy3d differences:** (1) curvy3d's `dBx_profile` is accumulated twice (solver l.2580 and l.2583) in the
initial save before the time loop (current_time = 0), but once per save inside the loop (l.3186, under
`save_step_subcrit_modes`). Read literally, curvy3d_saved = [Σ_saves dBx_k + dBx_k(t=0)]/N_saves, i.e. the mean plus
one extra copy of the initial-state term, **not** 2 × the mean as stated in PROGRESS part 1 (the docstring
l.597-598 and flucs_summary_nc.py:20 said the same; both corrected 2026-11-16; see §16.4). FLUCS stores the true instantaneous value; the summary writer
(`runs/tools/flucs_summary_nc.py:244-247`) writes the plain mean over records. `upar_profile`, turb profiles: one add. (2) curvy3d saves only run averages of the profiles (÷ total_saves_subcrit_modes); FLUCS saves every record.
(3) dt*: curvy3d forward difference over the step after the save (solver l.3066), FLUCS backward. (4) curvy3d's series
has no time axis (record i at t = i·subcrit_mode_hfmult·save_time_hf); the summary writer places FLUCS records on that
grid (NaN where absent) and refuses irregular grids (flucs_summary_nc.py:154-171).

## 11. (g) `zp_streamer_kzkx` (kytavg), `ZPStreamerTransferDiag`, l.717-761

Window averages (needs sample_time), **f4, zlib**, dims (time, kz_c, kx_c) in compact fft order (k_z = fftfreq(2K_z+1)Δk_z,
k_x likewise), plus `n_samples`. With A, B in the compact (k_z, k_x, k_y ≥ 0) layout and
pair(A,B)[k_z,k_x] = Σ_{j≥0} [A*(k_z,k_x,j) B(k_z,k_x,j+1) − A(k_z,k_x,j+1) B*(k_z,k_x,j)]:
`dBx_streamer_transfer_kzkx` = Re[k_x Ψ*(0,0,1) pair(Ψ,Φ)], `upar_streamer_transfer_reynolds_kzkx` =
Re[k_x u∥*(0,0,1) pair(u∥,Φ)], `upar_streamer_transfer_maxwell_kzkx` = −Re[k_x u∥*(0,0,1) pair(δB∥,Ψ)]: the k_y = 1
line transfers resolved by (k_z, k_x) of the triad leg with k_y = j+1 ≥ 1 (partner at (−k_z, −k_x, −j)); Σ over the
map = dBx_transfer / upar_reynolds_transfer / upar_maxwell_transfer at k_y = 1 (per mode, no w).
**Reader:** `t, A = load_diag(segs_kytavg, "zp_streamer_kzkx")`; run mean = Σ n_samples·map / Σ n_samples.
**Verification:** verify.txt (g): f4 ≤ 3.9e-8 vs curvy3d formula; K3 Σ map = k_y = 1 transfers 3e-8 (single-sample
records). **curvy3d:** same formula and layout (solver l.2632-2634) but one run average sampled at the subcrit cadence;
FLUCS: window averages at sample_time (1.0 in production); summary writer combines them weighted by n_samples.

## 12. (h) `zp_shells` (shells), `ZPShellTransferDiag`, l.767-896

Instantaneous, f8, **box averages** (= the plotters' per-snapshot "−2 Σ_half" values). Shell edges as the
hydra_shell_plotter_*_gpu.py scripts: K_i = k_min·2^i, i = 0…n; k⊥: k_min = 2π/max(L_x,L_y)/2.00001 = 0.249999,
k⊥,max = [(K_xΔk_x)² + (K_yΔk_y)²]^{1/2}; k_z: k_min = 2π/L_z/2.00001, k_z,max = K_zΔk_z; n = ⌈log₂(k_max/k_min)⌉ + 1.
Mode → shell j with K_j ≤ k < K_{j+1}, clipped at 0 (k < K_0 → 0). T1 edges: k⊥ 0.25, 0.5, 1, …, 128 (9 shells: shell 0
empty since min k⊥ = 0.5, shell 8 [64,128) empty); k_z 0.5, 1, …, 256 (9 shells: shell 0 = k_z = 0 only, shell 1 = \|k_z\| = 1,
shell 8 [128,256) empty). Dims (time, `<st>_from`, `<st>_to`) with coordinates K[:-1]; edges in `<st>_K_boundaries`
(`<st>_edges`, time-independent); st ∈ {kperp, kz} (option `shell_types`), sets option `sets` = elsasser, magkin.
Products computed once per source shell on the 3/2 padded grid; exact triad sums.

**Sign convention (derived from `_V` l.842-849 and `_S` l.851-855): T[j, l] = rate of energy transfer INTO shell l
(receiving field) FROM shell j (source field), positive = shell l gains.** X_l = X restricted to shell l;
u = ẑ×∇Φ, b = ẑ×∇Ψ, v± = ẑ×∇ζ±, ζ± = Φ ± Ψ, Z± = u∥ ± sδB∥, {a, f} = a_x f_y − a_y f_x:

| name | T[j, l] | energy |
|---|---|---|
| zplus / zminus | −⟨v±_l·(v∓·∇)v±_j⟩ | W_ζ± |
| zparplus / zparminus | −⟨Z±_l {Φ ∓ Ψ/s, Z±_j}⟩ | W_Z± |
| Tuu | −⟨u_l·(u·∇)u_j⟩ | ½\|u⊥\|² |
| TBB | −⟨b_l·(u·∇)b_j⟩ | ½\|δB⊥\|² |
| TuB (u_j → b_l) | +⟨b_l·(b·∇)u_j⟩ | ½\|δB⊥\|² |
| TBu (b_j → u_l) | +⟨u_l·(b·∇)b_j⟩ | ½\|u⊥\|² |
| TBparBpar | −c⟨δB∥_l {Φ, δB∥_j}⟩ | (c/2)δB∥² |
| Tuparupar | −⟨u∥_l {Φ, u∥_j}⟩ | ½u∥² |
| TBparupar (δB∥_j → u∥_l) | +⟨u∥_l {Ψ, δB∥_j}⟩ | ½u∥² |
| TuparBpar (u∥_j → δB∥_l) | +⟨δB∥_l {Ψ, u∥_j}⟩ | (c/2)δB∥² |

Identities: self-transfers antisymmetric T = −Tᵀ; TuB = −TBuᵀ; TBparupar = −TuparBparᵀ; Σ_{j,l} of each conserving
set = 0; per receiving shell ½Σ_j(zplus+zminus) = Σ_j(Tuu+TBu+TBB+TuB), ½Σ_j(zparplus+zparminus) = Σ_j(TBparBpar+
TBparupar+Tuparupar+TuparBpar) (E⊥ = (W_ζ++W_ζ−)/2, E_comp = (W_Z++W_Z−)/2). Curvature exchange, injection and
dissipation are not in these matrices.
Shell energies (Stage 4b): `<st>_E_<part>` (time, `<st>_to`), part ∈ {uperp, bperp, dbpar, upar}: Σ e_k (box average)
of the modes in shell l, so dE_l/dt = Σ_j T[j, l] + (linear exchange, injection, dissipation in shell l).
**Reader:** `t, S = load_diag(segs_shells, "zp_shells"); S["kperp_Tuu"].mean(0)`; plotter CSV/JSON:
`runs/tools/zp_shells_export.py` (CSV = time mean / L_y², as the scripts; sums uperp = Tuu+TBu, deltaBperp = TBB+TuB,
deltaBpar = TBparBpar+TuparBpar, upar = Tuparupar+TBparupar).
**Verification:** verify.txt (h): both plotters run unmodified on the same state, FLUCS = CSV × L_y², no transpose,
≤ 7.2e-14 (all 12 × 2 shell types; T2 grid). K4 (check_transfers.py) on REALTEST t0500: antisymmetries, Σ = 0 and the
two per-shell budgets ≤ 8e-14 [1e-10]. **Dynamical sign test (Stage 4b, `runs/stage4b/verify/V3_signtest`,
compare_4b.py):** A2-T1 turbulent state with κ_P = 0 and ν = 0 (then every linear term, curvature exchange included,
conserves each mode's energy), dt = 1e-4, zp_shells every step: the central difference of the shell energies
dE_l/dt = Σ_j T[j, l] (kin/mag set) = ½Σ_j T (Elsasser set) to 8.8e-7 (k⊥ shells) and 4.3e-7 (k_z shells) of max|T|
over 8 steps, same sign on every active shell; total E constant to 7e-11. So T[j, l] is the rate INTO shell l
FROM shell j (sign confirmed).

## 13. Other diagnostics in `zpinch_fourier_diagnostics.py`

`energy_1d` (`ZPinchEnergyDiag1D`, l.317-383): `<sp>_spectra/{E, E_uperp, E_bperp, E_dbpar, E_upar, Qhat, D}` for
sp ∈ {kperp (default), kx, ky, kz}, FLUCS core reduction bins (`shell_kperp` etc., not the zp unit bins), box averages.
Not used in production; compared with curvy3d in Stage 3 (`flucs/tests/zpinch/check_S31.py`), not re-verified in 4a.

## 14. Readers (`D/zpinch_readers.py`, CPU; importing it imports the flucs package → harmless CuPy warnings on the login node)

```python
import glob, sys; sys.path.insert(0, "/mnt/users/clarkew/flucs_work/flucs/src")
from flucs.systems.zpinch_fourier.zpinch_readers import load_diag, mean_profiles, kpar_table
segs = lambda o: sorted(glob.glob(f"RUN/t[0-9][0-9][0-9][0-9]/output.{o}.nc"))   # chain order
t, d = load_diag(segs("0d"), "energy")          # dict name -> (nt, ...); complex vars recombined; dims/static no time axis
```
`load_diag(paths, diag, names=None, dedup=True)` (l.58-109): concatenates all groups of all files in the given order
(no overlap handling: segments of a chain do not overlap); dedup as §1. Analysis consumer: `R/f_analyse.py`
(Q/Q_MHD series, two-state labels, spectra pile-up, maps per state, kpar per state). curvy3d-format summary for
build_registry.py / make_flux_table.py: `runs/tools/flucs_summary_nc.py` (`heatflux` = Q̂_es(κ_P(1+κ_β)−2)/(2(1+κ_β)),
`heatflux_em` = Q̂_em κ_P/2, interpolated onto t = i·save_time_hf, so `heat_flux_series` returns Q̂/L_y²; verified to
8e-13, PROGRESS part 2 (4); REALTEST via make_flux_table Q/Q_MHD 0.0882 = f_analyse).

## 15. Summary of conversion factors to curvy3d

| quantity | curvy3d saved | FLUCS | conversion |
|---|---|---|---|
| heatflux, heatflux_em | half-space sums × solver prefactors | Qhat_es, Qhat_em (box) | Q̂_es = 2 hf (1+κ_β)/(κ_P(1+κ_β)−2); Q̂_em = 2 hf_em/κ_P; Q = Q̂/L_y² |
| energies, invariants W | half-space sums | box averages | FLUCS = 2 × curvy3d |
| `<inv>_inj`, `<inv>_diss` | half-space, inj = exch + inj | `<inv>_lin`, `<inv>_diss` | FLUCS = 2 × curvy3d |
| subcrit balances (dBx_sq … upar_maxwell_transfer) | per mode | per mode | 1 (dt*: forward vs backward FD) |
| dBx_profile | run mean + initial-state term/N (double add only at t = 0, §16.4) | instantaneous true value | ≈ 1 for a noise start; check §16.4 |
| upar_profile, turb_* profiles, char wavenumbers, alfvenic comps | run mean / per save | per record | 1 (average FLUCS records) |
| streamer kzkx maps | run mean | window means, n_samples | 1 (n_samples-weighted mean) |
| shell transfers | CSV = mean / L_y² | per record, box average | CSV = mean(FLUCS)/L_y² |
| `X_spec_calc` (k_x, k_y) | time mean Σ_kz \|mask·X\|², half space (k_y = 0: k_x > 0 only), no ½, w, k⊥² | `zp_kxky` window means, box average | zplus_kxky = k⊥²·zplus_spec_calc (k_y > 0); ours(k_x,0)+ours(−k_x,0) = k⊥²·curvy(k_x,0) (§6a) |
| k_x spectra in plot scripts | only the +k_x half plotted | `<n>_kx`: \|k_x\| bins, ±k_x summed | curvy3d's k_x > 0 points ≈ ½ of ours |
| k⊥ unit shells (dissipation_map, kpar_spectrum) | rint(k⊥) / [lo, hi) | Paper 1 rule (§0) | differ only at exact half-integer / edge k⊥ |
| time axis | implicit i·save_time_hf (record at first t ≥ t_grid − dt/2) | stored actual t (first t ≥ t_grid − ε) | offsets < 1 dt |

## 16. Discrepancies / open points

1. `zp_scalars` option `kz_bands` holds **k⊥** band edges (`zpinch_runtime_diags.py:247, 267`); `kzrms_<lo>_<hi>`
   = k_z,rms in k⊥ ∈ [lo, hi). Name only; docstring (l.14-15) is right.
2. (Resolved in Stage 4b.) Unit k⊥ bins used `np.rint` (half to even: k⊥ = 0.5 in bin 0, 1.5 → 2, 2.5 → 2). Now the
   Paper 1 rule (§0) with stored edges; the reference tools (dissipation_map.py:183, kpar_spectrum.py) still use rint /
   [lo, hi), so their shells differ at exact half-integer (edge) k⊥ — expected. zp_scalars `kz_bands` keep [lo, hi).
3. `zp_kpar` field order Φ, Ψ, δB∥, u∥ (l.409, 465) vs kpar_spectrum.py (Φ, Ψ, u∥, δB∥; `C/kpar_spectrum.py:48`):
   per-field rows are permuted; combined values and chi (Φ) identical.
4. curvy3d `dBx_profile`: the duplicated `+=` is at `C/curvy3dsolver_rmhd_zpinch_cupy_ARS343.py:2580, 2583`, inside
   the initial-save block of `solve()` (after "Reset stuff", current_time = 0); the in-loop accumulation (l.3186)
   adds once. So curvy3d's saved profile = mean + dBx(t=0)/N_saves (negligible after a 1e-5 noise start, not after a
   kick/snapshot start), whereas PROGRESS Stage 4a part 1, the `ZPKyTransferDiag` docstring
   (`zpinch_runtime_diags.py:597-598`, "adds dBx_profile twice") and `runs/tools/flucs_summary_nc.py:20` ("twice per
   save") said 2 × mean (docstring and summary-writer comment corrected 2026-11-16; checked by the main session at
   solver l.2560-2590 and l.3170-3190). FLUCS (l.693) and the summary writer store the true average either way, so no
   FLUCS number is affected. Open for Will: whether the initial save block also runs on a curvy3d restart (not
   checked), and whether to delete the duplicate at curvy3d l.2583.
5. `dtdBx_sq`/`dtupar_sq`: FLUCS backward (`zpinch_runtime_diags.py:675-684`) vs curvy3d forward (solver l.3066);
   K1 residuals are O(dt) (1.6e-2 at dt 7.5e-4) for this reason, not round-off.
6. `dEdt`/`budget_error` (`zpinch_fourier_diagnostics.py:287-289`) are first-order one-step differences evaluated
   against the end-of-step RHS: `budget_error` is O(dt), not a conservation check (T4 used a 5-point offline FD).
7. Pre-fix outputs (before a21eb61): readers drop duplicate records (`zpinch_readers.py:37-55, 99`) but a window in
   which a sample was duplicated keeps the extra sample (n_samples = 11/26 instead of 10/25). `load_diag` takes the
   period from the first file's `input.toml` only (l.99); without an `input.toml` next to it, no dedup (silently).
8. Shell coordinates (`zpinch_runtime_diags.py:809-832`): `kz_from[0]` = 0.5 although shell 0 holds k_z = 0 only;
   `kperp` shell 0 and the last shell of each type are empty on T1 (n = ⌈log₂⌉ + 1 adds one). Same as the plotter scripts.
9. (Closed in Stage 4b.) Shell-transfer sign: dynamical test with κ_P = 0, ν = 0 (§12): T[j, l] = into l from j.
10. `zp_scalars` `cfl_max`/`dt_min` use system-level state reset by every zp_scalars execution
    (`zpinch_runtime_diags.py:297-300`, `zpinch_fourier.py:390-398`): listing zp_scalars in two outputs would make
    them interfere (production: 0d only). Not independently verified.
11. `mean_profiles` takes L_x, L_y as arguments (defaults 4π, 2π; `zpinch_readers.py:128`), not from the file.
12. `inj_*` in zp_spectra/zp_maps equals κ_P × the per-mode Q̂ exactly (l.195-199): `inj_kperp` ≡ κ_P·`Qhat_kperp`
    (redundant, kept for dissipation_map.py naming).
13. Code vs docstrings otherwise consistent: `ZPinchEnergyDiag` docstring (l.37-48) and `.cu` l.367-381 match the
    functors; the zp_kytransfer formulas (l.586-596) match the code including signs (only the curvy3d remark of
    item 4 is in doubt).
14. (Closed.) chaintest3a (A100) FAIL count 0 in P1-P7, recorded in PROGRESS Stage 4a part 4.
