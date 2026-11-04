/*
 * CUDA kernels for the four-field Z-pinch reduced-MHD system
 * (W. Clarke, paper section 2; ~/flucs_work/SPEC.md).
 *
 *   d_t Psi                 = dpar Phi
 *   d/dt lap Phi            = dpar lap Psi + 2 d_y dBpar
 *   (1 + 1/kb) d/dt dBpar   = [kP (1 + kb) - 2] d_y Phi + dpar upar
 *   d/dt upar               = dpar dBpar - kb kP d_y Psi
 *
 *   d/dt = d_t + {Phi, .},  dpar = d_z + {Psi, .},  {f, g} = f_x g_y - f_y g_x
 *
 * Field order: 0 Phi, 1 Psi, 2 dBpar, 3 upar.
 * FLUCS convention: d_t f = -M f - N, where M is what get_linear_matrix
 * returns (propagator = Pade approximant of exp(-M dt)) and N is what
 * add_nonlinear_terms adds. Below L = -M is the physical d_t f = L f.
 * Brackets are written in conservative form {f, g} = d_y(g f_x) - d_x(g f_y).
 *
 * kperp = 0 modes (kx = ky = 0): all brackets vanish identically there and
 * the vorticity equation is 0 = 0, so the Phi row of L is set to zero and
 * no nonlinear terms are added. Initial conditions zero these modes (as
 * curvy3d does), so they stay zero.
 *
 * Constants defined from Python: KB, KP, C_DBPAR (= 1 + 1/kb),
 * G_CURV (= kP (1 + kb) - 2), KBKP (= kb kP).
 */

#include "flucs/solvers/fourier/fourier_system.cuh"

extern "C" {

////////////////////////////////////////////////////////////////////////////////
// Core solver functions
////////////////////////////////////////////////////////////////////////////////

__device__ void get_linear_matrix(
    const size_t index,
    const FLUCS_FLOAT dt,
    const double current_time,
    const long long current_step,
    FLUCS_COMPLEX matrix[4][4]
){
    indices3d_t indices = get_indices3d<NZ, NX, HALF_NY>(index);

    const FLUCS_FLOAT kx = kx_from_ikx(indices.ikx);
    const FLUCS_FLOAT ky = ky_from_iky(indices.iky);
    const FLUCS_FLOAT kz = kz_from_ikz(indices.ikz);
    const FLUCS_FLOAT kperp2 = kx*kx + ky*ky;
    const bool kperp_zero = (indices.ikx == 0 && indices.iky == 0);

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            matrix[i][j] = FLUCS_COMPLEX(0, 0);
        }
    }

    // Phi: d_t Phi = i kz Psi - 2 i ky / kperp^2 dBpar
    if (!kperp_zero) {
        matrix[0][1] = FLUCS_COMPLEX(0, kz);
        matrix[0][2] = FLUCS_COMPLEX(0, -((FLUCS_FLOAT)2.0) * ky / kperp2);
    }

    // Psi: d_t Psi = i kz Phi
    matrix[1][0] = FLUCS_COMPLEX(0, kz);

    // dBpar: d_t dBpar = (i ky G Phi + i kz upar) / c
    matrix[2][0] = FLUCS_COMPLEX(0, ky * G_CURV / C_DBPAR);
    matrix[2][3] = FLUCS_COMPLEX(0, kz / C_DBPAR);

    // upar: d_t upar = -i ky kb kP Psi + i kz dBpar
    matrix[3][1] = FLUCS_COMPLEX(0, -ky * KBKP);
    matrix[3][2] = FLUCS_COMPLEX(0, kz);

    // FLUCS wants M = -L
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            matrix[i][j] = -matrix[i][j];
        }
    }
}

// Fourier-space inputs for the nonlinear terms:
// 0 dxPhi, 1 dyPhi, 2 dxPsi, 3 dyPsi, 4 kperp^2 Phi, 5 kperp^2 Psi,
// 6 dBpar, 7 upar
__global__ void find_derivatives(
    const FLUCS_COMPLEX fields_global[NUMBER_OF_FIELDS][HALFSIZE],
    FLUCS_COMPLEX dft_derivatives_global[NUMBER_OF_DFT_DERIVATIVES][HALFSIZE]
){
    const size_t index = blockDim.x * blockIdx.x + threadIdx.x;
    if (!(index < HALFSIZE))
        return;

    indices3d_t indices = get_indices3d<NZ, NX, HALF_NY>(index);
    const size_t ikx = indices.ikx;
    const size_t iky = indices.iky;
    const size_t ikz = indices.ikz;

    if (is_mode_padded(ikz, ikx, iky)) {
        #pragma unroll
        for (int i = 0; i < NUMBER_OF_DFT_DERIVATIVES; i++)
            dft_derivatives_global[i][index] = 0;
        return;
    }

    const FLUCS_FLOAT kx = kx_from_ikx(ikx);
    const FLUCS_FLOAT ky = ky_from_iky(iky);
    const FLUCS_COMPLEX dx = dx_from_ikx(ikx);
    const FLUCS_COMPLEX dy = dy_from_iky(iky);
    const FLUCS_FLOAT kperp2 = kx*kx + ky*ky;

    const FLUCS_COMPLEX phi = fields_global[0][index];
    const FLUCS_COMPLEX psi = fields_global[1][index];

    dft_derivatives_global[0][index] = dx * phi;
    dft_derivatives_global[1][index] = dy * phi;
    dft_derivatives_global[2][index] = dx * psi;
    dft_derivatives_global[3][index] = dy * psi;
    dft_derivatives_global[4][index] = kperp2 * phi;
    dft_derivatives_global[5][index] = kperp2 * psi;
    dft_derivatives_global[6][index] = fields_global[2][index];
    dft_derivatives_global[7][index] = fields_global[3][index];
}

// Real-space products (kphi = -lap Phi, kpsi = -lap Psi):
// 0 P  = Phi_x Psi_y - Phi_y Psi_x                  ({Phi, Psi})
// 1 Ax = Phi_x kphi - Psi_x kpsi,   2 Ay = Phi_y kphi - Psi_y kpsi
// 3 Bx = Phi_x dB - Psi_x upar/c,   4 By = Phi_y dB - Psi_y upar/c
// 5 Cx = Phi_x upar - Psi_x dB,     6 Cy = Phi_y upar - Psi_y dB
__global__ void find_nonlinear_bits(
    const FLUCS_FLOAT real_derivatives_global
        [NUMBER_OF_DFT_DERIVATIVES][FULLSIZE],
    FLUCS_FLOAT real_bits_global[NUMBER_OF_DFT_BITS][FULLSIZE],
    const bool calculate_cfl,
    FLUCS_FLOAT* cfl_rate_global
){
    const size_t index = blockDim.x * blockIdx.x + threadIdx.x;
    const bool in_bounds = index < FULLSIZE;

    // Inactive threads contribute zero but must take part in the CFL reduction
    const FLUCS_FLOAT phix = in_bounds ? real_derivatives_global[0][index] : (FLUCS_FLOAT)0;
    const FLUCS_FLOAT phiy = in_bounds ? real_derivatives_global[1][index] : (FLUCS_FLOAT)0;
    const FLUCS_FLOAT psix = in_bounds ? real_derivatives_global[2][index] : (FLUCS_FLOAT)0;
    const FLUCS_FLOAT psiy = in_bounds ? real_derivatives_global[3][index] : (FLUCS_FLOAT)0;

    if (calculate_cfl) {
        // Same as curvy3d: (|u_y| + |dB_y|) ny/Ly + (|u_x| + |dB_x|) nx/Lx,
        // with nx, ny the unpadded (solved) mode counts.
        const FLUCS_FLOAT cfl =
              (flucs_fabs(phix) + flucs_fabs(psix)) * (NY_UNPADDED / LY)
            + (flucs_fabs(phiy) + flucs_fabs(psiy)) * (NX_UNPADDED / LX);
        update_cfl(cfl, cfl_rate_global);
    }

    if (!in_bounds)
        return;

    // Arrays may alias: load everything before writing
    const FLUCS_FLOAT kphi = real_derivatives_global[4][index];
    const FLUCS_FLOAT kpsi = real_derivatives_global[5][index];
    const FLUCS_FLOAT dbpar = real_derivatives_global[6][index];
    const FLUCS_FLOAT upar = real_derivatives_global[7][index];
    const FLUCS_FLOAT upar_c = upar / C_DBPAR;

    real_bits_global[0][index] = phix * psiy - phiy * psix;
    real_bits_global[1][index] = phix * kphi - psix * kpsi;
    real_bits_global[2][index] = phiy * kphi - psiy * kpsi;
    real_bits_global[3][index] = phix * dbpar - psix * upar_c;
    real_bits_global[4][index] = phiy * dbpar - psiy * upar_c;
    real_bits_global[5][index] = phix * upar - psix * dbpar;
    real_bits_global[6][index] = phiy * upar - psiy * dbpar;
}

// N such that d_t f = L f - N (= -M f - N)
__device__ void add_nonlinear_terms(
    const size_t index,
    const FLUCS_FLOAT dt,
    const double current_time,
    const long long current_step,
    const FLUCS_COMPLEX dft_bits_global[NUMBER_OF_DFT_BITS][HALFSIZE],
    FLUCS_COMPLEX explicit_terms[NUMBER_OF_FIELDS]
){
    indices3d_t indices = get_indices3d<NZ, NX, HALF_NY>(index);
    const size_t ikx = indices.ikx;
    const size_t iky = indices.iky;

    // All brackets vanish at kperp = 0
    if (ikx == 0 && iky == 0)
        return;

    const FLUCS_FLOAT kx = kx_from_ikx(ikx);
    const FLUCS_FLOAT ky = ky_from_iky(iky);
    const FLUCS_COMPLEX dx = dx_from_ikx(ikx);
    const FLUCS_COMPLEX dy = dy_from_iky(iky);
    const FLUCS_FLOAT kperp2 = kx*kx + ky*ky;

    const FLUCS_COMPLEX P  = dft_bits_global[0][index];
    const FLUCS_COMPLEX Ax = dft_bits_global[1][index];
    const FLUCS_COMPLEX Ay = dft_bits_global[2][index];
    const FLUCS_COMPLEX Bx = dft_bits_global[3][index];
    const FLUCS_COMPLEX By = dft_bits_global[4][index];
    const FLUCS_COMPLEX Cx = dft_bits_global[5][index];
    const FLUCS_COMPLEX Cy = dft_bits_global[6][index];

    // Phi: d_t lapPhi = -{Phi, lapPhi} + {Psi, lapPsi} + ...
    explicit_terms[0] += DFT_FULLSIZE_FACTOR * (dy * Ax - dx * Ay) / kperp2;
    // Psi: d_t Psi = {Psi, Phi} + ... = -{Phi, Psi} + ...
    explicit_terms[1] += DFT_FULLSIZE_FACTOR * P;
    // dBpar: d_t dBpar = -{Phi, dBpar} + {Psi, upar}/c + ...
    explicit_terms[2] += DFT_FULLSIZE_FACTOR * (dy * Bx - dx * By);
    // upar: d_t upar = -{Phi, upar} + {Psi, dBpar} + ...
    explicit_terms[3] += DFT_FULLSIZE_FACTOR * (dy * Cx - dx * Cy);
}

////////////////////////////////////////////////////////////////////////////////
// Diagnostics (reductions give box averages: sum over the full Fourier space)
////////////////////////////////////////////////////////////////////////////////

__device__ __forceinline__
FLUCS_FLOAT zp_abs2(const FLUCS_COMPLEX a) {
    return a.real() * a.real() + a.imag() * a.imag();
}

__device__ __forceinline__
FLUCS_FLOAT zp_kperp2(const size_t index) {
    indices3d_t indices = get_indices3d<NZ, NX, HALF_NY>(index);
    const FLUCS_FLOAT kx = kx_from_ikx(indices.ikx);
    const FLUCS_FLOAT ky = ky_from_iky(indices.iky);
    return kx*kx + ky*ky;
}

// Energy density of a mode, E = (1/2)[kperp^2 (|Phi|^2 + |Psi|^2)
//                                     + |upar|^2 + c |dBpar|^2]
// part: -1 total, 0 u_perp, 1 dB_perp, 2 dBpar (incl. c), 3 upar
struct ZPEnergy_Functor {
    const FLUCS_COMPLEX* __restrict__ fields_global;
    const int part;
    __device__ __forceinline__ FLUCS_FLOAT operator()(size_t index) const {
        const FLUCS_FLOAT kperp2 = zp_kperp2(index);
        const FLUCS_FLOAT e0 = kperp2 * zp_abs2(fields_global[index]);
        const FLUCS_FLOAT e1 = kperp2 * zp_abs2(fields_global[index + HALFSIZE]);
        const FLUCS_FLOAT e2 = C_DBPAR * zp_abs2(fields_global[index + 2*HALFSIZE]);
        const FLUCS_FLOAT e3 = zp_abs2(fields_global[index + 3*HALFSIZE]);
        FLUCS_FLOAT e;
        switch (part) {
            case 0: e = e0; break;
            case 1: e = e1; break;
            case 2: e = e2; break;
            case 3: e = e3; break;
            default: e = e0 + e1 + e2 + e3;
        }
        return ((FLUCS_FLOAT)0.5) * e;
    }
};

struct ZPEnergyTotal_Functor {
    const FLUCS_COMPLEX* __restrict__ fields_global;
    __device__ __forceinline__ FLUCS_FLOAT operator()(size_t index) const {
        return ZPEnergy_Functor{fields_global, -1}(index);
    }
};

// Qhat = -(1 + kb) <dBpar u_x> + kb <upar dB_x>, u_x = -dyPhi, dB_x = -dyPsi
// part: -1 total, 0 electrostatic (dBpar u_x), 1 magnetic (upar dB_x)
struct ZPHeatFlux_Functor {
    const FLUCS_COMPLEX* __restrict__ fields_global;
    const int part;
    __device__ __forceinline__ FLUCS_FLOAT operator()(size_t index) const {
        indices3d_t indices = get_indices3d<NZ, NX, HALF_NY>(index);
        const FLUCS_COMPLEX dy = dy_from_iky(indices.iky);
        const FLUCS_COMPLEX phi = fields_global[index];
        const FLUCS_COMPLEX psi = fields_global[index + HALFSIZE];
        const FLUCS_COMPLEX dbpar = fields_global[index + 2*HALFSIZE];
        const FLUCS_COMPLEX upar = fields_global[index + 3*HALFSIZE];
        const FLUCS_COMPLEX ux = -dy * phi;
        const FLUCS_COMPLEX dbx = -dy * psi;
        const FLUCS_FLOAT q_es = -(FLOAT_ONE + KB) * (dbpar * conj(ux)).real();
        const FLUCS_FLOAT q_em = KB * (upar * conj(dbx)).real();
        switch (part) {
            case 0: return q_es;
            case 1: return q_em;
            default: return q_es + q_em;
        }
    }
};

// D = sum 2 nu(k) E_k  for one hyperdissipation component
struct ZPDissipationComponent_Functor {
    const FLUCS_COMPLEX* __restrict__ fields_global;
    const FLUCS_FLOAT adaptive_rate;
    const int hyperdissipation_type;
    __device__ __forceinline__ FLUCS_FLOAT operator()(size_t index) const {
        return (FLUCS_FLOAT)2.0
            * HyperdissipationSelector_Functor<ZPEnergyTotal_Functor>{
                ZPEnergyTotal_Functor{fields_global},
                adaptive_rate,
                hyperdissipation_type
            }(index);
    }
};

// Total D per mode (for spectra)
struct ZPDissipation_Functor {
    const FLUCS_COMPLEX* __restrict__ fields_global;
    const FLUCS_FLOAT adaptive_rate;
    __device__ __forceinline__ FLUCS_FLOAT operator()(size_t index) const {
        return (FLUCS_FLOAT)2.0
            * Hyperdissipation_Functor<ZPEnergyTotal_Functor>{
                ZPEnergyTotal_Functor{fields_global},
                adaptive_rate
            }(index);
    }
};

// Instantaneous nonlinear energy transfer, dE/dt|_NL = -Re sum w_f f* N_f
// (should vanish to roundoff when summed over all modes)
struct ZPEnergyNonlinear_Functor {
    const FLUCS_COMPLEX* __restrict__ fields_global;
    const FLUCS_FLOAT dt;
    const double current_time;
    const long long current_step;
    const FLUCS_COMPLEX (* __restrict__ dft_bits_global)[HALFSIZE];

    __device__ __forceinline__ FLUCS_FLOAT operator()(size_t index) const {
        FLUCS_COMPLEX nl[NUMBER_OF_FIELDS] = {0};
#ifdef NONLINEAR
        add_nonlinear_terms(index, dt, current_time, current_step,
                            dft_bits_global, nl);
#endif
        const FLUCS_FLOAT kperp2 = zp_kperp2(index);
        const FLUCS_COMPLEX phi = fields_global[index];
        const FLUCS_COMPLEX psi = fields_global[index + HALFSIZE];
        const FLUCS_COMPLEX dbpar = fields_global[index + 2*HALFSIZE];
        const FLUCS_COMPLEX upar = fields_global[index + 3*HALFSIZE];
        return -(
              kperp2 * (conj(phi) * nl[0]).real()
            + kperp2 * (conj(psi) * nl[1]).real()
            + C_DBPAR * (conj(dbpar) * nl[2]).real()
            + (conj(upar) * nl[3]).real()
        );
    }
};

////////////////////////////////////////////////////////////////////////////////
// The four quadratic invariants of the nonlinear terms (c = C_DBPAR, s = sqrt c):
//   inv 0 W_zeta+ = 1/2 <|grad zeta+|^2>,  zeta+- = Phi +- Psi
//   inv 1 W_zeta- = 1/2 <|grad zeta-|^2>
//   inv 2 W_Z+    = 1/2 <Z+^2>,            Z+- = upar +- s dBpar
//   inv 3 W_Z-    = 1/2 <Z-^2>
// (curvy3d: zplus, zminus, Zpplus, Zpminus; E = (W_z+ + W_z- + W_Z+ + W_Z-)/2.)
// NL(Z+-) = -{Phi -+ Psi/s, Z+-}; the Alfvenic part is RMHD Elsasser, so the
// nonlinear terms conserve each invariant separately.
// Linear source of each (parallel streaming drops out exactly):
//   zeta+-: exch = 2 <dBpar d_y zeta+->,                    inj = 0
//   Z+-   : exch = -+(2/s) <Z+- d_y Phi>,
//           inj  = -kb kP <Z+- d_y Psi> +- (kP (1+kb)/s) <Z+- d_y Phi>
// with sum_inv inj / 2 = kP Qhat and sum_inv exch = 0 (curvature exchange).
// curvy3d's "<inv>_inj" = exch + inj (as half-space sums).
// quantity: 0 W density, 1 inj, 2 exch
////////////////////////////////////////////////////////////////////////////////

struct ZPInvariant_Functor {
    const FLUCS_COMPLEX* __restrict__ fields_global;
    const int which;
    const int quantity;
    __device__ __forceinline__ FLUCS_FLOAT operator()(size_t index) const {
        indices3d_t indices = get_indices3d<NZ, NX, HALF_NY>(index);
        const FLUCS_FLOAT kx = kx_from_ikx(indices.ikx);
        const FLUCS_FLOAT ky = ky_from_iky(indices.iky);
        const FLUCS_COMPLEX dy = dy_from_iky(indices.iky);
        const FLUCS_FLOAT kperp2 = kx*kx + ky*ky;
        const FLUCS_FLOAT s = flucs_sqrt(C_DBPAR);
        const FLUCS_FLOAT sgn = (which % 2 == 0) ? FLOAT_ONE : -FLOAT_ONE;

        const FLUCS_COMPLEX phi = fields_global[index];
        const FLUCS_COMPLEX psi = fields_global[index + HALFSIZE];
        const FLUCS_COMPLEX dbpar = fields_global[index + 2*HALFSIZE];
        const FLUCS_COMPLEX upar = fields_global[index + 3*HALFSIZE];

        if (which < 2) {
            const FLUCS_COMPLEX zeta = phi + sgn * psi;
            switch (quantity) {
                case 0: return ((FLUCS_FLOAT)0.5) * kperp2 * zp_abs2(zeta);
                case 1: return (FLUCS_FLOAT)0;
                default: return ((FLUCS_FLOAT)2.0) * (dy * zeta * conj(dbpar)).real();
            }
        } else {
            const FLUCS_COMPLEX Z = upar + sgn * s * dbpar;
            switch (quantity) {
                case 0: return ((FLUCS_FLOAT)0.5) * zp_abs2(Z);
                case 1: return (
                      (-KBKP * dy * psi
                       + (sgn * KP * (FLOAT_ONE + KB) / s) * dy * phi) * conj(Z)
                    ).real();
                default: return (
                      ((-sgn * (FLUCS_FLOAT)2.0 / s) * dy * phi) * conj(Z)
                    ).real();
            }
        }
    }
};

struct ZPInvariantW_Functor {
    const FLUCS_COMPLEX* __restrict__ fields_global;
    const int which;
    __device__ __forceinline__ FLUCS_FLOAT operator()(size_t index) const {
        return ZPInvariant_Functor{fields_global, which, 0}(index);
    }
};

// Dissipation of one invariant by one hyperdissipation component: 2 nu(k) W_k
struct ZPInvariantDissipation_Functor {
    const FLUCS_COMPLEX* __restrict__ fields_global;
    const FLUCS_FLOAT adaptive_rate;
    const int hyperdissipation_type;
    const int which;
    __device__ __forceinline__ FLUCS_FLOAT operator()(size_t index) const {
        return (FLUCS_FLOAT)2.0
            * HyperdissipationSelector_Functor<ZPInvariantW_Functor>{
                ZPInvariantW_Functor{fields_global, which},
                adaptive_rate,
                hyperdissipation_type
            }(index);
    }
};

// Instantaneous nonlinear transfer into one invariant, from the nonlinear
// RHS alone (d_t f|_NL = -N_f). absval = 1 returns |per-mode transfer|
// (the gross transfer, a round-off reference scale).
struct ZPInvariantNonlinear_Functor {
    const FLUCS_COMPLEX* __restrict__ fields_global;
    const FLUCS_FLOAT dt;
    const double current_time;
    const long long current_step;
    const FLUCS_COMPLEX (* __restrict__ dft_bits_global)[HALFSIZE];
    const int which;
    const int absval;

    __device__ __forceinline__ FLUCS_FLOAT operator()(size_t index) const {
        FLUCS_COMPLEX nl[NUMBER_OF_FIELDS] = {0};
#ifdef NONLINEAR
        add_nonlinear_terms(index, dt, current_time, current_step,
                            dft_bits_global, nl);
#endif
        const FLUCS_FLOAT kperp2 = zp_kperp2(index);
        const FLUCS_FLOAT s = flucs_sqrt(C_DBPAR);
        const FLUCS_FLOAT sgn = (which % 2 == 0) ? FLOAT_ONE : -FLOAT_ONE;
        const FLUCS_COMPLEX phi = fields_global[index];
        const FLUCS_COMPLEX psi = fields_global[index + HALFSIZE];
        const FLUCS_COMPLEX dbpar = fields_global[index + 2*HALFSIZE];
        const FLUCS_COMPLEX upar = fields_global[index + 3*HALFSIZE];
        FLUCS_FLOAT t;
        if (which < 2) {
            t = -kperp2 * (conj(phi + sgn * psi) * (nl[0] + sgn * nl[1])).real();
        } else {
            t = -(conj(upar + sgn * s * dbpar) * (nl[3] + sgn * s * nl[2])).real();
        }
        return absval ? flucs_fabs(t) : t;
    }
};

// <|f|^2>-type quantities. which: 0 dBpar^2, 1 |dB_perp|^2, 2 dB_y^2 (= (dxPsi)^2)
struct ZPFieldSquare_Functor {
    const FLUCS_COMPLEX* __restrict__ fields_global;
    const int which;
    __device__ __forceinline__ FLUCS_FLOAT operator()(size_t index) const {
        indices3d_t indices = get_indices3d<NZ, NX, HALF_NY>(index);
        const FLUCS_FLOAT kx = kx_from_ikx(indices.ikx);
        const FLUCS_FLOAT ky = ky_from_iky(indices.iky);
        const FLUCS_FLOAT psi2 = zp_abs2(fields_global[index + HALFSIZE]);
        switch (which) {
            case 0: return zp_abs2(fields_global[index + 2*HALFSIZE]);
            case 1: return (kx*kx + ky*ky) * psi2;
            default: return kx*kx * psi2;
        }
    }
};

} // extern "C"
