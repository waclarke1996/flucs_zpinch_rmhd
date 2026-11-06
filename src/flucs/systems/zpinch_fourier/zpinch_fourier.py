"""
Pseudospectral Fourier implementation of the four-field Z-pinch
reduced-MHD system (W. Clarke; see zpinch_fourier.cu for the equations).

Fields: 0 Phi, 1 Psi, 2 dBpar, 3 upar.
Parameters: parameters.kb (kappa_beta), parameters.kP (kappa_P).
Dissipation: use the generic [hyperdissipation] section, e.g.
    kperp = nu_perp, kperp_power = 2 (k^4) or 4 (k^8), same for kz.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from flucs import cupy as cp
from flucs.diagnostic import FlucsDiagnostic
from flucs.input import InvalidFlucsInputFileError
from flucs.solvers.fourier.fourier_system import FourierSystem
from flucs.utilities.cupy import KernelWrapper
from flucs.utilities.messages import flucsprint

from .zpinch_fourier_diagnostics import ZPinchEnergyDiag, ZPinchEnergyDiag1D


class ZPinchFourier(FourierSystem):
    """Four-field Z-pinch reduced MHD (Phi, Psi, dBpar, upar)."""

    number_of_fields = 4
    number_of_dft_derivatives = 8
    number_of_dft_bits = 7

    find_derivatives_kernel: KernelWrapper
    find_nonlinear_bits_kernel: KernelWrapper

    diags: ClassVar[set[type[FlucsDiagnostic]]] = {
        ZPinchEnergyDiag,
        ZPinchEnergyDiag1D,
    }

    # Physical parameters
    kb: float
    kP: float
    c_dbpar: float
    g_curv: float
    Q_MHD: float
    Lx: float
    Ly: float
    Lz: float

    def register_kernels(self) -> None:
        super().register_kernels()

        self.find_derivatives_kernel = KernelWrapper(
            system=self,
            cuda_kernel_name="find_derivatives",
            grid=(self.half_cuda_grid_size,),
            block=(self.cuda_block_size,),
        )
        self.find_nonlinear_bits_kernel = KernelWrapper(
            system=self,
            cuda_kernel_name="find_nonlinear_bits",
            grid=(self.full_cuda_grid_size,),
            block=(self.cuda_block_size,),
            shared_mem=self.cuda_block_size * self.float().nbytes,
        )

        def find_derivatives_function(
            current_dt, current_time, current_step, fields, memory_dict
        ) -> None:
            self.find_derivatives_kernel(
                fields, memory_dict["first_intermediates_fourier"]
            )

        def find_nonlinear_bits_function(
            current_dt, current_time, current_step, calculate_cfl, memory_dict
        ) -> None:
            self.find_nonlinear_bits_kernel(
                memory_dict["first_intermediates_real"],
                memory_dict["second_intermediates_real"],
                calculate_cfl,
                self.cfl_rate,
            )

        if not self.input["setup.linear"]:
            self.dft_derivatives_operation, self.dft_bits = (
                self.create_dealiased_operation(
                    n_in=self.number_of_dft_derivatives,
                    n_out=self.number_of_dft_bits,
                    create_first_intermediates=find_derivatives_function,
                    create_second_intermediates=find_nonlinear_bits_function,
                    allocate_additional_memory=None,
                    combine_first_and_second_intermediates=True,
                )
            )

    # ------------------------------------------------------------------
    # Input
    # ------------------------------------------------------------------

    def _interpret_input(self):
        super()._interpret_input()

        self.kb = float(self.input["parameters.kb"])
        self.kP = float(self.input["parameters.kP"])
        if self.kb <= 0.0:
            raise InvalidFlucsInputFileError("parameters.kb must be positive.")
        if self.kP < 0.0:
            raise InvalidFlucsInputFileError(
                "parameters.kP must be non-negative."
            )

        self.c_dbpar = 1.0 + 1.0 / self.kb
        self.g_curv = self.kP * (1.0 + self.kb) - 2.0
        self.Q_MHD = (2.0 * self.kb * self.kP) ** 1.5 if self.kP > 0 else np.nan
        self.Lx = float(self.input["dimensions.Lx"])
        self.Ly = float(self.input["dimensions.Ly"])
        self.Lz = float(self.input["dimensions.Lz"])

        N2 = 2.0 * self.kb / (1.0 + self.kb) * self.g_curv
        flucsprint(
            f"Z-pinch: kb = {self.kb}, kP = {self.kP}, c = {self.c_dbpar:.6f},"
            f" N^2 = {N2:.6f}, N0^2 = {2 * self.kb * self.kP:.6f}",
            source=self,
        )

    def setup_cuda_definitions(self) -> None:
        self.module_options.define_float("KB", self.kb)
        self.module_options.define_float("KP", self.kP)
        self.module_options.define_float("KBKP", self.kb * self.kP)
        self.module_options.define_float("C_DBPAR", self.c_dbpar)
        self.module_options.define_float("G_CURV", self.g_curv)
        super().setup_cuda_definitions()

    # ------------------------------------------------------------------
    # Initial conditions
    # ------------------------------------------------------------------

    def _set_initial_conditions(self) -> None:
        if self.restart_manager.data is not None:
            super()._set_initial_conditions()
            return

        method = self.input["init.method"]
        match method:
            case "white_noise":
                # Plugin white noise (the core version draws phases on
                # [0, pi) only; see NOTES_flucs.md). Old: "white_noise_core".
                self.fields_initial = self._white_noise(
                    self.input["init.amplitude"], self.input["init.rand_seed"]
                )
            case "white_noise_core":
                # Copy of the FLUCS core white_noise (fourier_system.py),
                # phases exp(i pi U[0,1)), kept to reproduce Stage 3 runs.
                mask = self.get_solved_grid_mask().astype(bool)
                np.random.seed(self.input["init.rand_seed"])
                solved = np.random.random(
                    (self.number_of_fields, np.count_nonzero(mask))
                ).astype(dtype=self.complex)
                self.fields_initial = np.zeros(
                    (self.number_of_fields, *self.half_tuple), dtype=self.complex
                )
                self.fields_initial[:, mask] = (
                    self.input["init.amplitude"] * np.exp(1j * np.pi * solved)
                )
            case "streamer":
                self._init_streamer()
            case "single_mode":
                self._init_single_mode()
            case "curvy3d_snapshot":
                self._init_curvy3d_snapshot()
            case "kx0_noise":
                # Random phases, amplitude init.amplitude, only kx = 0 modes
                mask = self.get_solved_grid_mask().astype(bool)
                mask &= self.get_broadcast_wavenumbers()[1] == 0
                rng = np.random.default_rng(self.input["init.rand_seed"])
                phases = rng.uniform(
                    0, 2 * np.pi, (self.number_of_fields, np.count_nonzero(mask))
                )
                self.fields_initial = np.zeros(
                    (self.number_of_fields, *self.half_tuple), dtype=self.complex
                )
                self.fields_initial[:, mask] = (
                    self.input["init.amplitude"] * np.exp(1j * phases)
                )
            case _:
                super()._set_initial_conditions()

        # kperp = 0 modes are not evolved (as in curvy3d): zero them
        self.fields_initial[:, :, 0, 0] = 0

    def _init_curvy3d_snapshot(self) -> None:
        """
        init.file = curvy3d .nc output, init.index = snapshot index.
        curvy3d stores X_k of f = sum_k X_k exp(i k.x) (same as FLUCS) in the
        compact unpadded layout (nz, nx, half_ny), fields (Phi, deltaBpar,
        Psi, upar). Copied mode by mode into the padded FLUCS layout;
        normalisation factor 1. The ky = 0 plane is Hermitian-projected
        afterwards by FourierSystem._check_initial_conditions.
        """
        import netCDF4

        path = self.input["init.file"]
        it = int(self.input["init.index"])
        names = ("Phi", "Psi", "deltaBpar", "upar")  # FLUCS field order
        with netCDF4.Dataset(path, "r") as ds:
            data = [
                np.asarray(ds[f"{n}_rfft_r"][it], dtype=np.float64)
                + 1j * np.asarray(ds[f"{n}_rfft_i"][it], dtype=np.float64)
                for n in names
            ]
        nz_c, nx_c, hy_c = data[0].shape
        solved = self.get_solved_grid_mask().astype(bool)
        n_kz = int(np.count_nonzero(solved[:, 0, 1]))
        n_kx = int(np.count_nonzero(solved[0, :, 1]))
        n_ky = int(np.count_nonzero(solved[0, 1, :]))
        if (nz_c, nx_c, hy_c) != (n_kz, n_kx, n_ky):
            raise InvalidFlucsInputFileError(
                f"curvy3d snapshot modes {(nz_c, nx_c, hy_c)} do not match "
                f"the FLUCS solved modes {(n_kz, n_kx, n_ky)}."
            )

        def padded_index(n_c, n_pad):
            i = np.arange(n_c)
            return np.where(i < (n_c + 1) // 2, i, i + (n_pad - n_c))

        iz = padded_index(nz_c, self.nz)
        ix = padded_index(nx_c, self.nx)
        self.fields_initial = np.zeros(
            (self.number_of_fields, *self.half_tuple), dtype=self.complex
        )
        for f, X in enumerate(data):
            self.fields_initial[f, iz[:, None, None], ix[None, :, None],
                                np.arange(hy_c)[None, None, :]] = X
        if np.any(self.fields_initial[:, ~solved]):
            raise RuntimeError("curvy3d snapshot mapped onto unsolved modes.")
        flucsprint(
            f"Initialised from curvy3d snapshot {it} of {path}: "
            f"max |X| = {max(np.abs(X).max() for X in data):.3e}",
            source=self,
        )

    def _white_noise(self, amplitude: float, seed: int) -> np.ndarray:
        """
        |X_k| = amplitude, phase uniform on [0, 2 pi), on every solved mode
        of every field (as curvy3d: 1e-5 per mode). The ky = 0 plane is made
        exactly Hermitian (one phase per (k, -k) pair, partner = conj), so
        the core's 0.5 (f + conj f) projection leaves it unchanged.
        kperp = 0 modes are zeroed by the caller.
        """
        mask = self.get_solved_grid_mask().astype(bool)
        rng = np.random.default_rng(seed)
        phases = rng.uniform(
            0.0, 2.0 * np.pi, (self.number_of_fields, *self.half_tuple)
        )
        fields = np.zeros(
            (self.number_of_fields, *self.half_tuple), dtype=self.complex
        )
        fields[:] = amplitude * np.exp(1j * phases)
        # ky = 0: set the partner (-kz, -kx) of each mode to its conjugate,
        # keeping the member with the "positive" (kz, kx) index ordering.
        ikz = np.arange(self.nz)[:, None]
        ikx = np.arange(self.nx)[None, :]
        cikz, cikx = (-ikz) % self.nz, (-ikx) % self.nx
        lin = ikz * self.nx + ikx
        clin = cikz * self.nx + cikx
        partner = np.conj(fields[:, cikz, cikx, 0])
        ky0 = fields[:, :, :, 0]
        ky0[:] = np.where((lin > clin)[None], partner, ky0)
        fields[:, ~mask] = 0
        return fields

    def _box_energy(self, fields: np.ndarray) -> float:
        """Box-averaged E of half-plane fields (same as ZPEnergy_Functor)."""
        kz_b, kx_b, ky_b = self.get_broadcast_wavenumbers()
        kperp2 = kx_b**2 + ky_b**2
        weight = np.where(np.arange(self.half_ny) == 0, 1.0, 2.0)
        weight = weight[None, None, :] * self.get_solved_grid_mask()
        e = 0.5 * (
            kperp2 * (np.abs(fields[0]) ** 2 + np.abs(fields[1]) ** 2)
            + self.c_dbpar * np.abs(fields[2]) ** 2
            + np.abs(fields[3]) ** 2
        )
        return float(np.sum(weight * e))

    def _init_streamer(self) -> None:
        """
        One linear eigenmode (fastest root) at init.modes = [[kx, ky, kz]]
        with box energy init.energy, plus plugin white noise with total box
        energy init.noise_energy (seed init.rand_seed).
        """
        modes = self.input["init.modes"]
        if len(modes) != 1:
            raise InvalidFlucsInputFileError(
                "init.method = 'streamer' needs exactly one mode in init.modes."
            )
        kx, ky, kz = modes[0]
        ikz, ikx, iky = self._mode_index(kx, ky, kz)
        L = self.compute_physical_linear_operator()
        vals, vecs = np.linalg.eig(L[:, :, ikz, ikx, iky])
        v = vecs[:, np.argmax(vals.real)]
        mode = np.zeros(
            (self.number_of_fields, *self.half_tuple), dtype=self.complex
        )
        mode[:, ikz, ikx, iky] = v
        if iky == 0:
            mode[:, (-ikz) % self.nz, (-ikx) % self.nx, 0] = np.conj(v)
        mode[:, :, 0, 0] = 0
        e_mode = self._box_energy(mode)
        if e_mode == 0:
            raise InvalidFlucsInputFileError("Streamer mode has zero energy.")
        mode *= np.sqrt(float(self.input["init.energy"]) / e_mode)

        noise = self._white_noise(1.0, self.input["init.rand_seed"])
        noise[:, :, 0, 0] = 0
        noise *= np.sqrt(
            float(self.input["init.noise_energy"]) / self._box_energy(noise)
        )
        self.fields_initial = mode + noise
        flucsprint(
            f"Streamer ({kx}, {ky}, {kz}): growth rate {vals.real.max():.6f},"
            f" E_mode = {self._box_energy(mode):.6e},"
            f" E_noise = {self._box_energy(noise):.6e},"
            f" E_box = {self._box_energy(self.fields_initial):.16e}",
            source=self,
        )

    def _mode_index(self, kx: float, ky: float, kz: float):
        """Array indices (ikz, ikx, iky) of a mode with ky >= 0."""
        ikx = int(round(kx * self.Lx / (2 * np.pi))) % self.nx
        iky = int(round(ky * self.Ly / (2 * np.pi)))
        ikz = int(round(kz * self.Lz / (2 * np.pi))) % self.nz
        if not (0 <= iky < self.half_ny):
            raise InvalidFlucsInputFileError(f"ky = {ky} not on the grid.")
        if not self.get_solved_grid_mask()[ikz, ikx, iky]:
            raise InvalidFlucsInputFileError(
                f"Mode ({kx}, {ky}, {kz}) is not a solved mode."
            )
        return ikz, ikx, iky

    def _init_single_mode(self) -> None:
        """
        init.modes = [[kx, ky, kz], ...] (ky >= 0). With init.eigenvector
        = true, each mode is set to the eigenvector of L with the largest
        growth rate (largest Re of the eigenvalue), scaled to energy
        init.amplitude^2; otherwise all fields get amplitude init.amplitude.
        """
        modes = self.input["init.modes"]
        amplitude = self.input["init.amplitude"]
        use_eigenvector = self.input["init.eigenvector"]
        if len(modes) == 0:
            raise InvalidFlucsInputFileError(
                "init.method = 'single_mode' needs a non-empty init.modes."
            )

        self.fields_initial = np.zeros(
            (self.number_of_fields, *self.half_tuple), dtype=self.complex
        )
        L = self.compute_physical_linear_operator()
        kz_b, kx_b, ky_b = self.get_broadcast_wavenumbers()

        for kx, ky, kz in modes:
            ikz, ikx, iky = self._mode_index(kx, ky, kz)
            kperp2 = kx_b[ikz, ikx, iky] ** 2 + ky_b[ikz, ikx, iky] ** 2
            if use_eigenvector:
                vals, vecs = np.linalg.eig(L[:, :, ikz, ikx, iky])
                v = vecs[:, np.argmax(vals.real)]
                weights = np.array([kperp2, kperp2, self.c_dbpar, 1.0])
                v = v / np.sqrt(0.5 * np.sum(weights * np.abs(v) ** 2))
                v = amplitude * v
            else:
                v = amplitude * np.ones(self.number_of_fields)
            self.fields_initial[:, ikz, ikx, iky] = v
            if iky == 0:
                # Reality condition for ky = 0
                self.fields_initial[:, (-ikz) % self.nz, (-ikx) % self.nx, 0] \
                    = np.conj(v)

    # ------------------------------------------------------------------
    # Time stepping
    # ------------------------------------------------------------------

    def compute_nonlinear_terms(
        self,
        current_dt,
        current_time,
        current_step,
        fields: cp.ndarray,
        calculate_cfl,
    ) -> None:
        self.dft_derivatives_operation(
            current_dt,
            current_time,
            current_step,
            fields,
            calculate_cfl=calculate_cfl,
        )

    # ------------------------------------------------------------------
    # Linear theory
    # ------------------------------------------------------------------

    def compute_linear_matrix_reference(self) -> np.ndarray:
        """
        FLUCS linear matrix M (d_t f = -M f - N) without dissipation, i.e.
        M = -L with L the physical linear operator, d_t f = L f.
        """
        return -self.compute_physical_linear_operator()

    def compute_physical_linear_operator(self) -> np.ndarray:
        """L with d_t f = L f (linear, no dissipation), numpy."""
        L = np.zeros(
            (self.number_of_fields, self.number_of_fields, *self.half_tuple),
            dtype=self.complex,
        )
        kz, kx, ky = self.get_broadcast_wavenumbers()
        kperp2 = kx**2 + ky**2
        inv_kperp2 = np.divide(
            1.0, kperp2, out=np.zeros_like(kperp2), where=kperp2 > 0
        )
        kperp_nonzero = kperp2 > 0

        L[0, 1] = 1j * kz * kperp_nonzero
        L[0, 2] = -2j * ky * inv_kperp2
        L[1, 0] = 1j * kz
        L[2, 0] = 1j * ky * self.g_curv / self.c_dbpar
        L[2, 3] = 1j * kz / self.c_dbpar
        L[3, 1] = -1j * ky * self.kb * self.kP
        L[3, 2] = 1j * kz
        return L

    def linear_growth_rate_theory(self, kx, ky, kz) -> tuple[float, np.ndarray]:
        """
        SPEC.md dispersion relation. Returns (gamma, omega^2 roots).
        c w^4 + w^2 [kap c N^2 - (1 + c) kz^2] + kz^2 (kz^2 - kap N0^2) = 0
        """
        c = self.c_dbpar
        kap = ky**2 / (kx**2 + ky**2)
        N2 = 2 * self.kb / (1 + self.kb) * self.g_curv
        N02 = 2 * self.kb * self.kP
        roots = np.roots([c, kap * c * N2 - (1 + c) * kz**2,
                          kz**2 * (kz**2 - kap * N02)])
        roots = np.sort(roots.real)
        gamma = np.sqrt(-roots[0]) if roots[0] < 0 else 0.0
        return gamma, roots
