"""
Diagnostics for the Z-pinch system. All quantities are box averages
(FLUCS reductions sum over the full Fourier space), i.e. curvy3d's saved
half-space sums times 2.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, ClassVar

import numpy as np

from flucs import cupy as cp
from flucs.diagnostic import FlucsDiagnostic, FlucsDiagnosticVariable
from flucs.solvers.fourier.fourier_system_reductions import FourierReductions

if TYPE_CHECKING:
    from flucs.systems.zpinch_fourier.zpinch_fourier import ZPinchFourier


def _scalar(name: str) -> FlucsDiagnosticVariable:
    return FlucsDiagnosticVariable(
        name=name, shape=(), dimensions={}, is_complex=False
    )


class ZPinchEnergyDiag(FlucsDiagnostic):
    """
    Energy budget time series: E and its parts, Qhat, injection kP*Qhat,
    dissipation D (total and per hyperdissipation component), Q/Q_MHD,
    <dBpar^2>, <|dB_perp|^2>, <dB_y^2>, one-step finite-difference dE/dt and
    the budget residual dE/dt - (kP Qhat - D). Optionally the instantaneous
    nonlinear energy transfer (costs one extra nonlinear evaluation).
    """

    name = "energy"
    system: ZPinchFourier
    option_defaults: ClassVar[dict[str, object]] = {
        "save_nonlinear": False,
    }

    get_E: Callable[..., cp.ndarray]
    get_Q: Callable[..., cp.ndarray]
    get_D: Callable[..., cp.ndarray]
    get_sq: Callable[..., cp.ndarray]
    get_NL: Callable[..., cp.ndarray]

    energy_parts = ("E_uperp", "E_bperp", "E_dbpar", "E_upar")
    square_names = ("dbpar2", "dbperp2", "dby2")

    def init_vars(self) -> None:
        reductions = FourierReductions(self.system)

        names = [
            "E", *self.energy_parts,
            "Qhat", "Qhat_es", "Qhat_em", "injection", "Q_over_QMHD",
            "D", "dEdt", "budget_error", *self.square_names,
        ]
        names += [f"D_{c}" for c in self.system.hyperdissipation_components]
        if self.save_nonlinear:
            names.append("dEdt_nonlinear")
        for name in names:
            self.add_var(_scalar(name))

        self.get_E = reductions.get_reduction(
            reduction_output="scalar",
            functor="ZPEnergy_Functor",
            input_args="FLUCS_COMPLEX*,int",
            complex_output=False,
        )
        self.get_Q = reductions.get_reduction(
            reduction_output="scalar",
            functor="ZPHeatFlux_Functor",
            input_args="FLUCS_COMPLEX*,int",
            complex_output=False,
        )
        self.get_D = reductions.get_reduction(
            reduction_output="scalar",
            functor="ZPDissipationComponent_Functor",
            input_args="FLUCS_COMPLEX*,FLUCS_FLOAT,int",
            complex_output=False,
        )
        self.get_sq = reductions.get_reduction(
            reduction_output="scalar",
            functor="ZPFieldSquare_Functor",
            input_args="FLUCS_COMPLEX*,int",
            complex_output=False,
        )
        if self.save_nonlinear:
            self.get_NL = reductions.get_reduction(
                reduction_output="scalar",
                functor="ZPEnergyNonlinear_Functor",
                input_args=(
                    "FLUCS_COMPLEX*,FLUCS_FLOAT,double,long long,"
                    "const FLUCS_COMPLEX (*)[HALFSIZE]"
                ),
                complex_output=False,
            )

    def ready(self) -> None:
        pass

    def execute(self) -> None:
        system = self.system
        current_dt = system.float(system.current_dt)
        current_time = system.time_float(system.current_time)
        current_step = system.int(system.current_step)
        adaptive_rate = system.float(system.adaptive_rate)
        minus_one = np.int32(-1)

        fields = system.get_fields()
        fields_prev = system.get_fields(1)

        E = self.get_E(fields, minus_one).get().item()
        self.save_data("E", E)
        for i, name in enumerate(self.energy_parts):
            self.save_data(name, self.get_E(fields, np.int32(i)).get().item())

        Q = self.get_Q(fields, minus_one).get().item()
        self.save_data("Qhat", Q)
        self.save_data("Qhat_es", self.get_Q(fields, np.int32(0)).get().item())
        self.save_data("Qhat_em", self.get_Q(fields, np.int32(1)).get().item())
        injection = system.kP * Q
        self.save_data("injection", injection)
        self.save_data("Q_over_QMHD", Q / system.Ly**2 / system.Q_MHD)

        D = 0.0
        for i, c in enumerate(system.hyperdissipation_components):
            Dc = self.get_D(fields, adaptive_rate, np.int32(i)).get().item()
            self.save_data(f"D_{c}", Dc)
            D += Dc
        self.save_data("D", D)

        E_prev = self.get_E(fields_prev, minus_one).get().item()
        dEdt = (E - E_prev) / current_dt
        self.save_data("dEdt", dEdt)
        self.save_data("budget_error", dEdt - (injection - D))

        for i, name in enumerate(self.square_names):
            self.save_data(name, self.get_sq(fields, np.int32(i)).get().item())

        if self.save_nonlinear:
            if system.input["setup.linear"]:
                self.save_data("dEdt_nonlinear", 0.0)
            else:
                system.compute_nonlinear_terms(
                    current_dt, current_time, current_step, fields, False
                )
                self.save_data(
                    "dEdt_nonlinear",
                    self.get_NL(
                        fields, current_dt, current_time, current_step,
                        system.dft_bits,
                    ).get().item(),
                )


class ZPinchEnergyDiag1D(FlucsDiagnostic):
    """
    1D spectra (default kperp shells) of the energy, its parts, the heat
    flux and the dissipation.
    """

    name = "energy_1d"
    system: ZPinchFourier
    option_defaults: ClassVar[dict[str, object]] = {
        "spectra": ["kperp"],
    }

    def init_vars(self) -> None:
        reductions = FourierReductions(self.system)
        spectra = self.spectra
        if isinstance(spectra, str):
            spectra = [spectra]
        valid = ("kz", "kx", "ky", "kperp")
        if set(spectra) - set(valid):
            raise ValueError(f"{self.name} only supports spectra {valid}.")

        self.get = {}
        for spectrum in dict.fromkeys(spectra):
            dimensions = reductions.get_dimensions(spectrum)
            shape = tuple(dimensions)
            for name in ("E", *ZPinchEnergyDiag.energy_parts, "Qhat", "D"):
                self.add_var(FlucsDiagnosticVariable(
                    name=f"{spectrum}_spectra/{name}",
                    shape=shape,
                    dimensions=dimensions,
                    is_complex=False,
                ))
            self.get[spectrum] = {
                "E": reductions.get_reduction(
                    reduction_output=spectrum,
                    functor="ZPEnergy_Functor",
                    input_args="FLUCS_COMPLEX*,int",
                    complex_output=False,
                ),
                "Qhat": reductions.get_reduction(
                    reduction_output=spectrum,
                    functor="ZPHeatFlux_Functor",
                    input_args="FLUCS_COMPLEX*,int",
                    complex_output=False,
                ),
                "D": reductions.get_reduction(
                    reduction_output=spectrum,
                    functor="ZPDissipation_Functor",
                    input_args="FLUCS_COMPLEX*,FLUCS_FLOAT",
                    complex_output=False,
                ),
            }

    def ready(self) -> None:
        pass

    def execute(self) -> None:
        system = self.system
        fields = system.get_fields()
        adaptive_rate = system.float(system.adaptive_rate)
        for spectrum, get in self.get.items():
            pre = f"{spectrum}_spectra/"
            self.save_data(pre + "E", get["E"](fields, np.int32(-1)).get())
            for i, name in enumerate(ZPinchEnergyDiag.energy_parts):
                self.save_data(pre + name, get["E"](fields, np.int32(i)).get())
            self.save_data(pre + "Qhat", get["Qhat"](fields, np.int32(-1)).get())
            self.save_data(pre + "D", get["D"](fields, adaptive_rate).get())
