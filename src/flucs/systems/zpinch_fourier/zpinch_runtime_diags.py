"""
Run-time diagnostics for the Z-pinch system (Stage 4a). Diagnostics only:
nothing here changes the equations or the fields.

All quantities are box averages unless stated otherwise (FLUCS convention;
curvy3d's saved half-space sums are half of these). Per-mode sums run over
the solved half-space modes with weight w = 1 at k_y = 0 (both members of
each conjugate pair are stored) and w = 2 at k_y > 0.

Diagnostic names (use in [output.<name>].diags; cadence from save_time /
sample_time of the output, see zpinch_output.py):

  zp_scalars        (a) zonal/streamer fractions, shear, rms amplitudes,
                        Q_es/Q_em, energy-weighted k_z,rms in k_perp bands,
                        CFL (instantaneous and max since the last record)
  zp_spectra        (b) 1D spectra in unit bins of k_perp, k_y and |k_z|
  zp_maps           (c) 2D (k_perp, |k_z|) unit-bin maps, window averages
                        (needs sample_time), float32
  zp_kpar           (d) k_par(k_perp) accumulators as kpar_spectrum.py
  zp_profiles       (e) mean-field lines (k_y = k_z = 0 and k_x = k_z = 0),
                        complex Fourier coefficients of all four fields
  zp_slices         (f) float32 x-y slices at z = 0 of Phi and dBpar
  zp_kytransfer     (g) curvy3d "subcrit_mode_balances" (k_y transfers)
  zp_streamer_kzkx  (g) curvy3d streamer transfers in (k_z, k_x), window
                        averages (needs sample_time), float32
  zp_shells         (h) shell-to-shell transfers (Elsasser and kinetic /
                        magnetic sets of the hydra_shell_plotter_*_gpu.py)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import numpy as np

from flucs import cupy as cp
from flucs.diagnostic import FlucsDiagnostic, FlucsDiagnosticVariable

if TYPE_CHECKING:
    from flucs.systems.zpinch_fourier.zpinch_fourier import ZPinchFourier

FIELD_NAMES = ("Phi", "Psi", "dBpar", "upar")


def _var(name, dims=None, complex_=False, dtype=None, zlib=False,
         time_dependent=True):
    dims = dims or {}
    return FlucsDiagnosticVariable(
        name=name, shape=tuple(dims), dimensions=dims, is_complex=complex_,
        is_time_dependent=time_dependent, dtype=dtype, zlib=zlib,
    )


def _fast_len(n):
    """Smallest 2^a 3^b 5^c 7^d >= n (as kpar_spectrum.py)."""
    try:
        import scipy.fft
        return scipy.fft.next_fast_len(int(n), real=True)
    except Exception:
        m = int(n)
        while True:
            r = m
            for p in (2, 3, 5, 7):
                while r % p == 0:
                    r //= p
            if r == 1:
                return m
            m += 1


class ZPGrid:
    """Solved-mode geometry shared by the diagnostics (built once)."""

    def __init__(self, system: ZPinchFourier):
        s = system
        # Built in numpy: outputs (and these diagnostics) are set up before the
        # CUDA module (and its solved-mask kernel) exists; checked against the
        # kernel's mask in ready() (check_mask).
        mask = self.solved_mask(s)
        self.mask_np = mask
        kz, kx, ky = s.get_broadcast_wavenumbers()
        self.half_tuple = s.half_tuple
        self.full_tuple = s.full_tuple
        self.Lx, self.Ly, self.Lz = s.Lx, s.Ly, s.Lz
        self.c = s.c_dbpar
        self.kb, self.kP = s.kb, s.kP
        self.Q_MHD = s.Q_MHD

        idx = np.flatnonzero(mask.ravel())
        kx_s, ky_s, kz_s = kx[mask], ky[mask], kz[mask]
        self.n_modes = idx.size
        self.idx = cp.asarray(idx)
        self.kx = cp.asarray(kx_s)
        self.ky = cp.asarray(ky_s)
        self.kz = cp.asarray(kz_s)
        self.w = cp.asarray(np.where(ky_s > 0, 2.0, 1.0))
        self.k2 = self.kx**2 + self.ky**2
        self.kperp = cp.sqrt(self.k2)
        self.kz2 = self.kz**2

        # integer mode numbers and compact (curvy3d) layout
        self.dkx, self.dky, self.dkz = (2 * np.pi / L for L in (s.Lx, s.Ly, s.Lz))
        mx = np.rint(kx_s / self.dkx).astype(np.int64)
        my = np.rint(ky_s / self.dky).astype(np.int64)
        mz = np.rint(kz_s / self.dkz).astype(np.int64)
        self.Kx, self.Ky, self.Kz = int(np.abs(mx).max()), int(my.max()), int(np.abs(mz).max())
        self.nx_c, self.hy_c, self.nz_c = 2 * self.Kx + 1, self.Ky + 1, 2 * self.Kz + 1
        self.ny_c = 2 * self.Ky + 1
        self.mx_np, self.my_np, self.mz_np = mx, my, mz
        self.mx, self.my, self.mz = (cp.asarray(a) for a in (mx, my, mz))

        # unit bins (dk = 2 pi / L_y in k_perp and k_y, 2 pi / L_z in k_z)
        ikp = np.rint(np.sqrt(kx_s**2 + ky_s**2) / self.dky).astype(np.int64)
        self.n_kp = int(ikp.max()) + 1
        self.n_ky = self.Ky + 1
        self.n_kz = self.Kz + 1
        self.ikp = cp.asarray(ikp)
        self.iky = self.my
        self.ikz = cp.abs(self.mz)
        self.kp_centers = np.arange(self.n_kp) * self.dky
        self.ky_centers = np.arange(self.n_ky) * self.dky
        self.kz_centers = np.arange(self.n_kz) * self.dkz

        # mode sets
        self.zon = cp.asarray(np.flatnonzero(ky_s == 0))
        self.zon0 = cp.asarray(np.flatnonzero((ky_s == 0) & (kz_s == 0)))
        self.kx0 = cp.asarray(np.flatnonzero(kx_s == 0))

        # hyperdissipation rates (rate nu * k^(2 power); D = 2 rate E)
        rp = np.zeros(idx.size)
        rz = np.zeros(idx.size)
        for comp in s.hyperdissipation_components:
            nu = float(s.input[f"hyperdissipation.{comp}"])
            if nu <= 0:
                continue
            if (s.input[f"hyperdissipation.{comp}_adaptive"]
                    or s.input[f"hyperdissipation.{comp}_normalised"]):
                raise NotImplementedError(
                    "zp diagnostics: adaptive/normalised hyperdissipation")
            p = float(s.input[f"hyperdissipation.{comp}_power"])
            if comp == "kperp":
                rp += nu * (kx_s**2 + ky_s**2) ** p
            elif comp == "kz":
                rz += nu * (kz_s**2) ** p
            else:
                raise NotImplementedError(
                    f"zp diagnostics: hyperdissipation component {comp}")
        self.rate_perp = cp.asarray(rp)
        self.rate_par = cp.asarray(rz)

        # padded half-grid wavenumbers (broadcastable), for FFT-based diags
        self.kx_h = cp.asarray(s.kx)[None, :, None]
        self.ky_h = cp.asarray(s.ky)[None, None, :]
        self.kz_h = cp.asarray(s.kz)[:, None, None]

    # ------------------------------------------------------------------
    @staticmethod
    def solved_mask(s):
        """Two-thirds solved set (as is_mode_padded in cuda/indexing.cuh)."""
        def ok(n, nu):
            h = nu // 2 + 1
            i = np.arange(n)
            return (i < h) | (i >= n - (nu - h))
        mz = ok(s.nz, s.nz_unpadded)
        mx = ok(s.nx, s.nx_unpadded)
        my = np.arange(s.half_ny) < s.ny_unpadded // 2 + 1
        return mz[:, None, None] & mx[None, :, None] & my[None, None, :]

    def check_mask(self, s):
        if getattr(self, "_mask_checked", False):
            return
        if not np.array_equal(self.mask_np, s.get_solved_grid_mask().astype(bool)):
            raise RuntimeError("zp diagnostics: solved-mode mask differs from the "
                               "solver's (only two-thirds dealiasing is supported)")
        self._mask_checked = True

    def compact(self, fields):
        """(4, nz, nx, hy) padded half grid -> (4, n_modes) solved modes."""
        return fields.reshape(fields.shape[0], -1)[:, self.idx]

    def mode_energy_parts(self, F):
        """w-weighted energy per mode: (4, M) parts (u_perp, b_perp, dbpar, upar)."""
        a2 = F.real**2 + F.imag**2
        return 0.5 * self.w * cp.stack(
            (self.k2 * a2[0], self.k2 * a2[1], self.c * a2[2], a2[3]))

    def mode_heat_flux(self, F):
        """w-weighted Qhat per mode: (es, em), as ZPHeatFlux_Functor."""
        ux = -1j * self.ky * F[0]
        dbx = -1j * self.ky * F[1]
        q_es = -(1.0 + self.kb) * (F[2] * cp.conj(ux)).real
        q_em = self.kb * (F[3] * cp.conj(dbx)).real
        return self.w * q_es, self.w * q_em

    def mode_injection(self, F):
        """w-weighted energy injection per mode (dissipation_map.py; = kP Qhat)."""
        return self.w * (
            self.kP * (1 + self.kb) * (1j * self.ky * F[0] * cp.conj(F[2])).real
            - self.kP * self.kb * (1j * self.ky * F[1] * cp.conj(F[3])).real)

    def binned(self, index, weights, n):
        return cp.bincount(index, weights=weights, minlength=n)[:n]


def get_grid(system) -> ZPGrid:
    if getattr(system, "_zp_grid", None) is None:
        system._zp_grid = ZPGrid(system)
    return system._zp_grid


class _ZPDiag(FlucsDiagnostic):
    system: ZPinchFourier

    def ready(self) -> None:
        self.grid.check_mask(self.system)

    @property
    def grid(self) -> ZPGrid:
        return get_grid(self.system)


class _ZPWindowed(_ZPDiag):
    """Diagnostic that averages sample() calls over the record window."""

    def reset_window(self) -> None:
        self._acc = None
        self._count = 0

    def window_count(self) -> int:
        return getattr(self, "_count", 0)

    def _accumulate(self, arrays) -> None:
        if getattr(self, "_acc", None) is None:
            self._acc = [a.copy() for a in arrays]
        else:
            for acc, a in zip(self._acc, arrays):
                acc += a
        self._count += 1


# ----------------------------------------------------------------------
# (a) scalars
# ----------------------------------------------------------------------
class ZPScalarsDiag(_ZPDiag):
    name = "zp_scalars"
    option_defaults: ClassVar[dict[str, object]] = {
        "kz_bands": [1.0, 2.0, 4.0, 8.0, 16.0, 32.0],
    }
    names = (
        "E_kx0_frac", "E_ky0_frac", "E_ky0kz0_frac", "S_zon0", "S_zon_all",
        "W_rms", "b_rms", "u_rms", "upar_rms", "dbpar_rms",
        "Q_es_over_QMHD", "Q_em_over_QMHD",
    )

    def init_vars(self) -> None:
        edges = [float(e) for e in self.kz_bands]
        self.band_names = [f"kzrms_{lo:g}_{hi:g}" for lo, hi in zip(edges[:-1], edges[1:])]
        self.edges = edges
        for n in (*self.names, *self.band_names, "cfl", "cfl_max", "dt_min"):
            self.add_var(_var(n))
        self._band = None

    def execute(self) -> None:
        g = self.grid
        F = g.compact(self.system.get_fields())
        if self._band is None:
            b = cp.searchsorted(cp.asarray(self.edges), g.kperp, side="right") - 1
            n = len(self.edges) - 1
            self._band = cp.where((b >= 0) & (b < n), b, n)
        a2 = F.real**2 + F.imag**2
        ep = g.mode_energy_parts(F)
        e = ep.sum(axis=0)
        q_es, q_em = g.mode_heat_flux(F)
        vort2 = g.w * g.k2**2 * a2[0]
        n = len(self.edges) - 1
        eb = cp.bincount(self._band, weights=e, minlength=n + 1)
        ezb = cp.bincount(self._band, weights=e * g.kz2, minlength=n + 1)
        vals = cp.stack([
            e.sum(), e[g.kx0].sum(), e[g.zon].sum(), e[g.zon0].sum(),
            vort2[g.zon0].sum(), vort2[g.zon].sum(), vort2.sum(),
            2 * ep[1].sum(), 2 * ep[0].sum(), (g.w * a2[3]).sum(),
            (g.w * a2[2]).sum(), q_es.sum(), q_em.sum(),
        ]).get()
        E = vals[0]
        qn = 1.0 / (g.Ly**2 * g.Q_MHD)
        out = [vals[1] / E, vals[2] / E, vals[3] / E, np.sqrt(vals[4]),
               np.sqrt(vals[5]), np.sqrt(vals[6]), np.sqrt(vals[7]),
               np.sqrt(vals[8]), np.sqrt(vals[9]), np.sqrt(vals[10]),
               vals[11] * qn, vals[12] * qn]
        for name, v in zip(self.names, out):
            self.save_data(name, v)
        eb, ezb = eb.get(), ezb.get()
        for i, name in enumerate(self.band_names):
            self.save_data(name, np.sqrt(ezb[i] / eb[i]) if eb[i] > 0 else np.nan)
        s = self.system
        self.save_data("cfl", float(s.current_cfl))
        self.save_data("cfl_max", float(getattr(s, "_zp_cfl_max", s.current_cfl)))
        self.save_data("dt_min", float(getattr(s, "_zp_dt_min", s.current_dt)))
        s._zp_cfl_max = 0.0
        s._zp_dt_min = np.inf


# ----------------------------------------------------------------------
# (b) 1D spectra
# ----------------------------------------------------------------------
class ZPSpectraDiag(_ZPDiag):
    name = "zp_spectra"
    option_defaults: ClassVar[dict[str, object]] = {}
    kperp_names = ("E", "E_uperp", "E_bperp", "E_dbpar", "E_upar", "inj",
                   "D_perp", "D_par", "Qhat", "Qhat_es", "Qhat_em", "kz2E")
    ky_names = ("E", "Qhat", "Qhat_es", "Qhat_em")
    kz_names = ("E", "inj", "D_perp", "D_par")

    def init_vars(self) -> None:
        g = self.grid
        for n in self.kperp_names:
            self.add_var(_var(f"{n}_kperp", {"kperp": g.kp_centers}))
        for n in self.ky_names:
            self.add_var(_var(f"{n}_ky", {"ky": g.ky_centers}))
        for n in self.kz_names:
            self.add_var(_var(f"{n}_kz", {"kz": g.kz_centers}))

    def execute(self) -> None:
        g = self.grid
        F = g.compact(self.system.get_fields())
        ep = g.mode_energy_parts(F)
        e = ep.sum(axis=0)
        q_es, q_em = g.mode_heat_flux(F)
        inj = g.mode_injection(F)
        dperp, dpar = 2 * g.rate_perp * e, 2 * g.rate_par * e
        kp = {"E": e, "E_uperp": ep[0], "E_bperp": ep[1], "E_dbpar": ep[2],
              "E_upar": ep[3], "inj": inj, "D_perp": dperp, "D_par": dpar,
              "Qhat": q_es + q_em, "Qhat_es": q_es, "Qhat_em": q_em,
              "kz2E": g.kz2 * e}
        for n in self.kperp_names:
            self.save_data(f"{n}_kperp", g.binned(g.ikp, kp[n], g.n_kp).get())
        for n in self.ky_names:
            self.save_data(f"{n}_ky", g.binned(g.iky, kp[n], g.n_ky).get())
        for n in self.kz_names:
            self.save_data(f"{n}_kz", g.binned(g.ikz, kp[n], g.n_kz).get())


# ----------------------------------------------------------------------
# (c) 2D maps, window averages
# ----------------------------------------------------------------------
class ZPMapsDiag(_ZPWindowed):
    name = "zp_maps"
    option_defaults: ClassVar[dict[str, object]] = {}
    map_names = ("E", "inj", "D_perp", "D_par")

    def init_vars(self) -> None:
        g = self.grid
        dims = {"kperp": g.kp_centers, "kz": g.kz_centers}
        for n in self.map_names:
            self.add_var(_var(f"{n}_map", dims, dtype="f4", zlib=True))
        self.add_var(_var("n_samples"))
        self.reset_window()

    def sample(self) -> None:
        g = self.grid
        F = g.compact(self.system.get_fields())
        e = g.mode_energy_parts(F).sum(axis=0)
        inj = g.mode_injection(F)
        b2 = g.ikp * g.n_kz + g.ikz
        n = g.n_kp * g.n_kz
        self._accumulate([
            g.binned(b2, e, n), g.binned(b2, inj, n),
            g.binned(b2, 2 * g.rate_perp * e, n),
            g.binned(b2, 2 * g.rate_par * e, n)])

    def execute(self) -> None:
        g = self.grid
        for name, acc in zip(self.map_names, self._acc):
            m = (acc / self._count).get().reshape(g.n_kp, g.n_kz)
            self.save_data(f"{name}_map", m.astype(np.float32))
        self.save_data("n_samples", float(self._count))


# ----------------------------------------------------------------------
# (d) k_par(k_perp) as kpar_spectrum.py
# ----------------------------------------------------------------------
class ZPKparDiag(_ZPDiag):
    """
    Per record, for each octave shell s (edges 0.25, 0.75, 1.5, 3, ...) and
    field f, the accumulators of kpar_spectrum.py for ONE snapshot:
      den   = sum w |f_s|^2           lab  = sum w k_z^2 |f_s|^2
      full  = <(d_z f_s + {Psi, f_s})^2>        (full Psi)
      wrong = <(d_z f_s - {Psi, f_s})^2>
      large = <(d_z f_s + {Psi_<, f_s})^2>      (Psi at k_perp < lower edge)
      E     = energy of f_s                vort2 = sum w k_perp^4 |f_s|^2
    plus b2 = sum w k_perp^2 |Psi|^2 and the normalisation check norm_err.
    Averages over records: k_par = sqrt(sum full / sum den) etc. (ratios of
    record-summed numerators and denominators, as the offline tool).
    Products on a grid of >= 2 pad K + 1 points (pad 2: exact).
    """

    name = "zp_kpar"
    option_defaults: ClassVar[dict[str, object]] = {"pad": 2.0}
    acc_names = ("den", "lab", "full", "wrong", "large", "E", "vort2")

    def init_vars(self) -> None:
        g = self.grid
        kx_max = g.Kx * g.dkx
        ky_max = g.Ky * g.dky
        self.kperp_max = min(kx_max, ky_max)
        edges = [0.25, 0.75] + [1.5 * 2**j for j in range(12)]
        self.shells = [(lo, hi) for lo, hi in zip(edges[:-1], edges[1:])
                       if lo < self.kperp_max]
        dims = {"field": np.arange(4.0),
                "shell": np.array([lo for lo, _ in self.shells])}
        for n in self.acc_names:
            self.add_var(_var(n, dims))
        self.add_var(_var("b2"))
        self.add_var(_var("norm_err"))
        hi = _var("shell_hi", {"shell": dims["shell"]}, time_dependent=False)
        self.add_var(hi)
        self._setup = False

    def ready(self) -> None:
        super().ready()
        # time-independent data must be cached after FlucsOutput.ready()'s clear()
        self.save_data("shell_hi", np.array([h for _, h in self.shells]))

    def _prepare(self):
        g = self.grid
        pad = float(self.pad)
        self.Nx, self.Ny, self.Nz = (
            _fast_len(int(np.ceil(2 * pad * K)) + 1) for K in (g.Kx, g.Ky, g.Kz))
        self.shape_r = (self.Nz, self.Nx, self.Ny)
        self.shape_c = (self.Nz, self.Nx, self.Ny // 2 + 1)
        self.ntot = float(self.Nz * self.Nx * self.Ny)
        self.iz = g.mz % self.Nz
        self.ix = g.mx % self.Nx
        self.iy = g.my
        # Hermitian partner of each k_y = 0 mode (within the solved list)
        my, mx, mz = g.my_np, g.mx_np, g.mz_np
        zero = np.flatnonzero(my == 0)
        pos = {(int(mz[i]), int(mx[i])): i for i in zero}
        partner = np.array([pos[(-int(mz[i]), -int(mx[i]))] for i in zero])
        self.ky0 = cp.asarray(zero)
        self.ky0_partner = cp.asarray(partner)
        self.masks = [((g.kperp >= lo) & (g.kperp < hi)) for lo, hi in self.shells]
        self.lmasks = [(g.kperp < lo) for lo, _ in self.shells]
        self._setup = True

    def _real(self, Fk):
        out = cp.zeros(self.shape_c, dtype=complex)
        out[self.iz, self.ix, self.iy] = Fk
        return cp.fft.irfftn(out, s=self.shape_r, axes=(0, 1, 2)) * self.ntot

    def execute(self) -> None:
        if not self._setup:
            self._prepare()
        g = self.grid
        F = g.compact(self.system.get_fields()).copy()
        # exact Hermitian projection of the k_y = 0 plane (as the tool)
        F0 = F[:, self.ky0]
        F[:, self.ky0] = 0.5 * (F0 + cp.conj(F[:, self.ky0_partner]))
        Psi = F[1]
        ikx, iky, ikz = 1j * g.kx, 1j * g.ky, 1j * g.kz
        psx, psy = self._real(ikx * Psi), self._real(iky * Psi)
        b2 = float((g.w * g.k2 * (Psi.real**2 + Psi.imag**2)).sum().get())
        ns = len(self.shells)
        acc = {n: np.zeros((4, ns)) for n in self.acc_names}
        wE = (0.5 * g.k2, 0.5 * g.k2, 0.5 * g.c, 0.5)
        norm_err = 0.0
        for s in range(ns):
            m = self.masks[s]
            if bool(self.lmasks[s].any()):
                PL = Psi * self.lmasks[s]
                plx, ply = self._real(ikx * PL), self._real(iky * PL)
            else:
                plx = ply = None
            for i in range(4):
                fs = F[i] * m
                a2 = g.w * (fs.real**2 + fs.imag**2)
                den = float(a2.sum().get())
                if den <= 0:
                    continue
                lab = float((g.kz2 * a2).sum().get())
                fz, fx, fy = self._real(ikz * fs), self._real(ikx * fs), self._real(iky * fs)
                br = psx * fy - psy * fx
                m_lab = float(cp.mean(fz * fz).get())
                norm_err = max(norm_err, abs(m_lab / lab - 1) if lab > 0 else 0.0)
                acc["den"][i, s] = den
                acc["lab"][i, s] = lab
                acc["full"][i, s] = float(cp.mean((fz + br) ** 2).get())
                acc["wrong"][i, s] = float(cp.mean((fz - br) ** 2).get())
                if plx is not None:
                    acc["large"][i, s] = float(cp.mean((fz + plx * fy - ply * fx) ** 2).get())
                else:
                    acc["large"][i, s] = lab
                acc["E"][i, s] = float((wE[i] * a2).sum().get())
                acc["vort2"][i, s] = float((g.k2**2 * a2).sum().get())
                del fz, fx, fy, br
        for n in self.acc_names:
            self.save_data(n, acc[n])
        self.save_data("b2", b2)
        self.save_data("norm_err", norm_err)


# ----------------------------------------------------------------------
# (e) mean-field lines (spectral)
# ----------------------------------------------------------------------
class ZPProfilesDiag(_ZPDiag):
    """
    line_x[f, kx]: Fourier coefficients X_f(k_x, k_y = 0, k_z = 0), i.e. the
      (y,z)-average of field f as a function of x: f(x) = sum_kx X e^{i kx x}
      (k_x sorted ascending, both signs; X(-kx) = conj X(kx)).
    line_y[f, ky]: X_f(k_x = 0, k_y, k_z = 0), k_y >= 0: the (x,z)-average
      as a function of y: f(y) = X(0) + 2 Re sum_{ky>0} X e^{i ky y}.
    Fields f = 0 Phi, 1 Psi, 2 dBpar, 3 upar. Double precision.
    Reader: flucs.systems.zpinch_fourier.zpinch_readers.mean_profiles.
    """

    name = "zp_profiles"
    option_defaults: ClassVar[dict[str, object]] = {}

    def init_vars(self) -> None:
        g = self.grid
        mx, my, mz = g.mx_np, g.my_np, g.mz_np
        lx = np.flatnonzero((my == 0) & (mz == 0))
        lx = lx[np.argsort(mx[lx])]
        ly = np.flatnonzero((mx == 0) & (mz == 0))
        ly = ly[np.argsort(my[ly])]
        self.lx, self.ly = cp.asarray(lx), cp.asarray(ly)
        fd = np.arange(4.0)
        self.add_var(_var("line_x", {"field": fd, "kx_line": mx[lx] * g.dkx}, complex_=True))
        self.add_var(_var("line_y", {"field": fd, "ky_line": my[ly] * g.dky}, complex_=True))

    def execute(self) -> None:
        F = self.grid.compact(self.system.get_fields())
        self.save_data("line_x", F[:, self.lx].get())
        self.save_data("line_y", F[:, self.ly].get())


# ----------------------------------------------------------------------
# (f) x-y slices at z = 0
# ----------------------------------------------------------------------
class ZPSlicesDiag(_ZPDiag):
    """float32 Phi and dBpar on the x-y plane z = 0, real-space grid of the
    solved modes (nx_c = 2 Kx + 1 by ny_c = 2 Ky + 1 points, x = i Lx/nx_c)."""

    name = "zp_slices"
    option_defaults: ClassVar[dict[str, object]] = {"fields": ["Phi", "dBpar"]}

    def init_vars(self) -> None:
        g = self.grid
        self.fidx = [FIELD_NAMES.index(f) for f in self.fields]
        dims = {"x": np.arange(g.nx_c) * g.Lx / g.nx_c,
                "y": np.arange(g.ny_c) * g.Ly / g.ny_c}
        for f in self.fields:
            self.add_var(_var(f"{f}_z0", dims, dtype="f4", zlib=True))
        # compact k_x index q (fft order of nx_c) -> padded index
        m = np.rint(np.fft.fftfreq(g.nx_c, d=1.0 / g.nx_c)).astype(int)
        self.pix = cp.asarray(m % self.system.nx)

    def execute(self) -> None:
        g = self.grid
        X = self.system.get_fields()
        for f, i in zip(self.fields, self.fidx):
            # z = 0: sum over k_z (padded modes are zero), then compact layout
            X2 = X[i].sum(axis=0)[self.pix, : g.hy_c]
            r = cp.fft.irfft2(X2, s=(g.nx_c, g.ny_c)) * (g.nx_c * g.ny_c)
            self.save_data(f"{f}_z0", r.get().astype(np.float32))


# ----------------------------------------------------------------------
# helpers for padded-grid products
# ----------------------------------------------------------------------
def _c2r(system, X):
    """padded half-grid coefficients (..., nz, nx, hy) -> real padded grid."""
    return cp.fft.irfftn(X, s=system.full_tuple, axes=(-3, -2, -1), norm="forward")


def _r2c(r):
    return cp.fft.rfftn(r, axes=(-3, -2, -1), norm="forward")


# ----------------------------------------------------------------------
# (g) k_y transfers (curvy3d subcrit_mode_balances)
# ----------------------------------------------------------------------
class ZPKyTransferDiag(_ZPDiag):
    """
    curvy3d (curvy3dsolver_rmhd_zpinch_cupy_ARS343.py, subcrit_mode_balances)
    quantities for the modes k_x = k_z = 0, k_y = 1..n (per-mode values, same
    normalisation as curvy3d; curvy3d field X[2] = Psi, X[1] = dBpar):
      dBx_sq = 1/2 ky^2 |Psi|^2,  dBx_transfer = -ky^2 Re(Psi* N_Psi),
      dBx_diss = nu_perp ky^(2p+2) |Psi|^2,
      upar_sq = 1/2 |upar|^2, upar_transfer = -Re(upar* N_u),
      upar_inj = Re(upar* kb kP (-i ky) Psi), upar_diss = nu_perp ky^2p |upar|^2,
      upar_reynolds/maxwell_transfer = the two parts of upar_transfer,
      with N_Psi = i ky (Phi_x Psi)_ky, N_u = i ky [(Phi_x upar) - (Psi_x dBpar)]_ky
      (products: (x,z)-averages, exact on the 3/2 padded grid).
      dtdBx_sq, dtupar_sq: one-step finite difference (backward: (now - previous
      step)/dt; curvy3d uses the step after the save).
      dBx_profile = -i ky Psi(0,0,ky), upar_profile = upar(0,0,ky) (k_y = 0..Ky;
      instantaneous; curvy3d saves the run average, and adds dBx_profile twice).
      turb_*_sq_profile: (x,z)-averaged |u_perp|^2, dBpar^2, |dB_perp|^2, upar^2
      of the fields without the k_x = k_z = 0 line, Fourier coefficients in y.
      turb_char_wavenumbers (12), turb_alfvenic_comps (6): as curvy3d.
    """

    name = "zp_kytransfer"
    option_defaults: ClassVar[dict[str, object]] = {"n_subcrit_modes": -1}
    balance_names = ("dBx_sq", "dtdBx_sq", "dBx_transfer", "dBx_diss",
                     "upar_sq", "dtupar_sq", "upar_transfer", "upar_inj",
                     "upar_diss", "upar_reynolds_transfer", "upar_maxwell_transfer")
    profile_names = ("dBx_profile", "upar_profile", "turb_uperp_sq_profile",
                     "turb_dBpar_sq_profile", "turb_dBperp_sq_profile",
                     "turb_upar_sq_profile")

    def init_vars(self) -> None:
        g = self.grid
        n = int(self.n_subcrit_modes)
        self.n = g.Ky if n < 0 else min(n, g.Ky)
        kyb = np.arange(1, self.n + 1) * g.dky
        for nm in self.balance_names:
            self.add_var(_var(nm, {"ky_mode": kyb}))
        for nm in self.profile_names:
            self.add_var(_var(nm, {"ky_profile": g.ky_centers}, complex_=True))
        self.add_var(_var("turb_char_wavenumbers", {"turb_wavenumber_measurements": np.arange(12.0)}))
        self.add_var(_var("turb_alfvenic_comps", {"turb_alfvenic_comps_measurements": np.arange(6.0)}))
        s = self.system
        p = float(s.input["hyperdissipation.kperp_power"])
        self.nu = max(float(s.input["hyperdissipation.kperp"]), 0.0)
        self.p = p
        # curvy3d wavenumber_mask (one member of each conjugate pair) x
        # turb_mask (no k_x = k_z = 0 line)
        kx, ky, kz = g.kx, g.ky, g.kz
        half = (ky > 0) | (kx > 0) | ((kx == 0) & (kz > 0))
        turb = ~((kx == 0) & (kz == 0))
        self.wt = 2.0 * (half & turb).astype(float)
        self.turb_h = cp.asarray(~((s.kx[None, :, None] == 0) & (s.kz[:, None, None] == 0))
                                 * np.ones((1, 1, s.half_ny)))

    def execute(self) -> None:
        g = self.grid
        s = self.system
        X = s.get_fields()
        Xp = s.get_fields(1)
        n, Ky, ny = self.n, g.Ky, s.full_tuple[2]
        kyl = cp.asarray(np.arange(Ky + 1) * g.dky)
        dx, dy = 1j * g.kx_h, 1j * g.ky_h
        Phi, Psi, B, u = X[0], X[1], X[2], X[3]

        def xz_mean_coeffs(r):
            prof = r.mean(axis=(0, 1))
            return cp.fft.rfft(prof)[: Ky + 1] / ny

        # products for the k_x = k_z = 0 nonlinear terms
        R = _c2r(s, cp.stack((dx * Phi, Psi, dx * Psi, B, u)))
        P0 = xz_mean_coeffs(R[0] * R[1])
        P1 = xz_mean_coeffs(R[0] * R[4])
        P2 = xz_mean_coeffs(R[2] * R[3])
        del R
        tm = self.turb_h
        R = _c2r(s, cp.stack((dx * Phi * tm, -dy * Phi * tm, B * tm,
                              dx * Psi * tm, -dy * Psi * tm, u * tm)))
        T_u = xz_mean_coeffs(R[0] ** 2 + R[1] ** 2)
        T_B = xz_mean_coeffs(R[2] ** 2)
        T_b = xz_mean_coeffs(R[3] ** 2 + R[4] ** 2)
        T_up = xz_mean_coeffs(R[5] ** 2)
        del R

        line = slice(1, n + 1)
        ky = kyl[line]
        psi, uu = Psi[0, 0, line], u[0, 0, line]
        psi_p, uu_p = Xp[1][0, 0, line], Xp[3][0, 0, line]
        NPsi = 1j * ky * P0[line]
        Nu_r = 1j * ky * P1[line]
        Nu_m = -1j * ky * P2[line]
        a2 = lambda z: z.real**2 + z.imag**2
        dt = float(s.current_dt)
        dBx_sq = 0.5 * ky**2 * a2(psi)
        upar_sq = 0.5 * a2(uu)
        vals = {
            "dBx_sq": dBx_sq,
            "dtdBx_sq": (dBx_sq - 0.5 * ky**2 * a2(psi_p)) / dt,
            "dBx_transfer": -(ky**2 * cp.conj(psi) * NPsi).real,
            "dBx_diss": self.nu * (ky**2) ** (self.p + 1) * a2(psi),
            "upar_sq": upar_sq,
            "dtupar_sq": (upar_sq - 0.5 * a2(uu_p)) / dt,
            "upar_transfer": -(cp.conj(uu) * (Nu_r + Nu_m)).real,
            "upar_inj": (cp.conj(uu) * g.kb * g.kP * (-1j * ky) * psi).real,
            "upar_diss": self.nu * (ky**2) ** self.p * a2(uu),
            "upar_reynolds_transfer": -(cp.conj(uu) * Nu_r).real,
            "upar_maxwell_transfer": -(cp.conj(uu) * Nu_m).real,
        }
        for nm in self.balance_names:
            self.save_data(nm, vals[nm].get())
        self.save_data("dBx_profile", (-1j * kyl * Psi[0, 0, : Ky + 1]).get())
        self.save_data("upar_profile", u[0, 0, : Ky + 1].get())
        self.save_data("turb_uperp_sq_profile", T_u.get())
        self.save_data("turb_dBpar_sq_profile", T_B.get())
        self.save_data("turb_dBperp_sq_profile", T_b.get())
        self.save_data("turb_upar_sq_profile", T_up.get())

        F = g.compact(X)
        a2F = F.real**2 + F.imag**2
        wt = self.wt
        kx2, ky2, k2 = g.kx**2, g.ky**2, g.k2
        specs = (k2 * a2F[0], a2F[2], k2 * a2F[1], a2F[3])
        cw = []
        for q in specs:
            den = (wt * q).sum()
            cw += [(wt * kx2 * q).sum() / den, (wt * ky2 * q).sum() / den,
                   (wt * k2 * q).sum() / den]
        self.save_data("turb_char_wavenumbers", cp.stack(cw).get())
        comps = [(wt * k2 * a2F[0]).sum(), (wt * ky2 * a2F[0]).sum(),
                 (wt * kx2 * a2F[0]).sum(), (wt * k2 * a2F[1]).sum(),
                 (wt * ky2 * a2F[1]).sum(), (wt * kx2 * a2F[1]).sum()]
        self.save_data("turb_alfvenic_comps", cp.stack(comps).get())


class ZPStreamerTransferDiag(_ZPWindowed):
    """
    curvy3d dBx_streamer_transfer_kzkx, upar_streamer_transfer_reynolds_kzkx,
    upar_streamer_transfer_maxwell_kzkx (transfer into the k_y = 1, k_x = k_z
    = 0 mode, resolved in (k_z, k_x)), averaged over the samples of each
    record window; compact (curvy3d) layout, fft order in k_z and k_x; float32.
    """

    name = "zp_streamer_kzkx"
    option_defaults: ClassVar[dict[str, object]] = {}
    out_names = ("dBx_streamer_transfer_kzkx",
                 "upar_streamer_transfer_reynolds_kzkx",
                 "upar_streamer_transfer_maxwell_kzkx")

    def init_vars(self) -> None:
        g = self.grid
        kzc = np.fft.fftfreq(g.nz_c, d=1.0 / g.nz_c) * g.dkz
        kxc = np.fft.fftfreq(g.nx_c, d=1.0 / g.nx_c) * g.dkx
        for nm in self.out_names:
            self.add_var(_var(nm, {"kz_c": kzc, "kx_c": kxc}, dtype="f4", zlib=True))
        self.add_var(_var("n_samples"))
        self.cz = cp.asarray(g.mz_np % g.nz_c)
        self.cx = cp.asarray(g.mx_np % g.nx_c)
        self.kxs = cp.asarray(kxc)[None, :]
        self.reset_window()

    def sample(self) -> None:
        g = self.grid
        F = g.compact(self.system.get_fields())
        C = cp.zeros((4, g.nz_c, g.nx_c, g.hy_c), dtype=complex)
        C[:, self.cz, self.cx, g.my] = F
        Phi, Psi, B, u = C[0], C[1], C[2], C[3]

        def pair(a, b):
            return (cp.conj(a[:, :, :-1]) * b[:, :, 1:] - a[:, :, 1:] * cp.conj(b[:, :, :-1])).sum(axis=2)

        dbx = (self.kxs * cp.conj(Psi[0, 0, 1]) * pair(Psi, Phi)).real
        rey = (self.kxs * cp.conj(u[0, 0, 1]) * pair(u, Phi)).real
        mxw = -(self.kxs * cp.conj(u[0, 0, 1]) * pair(B, Psi)).real
        self._accumulate([dbx, rey, mxw])

    def execute(self) -> None:
        for nm, acc in zip(self.out_names, self._acc):
            self.save_data(nm, (acc / self._count).get().astype(np.float32))
        self.save_data("n_samples", float(self._count))


# ----------------------------------------------------------------------
# (h) shell-to-shell transfers
# ----------------------------------------------------------------------
class ZPShellTransferDiag(_ZPDiag):
    """
    Shell-to-shell transfers T[j, l] (from shell j to shell l) of the two
    hydra_shell_plotter_*_gpu.py scripts, per record, as box averages (the
    scripts' "-2 sum(mask ...)" half-space sums; their CSV files hold the time
    mean divided by L_y^2, see runs/tools/zp_shells_export.py).
    Shells (as the scripts): k_perp: edges kmin 2^i, kmin = 2 pi/max(Lx,Ly)/2.00001,
    n = ceil(log2(kperp_max/kmin)) + 1 with kperp_max = |(kx(nx//2), ky(ny//2))|;
    k_z: kmin = 2 pi/Lz/2.00001, kz_max = kz(nz//2). Shell 0 = k < edge[1].
    Elsasser set: zplus, zminus, zparplus, zparminus. Kinetic/magnetic set:
    Tuu, TBu, TBB, TuB, TBparBpar, TuparBpar, Tuparupar, TBparupar (and the
    sums uperp, deltaBperp, deltaBpar, upar).
    Method (exact, same triads as the scripts): for each source shell j the
    products of the advecting field's and f_j's derivatives are transformed
    once on the 3/2 padded grid; T[j, l] for all l follows from one weighted
    bincount over the receiving field's modes.
    """

    name = "zp_shells"
    option_defaults: ClassVar[dict[str, object]] = {
        "shell_types": ["kperp", "kz"],
        "sets": ["elsasser", "magkin"],
    }
    elsasser = ("zplus", "zminus", "zparplus", "zparminus")
    magkin = ("Tuu", "TBu", "TBB", "TuB", "TBparBpar", "TuparBpar",
              "Tuparupar", "TBparupar")

    def init_vars(self) -> None:
        g = self.grid
        s = self.system
        self.types = list(self.shell_types)
        self.outs = []
        if "elsasser" in self.sets:
            self.outs += list(self.elsasser)
        if "magkin" in self.sets:
            self.outs += list(self.magkin)
        nx_c, ny_c, nz_c = g.nx_c, g.ny_c, g.nz_c
        self.sid = {}
        self.sid_pad = {}
        self.edges = {}
        for st in self.types:
            if st == "kperp":
                kmin = 2 * np.pi / max(g.Lx, g.Ly) / 2.00001
                kmax = np.sqrt((2 * np.pi * (nx_c // 2) / g.Lx) ** 2
                               + (2 * np.pi * (ny_c // 2) / g.Ly) ** 2)
                kmod = g.kperp
            elif st == "kz":
                kmin = 2 * np.pi / g.Lz / 2.00001
                kmax = 2 * np.pi * (nz_c // 2) / g.Lz
                kmod = cp.abs(g.kz)
            else:
                raise ValueError(f"zp_shells: unknown shell type {st}")
            n = int(np.ceil(np.log2(kmax / kmin)) + 1)
            start = np.log2(kmin)
            K = np.logspace(start, start + n, n + 1, base=2)
            self.edges[st] = K
            sid = cp.searchsorted(cp.asarray(K), kmod, side="right") - 1
            sid = cp.maximum(sid, 0)
            sid = cp.where(kmod < K[-1], sid, n)        # n = outside (none)
            self.sid[st] = sid
            pad = cp.full(int(np.prod(s.half_tuple)), -1, dtype=cp.int64)
            pad[g.idx] = sid
            self.sid_pad[st] = pad.reshape(s.half_tuple)
            dims = {f"{st}_from": K[:-1], f"{st}_to": K[:-1]}
            for nm in self.outs:
                self.add_var(_var(f"{st}_{nm}", dims))
            ev = _var(f"{st}_K_boundaries", {f"{st}_edges": K}, time_dependent=False)
            self.add_var(ev)

    def ready(self) -> None:
        super().ready()
        for st in self.types:
            self.save_data(f"{st}_K_boundaries", self.edges[st])

    # T_V(j -> all l) for V(a, f_j, f_l) given real a_x, a_y, fj_x, fj_y
    def _V(self, ax, ay, fjx, fjy, fl, sid, n):
        g = self.grid
        hat = _r2c(cp.stack((ax * fjx - ay * fjy, ax * fjy, ay * fjx)))
        hat = hat.reshape(3, -1)[:, g.idx]
        kx, ky = g.kx, g.ky
        val = (cp.conj(hat[0]) * (-kx * ky * fl) + cp.conj(hat[1]) * (-ky * ky * fl)
               - cp.conj(hat[2]) * (-kx * kx * fl)).real
        return cp.bincount(sid, weights=g.w * val, minlength=n + 1)[:n]

    def _S(self, ax, ay, fj, fl, sid, n):
        g = self.grid
        hat = _r2c(cp.stack((ax * fj, ay * fj))).reshape(2, -1)[:, g.idx]
        val = (cp.conj(hat[0]) * (1j * g.ky * fl) - cp.conj(hat[1]) * (1j * g.kx * fl)).real
        return cp.bincount(sid, weights=g.w * val, minlength=n + 1)[:n]

    def execute(self) -> None:
        g = self.grid
        s = self.system
        X = s.get_fields()
        dx, dy = 1j * g.kx_h, 1j * g.ky_h
        Phi, Psi, B, u = X[0], X[1], X[2], X[3]
        c = g.c
        sq = np.sqrt(c)          # sqrt(1 + 1/kb)
        f = 1.0 / sq             # sqrt(kb/(1 + kb))
        Fc = g.compact(X)
        fPhi, fPsi, fB, fu = Fc[0], Fc[1], Fc[2], Fc[3]
        fzp, fzm = fPhi + fPsi, fPhi - fPsi
        fZp, fZm = fu + sq * fB, fu - sq * fB
        Phx, Phy, Psx, Psy = _c2r(s, cp.stack((dx * Phi, dy * Phi, dx * Psi, dy * Psi)))
        for st in self.types:
            sid, sidp = self.sid[st], self.sid_pad[st]
            n = len(self.edges[st]) - 1
            res = {nm: np.zeros((n, n)) for nm in self.outs}
            for j in range(n):
                m = (sidp == j)
                Rj = _c2r(s, cp.stack((dx * Phi * m, dy * Phi * m, dx * Psi * m,
                                       dy * Psi * m, B * m, u * m)))
                jPhx, jPhy, jPsx, jPsy, jB, ju = Rj
                if "zplus" in self.outs:
                    res["zplus"][j] = self._V(Phx - Psx, Phy - Psy, jPhx + jPsx, jPhy + jPsy, fzp, sid, n).get()
                    res["zminus"][j] = self._V(Phx + Psx, Phy + Psy, jPhx - jPsx, jPhy - jPsy, fzm, sid, n).get()
                    res["zparplus"][j] = self._S(Phx - f * Psx, Phy - f * Psy, ju + sq * jB, fZp, sid, n).get()
                    res["zparminus"][j] = self._S(Phx + f * Psx, Phy + f * Psy, ju - sq * jB, fZm, sid, n).get()
                if "Tuu" in self.outs:
                    res["Tuu"][j] = self._V(Phx, Phy, jPhx, jPhy, fPhi, sid, n).get()
                    res["TuB"][j] = -self._V(Psx, Psy, jPhx, jPhy, fPsi, sid, n).get()
                    res["TBu"][j] = -self._V(Psx, Psy, jPsx, jPsy, fPhi, sid, n).get()
                    res["TBB"][j] = self._V(Phx, Phy, jPsx, jPsy, fPsi, sid, n).get()
                    res["TBparBpar"][j] = c * self._S(Phx, Phy, jB, fB, sid, n).get()
                    res["TBparupar"][j] = -self._S(Psx, Psy, jB, fu, sid, n).get()
                    res["Tuparupar"][j] = self._S(Phx, Phy, ju, fu, sid, n).get()
                    res["TuparBpar"][j] = -self._S(Psx, Psy, ju, fB, sid, n).get()
                del Rj, jPhx, jPhy, jPsx, jPsy, jB, ju
            for nm in self.outs:
                self.save_data(f"{st}_{nm}", res[nm])


ZP_RUNTIME_DIAGS = {
    ZPScalarsDiag, ZPSpectraDiag, ZPMapsDiag, ZPKparDiag, ZPProfilesDiag,
    ZPSlicesDiag, ZPKyTransferDiag, ZPStreamerTransferDiag, ZPShellTransferDiag,
}
