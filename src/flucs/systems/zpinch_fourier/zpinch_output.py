"""
Time-based netCDF output for the Z-pinch system (Stage 4a).

An [output.<name>] section with save_time > 0 is handled by ZPTimedOutputNC
instead of the step-based FlucsOutputNC:

    save_time   > 0: write one record each time the simulation time crosses
                     a multiple of save_time (record time = actual time).
    sample_time > 0 (optional, < save_time): diagnostics that define
                     sample() are sampled at multiples of sample_time and
                     write the AVERAGE over the samples since the last record
                     (window average; windows end at multiples of save_time).
    save_steps       how often (in steps) the time is checked (default 1;
                     the check is one float comparison).

Grid points are absolute multiples of save_time / sample_time, so segments
of a restart chain continue the same grid. A fresh run (start time 0)
records/samples at t = 0; a restarted run starts at the first grid point
after its start time (the previous segment recorded that one). A window
that is not complete when a run ends is dropped (segments should be
multiples of save_time).
"""

from __future__ import annotations

import math
import time

from flucs import cupy as cp
from flucs.output import FlucsOutput, FlucsOutputNC


def _sync():
    try:
        cp.cuda.Device().synchronize()
    except Exception:
        pass


def _next_grid_point(t: float, period: float, include: bool) -> float:
    """Smallest multiple of period that is >= t (include) or > t."""
    n = t / period
    k = math.floor(n + 1e-9)
    if include and abs(n - round(n)) < 1e-9:
        return round(n) * period
    return (k + 1) * period


class ZPTimedOutputNC(FlucsOutputNC):
    """netCDF output written at fixed simulation-time intervals."""

    def __new__(cls, name, system):
        # Bypass FlucsOutput.__new__ (entry-point dispatch on output.type)
        return object.__new__(cls)

    def __init__(self, name: str, system) -> None:
        super().__init__(name, system)
        self.save_time = float(system.input[f"output.{name}.save_time"])
        self.sample_time = float(system.input[f"output.{name}.sample_time"])
        if self.save_time <= 0:
            raise ValueError(f"output.{name}.save_time must be > 0.")
        if self.sample_time > 0:
            ratio = self.save_time / self.sample_time
            if self.sample_time > self.save_time or abs(ratio - round(ratio)) > 1e-9:
                raise ValueError(
                    f"output.{name}: save_time must be a multiple of sample_time."
                )
        if self.save_steps <= 0:
            self.save_steps = 1
        self._next_record = None
        self._next_sample = None
        self._samplers = []
        self._timing = {}       # "diag.method" -> [calls, seconds] (incl. GPU sync)

    def _timed(self, diag, method):
        _sync()
        t0 = time.perf_counter()
        getattr(diag, method)()
        _sync()
        rec = self._timing.setdefault(f"{diag.name}.{method}", [0, 0.0])
        rec[0] += 1
        rec[1] += time.perf_counter() - t0

    def write(self):
        super().write()
        if not self._timing:
            return
        path = self.system.input.io_path / f"zp_timing_{self.name}.txt"
        lines = [f"{k:32s} {n:8d} {s:12.4f} {1e3 * s / max(n, 1):10.3f}"
                 for k, (n, s) in sorted(self._timing.items())]
        head = (f"# diag.method calls total_s ms/call (t = {float(self.system.current_time):.4f},"
                f" step {int(self.system.current_step)})")
        path.write_text(head + "\n" + "\n".join(lines) + "\n")

    def ready(self):
        super().ready()
        self._next_record = None
        self._next_sample = None
        self._samplers = [d for d in self.diagnostics if hasattr(d, "sample")]
        for diag in self._samplers:
            diag.reset_window()

    def _init_grid(self, t: float) -> None:
        fresh = t == 0.0
        self._next_record = _next_grid_point(t, self.save_time, fresh)
        if self.sample_time > 0:
            self._next_sample = _next_grid_point(t, self.sample_time, fresh)

    def execute(self):
        """Called every save_steps steps; records only on the time grid."""
        self.next_save += self.save_steps
        t = float(self.system.current_time)
        if self._next_record is None:
            self._init_grid(t)

        eps = 1e-9 * max(1.0, abs(t))
        if self.sample_time > 0 and t + eps >= self._next_sample:
            for diag in self._samplers:
                self._timed(diag, "sample")
            self._next_sample = _next_grid_point(t, self.sample_time, False)

        if t + eps < self._next_record:
            return
        self._next_record = _next_grid_point(t, self.save_time, False)

        if self._samplers and all(d.window_count() == 0 for d in self._samplers):
            # Window without samples (e.g. the start of a fresh run when
            # sampling starts later): nothing to write.
            return

        for diag in self.diagnostics:
            self._timed(diag, "execute")
        for diag in self._samplers:
            diag.reset_window()
        self.time_cache.append(self.system.current_time)
        self.dt_cache.append(self.system.current_dt)


def build_outputs(system) -> None:
    """Replacement for FlucsSystem.setup_output: timed or step outputs."""
    for output_name, output_opt in system.input["output"].items():
        if not isinstance(output_opt, dict):
            continue
        save_time = float(output_opt.get("save_time", -1.0))
        if save_time > 0:
            system.add_output(ZPTimedOutputNC(name=output_name, system=system))
            continue
        if output_opt["save_steps"] < 0:
            continue
        system.add_output(FlucsOutput(name=output_name, system=system))
    system.setup_netcdf_output_group()
