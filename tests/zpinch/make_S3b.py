"""Stage 3b streamer secondary-instability runs: write inputs, optionally run them.

python make_S3b.py <out dir>            write the 20 run directories
python make_S3b.py <out dir> --run [names...]   (on the GPU node) run them serially, linear controls
                                        first; skip runs that already finished (output.0d.nc + 'flucs given')
"""
import sys
import time
import pathlib as pl

from zp_common import POINTS, write_input, run

G1 = (216, 108, 256)
YG = (256, 128, 256)
NU4, NU8 = (2e-4, 8e-6), (4.44089e-12, 1.38778e-14)
CASES = {  # point, mode, dims, nu, power
    "C1": ("A2", (0, 1, 0), G1, NU4, 2),
    "C3": ("A3a", (0, 1, 1), G1, NU4, 2),
    "C2": ("A3a", (0, 1, 0), G1, NU4, 2),
    "C5": ("A4", (0, 1, 1), G1, NU4, 2),
    "C4": ("A3a", (0, 1, 1), YG, NU8, 4),
}
ES = (3.0, 30.0)


def runs():
    for lin in (True, False):
        for case, (pt, mode, dims, nu, power) in CASES.items():
            for es in ES:
                yield f"{case}-S{es:g}" + ("-lin" if lin else ""), case, lin, es


def write_all(out):
    for name, case, lin, es in runs():
        pt, mode, dims, nu, power = CASES[case]
        kb, kP = POINTS[pt]
        write_input(out / name, kb=kb, kP=kP, nu=nu, power=power, dims=dims, linear=lin,
                    tfinal=30.0, dt_max=0.01, max_cfl=0.5, save_steps=5,
                    init={"method": "streamer", "modes": [list(mode)], "energy": es,
                          "noise_energy": 1e-6 * es, "rand_seed": 257},
                    energy_opts={"stop_kxn0_over_kx0": 1.0, "stop_dt_min": 1e-4})


if __name__ == "__main__":
    out = pl.Path(sys.argv[1]).resolve()
    write_all(out)
    if "--run" in sys.argv:
        only = [a for a in sys.argv[2:] if not a.startswith("--")]
        for name, *_ in runs():
            if only and name not in only:
                continue
            d = out / name
            log = d / "run.log"
            if (d / "output.0d.nc").exists() and log.exists() and "flucs given" in log.read_text():
                print(f"{name}: done, skipped", flush=True)
                continue
            t0 = time.time()
            try:
                run(d)
                print(f"{name}: finished in {time.time() - t0:.0f} s", flush=True)
            except RuntimeError as e:
                print(f"{name}: FAILED ({e})", flush=True)
