import sys, os, glob
sys.path.insert(0, os.path.expanduser("~/flucs_work/reference/curvy3d"))
import zpinch_runtools as zr
from netCDF4 import Dataset
D = "/mnt/users/clarkew/data/rmhd_zpinch/output/3D/BP2_anchors_v2"
for rid in sys.argv[1:]:
    f = glob.glob(f"{D}/*_A3a-{rid}.nc")[0]
    with Dataset(f) as ds:
        P = zr.params_from_dataset(ds); t, a, b = zr.heat_flux_series(ds, P)
        print(f"{rid}: save_time {P.save_time} hf {P.save_time_hf} t_hf_end {t[-1]:.1f} last_snap {zr.last_saved_snapshot(ds)} "
              f"complete {zr.is_complete(ds)} restart {getattr(P,'restart',None)} rtime {getattr(P,'restart_time',None)} "
              f"rfile {os.path.basename(str(getattr(P,'restart_file','')))[-30:]} nu {P.perpvisc:.3g}/{P.parvisc:.3g} nz {ds.dimensions['nz'].size}")
