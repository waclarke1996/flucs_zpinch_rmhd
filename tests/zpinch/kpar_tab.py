"""Per-shell kz_rms / k_par / k_par,wr / chi_Phi from kpar_spectrum.py stdout (old script, 1 snapshot/call)."""
import re, os
S = "/mnt/users/clarkew/flucs_work/runs/stage3c/"
KZS = (0.717 / 1.38778e-14) ** (1 / 8)   # k_z* = (gamma/nu_z)^(1/2q), gamma = 0.717 as dissipation_map.py, q = 4
files = [("H", "C A3a-YP4 t400", "same_state/kpar_A3a-YP4_t400.txt"), ("H", "C A3a-YP4 t450", "same_state/kpar_A3a-YP4_t450.txt"),
         ("H", "C A3a-Yc2 t1000", "same_state/kpar_A3a-Yc2_t500.txt"),
         ("H", "F C300 t450", "a2_state/kpar_A3a-Y-k8-C300_t450.txt"),
         ("L", "C A3a-Y t450", "kpar_A3a-Y_t450.txt"), ("L", "C A3a-Y t500", "kpar_A3a-Y_t500.txt"),
         ("L", "C A3a-YP4 t250", "same_state/kpar_A3a-YP4_t250.txt"),
         ("L", "F C300 t600", "a2_state/kpar_A3a-Y-k8-C300_t600.txt"),
         ("L", "F C300-cfl025 t450", "a2_state/kpar_A3a-Y-k8-C300-cfl025_t450.txt"),
         ("L", "F C300-cfl025 t600", "a2_state/kpar_A3a-Y-k8-C300-cfl025_t600.txt")]
print(f"k_z* = {KZS:.1f} (gamma = 0.717, nu_z = 1.39e-14, q = 4); per shell: kz_rms / k_par / k_par,wr / chi_Phi")
print(f"{'st':2s} {'run':20s} " + " | ".join(f"{s:^23s}" for s in ("k~1.06", "2.12", "4.24", "8.49", "17.0", "33.9*")) + " | kz*-cross (lab, kpar)")
for st, name, f in files:
    rows = []
    take = False
    for ln in open(S + f):
        if ln.strip().startswith("shell"): take = True; continue
        if take:
            if not ln.strip() or ln.startswith("("): break
            v = re.findall(r"[-\d.e+]+", ln.replace("*", ""))
            rows.append([float(x) for x in v[-8:]])   # k_c Efrac kz_rms k_par k_par< k_par,wr kpar/kz chi
    rows = rows[1:]  # drop 0.25-0.75 shell
    cl = next((f"{r[0]:.0f}" for r in rows if r[2] >= KZS), "none")
    ck = next((f"{r[0]:.0f}" for r in rows if r[3] >= KZS), "none")
    print(f"{st:2s} {name:20s} " + " | ".join(f"{r[2]:5.1f}/{r[3]:5.1f}/{r[5]:5.1f}/{r[7]:4.2f}" for r in rows) + f" | {cl:>4s} {ck:>4s}")
