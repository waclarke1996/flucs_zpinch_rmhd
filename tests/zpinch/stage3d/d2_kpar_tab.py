"""Stage 3d D2: per-shell kz_rms / k_par / k_par,wr / chi_Phi, Z-grid FLUCS states vs Y-grid high/low states (Stage 3c).
k_z* = (gamma/nu_z)^(1/8), gamma 0.717 (as dissipation_map): Y nu_z 1.39e-14 -> 51.8, Z nu_z 8.7e-16 -> 73.2.
NB: Stage 3c rows come from the OLD kpar_spectrum (chi = k_perp u_s / k_par per snapshot); Z rows from the fixed one (ea3360c2,
state average, chi = shell rms vorticity / k_par): compare kz_rms / k_par / k_par,wr, treat chi only as indicative."""
import re
S = "/mnt/users/clarkew/flucs_work/runs/"
KY, KZ = (0.717 / 1.38778e-14) ** 0.125, (0.717 / 8.7e-16) ** 0.125
files = [("H", "Z F 550-700 (4)", "stage3d/d2_state/kpar_H_550-700.txt", KZ),
         ("H", "Z F 850-900 (2)", "stage3d/d2_state/kpar_H_850-900.txt", KZ),
         ("H", "Y C A3a-YP4 t450", "stage3c/same_state/kpar_A3a-YP4_t450.txt", KY),
         ("H", "Y C A3a-Yc2 t1000", "stage3c/same_state/kpar_A3a-Yc2_t500.txt", KY),
         ("H", "Y F C300 t450", "stage3c/a2_state/kpar_A3a-Y-k8-C300_t450.txt", KY),
         ("L", "Z F 750-800 (2)", "stage3d/d2_state/kpar_L_750-800.txt", KZ),
         ("L", "Z F 1000", "stage3d/d2_state/kpar_L_1000.txt", KZ),
         ("L", "Y C A3a-Y t450", "stage3c/kpar_A3a-Y_t450.txt", KY),
         ("L", "Y F C300 t600", "stage3c/a2_state/kpar_A3a-Y-k8-C300_t600.txt", KY)]
print(f"k_z* Y = {KY:.1f}, Z = {KZ:.1f}; per shell: Efrac kz_rms / k_par / k_par,wr / chi_Phi")
print(f"{'st':2s} {'run':18s} " + " | ".join(f"{s:^28s}" for s in ("k~1.06", "2.12", "4.24", "8.49", "17.0", "33.9*")) + " | k_z* cross (lab, kpar)")
for st, name, f, kzs in files:
    rows, take = [], False
    for ln in open(S + f):
        if ln.strip().startswith("shell"): take = True; continue
        if take:
            if not ln.strip() or ln.startswith("("): break
            v = re.findall(r"[-\d.e+]+", ln.replace("*", ""))
            rows.append([float(x) for x in v[-8:]])
    rows = rows[1:]
    cl = next((f"{r[0]:.0f}" for r in rows if r[2] >= kzs), "none")
    ck = next((f"{r[0]:.0f}" for r in rows if r[3] >= kzs), "none")
    print(f"{st:2s} {name:18s} " + " | ".join(f"{r[1]:.2f} {r[2]:5.1f}/{r[3]:5.1f}/{r[5]:5.1f}/{r[7]:4.2f}" for r in rows) + f" | {cl:>4s} {ck:>4s}")
