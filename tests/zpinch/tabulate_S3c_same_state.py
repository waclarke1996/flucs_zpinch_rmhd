"""Per-snapshot dissipation_map (+ kpar_spectrum where present) lines with the two-state label at that time."""
import glob, os, re
import numpy as np

H = os.path.dirname(os.path.abspath(__file__))
Z = np.load(os.path.join(H, "..", "two_state_series.npz"))
KEY = {"A3a-Y": ("C:A3a-Y+Yc2", 0), "A3a-Yc2": ("C:A3a-Y+Yc2", 500), "A3a-YP4": ("C:A3a-YP4", 0),
       "A3a-Z": ("C:A3a-Z", 0), "A3a-YHR2": ("C:A3a-YHR2", 0),
       "A3a-Y-k8-S30": ("F:A3a-Y-k8-S30+S30c", 0), "A3a-Y-k8-S30c": ("F:A3a-Y-k8-S30+S30c", 0)}
rows = []
for f in glob.glob(os.path.join(H, "dmap_*_t*.txt")):
    m = re.match(r"dmap_(.+)_t(\d+)\.txt", os.path.basename(f))
    run, T = m.group(1), int(m.group(2))
    key, off = KEY[run]
    t = T + off
    tg, lab = Z[key + "|t"], Z[key + "|l"]
    i = np.argmin(abs(tg - t))
    st = "HLU"[[1, -1, 0].index(int(lab[i]))] if abs(tg[i] - t) < 1 else "-"
    txt = open(f).read()
    snap = re.search(r"^\s+\d+\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.-]+)", txt, re.M)
    fpar = re.search(r"f_par = D_par / D = ([\d.]+)", txt)
    al = re.search(r"alpha = ([\d.]+)", txt)
    inj = re.search(r"positive part at k_perp < 1.5: ([\d.]+)", txt)
    kz = ""
    k = os.path.join(H, f"kpar_{run}_t{T}.txt")
    if os.path.exists(k):
        L = [l for l in open(k) if re.match(r"\s+(1.5-3|3-6|6-12)\s", l)]
        kz = " ".join(f"{float(l.split()[7]):.2f}" for l in L)        # kpar/kz in shells 1.5-3, 3-6, 6-12
    E, b, fkx, fk1, q = snap.groups() if snap else ("nan",) * 5
    rows.append((st, run, t, float(E), float(b), float(fkx), float(fk1), float(q),
                 float(fpar.group(1)) if fpar else np.nan, float(al.group(1)) if al else np.nan,
                 float(inj.group(1)) if inj else np.nan, kz))
rows.sort(key=lambda r: ("HLU-".index(r[0]), r[1].startswith("A3a-Y-k8"), r[1], r[2]))
print(f"{'st':2s} {'run':14s} {'t':>5s} {'E':>6s} {'b_rms':>5s} {'kx0/E':>5s} {'k<=1.5':>6s} {'Q/QMHD':>6s} "
      f"{'f_par':>5s} {'alpha':>5s} {'inj<1.5':>7s}  kpar/kz (shells 1.5-3, 3-6, 6-12)")
for r in rows:
    print(f"{r[0]:2s} {r[1]:14s} {r[2]:5.0f} {r[3]:6.0f} {r[4]:5.1f} {r[5]:5.2f} {r[6]:6.2f} {r[7]:6.2f} "
          f"{r[8]:5.2f} {r[9]:5.2f} {r[10]:7.2f}  {r[11]}")
for s in "HL":
    c = np.array([r[3:11] for r in rows if r[0] == s and not r[1].startswith("A3a-Y-k8")])
    if len(c):
        print(f"curvy3d {s} snapshots ({len(c)}): median E {np.median(c[:,0]):.0f} b {np.median(c[:,1]):.1f} "
              f"kx0/E {np.median(c[:,2]):.2f} [{c[:,2].min():.2f}-{c[:,2].max():.2f}] k<=1.5 {np.median(c[:,3]):.2f} "
              f"Q {np.median(c[:,4]):.2f} f_par {np.median(c[:,5]):.2f} [{c[:,5].min():.2f}-{c[:,5].max():.2f}] "
              f"alpha {np.median(c[:,6]):.2f} [{c[:,6].min():.2f}-{c[:,6].max():.2f}] inj<1.5 {np.median(c[:,7]):.2f}")
