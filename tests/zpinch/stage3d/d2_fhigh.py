"""Stage 3d D2: f_high variability (200-R/vA windows, blocks), episode lengths, Q/E means; FLUCS A3a-Z-k8-C300 vs curvy3d."""
import numpy as np
z = np.load("/mnt/users/clarkew/flucs_work/runs/stage3d/two_state_series.npz")
def f(l): c = l != 0; return np.mean(l[c] == 1) if c.any() else np.nan
for k in ["C:A3a-Y+Yc2", "C:A3a-Z", "C:A3a-YP4", "C:A3a-YHR2", "F:A3a-Z-k8-C300"]:
    t, q, E, l = [z[f"{k}|{v}"] for v in "tqEl"]
    w = [(a, a + 200) for a in np.arange(t[0], t[-1] - 199, 100)]
    fw = [f(l[(t >= a) & (t < b)]) for a, b in w]
    # episode lengths (classified runs of equal label, undecided gaps dropped)
    lab = l[l != 0]; tt = t[l != 0]; cuts = np.flatnonzero(np.diff(lab)) + 1
    seg = np.split(np.arange(len(lab)), cuts)
    LH = [len(s) * 0.1 for s in seg[1:-1] if lab[s[0]] == 1]; LL = [len(s) * 0.1 for s in seg[1:-1] if lab[s[0]] == -1]
    print(f"{k:18s} t {t[0]:.0f}-{t[-1]:.0f} f_high {f(l):.2f}  200-windows (step 100): {' '.join(f'{x:.2f}' for x in fw)} "
          f"[min {np.nanmin(fw):.2f} max {np.nanmax(fw):.2f}]  complete episodes H n{len(LH)} mean {np.mean(LH) if LH else np.nan:.0f} "
          f"L n{len(LL)} mean {np.mean(LL) if LL else np.nan:.0f}  <Q> {q.mean():.2f} <E> {E.mean():.0f}")
