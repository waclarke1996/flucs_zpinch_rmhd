"""Stage 3c B: streamer C6 (square box) vs C1-S30. sigma = d ln E(kx!=0)/dt."""
import numpy as np, netCDF4 as nc, sys
R = '/mnt/users/clarkew/flucs_work/runs/'
runs = ['stage3b/C1-S30', 'stage3b/C1-S30-lin', 'stage3c/C6-S30-k4', 'stage3c/C6-S30-k4-lin',
        'stage3c/C6-S30-k8', 'stage3c/C6-S30-k8-lin']
for r in runs:
    g = nc.Dataset(R + r + '/output.0d.nc')['0']
    t = g['time'][:]; dt = g['dt'][:]; e = g['energy']
    E, k0, kn = e['E'][:], e['E_kx0'][:], e['E_kxn0'][:]
    sig = np.gradient(np.log(kn), t)
    m = t > 0.3
    ipk = np.argmax(np.where(m, sig, -np.inf))
    i300 = np.argmax(k0 >= 300) if (k0 >= 300).any() else None
    f = kn / E
    t10 = t[np.argmax(f > 0.1)] if (f > 0.1).any() else np.nan
    late = sig[-max(5, len(t)//10):].mean()
    s300 = f'{sig[i300]:.1f}' if i300 is not None else '-'
    print(f'{r:24s} t_end {t[-1]:6.2f} t10% {t10:5.2f} sig_pk {sig[ipk]:6.2f} @t {t[ipk]:5.2f} (Ekx0 {k0[ipk]:.3g}) '
          f'sig@Ekx0=300 {s300:>5s} late_sig {late:5.2f} | end: E {E[-1]:.3g} Ekx0 {k0[-1]:.3g} kxn0/E {f[-1]:.2e} dt {dt[-1]:.2e}')
    if r.endswith('C6-S30-k4') or r.endswith('C6-S30-k8'):
        for tt in [1, 2, 3, 4, 4.5, 4.8, 5.0, t[-1]]:
            i = min(np.searchsorted(t, tt), len(t)-1)
            print(f'      t {t[i]:5.2f} E_kx0 {k0[i]:9.3g} kxn0/E {f[i]:.2e} sig {sig[i]:6.2f} dt {dt[i]:.2e}')
