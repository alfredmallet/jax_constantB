"""Analyticity-strip tracking (Sulem-Sulem-Frisch) for sm22_grow2d snapshots.
Fits the 2D shell spectrum E(k) ~ C k^-n exp(-2 delta k) on the band [kfit0, 0.6 kmax]
and reports a pointwise radius proxy rho_n = (||d^n B||_inf / n!)^(-1/n) for n=2..5
(max over l- and z-derivatives). Usage: python sm22_strip.py snaps_*.npz"""
import sys, numpy as np
from math import factorial
for f in sys.argv[1:]:
    z = np.load(f); keys = sorted(z.files, key=lambda s: float(s[1:]))
    print(f)
    for key in keys:
        B = z[key]; N = B.shape[-1]
        k1 = np.fft.fftfreq(N, 1.0 / N)
        KL, KZ = np.meshgrid(k1, k1, indexing="ij")
        Bh = np.fft.fft2(B) / N**2
        P = np.sum(np.abs(Bh)**2, axis=0)
        kk = np.sqrt(KL**2 + KZ**2); kb = np.rint(kk).astype(int)
        E = np.bincount(kb.ravel(), P.ravel())[: N // 2]
        ks = np.arange(len(E)); sel = (ks >= 4) & (ks <= int(0.6 * N / 2)) & (E > 1e-300)
        X = np.stack([np.ones(sel.sum()), -np.log(ks[sel]), -2 * ks[sel]], 1)
        c, *_ = np.linalg.lstsq(X, np.log(E[sel]), rcond=None)
        delta = c[2] * N / (2 * np.pi) / N  # k in units of 2pi/L, L=1 -> delta in box units /(2pi)
        rho = []
        for n in range(2, 6):
            dn = max(np.abs(np.real(np.fft.ifft2((1j * 2 * np.pi * K)**n * np.fft.fft2(B)))).max() for K in (KL, KZ))
            rho.append((dn / factorial(n))**(-1.0 / n))
        print(f"  A={float(key[1:]):.3f}  n={c[1]:5.2f}  delta*kmax={c[2]*2*N/2/1:7.2f}?  delta(1/k units)={c[2]:.4f}  "
              + " ".join(f"rho{n+2}={r:.4f}" for n, r in enumerate(rho)))
