"""Local Z-ray-lemma load on sm22_grow2d snapshots.

For points on Z = {w=0}: trace the orthogonal trajectory (ray, direction n = t^perp) both ways
while |w| < eps, and accumulate  Lam = int sqrt(max(F,0)) ds,  F = d_t^2 v / v,  v=(1-w^2)^(-1/2).
Lemma 1 (constant F=g^2): smooth needs g*l < pi; Lam is its WKB generalization.
Also reports, per snapshot: g_Z = median |d_t w| on Z (flattening route), cosZ = median
|d_t w|/|grad w| on Z (1 = Z along rays, 0 = Z along field lines: rotation route), and
d_top = median distance from the top-0.5% gradient voxels to Z (in box units).
Usage: python sm22_zray.py snaps_*.npz [--eps 0.05]
"""
import sys, numpy as np
from scipy.ndimage import map_coordinates, distance_transform_edt

EPS = 0.05
args = [a for a in sys.argv[1:] if not a.startswith("--")]
if "--eps" in sys.argv:
    EPS = float(sys.argv[sys.argv.index("--eps") + 1])


def fields(B):
    N = B.shape[-1]
    k = 2 * np.pi * np.fft.fftfreq(N, 1.0 / N)
    KL, KZ = np.meshgrid(k, k, indexing="ij")
    D = lambda f, K: np.real(np.fft.ifft2(1j * K * np.fft.fft2(f)))
    Bm = np.sqrt((B**2).sum(0)); b = B / Bm
    w = b[2]; m = b[:2]; r = np.sqrt((m**2).sum(0)); t = m / r
    n = np.stack([-t[1], t[0]])
    v = 1 / r
    dt = lambda f: t[0] * D(f, KL) + t[1] * D(f, KZ)
    F = dt(dt(v)) / v
    wl, wz = D(w, KL), D(w, KZ)
    dtw = t[0] * wl + t[1] * wz
    gradw = np.sqrt(wl**2 + wz**2)
    G = sum(D(B[i], K)**2 for i in range(3) for K in (KL, KZ))
    return dict(w=w, n=n, F=F, dtw=dtw, gradw=gradw, grad=np.sqrt(G) / Bm, N=N)


def interp(f, x):  # x: (2,P) in grid units, periodic
    return map_coordinates(f, x, order=1, mode="grid-wrap")


def trace(fd, x0, sgn, h=0.25, smax=None):
    N = fd["N"]; smax = smax or N  # up to one box length
    x = x0.copy(); lam = np.zeros(x.shape[1]); ell = np.zeros(x.shape[1])
    alive = np.ones(x.shape[1], bool)
    for _ in range(int(smax / h)):
        if not alive.any():
            break
        d1 = np.stack([interp(fd["n"][0], x), interp(fd["n"][1], x)]) * sgn
        xm = x + 0.5 * h * d1
        d2 = np.stack([interp(fd["n"][0], xm), interp(fd["n"][1], xm)]) * sgn
        # keep orientation continuous (n is a line field only up to sign where t flips; t is smooth here)
        xn = x + h * d2
        wn = interp(fd["w"], xn)
        Fm = interp(fd["F"], xm)
        step_ok = alive & (np.abs(wn) < EPS)
        lam += np.where(step_ok, np.sqrt(np.maximum(Fm, 0)) * h / N, 0)
        ell += np.where(step_ok, h / N, 0)
        x = np.where(step_ok, xn, x); alive = step_ok
    return lam, ell


def analyse(B):
    fd = fields(B); w = fd["w"]; N = fd["N"]
    zc = (np.sign(w) != np.sign(np.roll(w, -1, 0))) | (np.sign(w) != np.sign(np.roll(w, -1, 1)))
    if zc.sum() == 0:
        return None
    idx = np.array(np.nonzero(zc), float)
    # one Newton projection onto w=0 along grad w
    wl = interp(np.gradient(w, axis=0), idx); wz = interp(np.gradient(w, axis=1), idx)
    g2 = wl**2 + wz**2 + 1e-30; wv = interp(w, idx)
    x0 = idx - wv * np.stack([wl, wz]) / g2
    l1, e1 = trace(fd, x0, +1); l2, e2 = trace(fd, x0, -1)
    lam, ell = l1 + l2, e1 + e2
    g = np.abs(interp(fd["dtw"], x0)); cos = g / (interp(fd["gradw"], x0) + 1e-30)
    top = fd["grad"] >= np.quantile(fd["grad"], 0.995)
    dist = distance_transform_edt(~zc) / N  # non-periodic approx, fine for medians
    return dict(nZ=len(lam), lam_max=lam.max(), lam_99=np.quantile(lam, 0.99),
                ell_99=np.quantile(ell, 0.99), g_med=np.median(g), g_max=g.max(),
                cos_med=np.median(cos), d_top=np.median(dist[top]))


if __name__ == "__main__":
    for f in args:
        z = np.load(f); keys = sorted(z.files, key=lambda s: float(s[1:]))
        print(f)
        for key in keys:
            A = float(key[1:])
            if A < 0.2:
                continue
            r = analyse(z[key])
            if r is None:
                print(f"  A={A:.3f}  no Z"); continue
            print(f"  A={A:.3f} nZ={r['nZ']:5d} Lam_max={r['lam_max']:.2f} Lam99={r['lam_99']:.2f} "
                  f"ell99={r['ell_99']:.3f} g_med={r['g_med']:.2f} g_max={r['g_max']:.1f} "
                  f"cosZ={r['cos_med']:.2f} d_top={r['d_top']:.3f}", flush=True)
