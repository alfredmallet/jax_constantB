"""Per-seed convergence table parsed from sm22_grow2d logs (series npz only exist for finished runs).
Reports Z-birth, takeoff (maxgrad/A > 1.3x early value), maxgrad at matched A across N, and
chi = log2(maxgrad_2N / maxgrad_N)."""
import glob, re, numpy as np
from collections import defaultdict
runs = defaultdict(dict)
pat = re.compile(r"\[N(\d+)_s(\d+)_th30_k1\] A=([\d.]+) .*maxgrad=([\d.]+) .*Z=([\d.]+)")
for f in glob.glob("../sm22_2d/*.log"):
    for line in open(f):
        m = pat.search(line)
        if m:
            N, s = int(m[1]), int(m[2]); runs[(N, s)][float(m[3])] = (float(m[4]), float(m[5]))
for f in glob.glob("../sm22_2d/series_N*_s*_th30_k1.npz"):
    N, s_ = map(int, re.findall(r"N(\d+)_s(\d+)", f)[0]); z = np.load(f)
    for a, g, zf in zip(z["A"], z["maxgrad"], z["zfrac"]): runs[(N, s_)].setdefault(float(a), (float(g), float(zf)))
def series(N, s):
    d = runs[(N, s)]; A = np.array(sorted(d)); return A, np.array([d[a][0] for a in A]), np.array([d[a][1] for a in A])
for s in sorted({k[1] for k in runs}):
    Ns = sorted(N for (N, ss) in runs if ss == s)
    A, g, z = series(Ns[0], s)
    zb = A[np.argmax(z > 0)] if (z > 0).any() else np.nan
    lin = g / A; base = np.median(lin[A < 0.2]); tk = A[np.argmax(lin > 1.3 * base)] if (lin > 1.3 * base).any() else np.nan
    row = f"seed {s}: Zbirth {zb:.3f} takeoff {tk:.3f}"
    for Am in (0.6, 0.8, 1.0, 1.4, 2.0, 3.0):
        v = {}
        for N in Ns:
            a, gg, _ = series(N, s)
            if a.max() >= Am: v[N] = np.interp(np.log(Am), np.log(a), gg)
        if v:
            row += f" | A={Am}: " + " ".join(f"{N}:{x:.1f}" for N, x in v.items())
            ks = sorted(v)
            for n1, n2 in zip(ks, ks[1:]): row += f" chi{n1}-{n2}={np.log2(v[n2]/v[n1]):.2f}"
    print(row)
