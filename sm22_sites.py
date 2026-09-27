"""Per-site classification for an sm22 seed scan (two resolutions N and 2N of the same seeds).

Sharpening SITES (ground truth): pointwise resolution exponent at amplitude A1 (default 1.0),
    chi(x) = log2( maxpool G_2N / maxpool G_N ),  G = |grad B| / |B|,
on the N grid after 2x2 (then 3x3 periodic) max-pooling; a site is a periodic connected
component of {chi > CHI_S and G_2N > 3 median(G_2N)} with at least 2 cells.
Reversal SITES (candidates): periodic connected components of the reversed set
{w * sign(wbar) < 0} at amplitude A2 (default 0.8) on the N grid. Each gets
    lam  = max over its boundary points of the Z-ray load Lambda (sm22_zray.analyse points),
    sharp = whether any sharpening site lies within DMATCH (box units) of the patch.
Writes DIR_N/sites.csv (seed, patch id, area, lam, sharp) and DIR_N/boxes.csv.
Usage: python sm22_sites.py DIR_N DIR_2N N kcut [A1 A2]
"""
import sys, glob, re, os, numpy as np
from scipy import ndimage
from sm22_zray import fields, trace, interp

CHI_S = 0.5
DMATCH = float(os.environ.get("SITES_DMATCH", "0.05"))


def nearest(z, A):
    k = min(z.files, key=lambda k: abs(float(k[1:]) - A)); return z[k], float(k[1:])


def gradmag(B):
    N = B.shape[-1]; k = 2 * np.pi * np.fft.fftfreq(N, 1.0 / N)
    KL, KZ = np.meshgrid(k, k, indexing="ij")
    Bh = np.fft.fft2(B)
    G = sum(np.real(np.fft.ifft2(1j * K * Bh[i]))**2 for i in range(3) for K in (KL, KZ))
    return np.sqrt(G) / np.sqrt((B**2).sum(0))


def pool(G, f):
    n = G.shape[0] // f; return G[:n * f, :n * f].reshape(n, f, n, f).max(axis=(1, 3))


def wrap_label(mask):
    lab, n = ndimage.label(mask)
    # merge labels across periodic edges
    parent = list(range(n + 1))
    def find(a):
        while parent[a] != a: parent[a] = parent[parent[a]]; a = parent[a]
        return a
    for a, b in list(zip(lab[0], lab[-1])) + list(zip(lab[:, 0], lab[:, -1])):
        if a and b: parent[find(a)] = find(b)
    roots = np.array([find(i) for i in range(n + 1)])
    uniq = {r: j for j, r in enumerate(sorted(set(roots[1:])), 1)}
    return np.vectorize(lambda v: uniq[roots[v]] if v else 0)(lab), len(uniq)


def periodic_mindist(maskA, maskB):
    """min periodic distance (grid cells) from each labelled region of maskA to any cell of maskB."""
    if not maskB.any():
        return np.inf
    t = np.tile(~maskB, (3, 3)); d = ndimage.distance_transform_edt(t)
    n = maskB.shape[0]; return d[n:2 * n, n:2 * n]


def main():
    d1, d2, N, kc = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    A1 = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0
    A2 = float(sys.argv[6]) if len(sys.argv) > 6 else 0.8
    site_rows, box_rows = [], []
    for f in sorted(glob.glob(f"{d1}/snaps_N{N}_s*_k{kc}*.npz"), key=lambda p: int(re.findall(r"_s(\d+)_", p)[0])):
        s = int(re.findall(r"_s(\d+)_", f)[0])
        f2 = glob.glob(f"{d2}/snaps_N{2*N}_s{s}_*k{kc}*.npz")
        if not f2: continue
        z1, z2 = np.load(f), np.load(f2[0])
        B1, a1 = nearest(z1, A1); B2, a2 = nearest(z2, A1)
        G1 = pool(gradmag(B1), 1); G2 = pool(gradmag(B2), 2)
        G1p = ndimage.maximum_filter(G1, 3, mode="wrap"); G2p = ndimage.maximum_filter(G2, 3, mode="wrap")
        chi = np.log2(G2p / G1p)
        sharp_mask = (chi > CHI_S) & (G2p > 3 * np.median(G2p))
        slab, ns = wrap_label(sharp_mask)
        sizes = ndimage.sum(np.ones_like(slab), slab, range(1, ns + 1)) if ns else []
        keep = [i + 1 for i, a in enumerate(sizes) if a >= 2]
        sharp_mask = np.isin(slab, keep)
        # reversal patches at A2
        B, a_used = nearest(z1, A2)
        fd = fields(B); w = fd["w"]
        wbar_sign = np.sign(np.mean(B[2]))
        rev = w * wbar_sign < 0
        plab, npatch = wrap_label(rev)
        dist = periodic_mindist(rev, sharp_mask) / N if sharp_mask.any() else np.full(rev.shape, np.inf)
        # Lambda on boundary points, peak over the save points A_Z <= A <= A2, each point assigned
        # to the A2-patch it lies in (or the nearest one: reversed regions grow roughly monotonically)
        _, near = ndimage.distance_transform_edt(~rev, return_indices=True) if rev.any() else (None, None)
        lam_patch = np.zeros(npatch + 1)
        for key in sorted(z1.files, key=lambda k: float(k[1:])):
            A = float(key[1:])
            if A < 0.2 or A > A2 + 1e-6 or not rev.any():
                continue
            fdA = fields(z1[key]); wA = fdA["w"]
            revA = wA * wbar_sign < 0
            if not revA.any():
                continue
            bnd = revA & ~ndimage.binary_erosion(revA, iterations=1, border_value=0)
            idx = np.array(np.nonzero(bnd), float)
            l1, _ = trace(fdA, idx, +1); l2, _ = trace(fdA, idx, -1); lp = l1 + l2
            ii, jj = np.nonzero(bnd)
            lab_pts = plab[near[0][ii, jj], near[1][ii, jj]]
            np.maximum.at(lam_patch, lab_pts, lp)
        # one-to-one attribution: each sharp component -> the single patch nearest to it
        owned = np.zeros(npatch + 1)
        if rev.any() and keep:
            t = np.tile(~rev, (3, 3)); dt_, (ii, jj) = ndimage.distance_transform_edt(t, return_indices=True)
            dt_ = dt_[N:2 * N, N:2 * N]; ii = ii[N:2 * N, N:2 * N] % N; jj = jj[N:2 * N, N:2 * N] % N
            for c in keep:
                cells = np.nonzero(slab == c); k0 = np.argmin(dt_[cells])
                if dt_[cells][k0] / N <= 2 * DMATCH:
                    owned[plab[ii[cells][k0], jj[cells][k0]]] += 1
        for p in range(1, npatch + 1):
            m = plab == p
            lam = lam_patch[p]
            dmin = np.min(np.where(m, dist, np.inf)) if np.isfinite(dist).any() else np.inf
            site_rows.append((s, p, m.sum() / N**2, lam, float(dmin <= DMATCH), owned[p]))
        # sharp sites not near any reversed patch (periodic, same size filter)
        drev = periodic_mindist(sharp_mask, rev) / N if rev.any() else np.full(rev.shape, np.inf)
        norphan = sum(1 for i in keep if np.min(np.where(slab == i, drev, np.inf)) > DMATCH)
        box_rows.append((s, npatch, len(keep), norphan, a1, a2, a_used))
        print(f"seed {s}: patches {npatch}, sharp sites {len(keep)}, orphan sharp {box_rows[-1][3]:.0f}, "
              f"patch lam/sharp: " + " ".join(f"{r[3]:.2f}/{int(r[4])}" for r in site_rows if r[0] == s), flush=True)
    np.savetxt(os.path.join(d1, f"sites_k{kc}_A{A2:g}.csv"), np.array(site_rows), delimiter=",",
               header="seed,patch,area,lam,sharp,owned", comments="")
    np.savetxt(os.path.join(d1, f"boxes_k{kc}_A{A2:g}.csv"), np.array(box_rows), delimiter=",",
               header="seed,npatch,nsharp,norphan,A1_N,A1_2N,A2", comments="")


if __name__ == "__main__":
    main()
