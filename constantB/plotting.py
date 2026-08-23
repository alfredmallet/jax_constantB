"""Verification-cut and 3D-structure figures.

Host numpy; matplotlib is imported lazily inside each function (Agg backend)
so the package imports cleanly on headless/plot-free environments.
"""
import numpy as np

from .spectral import TWOPI


def plot_cuts(B, meta, out):
    """Three verification panels: (a) mean-field-aligned component and |B|
    along z through the deepest reversal; (b) Cartesian components along the
    same cut (spacecraft-style); (c) deflection map in the (x,y) plane through
    the reversal with the 90-degree contour."""
    B = np.asarray(B)
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    shape = B.shape[1:]
    z = np.linspace(0, TWOPI, shape[2], endpoint=False)
    x = np.linspace(0, TWOPI, shape[0], endpoint=False)
    Bbar = B.mean(axis=(1, 2, 3)); bb = Bbar / np.linalg.norm(Bbar)
    nrm = np.sqrt((B ** 2).sum(0))
    Bpar = (B * bb[:, None, None, None]).sum(0)
    defl = np.degrees(np.arccos(np.clip(Bpar / nrm, -1, 1)))
    ix, iy, iz = np.unravel_index(np.argmax(defl), defl.shape)
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
    ax[0].plot(z, Bpar[ix, iy, :], 'C0', lw=1.5, label=r'$\mathbf{B}\cdot\hat{\bar{\mathbf{B}}}$')
    ax[0].plot(z, nrm[ix, iy, :], 'k--', lw=1, label=r'$|\mathbf{B}|$')
    ax[0].axhline(0, color='gray', lw=0.6); ax[0].axvline(z[iz], color='r', lw=0.6, ls=':')
    ax[0].set_xlabel('z'); ax[0].legend(fontsize=8)
    ax[0].set_title('cut along z through deepest reversal', fontsize=9)
    for i, lab in enumerate(('$B_x$', '$B_y$', '$B_z$')):
        ax[1].plot(z, B[i, ix, iy, :], f'C{i}', lw=1.2, label=lab)
    ax[1].plot(z, nrm[ix, iy, :], 'k--', lw=1, label='$|B|$')
    ax[1].set_xlabel('z'); ax[1].legend(fontsize=8, ncol=2)
    ax[1].set_title('components along same cut', fontsize=9)
    im = ax[2].pcolormesh(x, x, defl[:, :, iz].T, shading='auto', cmap='RdBu_r')
    ax[2].contour(x, x, defl[:, :, iz].T, levels=[90], colors='k', linewidths=1.2)
    plt.colorbar(im, ax=ax[2], label='deflection (deg)')
    ax[2].plot(x[ix], x[iy], 'k+', ms=10); ax[2].set_xlabel('x'); ax[2].set_ylabel('y')
    ax[2].set_title(f'slice z={z[iz]:.2f}; black: 90$^\\circ$', fontsize=9)
    plt.tight_layout(); plt.savefig(out, dpi=200)
    print(f"wrote {out}")


def plot_3d(B, meta, out, iso_alpha=0.55, annotate_reversed=False):
    """3D visualisation: box faces coloured by the mean-field-aligned
    component (left) and, if scikit-image is available, the reversal
    isosurface B.Bbar-hat = 0 (right).

    iso_alpha: transparency of the isosurface (lower it for high-amplitude
    states where the reversal surface is space-filling rather than sparse).
    annotate_reversed: also compute/print the reversed-volume fraction and
    append it to the isosurface panel title.
    """
    B = np.asarray(B)
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import cm
    shape = B.shape[1:]; L = TWOPI
    x = np.linspace(0, L, shape[0], endpoint=False)
    z = np.linspace(0, L, shape[2], endpoint=False)
    Bbar = B.mean(axis=(1, 2, 3)); bb = Bbar / np.linalg.norm(Bbar)
    Bpar = (B * bb[:, None, None, None]).sum(0)
    iso_title = r'reversal surfaces $\mathbf{B}\cdot\hat{\bar{\mathbf{B}}}=0$'
    if annotate_reversed:
        vol_rev = 100.0 * (Bpar < 0).mean()
        print(f"reversed-volume fraction (Bpar<0): {vol_rev:.2f}%")
        iso_title += f'  ({vol_rev:.0f}% of volume reversed)'
    norm = plt.matplotlib.colors.TwoSlopeNorm(vmin=Bpar.min(), vcenter=0, vmax=Bpar.max())
    cmap = cm.RdBu_r
    edges = ([[0,L],[L,L],[L,L]],[[0,L],[0,0],[L,L]],[[0,0],[0,L],[L,L]],
             [[L,L],[0,L],[L,L]],[[L,L],[L,L],[0,L]],[[L,L],[0,0],[0,L]],
             [[0,0],[L,L],[0,L]],[[0,L],[L,L],[0,0]],[[L,L],[0,L],[0,0]])
    fig = plt.figure(figsize=(11, 4.6))
    ax = fig.add_subplot(121, projection='3d')
    Xf, Yf = np.meshgrid(x, x, indexing='ij'); Xz, Zz = np.meshgrid(x, z, indexing='ij')
    ax.plot_surface(Xf, Yf, np.full_like(Xf, L), facecolors=cmap(norm(Bpar[:, :, -1])),
                    shade=False, rstride=1, cstride=1)
    ax.plot_surface(Xz, np.full_like(Xz, L), Zz, facecolors=cmap(norm(Bpar[:, -1, :])),
                    shade=False, rstride=1, cstride=2)
    ax.plot_surface(np.full_like(Xz, L), Xz, Zz, facecolors=cmap(norm(Bpar[-1, :, :])),
                    shade=False, rstride=1, cstride=2)
    for e in edges: ax.plot(*e, 'k', lw=0.8, zorder=10)
    ax.set_xlim(0, L); ax.set_ylim(0, L); ax.set_zlim(0, L)
    ax.view_init(elev=28, azim=42); ax.set_axis_off()
    ax.set_title(r'$\mathbf{B}\cdot\hat{\bar{\mathbf{B}}}$ on box faces', fontsize=10)
    m = cm.ScalarMappable(norm=norm, cmap=cmap); m.set_array([])
    fig.colorbar(m, ax=ax, shrink=0.6, pad=0.02)
    ax2 = fig.add_subplot(122, projection='3d')
    try:
        from skimage import measure
        verts, faces, _, _ = measure.marching_cubes(
            Bpar, level=0.0, spacing=(L/shape[0], L/shape[1], L/shape[2]))
        ax2.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2],
                         color='crimson', alpha=iso_alpha, lw=0)
        ax2.set_title(iso_title, fontsize=10)
    except ImportError:
        idx = np.argwhere(Bpar < 0)
        ax2.scatter(idx[:, 0]*L/shape[0], idx[:, 1]*L/shape[1], idx[:, 2]*L/shape[2],
                    s=2, c='crimson', alpha=0.4)
        ax2.set_title('reversal region (scatter; install scikit-image for isosurface)',
                      fontsize=9)
    for e in edges: ax2.plot(*e, 'k', lw=0.8, zorder=10)
    ax2.set_xlim(0, L); ax2.set_ylim(0, L); ax2.set_zlim(0, L)
    ax2.view_init(elev=28, azim=42); ax2.set_axis_off()
    plt.tight_layout(); plt.savefig(out, dpi=200)
    print(f"wrote {out}")
