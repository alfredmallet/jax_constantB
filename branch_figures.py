#!/usr/bin/env python3
"""
================================================================================
branch_figures.py

Figure generation for the constant-|B| Alfvenic-states paper (constantB_results
.tex), covering the Sobolev-weighted ("smooth") continuation branch at its
current endpoint eps=5.06 (grid 72^2x144) and its comparison against the
plain (L2 minimum-norm) branch.

Per project policy (CLAUDE.md) new code lives in jax_constantB/, but this is
plain plotting code (numpy/matplotlib), so it simply imports the frozen
reference toolkit `constantB_tools.py` from the project root -- NOT the JAX
port that also lives in this directory -- by putting the project root ahead
of this script's own directory on sys.path.

Produces, into the project root (/Users/alfy/aw_papers/):
  fig_smooth_cuts.png     -- verification cuts through the weighted-branch
                             final state (reuses constantB_tools.plot_cuts
                             unmodified: it works directly on this state).
  fig_smooth_3d.png       -- 3D structure of the same state: box faces +
                             reversal isosurface (imitates plot_3d, with a
                             reduced isosurface alpha and an explicit
                             reversed-volume annotation, because at this
                             amplitude ~28% of the volume is reversed and the
                             surface is much more extensive than in the
                             lower-amplitude fig_switchback_3d.png reference).
  fig_branch_compare.png  -- 3-panel comparison built from quest.csv (plain
                             branch, collocation, to eps=4.3) and
                             quest_smooth.csv (weighted branch, dealias, to
                             eps=5.06): maxgrad vs eps, maxgrad vs maxdefl
                             (the matched-amplitude headline panel), and the
                             weighted branch's honest tail (gal_tail_rms) vs
                             eps with its two grid-refinement events marked.

Run from anywhere (paths are resolved relative to this file):
    python3 jax_constantB/branch_figures.py
================================================================================
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm

# ---------------------------------------------------------------------------
# Path setup: import the NUMPY REFERENCE constantB_tools.py from the project
# root, not the JAX port that also lives alongside this script in
# jax_constantB/.  Insert the root ahead of this file's own directory.
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))          # .../aw_papers/jax_constantB
ROOT = os.path.dirname(HERE)                                # .../aw_papers
sys.path.insert(0, ROOT)
import constantB_tools as ct                                # noqa: E402  (reference impl)

STATE_FILE = os.path.join(ROOT, 'quest_smooth.npz')
CSV_SMOOTH = os.path.join(ROOT, 'quest_smooth.csv')
CSV_PLAIN = os.path.join(ROOT, 'quest.csv')

OUT_CUTS = os.path.join(ROOT, 'fig_smooth_cuts.png')
OUT_3D = os.path.join(ROOT, 'fig_smooth_3d.png')
OUT_COMPARE = os.path.join(ROOT, 'fig_branch_compare.png')

# Grid-refinement events on the weighted-branch run (from the task brief /
# visible jumps in quest_smooth.csv's "grid" column):
#   eps=0.5792: 32^2x64  -> 48^2x96
#   eps=4.4192: 48^2x96  -> 72^2x144
REFINE_EPS = (0.5792, 4.4192)


# =============================================================================
# Figure 1: verification cuts (direct reuse of constantB_tools.plot_cuts)
# =============================================================================

def make_fig_cuts():
    B, eps, meta = ct.load_state(STATE_FILE)
    print(f"[fig_smooth_cuts] loaded {STATE_FILE}: grid {B.shape[1:]}, eps={eps:.4f}")
    # plot_cuts works directly on this state -- no modification needed. It
    # picks the cut through the deepest reversal (max deflection point) itself.
    ct.plot_cuts(B, meta, OUT_CUTS)
    return B, eps, meta


# =============================================================================
# Figure 2: 3D structure (imitates constantB_tools.plot_3d)
#
# plot_3d's isosurface alpha (0.55) and title were tuned for the reference
# switchback state, where only ~3.5% of the volume is reversed and the
# isosurface is a handful of separated blobs (see fig_switchback_3d.png).
# At eps=5.06 on the weighted branch ~28% of the volume is reversed and the
# surface is one connected, space-filling structure -- at alpha=0.55 it
# renders as a dense red mass.  We lower alpha to 0.32 and annotate the
# reversed-volume fraction explicitly in the title so the figure is honest
# about what's being shown rather than pretending it's as sparse as the
# switchback case.
# =============================================================================

def make_fig_3d(B, meta, alpha=0.32):
    shape = B.shape[1:]
    L = ct.TWOPI
    x = np.linspace(0, L, shape[0], endpoint=False)
    z = np.linspace(0, L, shape[2], endpoint=False)
    Bbar = B.mean(axis=(1, 2, 3))
    bb = Bbar / np.linalg.norm(Bbar)
    Bpar = (B * bb[:, None, None, None]).sum(0)
    vol_rev = 100.0 * (Bpar < 0).mean()
    print(f"[fig_smooth_3d] reversed-volume fraction (Bpar<0): {vol_rev:.2f}%")

    norm = plt.matplotlib.colors.TwoSlopeNorm(vmin=Bpar.min(), vcenter=0, vmax=Bpar.max())
    cmap = cm.RdBu_r
    edges = ([[0, L], [L, L], [L, L]], [[0, L], [0, 0], [L, L]], [[0, 0], [0, L], [L, L]],
             [[L, L], [0, L], [L, L]], [[L, L], [L, L], [0, L]], [[L, L], [0, 0], [0, L]],
             [[0, 0], [L, L], [0, L]], [[0, L], [L, L], [0, 0]], [[L, L], [0, L], [0, 0]])

    fig = plt.figure(figsize=(11, 4.6))

    ax = fig.add_subplot(121, projection='3d')
    Xf, Yf = np.meshgrid(x, x, indexing='ij')
    Xz, Zz = np.meshgrid(x, z, indexing='ij')
    ax.plot_surface(Xf, Yf, np.full_like(Xf, L), facecolors=cmap(norm(Bpar[:, :, -1])),
                     shade=False, rstride=1, cstride=1)
    ax.plot_surface(Xz, np.full_like(Xz, L), Zz, facecolors=cmap(norm(Bpar[:, -1, :])),
                     shade=False, rstride=1, cstride=2)
    ax.plot_surface(np.full_like(Xz, L), Xz, Zz, facecolors=cmap(norm(Bpar[-1, :, :])),
                     shade=False, rstride=1, cstride=2)
    for e in edges:
        ax.plot(*e, 'k', lw=0.8, zorder=10)
    ax.set_xlim(0, L); ax.set_ylim(0, L); ax.set_zlim(0, L)
    ax.view_init(elev=28, azim=42); ax.set_axis_off()
    ax.set_title(r'$\mathbf{B}\cdot\hat{\bar{\mathbf{B}}}$ on box faces', fontsize=10)
    m = cm.ScalarMappable(norm=norm, cmap=cmap); m.set_array([])
    fig.colorbar(m, ax=ax, shrink=0.6, pad=0.02)

    ax2 = fig.add_subplot(122, projection='3d')
    try:
        from skimage import measure
        verts, faces, _, _ = measure.marching_cubes(
            Bpar, level=0.0, spacing=(L / shape[0], L / shape[1], L / shape[2]))
        ax2.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2],
                          color='crimson', alpha=alpha, lw=0)
        ax2.set_title(r'reversal surfaces $\mathbf{B}\cdot\hat{\bar{\mathbf{B}}}=0$'
                       f'  ({vol_rev:.0f}% of volume reversed)', fontsize=10)
    except ImportError:
        idx = np.argwhere(Bpar < 0)
        ax2.scatter(idx[:, 0] * L / shape[0], idx[:, 1] * L / shape[1], idx[:, 2] * L / shape[2],
                    s=2, c='crimson', alpha=0.4)
        ax2.set_title('reversal region (scatter; install scikit-image for isosurface)',
                       fontsize=9)
    for e in edges:
        ax2.plot(*e, 'k', lw=0.8, zorder=10)
    ax2.set_xlim(0, L); ax2.set_ylim(0, L); ax2.set_zlim(0, L)
    ax2.view_init(elev=28, azim=42); ax2.set_axis_off()

    plt.tight_layout()
    plt.savefig(OUT_3D, dpi=200)
    plt.close(fig)
    print(f"wrote {OUT_3D}  "
          f"[note: at ~28% reversed volume the isosurface is a single connected, "
          f"space-filling structure, not isolated blobs as in the lower-amplitude "
          f"fig_switchback_3d.png reference; alpha lowered to {alpha} for readability]")


# =============================================================================
# Figure 3: branch comparison (3 panels from the CSV histories)
# =============================================================================

def make_fig_compare():
    import pandas as pd
    # BOTH branches are now Galerkin (dealias) quest runs with identical schemas:
    #   quest.csv        rough branch (smooth=0), ended SHARPENING at eps=2.327
    #   quest_smooth.csv weighted branch (smooth=1), reached eps=5.06
    dfp = pd.read_csv(CSV_PLAIN)
    dfs = pd.read_csv(CSV_SMOOTH)

    def refine_eps(df):
        """eps values where the 'grid' column changes (refinement events)."""
        g = df['grid'].astype(str)
        return df['eps'][g.ne(g.shift()) & g.shift().notna()].tolist()

    re_p, re_s = refine_eps(dfp), refine_eps(dfs)

    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))

    # (a) maxgrad vs eps, refinements marked per branch
    ax[0].plot(dfp['eps'], dfp['maxgrad'], 'C0-', lw=1.3, label='rough (L2)')
    ax[0].plot(dfs['eps'], dfs['maxgrad'], 'C3-', lw=1.3, label='Sobolev-weighted')
    for e in re_p: ax[0].axvline(e, color='C0', lw=1, ls=':', alpha=0.6)
    for e in re_s: ax[0].axvline(e, color='C3', lw=1, ls=':', alpha=0.6)
    ax[0].set_yscale('log')
    ax[0].set_xlabel(r'$\varepsilon$'); ax[0].set_ylabel('max$|\\nabla B|_F$')
    ax[0].set_title('gradient vs amplitude (dotted: refinements)', fontsize=9)
    ax[0].legend(fontsize=8)

    # (b) maxgrad vs maxdefl -- matched-deflection headline panel
    ax[1].plot(dfp['maxdefl'], dfp['maxgrad'], 'C0-', lw=1.3, label='rough (L2)')
    ax[1].plot(dfs['maxdefl'], dfs['maxgrad'], 'C3-', lw=1.3, label='Sobolev-weighted')
    ax[1].set_yscale('log')
    ax[1].set_xlabel('max deflection (deg)'); ax[1].set_ylabel('max$|\\nabla B|_F$')
    ax[1].set_title('gradient vs deflection (matched)', fontsize=9)
    ax[1].legend(fontsize=8)

    # (c) honest unresolved tail vs eps, BOTH branches: the cadence panel.
    # The rough branch's tail regrows ~4-9x faster in eps on matched grids,
    # forcing refinements at eps=0.41/0.73/1.55 vs 0.58/4.42 (weighted).
    ax[2].plot(dfp['eps'], dfp['gal_tail_rms'], 'C0-', lw=1.3, label='rough (L2)')
    ax[2].plot(dfs['eps'], dfs['gal_tail_rms'], 'C3-', lw=1.3, label='Sobolev-weighted')
    for e in re_p: ax[2].axvline(e, color='C0', lw=1, ls=':', alpha=0.6)
    for e in re_s: ax[2].axvline(e, color='C3', lw=1, ls=':', alpha=0.6)
    ax[2].set_yscale('log')
    ax[2].set_xlabel(r'$\varepsilon$'); ax[2].set_ylabel('gal_tail_rms')
    ax[2].set_title('honest unresolved tail (refinement cadence)', fontsize=9)
    ax[2].legend(fontsize=8)

    for a in ax:
        a.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_COMPARE, dpi=200)
    plt.close(fig)
    print(f"wrote {OUT_COMPARE}")

    # Headline matched-deflection numbers at the largest common deflection.
    target = min(dfp['maxdefl'].max(), dfs['maxdefl'].max())
    j_p = (dfp['maxdefl'] - target).abs().idxmin()
    j_s = (dfs['maxdefl'] - target).abs().idxmin()
    gp, gs = dfp.loc[j_p, 'maxgrad'], dfs.loc[j_s, 'maxgrad']
    print(f"[fig_branch_compare] at matched maxdefl~{target:.1f}deg: "
          f"rough maxgrad={gp:.2f} (eps={dfp.loc[j_p,'eps']:.3f}), "
          f"weighted maxgrad={gs:.2f} (eps={dfs.loc[j_s,'eps']:.3f}) "
          f"-> ratio {gp/gs:.2f}x")


# =============================================================================
# Verification printout
# =============================================================================

def verify_state_numbers(B, eps, meta):
    """Cross-check the headline numbers quoted in the task brief against the
    loaded state, using the same definitions as constantB_tools.diagnose."""
    defl = ct.diagnose(B, eps, meta, full=True)
    maxdefl = float(defl.max())
    vol_rev = 100.0 * (defl > 90).mean()
    S = ct.Solver(B.shape[1:])
    g2 = sum(S.dif(B[i], j) ** 2 for i in range(3) for j in range(3))
    maxgrad = float(np.sqrt(g2.max()))
    print("\n[verify] headline numbers for quest_smooth.npz (eps=%.4f):" % eps)
    print(f"  max deflection : {maxdefl:.1f} deg   (expected ~140.6 deg)")
    print(f"  reversed volume: {vol_rev:.1f} %      (expected ~28%)")
    print(f"  max|gradB|_F   : {maxgrad:.2f}        (expected ~7.49)")
    ok = (abs(maxdefl - 140.6) < 1.0 and abs(vol_rev - 28) < 2.0
          and abs(maxgrad - 7.49) < 0.1)
    print("  -> " + ("MATCH" if ok else "DISAGREES WITH BRIEF -- see numbers above"))


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--which', choices=['cuts', '3d', 'compare', 'all'], default='all',
                    help="which figure(s) to make (split runs to stay under a "
                         "shell timeout: the 3D marching-cubes render is the "
                         "slow one, ~20s at this grid)")
    args = p.parse_args()

    if args.which in ('cuts', '3d', 'all'):
        B, eps, meta = ct.load_state(STATE_FILE)
    if args.which in ('cuts', 'all'):
        ct.plot_cuts(B, meta, OUT_CUTS)
        verify_state_numbers(B, eps, meta)
    if args.which in ('3d', 'all'):
        make_fig_3d(B, meta)
    if args.which in ('compare', 'all'):
        make_fig_compare()
