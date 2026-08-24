#!/usr/bin/env python3
"""
poincare_map.py -- Poincare puncture map of the field lines of a saved
constant-|B| state: are there invariant surfaces, or is the line flow chaotic?

WHY.  `fieldline_topology.py` already established that lines are OPEN (they
drift along Bbar at rate |Bbar|, to 3 s.f.).  Open is a statement about the
NET progress; it says nothing about the transverse structure.  A unit-speed
divergence-free flow makes the z = z0 return map area-preserving, so the
transverse structure is the standard KAM alternative: punctures of one line
lie on a closed curve (an invariant torus survives), or they fill a patch
(chaos).  Which one holds inside the switchback-like deflected region is the
question this driver answers, and it is a genuinely new observable for the
paper -- the rough/smooth branch dichotomy is a statement about spectra, and
this asks whether it shows up in the LINE GEOMETRY as well.

    python3 poincare_map.py --state kag2/state.npz --lines 512 --transits 300
    python3 poincare_map.py --state s.npz --seed-mode reversed --lines 256
    python3 poincare_map.py --state s.npz --seed-mode uniform --refine 4

Seeds for `ring` and `uniform` start ON the section plane z = z0 (the
Poincare convention: every line then has the same number of punctures);
`reversed` seeds inside the reversed region B.Bbar_hat < 0, wherever it is,
so its lines may take up to one transit to reach the plane.

Reads only; writes the figure and a companion .npz of the punctures.  Memory
is bounded by the chunked tracer (constantB/poincare.py), not by the number
of transits.
"""
import argparse
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from constantB import load_state                     # enables x64 first
from constantB.spectral import TWOPI
from constantB.poincare import (trace_punctures, pad_punctures, rotation_number,
                                nn_spread, transverse_dispersion)


# ---------------------------------------------------------------------------
# setup: geometry of the state, section plane, seeds
# ---------------------------------------------------------------------------

def parse_plane(s):
    """--z0 accepts 'pi', 'pi/2', '2*pi/3' or a plain float."""
    return float(eval(s, {"__builtins__": {}}, {"pi": np.pi, "np": np}))


def deflection(B):
    """(Bbar, Bbar_hat, cos_angle field) -- carrier-free: the reference
    direction is the VOLUME MEAN of B, nothing else."""
    Bbar = np.asarray(B).mean(axis=(1, 2, 3))
    nrm = np.linalg.norm(Bbar)
    if nrm <= 0:
        raise ValueError("state has zero mean field: no deflection reference")
    bh = Bbar / nrm
    mag = np.sqrt((B ** 2).sum(0))
    cos = (B * bh[:, None, None, None]).sum(0) / np.maximum(mag, 1e-300)
    return Bbar, bh, cos


def plane_slice(f, z0, Nz):
    """The x-y slice of a scalar field at the grid plane nearest z = z0."""
    return f[:, :, int(round(z0 / (TWOPI / Nz))) % Nz]


def structure_center(cos, shape, z0):
    """(x, y) of the strongest deflection in the slab nearest z = z0 -- the
    natural centre for the zoom panel and for rotation numbers."""
    Nx, Ny, Nz = shape
    i, j = np.unravel_index(np.argmin(plane_slice(cos, z0, Nz)), (Nx, Ny))
    return np.array([i * TWOPI / Nx, j * TWOPI / Ny])


def structure_radius(cos, shape, z0, center):
    """Radius about `center` of the deflected patch in the section plane:
    the reversed region if there is one, else where the deflection exceeds
    half its maximum.  Field-based, so the zoom window does not depend on the
    punctures it is meant to display."""
    Nx, Ny, Nz = shape
    c = plane_slice(cos, z0, Nz)
    thr = 0.0 if (c < 0).any() else float(np.cos(0.5 * np.arccos(c.min())))
    i, j = np.nonzero(c < thr)
    if i.size == 0:
        return np.pi
    dx = (i * TWOPI / Nx - center[0] + np.pi) % TWOPI - np.pi
    dy = (j * TWOPI / Ny - center[1] + np.pi) % TWOPI - np.pi
    return float(np.hypot(dx, dy).max())


def make_seeds(mode, n, z0, center, cos, shape, rng, rmax, nang):
    """(n, 3) seed positions for the three seeding modes."""
    if mode == "uniform":
        # uniform in the VOLUME, not in the plane.  A plane-uniform sample is
        # weighted wrongly for the return map -- the invariant measure on
        # z = z0 carries the flux factor b_z -- and it biases the drift
        # check low (measured: 0.799 instead of 0.822 on the s=0 state).
        # This is the mode to use when the drift theorem is the question.
        return rng.uniform(0.0, TWOPI, size=(n, 3))
    if mode == "ring":
        # concentric rings about `center` in the section plane: radius-major,
        # so the per-line colour of the figure runs outward from the centre.
        nr = max(1, n // nang)
        r = np.repeat(np.linspace(rmax / nr, rmax, nr), nang)[:n]
        th = np.tile(np.arange(nang) * TWOPI / nang, nr)[:n]
        p = np.empty((len(r), 3))
        p[:, 0] = np.mod(center[0] + r * np.cos(th), TWOPI)
        p[:, 1] = np.mod(center[1] + r * np.sin(th), TWOPI)
        p[:, 2] = z0
        return p
    # reversed: cells where the field points back against the mean field
    idx = np.argwhere(cos < 0.0)
    if idx.shape[0] == 0:
        raise ValueError("no reversed region in this state (min cos > 0): "
                         "use --seed-mode ring or uniform")
    cell = TWOPI / np.array(shape, float)
    pick = idx[rng.integers(0, idx.shape[0], size=n)].astype(float)
    return (pick + rng.uniform(0.0, 1.0, size=(n, 3))) * cell[None, :]


# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------

def plot_map(xy_list, center, out, title, zoom, bz=None, dpi=170):
    """Two equal-aspect panels -- the full section plane and a zoom on the
    structure -- with one colour per line (turbo, in seed order).

    The B_z = 0 contour of the section plane is overlaid because it explains
    the most striking feature of these maps: an upward crossing needs
    b_z > 0, so the interior of that contour is FORBIDDEN to punctures and
    shows up as a clean hole, not as an island.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    X = np.concatenate([a[:, 0] for a in xy_list]) if xy_list else np.zeros(0)
    Y = np.concatenate([a[:, 1] for a in xy_list]) if xy_list else np.zeros(0)
    C = np.concatenate([np.full(len(a), m) for m, a in enumerate(xy_list)])

    fig, ax = plt.subplots(1, 2, figsize=(11, 5.4))
    for a, sz in zip(ax, (0.6, 4.0)):
        a.scatter(X, Y, c=C, cmap='turbo', s=sz, lw=0, alpha=0.85)
        if bz is not None and (bz < 0).any():
            a.contour(np.arange(bz.shape[0]) * TWOPI / bz.shape[0],
                      np.arange(bz.shape[1]) * TWOPI / bz.shape[1],
                      bz.T, levels=[0.0], colors='k', linewidths=1.0)
        a.set_aspect('equal')
        a.set_xlabel('$x$')
        a.tick_params(labelsize=8)
    ax[0].set_xlim(0, TWOPI); ax[0].set_ylim(0, TWOPI)
    ax[0].set_ylabel('$y$')
    ax[0].set_title(title, fontsize=9)
    ax[0].plot(center[0], center[1], 'k+', ms=9, mew=1.4)
    ax[1].set_xlim(center[0] - zoom, center[0] + zoom)
    ax[1].set_ylim(center[1] - zoom, center[1] + zoom)
    ax[1].set_title(f'zoom on the deflection centre '
                    f'({center[0]:.2f}, {center[1]:.2f}), '
                    f'half-width {zoom:.2f}; black: $B_z=0$', fontsize=9)
    plt.tight_layout(); plt.savefig(out, dpi=dpi); plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def run(args):
    if args.lines < 1:
        raise ValueError("--lines must be at least 1")
    B, eps, _ = load_state(args.state)
    B = np.asarray(B, dtype=float)
    shape = B.shape[1:]
    z0 = parse_plane(args.z0)
    Bbar, bh, cos = deflection(B)
    nBbar = float(np.linalg.norm(Bbar))
    center = (np.array(args.center, float) if args.center
              else structure_center(cos, shape, z0))
    rng = np.random.default_rng(args.seed_rng)
    seeds = make_seeds(args.seed_mode, args.lines, z0, center, cos, shape,
                       rng, args.ring_rmax, args.ring_angles)
    M = seeds.shape[0]

    # <dz/ds> = <b_z> = Bbar_z EXACTLY for a |B| = 1 field, so a transit
    # (2 pi in z) is 2 pi / |Bbar_z| of arc length.  Overshoot by 1.3x and let
    # punctures() report what is actually there.
    vz = max(abs(float(Bbar[2])), 0.05)
    per_transit = TWOPI / (vz * args.h)
    n_steps = int(np.ceil(1.3 * args.transits * per_transit))
    chunk = args.chunk or int(np.clip(8e7 / (M * 24), 1024, 16384))

    print(f"state {args.state}: {shape[0]}x{shape[1]}x{shape[2]}, "
          f"eps={eps:.4f}")
    print(f"  Bbar = [{Bbar[0]:+.6f} {Bbar[1]:+.6f} {Bbar[2]:+.6f}], "
          f"|Bbar| = {nBbar:.6f}")
    print(f"  max deflection {np.degrees(np.arccos(cos.min())):.1f} deg, "
          f"reversed volume fraction {float((cos < 0).mean()):.4f}")
    print(f"  section z0 = {z0:.4f}, centre = ({center[0]:.3f}, "
          f"{center[1]:.3f}), seeds = {M} ({args.seed_mode})")
    print(f"  h = {args.h}, refine = {args.refine} "
          f"(dx/refine = {TWOPI / shape[0] / args.refine:.4f}), "
          f"{n_steps} steps in chunks of {chunk}")

    t0 = time.time()
    xy, idx, endpos, n_steps = trace_punctures(
        B, seeds, h=args.h, n_steps=n_steps, refine=args.refine, z0=z0,
        chunk=chunk,
        progress=lambda d, n: print(f"    {d}/{n} steps "
                                    f"({time.time() - t0:.1f}s)", flush=True)
        if args.verbose else None)
    S = n_steps * args.h
    print(f"  traced {M} lines x {S:.1f} arc length in "
          f"{time.time() - t0:.1f}s")

    # --- the drift theorem: <dr/ds> = <b> = Bbar (exact for |B| = 1) -------
    disp = endpos - seeds
    dz_rate = float(disp[:, 2].mean() / S)
    dpar_rate = float((disp @ bh).mean() / S)
    npunct = np.array([len(a) for a in xy])
    stats = "n/a" if npunct.max() == 0 else (
        f"{npunct.mean():.1f} (min {npunct.min()}, max {npunct.max()})")

    print("\n  --- summary " + "-" * 52)
    print(f"  lines                          {M}")
    print(f"  punctures per line             {stats}")
    print(f"  mean z-advance / arclength     {dz_rate:.6f}   "
          f"(Bbar_z = {Bbar[2]:.6f}, rel err "
          f"{abs(dz_rate / Bbar[2] - 1):.2e})")
    print(f"  mean Bbar-advance / arclength  {dpar_rate:.6f}   "
          f"(|Bbar| = {nBbar:.6f}, rel err "
          f"{abs(dpar_rate / nBbar - 1):.2e})")
    print(f"    +- {disp[:, 2].std() / np.sqrt(M) / S / abs(Bbar[2]):.2e} "
          f"(1 s.e. over lines).  The drift theorem <dr/ds> = Bbar is an "
          f"ERGODIC average:")
    print(f"    unbiased only for --seed-mode uniform (volume-uniform); ring "
          f"and reversed seeds")
    print("    sample the plane / the reversed region and read low.")
    if npunct.max():
        print(f"  arclength per transit          "
              f"{S / max(npunct.mean(), 1e-30):.4f}   "
              f"(2pi/|Bbar_z| = {TWOPI / vz:.4f})")
        disp_t = transverse_dispersion(xy)
        good = np.isfinite(disp_t)
        if good.any():
            print(f"  transverse move per transit    "
                  f"median {np.median(disp_t[good]):.4f}, "
                  f"mean {disp_t[good].mean():.4f}")
        flag, rho, crit = nn_spread(xy, min_punct=args.min_punct)
        cls = np.isfinite(rho)
        if cls.any():
            print(f"  chaotic fraction               "
                  f"{flag[cls].mean():.3f}  ({int(flag[cls].sum())}/"
                  f"{int(cls.sum())} lines classified)")
            print(f"    criterion: median nearest-neighbour spacing d within "
                  f"a line's own punctures,")
            print(f"    normalised by the closed-curve value 2 pi R / P "
                  f"(R = RMS cloud radius,")
            print("    P = punctures); chaotic when rho = d P / (2 pi R) "
                  "exceeds")
            print(f"    rho_crit = sqrt(0.25 sqrt(P/pi)) = "
                  f"{np.nanmedian(crit):.2f} (geometric mean of the "
                  f"curve and")
            print(f"    area-filling predictions).  median rho = "
                  f"{np.nanmedian(rho):.2f}")
        rot = rotation_number(xy, center, turns=True)
        fin = np.isfinite(rot)
        if fin.any():
            print(f"  rotation number (turns/transit, about the centre) "
                  f"median {np.median(rot[fin]):+.4f}")
    print("  " + "-" * 64)

    pad, cnt = pad_punctures(xy)
    step = np.full((M, pad.shape[1]), -1, dtype=np.int64)
    for m, i in enumerate(idx):
        step[m, :len(i)] = i
    npz = args.npz or os.path.splitext(args.out)[0] + ".npz"
    np.savez_compressed(npz, punct_xy=pad, punct_n=cnt, step_idx=step,
                        seeds=seeds, endpos=endpos, center=center, z0=z0,
                        h=args.h, refine=args.refine, n_steps=n_steps,
                        Bbar=Bbar, eps=eps, seed_mode=args.seed_mode,
                        grid=np.array(shape))
    title = (f'{args.seed_mode} seeds, {M} lines, '
             f'$\\varepsilon$={eps:.2f}, {shape[0]}x{shape[1]}x{shape[2]}, '
             f'$z_0$={z0:.2f}, h={args.h}, refine={args.refine}')
    zoom = args.zoom or float(np.clip(                # >= 2x magnification
        1.3 * structure_radius(cos, shape, z0, center), 0.25, 0.5 * np.pi))
    plot_map(xy, center, args.out, title, zoom,
             bz=plane_slice(B[2], z0, shape[2]), dpi=args.dpi)
    print(f"  wrote {args.out} and {npz}")


def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--state", required=True)
    p.add_argument("--lines", type=int, default=512)
    p.add_argument("--transits", type=int, default=300)
    p.add_argument("--z0", default="pi")
    p.add_argument("--seed-mode", choices=("ring", "uniform", "reversed"),
                   default="ring")
    p.add_argument("--out", default="fig_poincare.png")
    p.add_argument("--npz", default="")
    p.add_argument("--h", type=float, default=0.02)
    p.add_argument("--refine", type=int, default=2)
    p.add_argument("--chunk", type=int, default=0)
    p.add_argument("--center", type=float, nargs=2, default=None)
    p.add_argument("--ring-rmax", type=float, default=3.0)
    p.add_argument("--ring-angles", type=int, default=8)
    p.add_argument("--zoom", type=float, default=0.0)
    p.add_argument("--min-punct", type=int, default=8)
    p.add_argument("--seed-rng", type=int, default=0)
    p.add_argument("--dpi", type=int, default=170)
    p.add_argument("--verbose", action="store_true")
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
