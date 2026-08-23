#!/usr/bin/env python3
"""
constantB_tools.py -- command-line driver for the `constantB` package.

Constant-|B| (|B| = 1 pointwise), divergence-free magnetic fields on the
periodic box [0,2pi)^3, continued to large amplitude from a 1D arc-polarised
carrier plus a genuinely 3D seed.  All reusable logic (spectral operators,
the jit-compiled minimum-norm Gauss-Newton solver, seeds, state files,
series diagnostics, plotting) lives in the `constantB` package -- see
constantB/ and DESIGN.md for the numerical rationale.  This file is the thin
CLI plus a re-export shim so old `from constantB_tools import ...` call
sites keep working.

Subcommands: series, init, cont, refine, polish, diagnose, plotcuts, plot3d.

    python3 constantB_tools.py init --grid 32 32 64 --eps 0.02
    python3 constantB_tools.py cont --steps 10 --de 0.02 --dealias --pcg
    python3 constantB_tools.py refine --grid 48 48 96
    python3 constantB_tools.py diagnose
    python3 constantB_tools.py series --Nord 16
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from constantB import *          # noqa: F401,F403  (re-export the package API)
from constantB import (TWOPI, Solver, WeightedSolver, build_seed, carrier,
                       diagnose, domb_sykes, load_state, meta_args,
                       numpy_dif, numpy_wavenumbers, pade_poles, rebuild_seed,
                       save_state, series, zero_pad)
from constantB.plotting import plot_cuts, plot_3d

import jax.numpy as jnp          # after constantB: x64 already enabled


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('cmd', choices=['series', 'init', 'cont', 'refine',
                                   'polish', 'diagnose', 'plotcuts', 'plot3d'])
    p.add_argument('--A', type=float, default=1.2, help='carrier arc amplitude (rad)')
    p.add_argument('--c', type=float, default=0.2, help='carrier mean-aligned component')
    p.add_argument('--grid', type=int, nargs=3, default=[32, 32, 64],
                   help='Nx Ny Nz (init: working grid; refine: target grid)')
    p.add_argument('--modes', type=float, nargs=3, action='append',
                   help='kx ky amp (repeatable); default (1,1,.15) and (1,-1,.15)')
    p.add_argument('--prof', type=float, nargs=2, default=[1.0, 0.7],
                   help='free profile u(z) = amp*(a cos z + b)')
    p.add_argument('--eps', type=float, default=0.02, help='initial seed amplitude')
    p.add_argument('--de', type=float, default=0.02, help='continuation step')
    p.add_argument('--steps', type=int, default=1, help='continuation steps')
    p.add_argument('--sweeps', type=int, default=4, help='GN sweeps (polish)')
    p.add_argument('--cgit', type=int, default=500, help='CG budget per sweep')
    p.add_argument('--dealias', action='store_true',
                   help='Galerkin 2/3-rule solve: alias-free retained-band '
                        'equations; honest tail measured on the same grid')
    p.add_argument('--pcg', action='store_true',
                   help='preconditioned CG (recommended for cold or localized starts)')
    p.add_argument('--state', default='state.npz', help='state file')
    p.add_argument('--out', default='fig.png', help='output figure file')
    # series-only options
    p.add_argument('--kap', type=float, default=np.sqrt(2.0))
    p.add_argument('--chi', type=float, default=np.pi/4)
    p.add_argument('--Nord', type=int, default=16)
    p.add_argument('--Nz', type=int, default=256)
    args = p.parse_args()
    if args.modes is None:
        args.modes = [[1, 1, 0.15], [1, -1, 0.15]]

    if args.cmd == 'series':
        a, scal, om = series(args.A, args.c, args.kap, args.chi,
                             Nord=args.Nord, Nz=args.Nz)
        invr, r = domb_sykes(a)
        poles = pade_poles(scal)
        print(f"omega = {om:+.5f}   (divisors 2|sin(pi j omega)|)")
        print("a_n ratios:", " ".join(f"{x:.2f}" for x in r))
        print(f"Domb-Sykes 1/rho = {invr:.2f}  =>  rho ~ {1/invr:.4g}")
        print(f"nearest Pade pole: {poles[np.argmin(np.abs(poles))]:.4g}")

    elif args.cmd == 'init':
        car = carrier(args.A, args.c, args.grid[2])
        seed = build_seed(car, [tuple(m) for m in args.modes], args.grid,
                          prof=tuple(args.prof))
        B = car['B0'][:, None, None, :] + args.eps * seed
        B, res, ci = Solver(args.grid, dealias=args.dealias).gn(B, sweeps=6, cgit=args.cgit, pcg=args.pcg)
        print(f"init eps={args.eps}: residual {res:.1e} (cg {ci})")
        save_state(args.state, B, args.eps, meta_args(args))

    elif args.cmd == 'cont':
        B, eps, meta = load_state(args.state)
        shape = B.shape[1:]
        seed, car = rebuild_seed(meta, shape)
        S = Solver(shape, dealias=args.dealias)
        for k in range(args.steps):
            eps += args.de
            B = B + args.de * seed
            B, res, ci = S.gn(B, sweeps=4, cgit=args.cgit, pcg=args.pcg)
            g = max(np.abs(S.dif(B[i], j)).max() for i in range(3) for j in range(3))
            print(f"eps={eps:.2f}: res={res:.1e} maxgrad={g:.2f} cg={ci}")
        save_state(args.state, B, eps, meta)

    elif args.cmd == 'refine':
        B, eps, meta = load_state(args.state)
        Bf = jnp.stack([zero_pad(B[i], tuple(args.grid)) for i in range(3)])
        r1, r2 = Solver(args.grid).residual(Bf)
        print(f"HONEST residual on {tuple(args.grid)}: div {np.abs(r1).max():.2e}, "
              f"|B|^2-1 {np.abs(2*r2).max():.2e}   [this is the number that counts]")
        save_state(args.state, Bf, eps, meta)

    elif args.cmd == 'polish':
        B, eps, meta = load_state(args.state)
        B, res, ci = Solver(B.shape[1:], dealias=args.dealias).gn(B, sweeps=args.sweeps,
                                            cgit=args.cgit, verbose=True, pcg=args.pcg)
        print(f"polished: residual {res:.2e}")
        save_state(args.state, B, eps, meta)

    elif args.cmd == 'diagnose':
        B, eps, meta = load_state(args.state)
        diagnose(B, eps, meta)

    elif args.cmd == 'plotcuts':
        B, eps, meta = load_state(args.state)
        plot_cuts(B, meta, args.out)

    elif args.cmd == 'plot3d':
        B, eps, meta = load_state(args.state)
        plot_3d(B, meta, args.out)


if __name__ == '__main__':
    main()
