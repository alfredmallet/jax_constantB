#!/usr/bin/env python3
"""
amplitude_quest.py (JAX port) -- automated continuation to arbitrarily large
amplitude, with adaptive stepping, automatic grid refinement, an optional
Sobolev-weighted (smoothing) step, and an online gradient-divergence
diagnostic. Designed to answer: DO THE GRADIENTS DIVERGE AT FINITE EPSILON,
OR ONLY GROW (e.g. exponentially) WITHOUT BOUND?

This is a COPY of ../amplitude_quest.py, ported to jax_constantB's jax-backed
Solver. See constantB_tools.py's module docstring for the host/device split
(seed construction stays numpy/scipy; the Gauss-Newton + CG solve is
jit-compiled). This file's own adaptive continuation loop -- accept/reject a
step, halve/grow d_eps, refine the grid on spectral-tail growth, fit the
divergence verdict -- is genuinely data-dependent host-side control flow (CSV
writes, grid-ladder switching driven by runtime values) and is therefore left
as an eager Python driver, exactly as in the original; only the per-step
`gn` call is jit-compiled.

    python3 amplitude_quest.py                  # defaults: the paper's branch
    python3 amplitude_quest.py --eps-max 3.0 --grid-max 96 96 192 --smooth 1.0

WHAT IT DOES per continuation step:
  1. add  d_eps * seed  to the current exact solution (push off the manifold);
  2. re-converge with (optionally Sobolev-weighted) minimum-norm Gauss-Newton;
  3. measure: residual, max|grad B| (Frobenius), |Bbar|, max deflection,
     reversal volume, spectral tails;
  4. ADAPT:
       - if GN struggled (residual above --res-ok), halve d_eps and retry
         from the saved previous state; if d_eps underflows -> stop: "FOLD?"
       - if spectral tails exceed --tail-max, refine the grid (zero-pad to
         the next grid in the ladder) and re-polish; if already at the top
         grid -> stop: "SHARPENING (resolution exhausted)"
       - otherwise, if convergence was easy, grow d_eps (up to --de-max);
  5. append everything to a CSV (resumable: rerun continues from the state).

THE DIVERGENCE DIAGNOSTIC. Over a trailing window of accepted steps the
script fits the inverse logarithmic growth rate of the gradient,
      Q(eps) = [ d ln(maxgrad) / d eps ]^{-1} .
  - Q roughly CONSTANT  -> exponential growth: gradients grow without bound
    but there is NO finite-eps singularity on this branch;
  - Q DECREASING LINEARLY toward zero -> finite-eps blow-up
    maxgrad ~ (eps* - eps)^(-alpha); the linear extrapolation of Q to zero
    estimates eps*, printed each step. (Same Domb-Sykes logic as for series.)

THE SOBOLEV-WEIGHTED STEP (--smooth s > 0). The default minimum-norm step
minimizes the L2 size of the correction; with --smooth s it minimizes
||(1+k^2)^(s/2) dB||_2 instead, i.e. it penalises small scales, choosing the
SMOOTHEST correction that cancels the linearised residual. Implementation:
dB = W^-2 J^T (J W^-2 J^T)^{-1} (-F) with W^-2 = spectral multiplication by
(1+k^2)^(-s); the CG operator remains SPD. If the plain branch's gradients
blow up but the weighted branch's do not, the blow-up convicts the PATH, not
the solution manifold -- the central interpretive point (paper Sec. 7.6).

CAVEATS. Residuals quoted per-step are working-grid residuals; the tail
monitor plus refinement ladder is what keeps them honest (each refinement
re-measures the true residual of the incoming state; see the CSV column
'incoming_res' after each grid change). For final states, run
constantB_tools.py refine/diagnose as usual.

JAX NOTE. The Sobolev weight `s` and the `pcg` flag are resolved as static
(trace-time) arguments of the underlying jit-compiled solve -- see
WeightedSolver below, which only overrides the `_Wm2`/`_weighted` hooks of
the jax-backed `Solver` in constantB_tools.py; the compiled GN/CG machinery
itself is shared, not duplicated.
"""
import argparse, csv, os, sys, time
import numpy as np
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from constantB_tools import (Solver, carrier, build_seed, zero_pad,
                             save_state, load_state, numpy_wavenumbers, numpy_dif)


class WeightedSolver(Solver):
    """Minimum-norm Gauss-Newton in the Sobolev norm ||(1+k^2)^(s/2) dB||.

    Subclasses the jax-backed Solver and only overrides the two hooks that
    parameterise the weighted adjoint (`_Wm2`, `_weighted`); the jit-compiled
    GN-sweep/CG machinery (`_gn_jit` in constantB_tools.py) is shared and
    re-traces once per (shape, weighted, pcg) combination -- `s` itself never
    enters the trace as a value, only through the concrete Wm2 array built
    here in plain Python before the jitted call."""

    def __init__(self, shape, smooth=0.0):
        super().__init__(shape)
        self.s = float(smooth)

    @property
    def _weighted(self):
        return self.s > 0

    @property
    def _Wm2(self):
        if self.s > 0:
            return (1.0 + self.K2) ** (-self.s)
        return jnp.ones_like(self.K2)


def diagnostics(S, B, car):
    """Host-numpy diagnostics (one-shot per step; no benefit from jit).
    Accepts B as a jax or numpy array."""
    B = np.asarray(B)
    nrm = np.sqrt((B ** 2).sum(0))
    Bbar = B.mean(axis=(1, 2, 3)); nb = np.linalg.norm(Bbar)
    cosM = np.clip((B * Bbar[:, None, None, None]).sum(0) / (nrm * nb), -1, 1)
    defl = np.degrees(np.arccos(cosM))
    K = numpy_wavenumbers(S.shape)
    g2 = sum(numpy_dif(B[i], j, K) ** 2 for i in range(3) for j in range(3))
    bh = np.abs(np.fft.fftn(B - car['B0'][:, None, None, :], axes=(1, 2, 3))) ** 2
    tails = []
    for ax, N in ((1, S.shape[0]), (2, S.shape[1]), (3, S.shape[2])):
        E = bh.sum(axis=tuple(i for i in range(4) if i != ax))[:N // 2]
        tails.append(float(E[-3:].max() / E.max()))
    return dict(maxgrad=float(np.sqrt(g2.max())), Bbar=float(nb),
                maxdefl=float(defl.max()), vol_rev=float((defl > 90).mean()),
                tail=max(tails))


def divergence_verdict(hist, window=6):
    """Fit Q = [d ln g/d eps]^{-1} over the trailing window; extrapolate."""
    pts = [(h['eps'], h['maxgrad']) for h in hist][-window-1:]
    if len(pts) < 4:
        return "insufficient data"
    e = np.array([p[0] for p in pts]); g = np.log([p[1] for p in pts])
    rate = np.diff(g) / np.diff(e)                 # d ln g / d eps at midpoints
    em = 0.5 * (e[1:] + e[:-1]); Q = 1.0 / np.maximum(rate, 1e-12)
    sl, ic = np.polyfit(em, Q, 1)
    if sl >= -0.05 * abs(ic) / max(em[-1] - em[0], 1e-9):
        return f"Q~const ({Q[-1]:.2f}): exponential growth, no finite-eps blow-up detected"
    eps_star = -ic / sl
    return (f"Q declining (slope {sl:.2f}): finite-eps blow-up candidate, "
            f"extrapolated eps* ~ {eps_star:.2f}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--state', default='quest.npz')
    p.add_argument('--csv', default='quest.csv')
    p.add_argument('--A', type=float, default=1.2)
    p.add_argument('--c', type=float, default=0.2)
    p.add_argument('--modes', type=float, nargs=3, action='append',
                   help='kx ky amp (repeatable); default (1,1,.15),(1,-1,.15)')
    p.add_argument('--prof', type=float, nargs=2, default=[1.0, 0.7])
    p.add_argument('--grid0', type=int, nargs=3, default=[32, 32, 64])
    p.add_argument('--grid-max', type=int, nargs=3, default=[96, 96, 192])
    p.add_argument('--eps-max', type=float, default=5.0)
    p.add_argument('--de', type=float, default=0.04, help='initial step')
    p.add_argument('--de-min', type=float, default=1e-4)
    p.add_argument('--de-max', type=float, default=0.08)
    p.add_argument('--sweeps', type=int, default=6)
    p.add_argument('--cgit', type=int, default=600)
    p.add_argument('--res-ok', type=float, default=1e-6,
                   help='step accepted if working-grid residual below this')
    p.add_argument('--tail-max', type=float, default=1e-5,
                   help='refine grid when spectral tails exceed this')
    p.add_argument('--smooth', type=float, default=0.0,
                   help='Sobolev exponent s of the weighted step (0 = plain L2)')
    p.add_argument('--pcg', action='store_true', default=True)
    args = p.parse_args()
    modes = [tuple(m) for m in (args.modes or [[1, 1, 0.15], [1, -1, 0.15]])]
    meta = dict(A=args.A, c=args.c, modes=np.array(modes, float),
                prof=np.array(args.prof, float))

    # ---- resume or initialise -------------------------------------------------
    if os.path.exists(args.state):
        B, eps, meta = load_state(args.state)
        grid = B.shape[1:]
        print(f"resuming: eps={eps:.3f}, grid {grid}")
    else:
        grid = tuple(args.grid0)
        car = carrier(args.A, args.c, grid[2])
        B = car['B0'][:, None, None, :] + 0.02 * build_seed(car, modes, grid,
                                                             tuple(args.prof))
        B, res, _ = WeightedSolver(grid, args.smooth).gn(
            B, sweeps=args.sweeps, cgit=args.cgit, pcg=args.pcg)
        eps = 0.02
        print(f"initialised: eps=0.02, residual {res:.1e}")

    grids = [grid]
    g = list(grid)
    while tuple(g) != tuple(args.grid_max):
        g = [min(int(gi * 1.5) - int(gi * 1.5) % 2, gm) for gi, gm in zip(g, args.grid_max)]
        grids.append(tuple(g))
    de = args.de
    hist = []
    fresh = not os.path.exists(args.csv)

    while eps < args.eps_max and de > args.de_min:
        car = carrier(float(meta['A']), float(meta['c']), B.shape[3])
        seed = build_seed(car, [tuple(m) for m in np.atleast_2d(meta['modes'])],
                          B.shape[1:], tuple(np.array(meta['prof']).ravel()))
        S = WeightedSolver(B.shape[1:], args.smooth)
        t0 = time.time()
        Btry, res, ci = S.gn(np.asarray(B) + de * seed, sweeps=args.sweeps,
                             cgit=args.cgit, pcg=args.pcg)
        if res > args.res_ok:
            de *= 0.5
            print(f"  [step rejected: res {res:.1e}; halving d_eps -> {de:.4f}]")
            continue
        B, eps = Btry, eps + de
        d = diagnostics(S, B, car)
        row = dict(eps=round(eps, 4), de=de, grid=str(B.shape[1:]), res=res,
                   cg=ci, minutes=round((time.time()-t0)/60, 2), **d)
        hist.append(row)
        verdict = divergence_verdict(hist)
        print(f"eps={eps:.3f} grid={B.shape[1:]} res={res:.1e} "
              f"maxgrad={d['maxgrad']:.2f} |Bbar|={d['Bbar']:.3f} "
              f"defl={d['maxdefl']:.1f} vol>90={100*d['vol_rev']:.1f}% "
              f"tail={d['tail']:.1e}")
        print(f"   verdict: {verdict}")
        with open(args.csv, 'a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if fresh: w.writeheader(); fresh = False
            w.writerow(row)
        save_state(args.state, B, eps, meta)

        # ---- adaptive refinement ---------------------------------------------
        if d['tail'] > args.tail_max:
            cur = grids.index(B.shape[1:]) if B.shape[1:] in grids else 0
            if cur + 1 < len(grids):
                new = grids[cur + 1]
                Bf = jnp.stack([zero_pad(B[i], new) for i in range(3)])
                rin = max(float(jnp.abs(r).max()) for r in Solver(new).residual(Bf))
                print(f"   [refining {B.shape[1:]} -> {new}; incoming honest "
                      f"residual {rin:.1e}]")
                B, res, _ = WeightedSolver(new, args.smooth).gn(
                    Bf, sweeps=args.sweeps + 4, cgit=args.cgit, pcg=args.pcg)
                save_state(args.state, B, eps, meta)
            else:
                print("STOP: tails exceed limit at the largest allowed grid.")
                print("  Interpretation: SHARPENING -- the solution is leaving")
                print("  the resolvable smoothness class. If this recurs at the")
                print("  same eps for larger --grid-max, that eps marks genuine")
                print("  gradient blow-up of THIS BRANCH (retry with --smooth,")
                print("  other seeds/pushes before blaming the manifold).")
                break
        elif ci < args.cgit // 2:
            de = min(de * 1.3, args.de_max)

    if de <= args.de_min:
        print("STOP: d_eps underflow -- Gauss-Newton cannot re-converge even for")
        print("  tiny pushes. Interpretation: FOLD CANDIDATE (branch may end).")
        print("  Check: does the stall eps move with resolution/--smooth/seed?")
        print("  Resolution-independent stall across reroutes = true obstruction.")
    print("final:", divergence_verdict(hist, window=10))


if __name__ == '__main__':
    main()
