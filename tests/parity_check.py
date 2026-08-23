#!/usr/bin/env python3
"""Parity validation: numpy reference vs legacy jax port vs new constantB
package, compared AT CONVERGENCE (converged residual, maxgrad, tail norms)
-- never iterate-by-iterate (project CLAUDE.md).

Run from the jax_constantB directory:
    python3 tests/parity_check.py [--grid 24 24 48] [--eps 0.3]

Implementations compared:
    ref    numpy reference        ../constantB_tools.py   (frozen spec)
    old    legacy jax port        legacy/constantB_tools.py
    new    constantB package      ./constantB/

Checks:
  1. seed construction bitwise-identical (same host numpy code path);
  2. collocation GN (pcg off/on): converged residual, maxgrad, max||B|-1|;
  3. Galerkin (dealias) GN + tail_norm: residual, maxgrad, tail rms/max;
  4. Sobolev-weighted GN (smooth=1): old-jax WeightedSolver vs new smooth=;
  5. residual + tail_norm of a saved state (mlstate_fine.npz if present);
  6. zero_pad refinement residual.

PASS thresholds are relative and generous to FFT round-off reordering
(rfft vs fft), but far tighter than any physical effect: 1e-6 relative on
converged scalars (residuals compared as "both below tol or equal floor").
"""
import argparse
import importlib.util
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
ROOT = os.path.dirname(REPO)

sys.path.insert(0, REPO)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def maxgrad_of(S, B):
    B = np.asarray(B)
    return max(float(np.abs(np.asarray(S.dif(B[i], j))).max())
               for i in range(3) for j in range(3))


def rel(a, b):
    d = abs(a - b) / max(abs(a), abs(b), 1e-300)
    return d


def check(label, vals, rtol=1e-6, floor=None):
    """vals: dict name->scalar. PASS if pairwise rel diff < rtol, or (floor
    given) all below floor."""
    names = list(vals)
    if floor is not None and all(abs(v) < floor for v in vals.values()):
        print(f"  {label:34s} all < {floor:.0e}  "
              + "  ".join(f"{n}={vals[n]:.3e}" for n in names) + "  PASS")
        return True
    worst = max(rel(vals[a], vals[b])
                for i, a in enumerate(names) for b in names[i+1:])
    ok = worst < rtol
    print(f"  {label:34s} reldiff {worst:.2e}  "
          + "  ".join(f"{n}={vals[n]:.9e}" for n in names)
          + ("  PASS" if ok else "  FAIL"))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, nargs=3, default=[24, 24, 48])
    ap.add_argument("--eps", type=float, default=0.3)
    ap.add_argument("--sweeps", type=int, default=8)
    ap.add_argument("--cgit", type=int, default=400)
    args = ap.parse_args()
    grid = tuple(args.grid)

    ref = load_module("ref_tools", os.path.join(ROOT, "constantB_tools.py"))
    old = load_module("old_tools", os.path.join(REPO, "legacy",
                                                "constantB_tools.py"))
    import constantB as new
    oldq = load_module("old_quest", os.path.join(REPO, "legacy",
                                                 "amplitude_quest.py"))

    ok = True
    modes = [(1, 1, 0.15), (1, -1, 0.15)]

    # ---- 1. seed construction --------------------------------------------
    print("== 1. seed construction (host numpy; expect bitwise) ==")
    car_r = ref.carrier(1.2, 0.2, grid[2])
    car_o = old.carrier(1.2, 0.2, grid[2])
    car_n = new.carrier(1.2, 0.2, grid[2])
    s_r = ref.build_seed(car_r, modes, grid)
    s_o = old.build_seed(car_o, modes, grid)
    s_n = new.build_seed(car_n, modes, grid)
    print(f"  max|seed_ref-seed_old| = {np.abs(s_r-s_o).max():.1e}, "
          f"max|seed_ref-seed_new| = {np.abs(s_r-s_n).max():.1e}")
    ok &= np.array_equal(s_r, s_n)
    B0 = car_r["B0"][:, None, None, :] + args.eps * s_r

    def converge(S, B, **kw):
        B, res, ci = S.gn(np.array(B), sweeps=args.sweeps, cgit=args.cgit,
                          **kw)
        return np.asarray(B), res, ci

    # ---- 2. collocation GN -----------------------------------------------
    for pcg in (False, True):
        print(f"== 2. collocation GN, pcg={pcg} ==")
        Br, rr, cr = converge(ref.Solver(grid), B0, pcg=pcg)
        Bo, ro, co = converge(old.Solver(grid), B0, pcg=pcg)
        Bn, rn, cn = converge(new.Solver(grid), B0, pcg=pcg)
        print(f"  cg iters: ref {cr}  old {co}  new {cn}")
        ok &= check("converged residual", dict(ref=rr, old=ro, new=rn),
                    floor=1e-10)
        ok &= check("maxgrad", dict(ref=maxgrad_of(ref.Solver(grid), Br),
                                    old=maxgrad_of(old.Solver(grid), Bo),
                                    new=maxgrad_of(new.Solver(grid), Bn)))
        # At convergence max||B|-1| sits at the round-off floor far below
        # the GN tolerance; only "all tiny" is meaningful.
        nrm = lambda B: float(np.abs(np.sqrt((B**2).sum(0)) - 1).max())
        ok &= check("max||B|-1|", dict(ref=nrm(Br), old=nrm(Bo), new=nrm(Bn)),
                    floor=1e-10)

    # ---- 3. Galerkin (dealias) GN + tail ----------------------------------
    print("== 3. Galerkin GN (dealias, pcg) ==")
    Sr = ref.Solver(grid, dealias=True)
    So = old.Solver(grid, dealias=True)
    Sn = new.Solver(grid, dealias=True)
    Br, rr, _ = converge(Sr, B0, pcg=True)
    Bo, ro, _ = converge(So, B0, pcg=True)
    Bn, rn, _ = converge(Sn, B0, pcg=True)
    ok &= check("converged residual", dict(ref=rr, old=ro, new=rn),
                floor=1e-10)
    ok &= check("maxgrad", dict(ref=maxgrad_of(Sr, Br),
                                old=maxgrad_of(So, Bo),
                                new=maxgrad_of(Sn, Bn)))
    tr, to, tn = Sr.tail_norm(Br), So.tail_norm(Bo), Sn.tail_norm(Bn)
    ok &= check("tail rms", dict(ref=tr[0], old=to[0], new=tn[0]), rtol=1e-4)
    ok &= check("tail max", dict(ref=tr[1], old=to[1], new=tn[1]), rtol=1e-4)

    # ---- 4. Sobolev-weighted GN ------------------------------------------
    print("== 4. weighted GN (smooth=1, dealias, pcg): old WeightedSolver "
          "vs new Solver(smooth=1) ==")
    Wo = oldq.WeightedSolver(grid, 1.0, dealias=True)
    Wn = new.Solver(grid, dealias=True, smooth=1.0)
    Bo, ro, _ = converge(Wo, B0, pcg=True)
    Bn, rn, _ = converge(Wn, B0, pcg=True)
    # Stalled inexact-GN endgame value: same floor, round-off-path sensitive.
    ok &= check("converged residual", dict(old=ro, new=rn), rtol=1e-3)
    ok &= check("maxgrad", dict(old=maxgrad_of(Wo, Bo),
                                new=maxgrad_of(Wn, Bn)))
    ok &= check("tail rms", dict(old=Wo.tail_norm(Bo)[0],
                                 new=Wn.tail_norm(Bn)[0]), rtol=1e-4)

    # ---- 5. saved state ----------------------------------------------------
    state = os.path.join(REPO, "mlstate_fine.npz")
    if os.path.exists(state):
        print("== 5. saved state residual/tail (mlstate_fine.npz) ==")
        B, eps, meta = new.load_state(state)
        g = B.shape[1:]
        for name, S in (("ref", ref.Solver(g)), ("old", old.Solver(g)),
                        ("new", new.Solver(g))):
            r1, r2 = S.residual(B)
            print(f"  {name}: div {float(np.abs(np.asarray(r1)).max()):.12e}"
                  f"  quad {float(np.abs(np.asarray(r2)).max()):.12e}")
        vals_div, vals_quad = {}, {}
        for name, S in (("ref", ref.Solver(g)), ("old", old.Solver(g)),
                        ("new", new.Solver(g))):
            r1, r2 = S.residual(B)
            vals_div[name] = float(np.abs(np.asarray(r1)).max())
            vals_quad[name] = float(np.abs(np.asarray(r2)).max())
        ok &= check("state div residual", vals_div, rtol=1e-8)
        ok &= check("state quad residual", vals_quad, rtol=1e-8)
        td = {name: S.tail_norm(B)[0]
              for name, S in (("ref", ref.Solver(g, dealias=True)),
                              ("old", old.Solver(g, dealias=True)),
                              ("new", new.Solver(g, dealias=True)))}
        ok &= check("state gal tail rms", td, rtol=1e-8)
    else:
        print("== 5. skipped (mlstate_fine.npz not present) ==")

    # ---- 6. zero_pad --------------------------------------------------------
    print("== 6. zero_pad refinement ==")
    fine = tuple(2 * n for n in grid)
    zr = np.stack([np.asarray(ref.zero_pad(B0[i], fine)) for i in range(3)])
    zn = np.stack([np.asarray(new.zero_pad(B0[i], fine)) for i in range(3)])
    ok &= check("zero_pad max|ref-new|",
                dict(diff=float(np.abs(zr - zn).max()), zero=0.0),
                floor=1e-12)

    print()
    print("OVERALL:", "ALL PASS" if ok else "FAILURES -- see above")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
