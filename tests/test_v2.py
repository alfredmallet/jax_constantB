#!/usr/bin/env python3
"""Acceptance tests for the v2 carrier-free stack (SPEC_v2 section 8), in the
style of tests/parity_check.py: a plain script, no pytest, PASS/FAIL per case
with the measured numbers printed, nonzero exit on any failure.

    python3 tests/test_v2.py                    # ~2-3 min on CPU
    python3 tests/test_v2.py --grid 32 32 64    # the spec's parity grid (slow)

CASES
  1  parity s=0: legacy Solver(dealias) vs MuSolver from the same start
  2  parity s=1: same, with the Sobolev-weighted step
  3  quadratic convergence of the mu-form sweep
  4  divergence preservation, and project() cleaning an injected div error
  5  freeze / fix_mean invariance (incl. a kz=0 pin and its rfft partner)
     and the noncoplanar truth table
  6  tail honesty: the coarse-grid Galerkin tail IS the true tail, sampled
  7  random_seed grid independence
  8  grow.run end to end (4 steps, then a refinement rung) in a tmp dir
  9  descent.sqp_step (EXPERIMENTAL, SPEC_v2 section 7)

TWO DELIBERATE DEVIATIONS FROM SPEC v2 section 8, both forced by measurement
(numbers from the mu-solver prototype against the real legacy Solver):

CASES 1-2, the "converged max|dB| < 1e-9" threshold is unreachable in
principle, not merely in practice.  Legacy dealias-mode GN and mu-form GN are
DIFFERENT algorithms: the legacy CG step model uses the plain collocation
J / J^T while its target is the Galerkin residual (DESIGN.md), so it is
inexact GN -- on the blob push at eps=0.3 it stalls at residual ~3e-9 with its
CG fully converged (more CG budget changes nothing: 3.20e-09 at cgit 4000 and
at cgit 20000, identically), while the mu form reaches 2e-14.  And because the
constraint is QUADRATIC, a residual r admits a state displacement |d| ~
sqrt(2r) ORTHOGONAL to B at no residual cost, so two states converged to
different residual floors cannot agree better than that: measured max|dB| is
5.3e-4 at r = 3.2e-9, i.e. 6.6 x sqrt(2r).  The parity asserted here is
therefore the honest one -- the states agree to the accuracy the LESS
converged one allows, and the converged observables (maxgrad, tail_norm)
agree -- with the sqrt(2r) slack printed on every line.

CASE 3, "res_3 < 1e-12" holds for a random push (2.9e-2, 3.9e-4, 7.5e-8,
9.1e-14 -- textbook quadratic) but NOT for the blob push of case 1, whose
quadratic prefactor is large because the min-norm correction to a LOCALISED
constraint violation is spread out by the div-free projection: 2.3e-2, 5.9e-3,
1.4e-3, 2.0e-4, 6.6e-6, 1.0e-8, 2.8e-14 -- quadratic, but only from sweep 5.
Both are therefore tested: the literal spec number on the random push, and
"reaches 1e-12 with a quadratic tail" on the blob push.

CASE 6 note.  The exactly true statement is a SAMPLING identity, and that is
what is asserted: for in-band B the constraint q is a pointwise product, so
the coarse-grid tail field q - Tq equals the 2x-grid tail field evaluated at
the coarse points, to round-off.  Their rms values are NOT equal -- alias
folding adds pairs of true tail coefficients coherently, which is exactly why
the measured tail is a one-sided proxy (true tail >= measured/sqrt(2) per
axis, project CLAUDE.md); the ratio is reported, not asserted.
"""
import argparse
import os
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

import constantB                                     # noqa: F401  (x64 first)
from constantB import Solver, zero_pad, load_state
from constantB.spectral import rfft3, numpy_wavenumbers, numpy_dif
from constantB.solver_mu import MuSolver, noncoplanar
from constantB.seeds_free import blob, random_seed, top_modes
from constantB.descent import sqp_step, pinned_energies

RESULTS = []


# ---------------------------------------------------------------------------
# harness
# ---------------------------------------------------------------------------

def report(name, ok, msg):
    RESULTS.append((name, bool(ok)))
    print(f"  {name:38s} {msg}   {'PASS' if ok else 'FAIL'}")


def rel(a, b):
    return abs(a - b) / max(abs(a), abs(b), 1e-300)


def uniform_push(S, eps, seed=None):
    """Carrier-free start: uniform z-hat plus eps * seed (the standard blob
    unless another seed is passed).  Host numpy."""
    if seed is None:
        seed = blob(S, w=(0.8, 0.8, 0.8), kz=1)[0]
    B = np.zeros((3,) + tuple(S.shape))
    B[2] = 1.0
    return B + eps * np.asarray(seed, float)


def maxgrad(B):
    """max_ij |d_j B_i| (host numpy spectral derivatives -- solver agnostic)."""
    B = np.asarray(B)
    K = numpy_wavenumbers(B.shape[1:])
    return max(float(np.abs(numpy_dif(B[i], j, K)).max())
               for i in range(3) for j in range(3))


def roughness(B, smooth):
    """R(B) = 1/2 sum_k (1+k^2)^s |B_k|^2, host numpy, grid-normalised."""
    B = np.asarray(B)
    shape = B.shape[1:]
    K = np.meshgrid(*[np.fft.fftfreq(n, d=1.0 / n) for n in shape],
                    indexing="ij")
    K2 = sum(k ** 2 for k in K)
    Bh = np.fft.fftn(B, axes=(1, 2, 3)) / np.prod(shape)
    return float((0.5 * (1.0 + K2) ** smooth
                  * (np.abs(Bh) ** 2).sum(0)).sum())


def band_mask(shape, ncut):
    """|k_i| < ncut_i/3 (strict) on `shape`, full FFT layout, host numpy."""
    ms = [np.abs(np.fft.fftfreq(n, d=1.0 / n)) < c / 3 - 1e-12
          for n, c in zip(shape, ncut)]
    return (ms[0][:, None, None] * ms[1][None, :, None]
            * ms[2][None, None, :]).astype(float)


# ---------------------------------------------------------------------------
# cases
# ---------------------------------------------------------------------------

def parity(tag, L, S, Bl, rl, Bm, rm, grad_rtol, tail_rtol, state=True):
    """Compare two converged states honestly (see the module docstring): the
    quadratic constraint admits a displacement ~sqrt(2r) orthogonal to B at no
    residual cost, so sqrt(2 max(r)) is the floor of any state comparison.

    `state=False` for the weighted step: there the two algorithms minimise
    DIFFERENT weighted norms (collocation vs projected constraint row), which
    is a different path, and different paths select different members of the
    family at equal amplitude -- the paper's own path-selection effect.  Only
    the converged observables are then comparable.
    """
    dB = float(np.abs(np.asarray(Bl) - np.asarray(Bm)).max())
    slack = 30.0 * np.sqrt(2.0 * max(rl, rm))
    report(f"{tag} mu reaches the Newton floor", rm < 1e-12, f"res {rm:.2e}")
    if state:
        report(f"{tag} same state within sqrt(2r)", dB < slack,
               f"max|dB| {dB:.2e} < {slack:.2e} = 30 sqrt(2 x {max(rl, rm):.1e})")
    else:
        print(f"  {'(' + tag + ' max|dB|)':38s} {dB:.2e}  "
              f"(vs sqrt(2r) slack {slack:.2e}: different paths, not compared)")
    gl, gm = maxgrad(Bl), maxgrad(Bm)
    report(f"{tag} maxgrad", rel(gl, gm) < grad_rtol,
           f"rel {rel(gl, gm):.2e}  ({gl:.6f} / {gm:.6f})")
    tl, tm = L.tail_norm(Bl)[0], S.tail_norm(Bm)[0]
    report(f"{tag} tail_norm", rel(tl, tm) < tail_rtol,
           f"rel {rel(tl, tm):.2e}  ({tl:.4e} / {tm:.4e})")


def quadratic(res, floor=1e-12, entry=1e-4, budget=4):
    """Assert quadratic (not linear) convergence WITHOUT assuming a prefactor:
    once the residual is below `entry` it must reach `floor` within `budget`
    more sweeps.  Linear convergence at the observed 4x/sweep would need ~20.
    The per-transition constants res_{n+1}/res_n^2 are printed: their being
    roughly constant is the signature, their VALUE is problem dependent (~250
    for the localised blob push, ~0.5 for a random one)."""
    i0 = next((i for i, r in enumerate(res) if r < entry), None)
    hit = next((i for i, r in enumerate(res) if r < floor), None)
    C = [res[i + 1] / res[i] ** 2 for i in range(len(res) - 1)
         if 1e-15 < res[i + 1] < res[i] < entry]
    print("  quadratic constants res_n+1/res_n^2: "
          + "  ".join(f"{c:.1f}" for c in C))
    report(f"{floor:.0e} within {budget} sweeps of {entry:.0e}",
           i0 is not None and hit is not None and hit - i0 <= budget,
           f"sweep {None if i0 is None else i0 + 1} -> "
           f"{None if hit is None else hit + 1}")


def case1(grid, eps, cgl):
    print("== 1. legacy consistency s=0 (SMOKE: thresholds sit above the\n      legitimate ~7% member-to-member tail difference -- tail correctness\n      is proven by the tight invariants of cases 3 and 6, not here) ==")
    S, L = MuSolver(grid), Solver(grid, dealias=True)
    B0 = uniform_push(S, eps)
    # legacy gn DONATES its input buffer -- always hand it a fresh array.
    Bl, rl, cl = L.gn(np.array(B0), 14, cgl, 1e-13, False, True)
    Bm, rm, cm = S.gn(np.array(B0), sweeps=14, cgit=2000, tol=1e-13)
    print(f"  residuals: legacy {rl:.2e} (cg {cl})  mu {rm:.2e} (cg {cm})")
    parity("s=0", L, S, Bl, rl, Bm, rm, grad_rtol=1e-2, tail_rtol=2e-1)


def case2(grid, eps, cgl):
    print("== 2. legacy consistency s=1 (SMOKE; see case 1 header) ==")
    S = MuSolver(grid, smooth=1.0)
    L = Solver(grid, dealias=True, smooth=1.0)
    B0 = uniform_push(S, eps)
    Bl, rl, cl = L.gn(np.array(B0), 14, cgl, 1e-13, False, True)
    Bm, rm, cm = S.gn(np.array(B0), sweeps=14, cgit=2000, tol=1e-13)
    print(f"  residuals: legacy {rl:.2e} (cg {cl})  mu {rm:.2e} (cg {cm})")
    report("both converged", max(rl, rm) < 1e-6,
           f"legacy {rl:.2e}  mu {rm:.2e}")
    parity("s=1", L, S, Bl, rl, Bm, rm, grad_rtol=1e-1, tail_rtol=1.5e-1,
           state=False)


def case3(grid, eps):
    print("== 3. quadratic convergence (one sweep at a time) ==")
    S = MuSolver(grid)
    res, B = [], uniform_push(S, eps)                    # the case-1 blob push
    for _ in range(8):
        B, r, _ = S.gn(B, sweeps=1, cgit=2000, tol=0.0)  # tol=0: never a no-op
        res.append(r)
    print("  blob push, residual per sweep: "
          + "  ".join(f"{r:.1e}" for r in res))
    report("blob push reaches 1e-12 in <= 8 sweeps", min(res) < 1e-12,
           f"min {min(res):.2e} at sweep {int(np.argmin(res)) + 1}")
    quadratic(res)
    res, B = [], uniform_push(S, eps, random_seed(S, kmax=4, key=1)[0])
    for _ in range(4):
        B, r, _ = S.gn(B, sweeps=1, cgit=2000, tol=0.0)
        res.append(r)
    print("  random push, residual per sweep: "
          + "  ".join(f"{r:.1e}" for r in res))
    report("random push: res after 4 sweeps < 1e-12", res[3] < 1e-12,
           f"{res[3]:.2e}")
    # Quadratic ORDER assertion (kills a fast-linear impostor): on the O(1)-
    # prefactor random push the constants C_n = res_{n+1}/res_n^2 must be
    # bounded and roughly constant; a linear contraction q^n has C_n = q/res_n
    # growing without bound as res_n -> 0.
    C = [res[i + 1] / res[i] ** 2 for i in range(len(res) - 1)
         if 1e-14 < res[i + 1] < res[i] < 1e-1]
    report("random push: quadratic order (C_n bounded)",
           len(C) >= 2 and max(C) < 100.0 and max(C) / min(C) < 30.0,
           "C = " + "  ".join(f"{c:.2f}" for c in C))


def case4(grid, eps):
    print("== 4. divergence preservation and project() ==")
    S = MuSolver(grid)
    B = np.asarray(S.project(uniform_push(S, eps)))
    worst = 0.0
    for _ in range(3):
        B, _r, _c = S.gn(B, sweeps=1, cgit=800, tol=0.0)
        worst = max(worst, float(np.abs(np.asarray(S.residual(B)[0])).max()))
    report("max|div| after each sweep", worst < 1e-12, f"{worst:.2e}")
    rng = np.random.default_rng(7)
    x = [np.linspace(0, 2 * np.pi, n, endpoint=False) for n in grid]
    X = np.meshgrid(*x, indexing="ij")
    phi = sum(rng.normal() * np.sin(X[0] + k * X[1] - X[2]) for k in (1, 2))
    Bd = np.asarray(B) + 0.01 * np.stack(np.gradient(phi, *[xx[1] for xx in x]))
    d_before = float(np.abs(np.asarray(S.residual(Bd)[0])).max())
    Bp = S.project(Bd)
    d_after = float(np.abs(np.asarray(S.residual(Bp)[0])).max())
    dm = float(np.abs(np.asarray(Bp).mean(axis=(1, 2, 3))
                      - np.asarray(Bd).mean(axis=(1, 2, 3))).max())
    report("project() kills injected div", d_before > 1e-4 and d_after < 1e-12,
           f"{d_before:.2e} -> {d_after:.2e}")
    report("project() preserves the mean", dm < 1e-14, f"{dm:.2e}")


def case5(grid, eps):
    print("== 5. freeze / fix_mean invariance and noncoplanar ==")
    frz = [(1, 0, 0), (0, 1, 1)]            # (1,0,0): kz=0, has an rfft partner
    S = MuSolver(grid, fix_mean=True, freeze=frz)
    B0 = np.asarray(S.trunc3(uniform_push(S, eps)))
    B1, _r, _c = S.gn(B0, sweeps=2, cgit=400, tol=0.0)
    n = float(np.prod(grid))
    h0, h1 = np.asarray(rfft3(np.asarray(B0))) / n, np.asarray(rfft3(np.asarray(B1))) / n
    idx = [(1, 0, 0), (grid[0] - 1, 0, 0), (0, 1, 1)]   # pin + partner + pin
    worst = max(float(np.abs(h1[:, i[0], i[1], i[2]]
                             - h0[:, i[0], i[1], i[2]]).max()) for i in idx)
    report("frozen bins (incl. partner)", worst < 1e-14, f"{worst:.2e}")
    dm = float(np.abs(np.asarray(B1).mean(axis=(1, 2, 3))
                      - np.asarray(B0).mean(axis=(1, 2, 3))).max())
    report("fix_mean holds Bbar", dm < 1e-14, f"{dm:.2e}")
    truth = [([(1, 0, 0), (0, 1, 0), (0, 0, 1)], True),
             ([(1, 0, 0), (2, 0, 0), (3, 0, 0)], False),
             ([(1, 0, 0), (0, 1, 0), (1, 1, 0)], False),
             ([(1, 0, 0), (0, 1, 0)], False),
             ([(1, 0, 0), (0, 1, 0), (1, 1, 0), (0, 0, 1)], True)]
    got = [bool(noncoplanar(t)) for t, _ in truth]
    report("noncoplanar truth table",
           got == [w for _, w in truth], f"{got} vs {[w for _, w in truth]}")


def case6(grid):
    print("== 6. tail honesty: coarse tail == true tail, sampled ==")
    S = MuSolver(grid)
    rng = np.random.default_rng(11)
    B = np.array(S.trunc3(rng.normal(size=(3,) + tuple(grid))))
    B *= 0.9 / np.sqrt((B ** 2).sum(0)).max()              # in-band, |B| < 1
    q = 0.5 * ((B ** 2).sum(0) - 1.0)
    mc = band_mask(grid, grid)
    qt_c = q - np.real(np.fft.ifftn(mc * np.fft.fftn(q)))
    rms_c, max_c = S.tail_norm(B)
    report("tail_norm vs host reimplementation",
           rel(rms_c, float(np.sqrt((qt_c ** 2).mean()))) < 1e-12,
           f"{rel(rms_c, float(np.sqrt((qt_c ** 2).mean()))):.2e}")
    fine = tuple(2 * n for n in grid)
    Bf = np.stack([np.asarray(zero_pad(B[i], fine)) for i in range(3)])
    qf = 0.5 * ((Bf ** 2).sum(0) - 1.0)
    qt_f = qf - np.real(np.fft.ifftn(band_mask(fine, grid) * np.fft.fftn(qf)))
    sub = float(np.abs(qt_f[::2, ::2, ::2] - qt_c).max())
    report("coarse tail == 2x tail sampled", sub < 1e-12, f"{sub:.2e}")
    rms_f, max_f = float(np.sqrt((qt_f ** 2).mean())), float(np.abs(qt_f).max())
    report("2x max >= coarse max", max_f >= max_c - 1e-12,
           f"{max_f:.3e} >= {max_c:.3e}")
    print(f"  (fold ratio rms_true/rms_measured = {rms_f / rms_c:.3f}; "
          "coherent folding, not asserted)")


def case7():
    print("== 7. random_seed grid independence ==")
    c, f = (16, 16, 32), (24, 24, 48)
    bc = np.asarray(random_seed(MuSolver(c), kmax=4, key=5)[0], float)
    bf = np.asarray(random_seed(MuSolver(f), kmax=4, key=5)[0], float)
    up = np.stack([np.asarray(zero_pad(bc[i], f)) for i in range(3)])
    err = float(np.abs(bf - up).max()) / max(float(np.abs(bf).max()), 1e-300)
    report("fine seed == zero_pad(coarse seed)", err < 1e-10, f"{err:.2e}")
    # separates "same shape of field" from "same field": a grid-dependent
    # normalisation would leave the shapes equal and this ratio off 1.
    a = float((bf * up).sum() / max((up * up).sum(), 1e-300))
    report("grid-independent normalisation", abs(a - 1.0) < 1e-10,
           f"ratio {a:.12f}")


def case8():
    print("== 8. grow.run end to end (4 steps, then a refinement) ==")
    import csv
    import grow
    with tempfile.TemporaryDirectory() as td:
        state, csvf = os.path.join(td, "g.npz"), os.path.join(td, "g.csv")
        common = ["--state", state, "--csv", csvf, "--seed", "random",
                  "--key", "1", "--kmax", "4", "--de", "0.1", "--de-max",
                  "0.1", "--sweeps", "8", "--cgit", "400", "--res-ok", "1e-9",
                  "--snap-de", "0.2"]
        rc = grow.run(grow.build_parser().parse_args(
            common + ["--grid0", "16", "16", "32", "--grid-max", "16", "16",
                      "32", "--eps-max", "0.4", "--gtail-max", "1",
                      "--edge-max", "1"]))
        rows = list(csv.DictReader(open(csvf)))
        ok = rc == 0 and len(rows) >= 4 and all(
            float(r["res"]) <= 1e-9 for r in rows)
        report("4 accepted steps, res-ok met",
               ok, f"rc={rc} rows={len(rows)} "
                   f"maxres={max(float(r['res']) for r in rows):.1e}")
        report("CSV schema", list(rows[0]) == grow.CSV_FIELDS,
               f"{len(rows[0])} columns")
        # resume (no seed flags) and force one refinement rung
        rc = grow.run(grow.build_parser().parse_args(
            common + ["--grid-max", "24", "24", "48", "--eps-max", "0.6",
                      "--gtail-max", "1e-12", "--edge-max", "1"]))
        B, eps, meta = load_state(state)
        rows2 = list(csv.DictReader(open(csvf)))
        report("refined to 24x24x48 on resume",
               tuple(B.shape[1:]) == (24, 24, 48) and len(rows2) > len(rows),
               f"grid={tuple(B.shape[1:])} eps={eps:.2f} rows={len(rows2)}")
        S = MuSolver(tuple(B.shape[1:]))
        r = max(float(np.abs(np.asarray(x)).max()) for x in S.residual(B))
        report("final state still converged", r <= 1e-9, f"res={r:.2e}")
        snaps = [f for f in os.listdir(td) if f.startswith("g_eps")]
        report("snapshots written beside --state", len(snaps) >= 1,
               f"{sorted(snaps)}")


def case9(eps):
    print("== 9. descent.sqp_step (EXPERIMENTAL) ==")
    grid = (16, 16, 32)
    S = MuSolver(grid, smooth=1.0)
    B, r0, _ = S.gn(uniform_push(S, eps), sweeps=10, cgit=800, tol=1e-13)
    pins = [tuple(int(v) for v in k)
            for k in top_modes(np.asarray(B) - np.asarray(B).mean(
                axis=(1, 2, 3), keepdims=True), 2)]
    e0, R0 = pinned_energies(B, pins), roughness(B, S.smooth)
    # alpha small on purpose: the roughness gain is O(alpha) while the pinned
    # energies drift only at O(alpha^2) (the step is exactly tangent), so a
    # small alpha is what makes the 1e-8 pin tolerance meaningful.
    Bp = sqp_step(S, B, pins, alpha=5e-5)
    e1, R1 = pinned_energies(Bp, pins), roughness(Bp, S.smooth)
    res = max(float(np.abs(np.asarray(x)).max()) for x in S.residual(Bp))
    de = float(np.abs(e1 - e0).max() / max(np.abs(e0).max(), 1e-300))
    print(f"  pins {pins}; start res {r0:.1e}")
    report("roughness decreased", R1 < R0,
           f"{R0:.12f} -> {R1:.12f}  (d = {R1 - R0:.2e})")
    report("pinned energies held", de < 1e-8, f"rel {de:.2e}")
    report("state back on the manifold", res < 1e-10, f"res {res:.2e}")


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    # The legacy solver is the runtime bottleneck of cases 1-2 (~70 s per
    # solve at 32^2x64 vs ~2 s for the mu form), and the parity question is
    # grid independent, so the default parity grid is one rung down from the
    # spec's; pass --grid 32 32 64 for the spec's exact configuration.
    ap.add_argument("--grid", type=int, nargs=3, default=[24, 24, 48])
    ap.add_argument("--small", type=int, nargs=3, default=[16, 16, 32])
    ap.add_argument("--cgit-legacy", type=int, default=1000)
    ap.add_argument("--eps", type=float, default=0.3)
    ap.add_argument("--only", type=int, nargs="*", default=None)
    args = ap.parse_args()
    grid, small, cgl = tuple(args.grid), tuple(args.small), args.cgit_legacy

    cases = {1: lambda: case1(grid, args.eps, cgl),
             2: lambda: case2(grid, args.eps, cgl),
             3: lambda: case3(grid, args.eps), 4: lambda: case4(small, args.eps),
             5: lambda: case5(small, args.eps), 6: lambda: case6(small),
             7: case7, 8: case8, 9: lambda: case9(args.eps)}
    for i in sorted(cases):
        if args.only and i not in args.only:
            continue
        t0 = time.time()
        try:
            cases[i]()
        except Exception as exc:                       # keep going, report it
            report(f"case {i} raised", False, f"{type(exc).__name__}: {exc}")
        print(f"  [{time.time() - t0:.1f} s]")

    bad = [n for n, ok in RESULTS if not ok]
    print()
    print(f"OVERALL: {len(RESULTS) - len(bad)}/{len(RESULTS)} checks passed")
    if bad:
        print("FAILED: " + ", ".join(bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
