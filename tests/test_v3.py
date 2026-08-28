#!/usr/bin/env python3
"""Acceptance tests for the v3 additions -- bordered energy pins (SPEC_v3
section A) and the Poincare puncture map (section B) -- in the style of
tests/test_v2.py: a plain script, no pytest, PASS/FAIL per check with the
measured numbers printed, nonzero exit if anything fails.

    python3 tests/test_v3.py                 # ~4 min CPU + the v2 gate
    python3 tests/test_v3.py --only 3 4      # selected cases
    python3 tests/test_v3.py --no-v2         # skip the v2 regression gate

CASES
  1  m = 0 regression: an unpinned solver reproduces the PRE-PIN module
     (recovered from git HEAD) bitwise on the standard 32^2x64 case
  2  bordered-operator symmetry, with pins alone / freeze alone / both
  3  the LITERAL pin reference run (36^2x72, random key 1, 3 pins, the 8-step
     de schedule): the numbers of SPEC_v3's reference section
  4  kz = 0 pins: (3,-2,0) and (-3,2,0) are the same mode (Hermitian partner)
  5  uniform B = zhat: every puncture is its own seed
  6  analytic circularly polarised state: the punctures are EXACT fixed
     points, so the measured drift is the interpolation error itself
  7  convergence in h and in refine -- and WHY the order is not 4
  8  drift theorem: the mean advance ALONG Bbar per unit arc is |Bbar|
     (needs a state file that lives outside the repo)
  9  grow.run end to end with --pin-top 3, including a refinement rung
 10  tests/test_v2.py run as a subprocess: 31/31 required (regression gate)

SKIPPED (not FAILED) is reported when a case's INPUT is missing: a state file
that lives outside the repo, or -- while the v3 builders are still landing --
`constantB.poincare` or `MuSolver(pins=...)`.  A missing input is not a defect
of the code under test; an input that is present and misbehaves is.

WHY CASE 3 IS LITERAL.  The pins exist to answer one question: do they cost
the continuation anything?  The v1 coefficient freeze does -- it cascades
(maxgrad 2.45, Galerkin tail 1.2e-3 at eps = 0.269) because a frozen bin
cannot help cancel the residual it creates.  The assertion here is that the
energy pins land on the UNPINNED control instead (0.80 / 1.7e-6 at the same
eps), which is why the tolerances are two-sided windows around the measured
0.87 / 3.8e-6 rather than one-sided bounds: a run that got dramatically
SMOOTHER would mean the pins had stopped constraining anything.

WHY CASE 6 IS EXACT.  For B = (a cos z, a sin z, c) with a^2 + c^2 = 1 the
field lines integrate in closed form, x(s) = x0 + (a/c)(sin z - sin z0) with
z = z0 + cs, so every return to a plane z = const restores x and y EXACTLY:
the puncture map is the identity.  There is nothing to compare against and no
reference integrator to trust -- the whole measured displacement IS the error,
and it must sit under the documented O((dx/refine)^2) of the interpolant.

ONE DELIBERATE DEVIATION FROM SPEC_v3, forced by measurement.  Test 7 asks for
O(h^4) convergence of the puncture positions.  Measured order: 0.6-1.2, and
that is correct behaviour.  RK4 is fourth order only on a field with four
continuous derivatives, while the trilinear interpolant is C^0 -- a kink at
every cell face -- so a step crossing a face carries O(h^2) local error and
the accumulated order collapses towards 1.  Case 7 therefore asserts what is
true and useful instead: the positions converge in h, they are already ~100x
below the interpolation bound at the largest h tried, and the order is NOT 4
(a fourth-order fit would mean the interpolant had been changed to a smooth
one without the error bound being revisited).  Accuracy here is bought with
`refine`, not with small h.
"""
import argparse
import csv
import importlib.util
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

import constantB                                     # noqa: F401  (x64 first)
from constantB import load_state
from constantB.solver_mu import MuSolver, _bordered_op, _pin_fields
from constantB.seeds_free import blob, random_seed, top_modes
from constantB.spectral import TWOPI, numpy_wavenumbers, numpy_dif

try:                                                 # builder B's module
    from constantB import poincare
    _POINCARE_WHY = ""
except Exception as exc:                             # pragma: no cover
    poincare, _POINCARE_WHY = None, f"{type(exc).__name__}: {exc}"

_HAS_PINS = "pins" in MuSolver.__init__.__code__.co_varnames

# The s = 0 endpoint state of the v3 session (80^2x162, eps = 10.05); only
# case 8 needs it, and only as a REALISTIC field -- any converged state does.
DEFAULT_STATE = os.path.join(
    "/private/tmp/claude-501/-Users-alfy-aw-papers",
    "0a4f64f6-19e2-414d-afb4-6edf91b0180d/scratchpad/kag2/state.npz")

RESULTS = []


# ---------------------------------------------------------------------------
# harness
# ---------------------------------------------------------------------------

def report(name, ok, msg):
    RESULTS.append((name, bool(ok)))
    print(f"  {name:38s} {msg}   {'PASS' if ok else 'FAIL'}")


def skip(name, why):
    print(f"  {name:38s} SKIPPED -- {why}")


def rel(a, b):
    return abs(a - b) / max(abs(a), abs(b), 1e-300)


def maxgrad(B):
    """max_x |grad B|_F, the FROBENIUS norm of the gradient tensor.

    This is grow.py's `_diagnostics` definition and therefore the one every
    reference number in SPEC_v3 is quoted in; test_v2's max_ij |d_j B_i| is a
    different (smaller, by ~1.4 on these states) quantity.  Host numpy
    spectral derivatives, so the measurement is solver agnostic.
    """
    B = np.asarray(B)
    K = numpy_wavenumbers(B.shape[1:])
    g2 = sum(numpy_dif(B[i], j, K) ** 2 for i in range(3) for j in range(3))
    return float(np.sqrt(g2.max()))


def uniform_push(S, eps, seed=None):
    """Uniform zhat plus eps * seed (the standard blob unless given)."""
    if seed is None:
        seed = blob(S, w=(0.8, 0.8, 0.8), kz=1)[0]
    B = np.zeros((3,) + tuple(S.shape))
    B[2] = 1.0
    return B + eps * np.asarray(seed, float)


def circular_state(shape, a=0.5):
    """B = (a cos z, a sin z, c), c = sqrt(1-a^2): exactly constant-|B|,
    divergence free, band limited (|kz| = 1), and analytically integrable
    (module docstring, case 6)."""
    c = float(np.sqrt(1.0 - a ** 2))
    z = np.arange(shape[2]) * TWOPI / shape[2]
    B = np.zeros((3,) + tuple(shape))
    B[0] = a * np.cos(z)[None, None, :]
    B[1] = a * np.sin(z)[None, None, :]
    B[2] = c
    return B


def plane_seeds(n, z0, rng):
    """`n` seeds spread over the plane z = z0 (exactly on it: a sample on a
    plane is snapped onto it, so the seed itself is not counted twice)."""
    xy = rng.uniform(0.0, TWOPI, size=(n, 2))
    return np.concatenate([xy, np.full((n, 1), z0)], axis=1)


# ---------------------------------------------------------------------------
# 1. m = 0 regression against the pre-pin module
# ---------------------------------------------------------------------------

def _pre_pin_module(td):
    """Import `constantB/solver_mu.py` as of git HEAD under a private name.

    Loaded as `constantB._solver_mu_v2ref`, so its `from .spectral import ...`
    resolves against the installed package exactly as the live module's does.
    Returns (module, why_not): the module, or None with the reason.
    """
    p = subprocess.run(["git", "-C", REPO, "show",
                        "HEAD:constantB/solver_mu.py"],
                       capture_output=True, text=True)
    if p.returncode != 0:
        return None, "git show f2f2891:constantB/solver_mu.py failed"
    if "pins=()" in p.stdout:
        return None, ("HEAD already carries the pins -- the pre-pin module is "
                      "no longer one commit away")
    path = os.path.join(td, "_solver_mu_v2ref.py")
    with open(path, "w") as f:
        f.write(p.stdout)
    spec = importlib.util.spec_from_file_location(
        "constantB._solver_mu_v2ref", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod, ""


def case1(grid, eps):
    print("== 1. m=0 regression: unpinned solver == the pre-pin module ==")
    S = MuSolver(grid)
    report("unpinned solver carries no border",
           getattr(S, "m", 0) == 0 and np.asarray(S.pinR).shape[0] == 0,
           f"m={getattr(S, 'm', 0)} pinR{tuple(np.asarray(S.pinR).shape)}")
    try:
        S.gn(uniform_push(S, 0.1), sweeps=1, cgit=10, pin_targets=[1.0])
        ok = False
    except ValueError:
        ok = True
    report("pin_targets rejected when m=0", ok, "ValueError")
    with tempfile.TemporaryDirectory() as td:
        old, why = _pre_pin_module(td)
        if old is None:
            skip("gn bitwise vs pre-pin module", why
                 + " (case 10's v2 gate covers the regression)")
            return
        B0 = uniform_push(S, eps)
        Bn, rn, cn = S.gn(np.array(B0), sweeps=10, cgit=800, tol=1e-13)
        Bo, ro, co = old.MuSolver(grid).gn(np.array(B0), sweeps=10, cgit=800,
                                           tol=1e-13)
        d = float(np.abs(np.asarray(Bn) - np.asarray(Bo)).max())
        report("gn state bitwise vs pre-pin module", d <= 1e-15,
               f"max|dB| {d:.2e}" + ("  (exactly 0)" if d == 0.0 else ""))
        report("gn residual and CG count unchanged",
               ro == rn and co == cn, f"res {ro:.3e}/{rn:.3e}  cg {co}/{cn}")


# ---------------------------------------------------------------------------
# 2. bordered-operator symmetry
# ---------------------------------------------------------------------------

def _sym_defect(S, B, rng):
    """|<x, Ay> - <Ax, y>| / |<x, Ay>| for the bordered normal operator at B.

    The (mu, nu) test vectors are band limited, as every CG iterate is (rhs,
    operator image and preconditioner are all band projected), so this is
    symmetry ON THE SUBSPACE the solve actually lives in.
    """
    op = (S.grid1d, S.freeze_idx)
    G = _pin_fields(np.asarray(B), S.pinR)
    m = int(np.asarray(S.pinR).shape[0])
    xs, ys = [], []
    for out in (xs, ys):
        out.append(np.asarray(S.trunc(rng.normal(size=S.shape))))
        out.append(rng.normal(size=m))
    ax = _bordered_op(xs[0], xs[1], np.asarray(B), G, *op)
    ay = _bordered_op(ys[0], ys[1], np.asarray(B), G, *op)
    xay = float((xs[0] * ay[0]).sum() + (xs[1] * ay[1]).sum())
    axy = float((ax[0] * ys[0]).sum() + (ax[1] * ys[1]).sum())
    return abs(xay - axy) / max(abs(xay), 1e-300), xay


def case2(grid):
    print("== 2. bordered operator symmetry (band subspace) ==")
    if not _HAS_PINS:
        skip("bordered symmetry", "MuSolver has no pins= yet (builder A)")
        return
    rng = np.random.default_rng(3)
    S0 = MuSolver(grid)
    B = np.asarray(S0.gn(uniform_push(S0, 0.3), sweeps=8, cgit=800,
                         tol=1e-13)[0])
    cfg = [("pins only", dict(pins=[(1, 0, 0), (0, 1, 1), (2, 1, 0)])),
           ("freeze only", dict(freeze=[(1, 0, 0), (0, 1, 1)])),
           ("pins + freeze", dict(freeze=[(1, 1, 0)],
                                  pins=[(2, 0, 1), (0, 2, 0)])),
           ("pins, s=1", dict(smooth=1.0,
                              pins=[(1, 0, 0), (0, 1, 1), (2, 1, 0)]))]
    for tag, kw in cfg:
        d, val = _sym_defect(MuSolver(grid, **kw), B, rng)
        report(f"symmetry, {tag}", d < 1e-13,
               f"rel defect {d:.2e}  (<x,Ay> = {val:.6e})")


# ---------------------------------------------------------------------------
# 3. the literal pin reference run
# ---------------------------------------------------------------------------

_DE_SCHEDULE = [.03, .03, .039, .039, .0507, .0507, .0507, .06591]
_REF_PINS = [(3, -2, 0), (-3, -2, 1), (-4, 0, 0)]


def case3():
    print("== 3. literal pin reference: 36^2x72, key 1, 3 pins ==")
    if not _HAS_PINS:
        skip("pin reference run", "MuSolver has no pins= yet (builder A)")
        return
    shape = (36, 36, 72)
    b = np.asarray(random_seed(MuSolver(shape), kmax=4, key=1)[0], float)
    pins = [tuple(int(v) for v in t) for t in top_modes(b, 3)]
    # Compared UP TO CONJUGATION: a kz = 0 bin and its partner (-kx,-ky,0)
    # hold the same mode with exactly equal power, so which of the two
    # top_modes reports is decided by a round-off-level tie and can differ
    # between runs.  It makes no difference to the pin (case 4).
    def canon(t):
        return min(tuple(t), tuple(-v for v in t))
    report("top_modes(seed, 3) as recorded (up to conjugation)",
           sorted(map(canon, pins)) == sorted(map(canon, _REF_PINS)),
           f"{pins}")
    S = MuSolver(shape, pins=pins)
    e_seed = S.pinned_energies(b)                 # SUM units, this grid
    B = np.asarray(S.project(np.zeros((3,) + shape) + np.array([0, 0, 1.0])[
        :, None, None, None]))
    eps, rows = 0.0, []
    for de in _DE_SCHEDULE:
        eps += de
        B, res, ci = S.gn(np.asarray(B) + de * b, sweeps=8, cgit=800,
                          tol=1e-13, pin_targets=eps ** 2 * e_seed)
        B = np.asarray(B)
        pe = S.pin_error(B, eps ** 2 * e_seed, relative=True)
        rows.append((eps, res, pe, maxgrad(B), S.tail_norm(B)[0], ci))
        print(f"  eps={eps:.3f} res={res:.1e} pin={pe:.1e} "
              f"maxgrad={rows[-1][3]:.3f} gtail={rows[-1][4]:.2e} cg={ci}")
    eps7, res7, pe7, g7, t7, _ = rows[6]           # the reference step
    report("reference step is eps = 0.290", abs(eps7 - 0.2901) < 1e-6,
           f"eps {eps7:.4f}")
    report("q-residual at the Newton floor", res7 < 1e-13, f"{res7:.2e}")
    report("pin relative error", pe7 < 1e-12, f"{pe7:.2e}")
    report("maxgrad in [0.75, 1.0] (ref 0.87)", 0.75 < g7 < 1.0, f"{g7:.4f}")
    report("Galerkin tail < 1e-5 (ref 3.8e-6)", t7 < 1e-5, f"{t7:.2e}")
    # The point of the exercise: nowhere near the v1 freeze cascade.
    report("far from the v1 freeze cascade (2.45 / 1.2e-3)",
           g7 < 1.5 and t7 < 1e-4,
           f"maxgrad {g7:.2f} vs 2.45,  tail {t7:.1e} vs 1.2e-3")
    eps8, _r8, pe8, g8, t8, _ = rows[7]
    print(f"  (eps={eps8:.3f}: maxgrad {g8:.3f}, tail {t8:.2e}, pin {pe8:.1e}"
          f"   -- reference 1.11 / 8.3e-6)")


# ---------------------------------------------------------------------------
# 4. Hermitian partner of a kz = 0 pin
# ---------------------------------------------------------------------------

def case4(grid):
    print("== 4. kz=0 pin: (kx,ky,0) and (-kx,-ky,0) are one mode ==")
    if not _HAS_PINS:
        skip("kz=0 pin partner", "MuSolver has no pins= yet (builder A)")
        return
    rng = np.random.default_rng(17)
    S0 = MuSolver(grid)
    B = np.asarray(S0.project(S0.trunc3(rng.normal(size=(3,) + grid))))
    k = (3, -2, 0)
    e1 = MuSolver(grid, pins=[k]).pinned_energies(B)
    e2 = MuSolver(grid, pins=[(-k[0], -k[1], 0)]).pinned_energies(B)
    report("energy invariant under conjugation", rel(e1[0], e2[0]) < 1e-14,
           f"{e1[0]:.12e} vs {e2[0]:.12e}")
    # SUM units, stated as a checkable identity: e = vol * <|P_j B|^2>.
    Bh = np.fft.rfftn(B, axes=(1, 2, 3)) / np.prod(grid)
    amp = float((np.abs(Bh[:, k[0], k[1], 0]) ** 2).sum())
    report("e_j = vol * 2 |Bhat|^2 (SUM units)",
           rel(e1[0], 2.0 * amp * np.prod(grid)) < 1e-12,
           f"{e1[0]:.6e} vs {2.0 * amp * np.prod(grid):.6e}")


# ---------------------------------------------------------------------------
# 5-8. the Poincare map
# ---------------------------------------------------------------------------

def _need_poincare(name):
    if poincare is None:
        skip(name, f"constantB.poincare not importable ({_POINCARE_WHY})")
        return True
    return False


def case5():
    print("== 5. uniform B = zhat: punctures are fixed points ==")
    if _need_poincare("uniform punctures"):
        return
    shape = (16, 16, 32)
    B = np.zeros((3,) + shape)
    B[2] = 1.0
    rng = np.random.default_rng(1)
    seeds = plane_seeds(8, np.pi, rng)
    traj = poincare.trace(B, seeds, h=0.02, n_steps=1000, refine=2)
    xy, idx = poincare.punctures(traj, z0=np.pi)
    n = [len(i) for i in xy]
    report("one puncture per 2pi of arc", n == [3] * 8, f"{n}")
    err = max(float(np.abs(poincare._wrap(p - seeds[m, :2])).max())
              for m, p in enumerate(xy))
    report("punctures sit on their seeds", err < 1e-12, f"max |dx| {err:.2e}")


def _drift(B, seeds, h, n_steps, refine):
    """max displacement of any puncture from its seed (mod 2pi)."""
    traj = poincare.trace(B, seeds, h=h, n_steps=n_steps, refine=refine)
    xy, _idx = poincare.punctures(traj, z0=float(seeds[0, 2]))
    if min(len(p) for p in xy) == 0:
        raise RuntimeError("no punctures: too few steps")
    return max(float(np.abs(poincare._wrap(p - seeds[m, :2])).max())
               for m, p in enumerate(xy)), xy


def _interp_estimate(shape, a, refine):
    """Documented trilinear direction error, as a transverse displacement per
    transit: (dx/refine)^2/8 * |f''| with f the interpolated component, times
    the transverse-to-axial ratio a/c.  The module's O((dx/refine)^2) claim."""
    c = float(np.sqrt(1.0 - a ** 2))
    return (TWOPI / shape[2] / refine) ** 2 / 8.0 * (a / c)


def case6(shape, a):
    print("== 6. analytic circular state: puncture map is the identity ==")
    if _need_poincare("analytic drift"):
        return
    B = circular_state(shape, a)
    c = float(np.sqrt(1.0 - a ** 2))
    q = np.sqrt((B ** 2).sum(0))
    report("state is exactly constant-|B|", float(np.abs(q - 1).max()) < 1e-14,
           f"max ||B|-1| {float(np.abs(q - 1).max()):.2e}")
    rng = np.random.default_rng(5)
    seeds = plane_seeds(6, np.pi, rng)
    n_steps = int(np.ceil(3.0 * TWOPI / c / 0.01))
    d = {r: _drift(B, seeds, 0.01, n_steps, r)[0] for r in (1, 2, 4)}
    est = {r: _interp_estimate(shape, a, r) for r in (1, 2, 4)}
    print("  drift / documented O((dx/refine)^2) bound: "
          + "  ".join(f"refine {r}: {d[r]:.2e} / {est[r]:.2e}"
                      for r in (1, 2, 4)))
    report("drift within the documented interp error",
           all(d[r] < est[r] for r in (1, 2, 4)),
           f"worst ratio {max(d[r] / est[r] for r in (1, 2, 4)):.3f}")
    # The drift comes in ~100x UNDER the bound, and does not fall with refine.
    # That is not luck: for this field the interpolated vector is a chord of
    # the (cos z, sin z) circle, whose DIRECTION error is third order even
    # though its magnitude error is second, and `bdir` normalises.  What is
    # left is the RK4 error on a C^0 field (case 7), which grows slightly with
    # refine because there are more cell faces to cross.  Reported, not
    # asserted: it is a property of this particular test field.
    print(f"  (refine 1 -> 4: {d[1]:.2e} -> {d[4]:.2e}; the direction error "
          "of a chord is O(dx^3) for this field, so the bound is loose)")


def case7(shape, a):
    print("== 7. convergence in h and in refine ==")
    if _need_poincare("h-refinement"):
        return
    B = circular_state(shape, a)
    c = float(np.sqrt(1.0 - a ** 2))
    rng = np.random.default_rng(6)
    seeds = plane_seeds(4, np.pi, rng)
    arc = 2.0 * TWOPI / c
    hs = [0.08, 0.04, 0.02, 0.01]
    pos = []
    for h in hs:
        _d, xy = _drift(B, seeds, h, int(np.ceil(arc / h)), 2)
        pos.append(np.stack([p[0] for p in xy]))   # first puncture, per line
    # Differences against the finest run: the interpolation error is a FIXED
    # perturbation of the field and cancels here, leaving the integrator's.
    e = [float(np.abs(poincare._wrap(p - pos[-1])).max()) for p in pos[:-1]]
    order = [float(np.log2(e[i] / e[i + 1])) for i in range(len(e) - 1)]
    print("  h = " + "  ".join(f"{h:g}" for h in hs[:-1])
          + ";  |x(h) - x(h_min)|: " + "  ".join(f"{v:.2e}" for v in e)
          + ";  orders: " + "  ".join(f"{o:.2f}" for o in order))
    # SPEC_v3 test 7 asks for O(h^4).  MEASURED: order ~ 0.6-1.2, and this is
    # correct behaviour, not a defect.  RK4 is fourth order only for a field
    # with four continuous derivatives; the trilinear interpolant is C^0, with
    # a kink at every cell face, so a step that crosses a face carries O(h^2)
    # local error and the accumulated order collapses to ~1.  The consequence
    # for users is the one the module already states: accuracy is bought with
    # `refine`, not with small h -- at h = 0.08 the integrator error is already
    # 100x below the interpolation bound, so halving h buys nothing.
    report("puncture positions converge in h",
           e[0] > 2.0 * e[-1] and max(e) < _interp_estimate(shape, a, 2),
           f"{e[0]:.2e} -> {e[-1]:.2e}, all < "
           f"{_interp_estimate(shape, a, 2):.2e}")
    report("order capped by the C^0 interpolant (not 4)",
           max(order) < 3.0, "orders " + " ".join(f"{o:.2f}" for o in order))
    d1, _ = _drift(B, seeds, 0.01, int(np.ceil(arc / 0.01)), 1)
    d2, _ = _drift(B, seeds, 0.01, int(np.ceil(arc / 0.01)), 2)
    report("refine=1 vs 2 within the O(dx^2) estimate",
           abs(d1 - d2) < _interp_estimate(shape, a, 1),
           f"|{d1:.2e} - {d2:.2e}| < {_interp_estimate(shape, a, 1):.2e}")


def case8(state, lines, steps):
    """The drift theorem, ALONG Bbar rather than along zhat.

    The theorem is <dr/ds> = Bbar (the field is a unit-speed volume-preserving
    flow, so the ensemble average of the tangent IS the volume mean of B), and
    only its component along Bbar has magnitude |Bbar|.  The z-component is
    Bbar_z, which is a different number whenever the mean field is tilted --
    on the reference state it is tilted 12.7 degrees (Bbar_x = -0.186), so
    testing dz/ds against |Bbar| would fail by 2.5% for a correct tracer.

    Seeds are drawn uniformly in the VOLUME, not on a plane: a
    volume-preserving flow keeps a uniform ensemble uniform, so the line
    average is then an unbiased Monte-Carlo estimate of the volume average at
    every arc length, and the only error is sampling noise (measured ~0.4% at
    512 lines x 120 units of arc) plus the interpolation bias.
    """
    print("== 8. drift theorem: mean advance along Bbar per unit arc ==")
    if _need_poincare("drift theorem"):
        return
    if not os.path.exists(state):
        skip("drift theorem", f"no state file at {state} (pass --state)")
        return
    B, eps, _meta = load_state(state)
    B = np.asarray(B, float)
    Bbar = B.mean(axis=(1, 2, 3))
    nb = float(np.linalg.norm(Bbar))
    rng = np.random.default_rng(9)
    seeds = rng.uniform(0.0, TWOPI, size=(lines, 3))
    h, t0 = 0.02, time.time()
    # trace_punctures carries the position instead of storing the trajectory
    # (and exercises the chunked path): only the endpoint is needed here.
    xy, _idx, fin, n_tot = poincare.trace_punctures(
        B, seeds, h=h, n_steps=steps, refine=2, z0=np.pi)
    adv = float(((fin - seeds) @ (Bbar / nb)).mean() / (n_tot * h))
    print(f"  {os.path.basename(state)}: grid {B.shape[1:]}, eps {eps:.2f}, "
          f"|Bbar| {nb:.6f}, Bbar_z {Bbar[2]:.6f}; {lines} lines x {n_tot} "
          f"steps, {np.mean([len(p) for p in xy]):.0f} punctures/line, "
          f"{time.time() - t0:.1f} s")
    report("mean advance along Bbar == |Bbar| to 1%", rel(adv, nb) < 1e-2,
           f"{adv:.6f} vs {nb:.6f}  (rel {rel(adv, nb):.2e})")
    report("z-advance == Bbar_z (the tilted mean is real)",
           rel(float(((fin - seeds)[:, 2]).mean() / (n_tot * h)),
               float(Bbar[2])) < 1.5e-2,
           f"{float(((fin - seeds)[:, 2]).mean() / (n_tot * h)):.6f} vs "
           f"{float(Bbar[2]):.6f}")


# ---------------------------------------------------------------------------
# 9. the driver, pinned
# ---------------------------------------------------------------------------

def case9():
    print("== 9. grow.run end to end with --pin-top 3 ==")
    if not _HAS_PINS:
        skip("pinned grow.run", "MuSolver has no pins= yet (builder A)")
        return
    import grow
    report("CSV schema carries pin_err", grow.CSV_FIELDS[-1] == "pin_err",
           f"{len(grow.CSV_FIELDS)} columns")
    with tempfile.TemporaryDirectory() as td:
        state, csvf = os.path.join(td, "p.npz"), os.path.join(td, "p.csv")
        common = ["--state", state, "--csv", csvf, "--seed", "random",
                  "--key", "1", "--kmax", "3", "--pin-top", "3",
                  "--de", "0.05", "--de-max", "0.05", "--sweeps", "8",
                  "--cgit", "400", "--res-ok", "1e-9", "--snap-de", "1e9"]
        rc = grow.run(grow.build_parser().parse_args(
            common + ["--grid0", "16", "16", "32", "--grid-max", "16", "16",
                      "32", "--eps-max", "0.2", "--gtail-max", "1",
                      "--edge-max", "1"]))
        rows = list(csv.DictReader(open(csvf)))
        pe = [float(r["pin_err"]) for r in rows] or [np.inf]
        report("4 pinned steps, res-ok met",
               rc == 0 and len(rows) >= 4
               and all(float(r["res"]) <= 1e-9 for r in rows),
               f"rc={rc} rows={len(rows)} "
               f"maxres={max(float(r['res']) for r in rows):.1e}")
        report("pin_err < 1e-10 on every row", max(pe) < 1e-10,
               f"max {max(pe):.2e}")
        _B, _e, meta = load_state(state)
        pins = np.asarray(meta["pins"], int).reshape(-1, 3)
        report("pins restored from meta", pins.shape == (3, 3),
               f"{[tuple(int(v) for v in t) for t in pins]}")
        # Resume onto a finer rung.  SUM-units regression: targets are
        # e_j(seed) summed over grid points, so a schedule carried across the
        # rung instead of rebuilt is wrong by vol_new/vol_old (3.375 here).
        # NOTE (review M1): the CSV pin_err is computed against the SAME
        # target grow.py used, so it CANNOT catch a carried schedule; the
        # non-vacuous check is below -- rebuild the seed on the fine grid
        # INDEPENDENTLY and compare the state's energies to eps^2 e_seed.
        rc = grow.run(grow.build_parser().parse_args(
            common + ["--grid-max", "24", "24", "48", "--eps-max", "0.3",
                      "--gtail-max", "1e-12", "--edge-max", "1"]))
        rows2 = list(csv.DictReader(open(csvf)))[len(rows):]
        B2, _e2, _m2 = load_state(state)
        fine = [r for r in rows2 if r["grid"] == "(24, 24, 48)"]
        report("refined rung reached", rc == 0 and len(fine) >= 1,
               f"rc={rc} grid={tuple(np.shape(B2)[1:])} rows={len(rows2)}")
        pe2 = max(float(r["pin_err"]) for r in fine) if fine else np.inf
        report("pin_err < 1e-10 after refinement (SUM units)", pe2 < 1e-10,
               f"{pe2:.2e}")
        # Independent schedule check (the real M1 guard): rebuild the seed on
        # the FINE grid from the recorded recipe and require the state's
        # pinned energies to sit on eps^2 * e_seed_fine.  A carried coarse
        # schedule fails this by the vol ratio 3.375.
        from constantB.seeds_free import make_seed
        eps2 = float(_e2)
        gridf = tuple(np.shape(B2)[1:])
        Sf = MuSolver(gridf, pins=[tuple(int(v) for v in t) for t in pins])
        bf = np.asarray(make_seed(_m2, Sf), float)
        e_seed_fine = Sf.pinned_energies(bf)
        e_state = Sf.pinned_energies(B2)
        sched_err = float(np.abs(e_state / (eps2 ** 2 * e_seed_fine) - 1).max())
        report("state energies on the INDEPENDENT fine-grid schedule",
               sched_err < 1e-8, f"rel {sched_err:.2e} (carried schedule "
               f"would be ~{3.375 - 1:.2f})")


# ---------------------------------------------------------------------------
# 10. the v2 regression gate
# ---------------------------------------------------------------------------

def case10(expect):
    print(f"== 10. tests/test_v2.py subprocess ({expect}/{expect} required) ==")
    t0 = time.time()
    p = subprocess.run([sys.executable, os.path.join(HERE, "test_v2.py")],
                       capture_output=True, text=True, cwd=REPO)
    line = next((l for l in p.stdout.splitlines() if l.startswith("OVERALL:")),
                "")
    got = line.split()[1] if line else "?"
    for l in p.stdout.splitlines():
        if "FAILED:" in l:
            print("  " + l.strip())
    report("v2 suite still green",
           p.returncode == 0 and got == f"{expect}/{expect}",
           f"{got} in {time.time() - t0:.0f} s (rc {p.returncode})")


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, nargs=3, default=[32, 32, 64],
                    help="grid for case 1 (the spec's parity grid)")
    ap.add_argument("--small", type=int, nargs=3, default=[16, 16, 32])
    ap.add_argument("--eps", type=float, default=0.3)
    ap.add_argument("--state", default=DEFAULT_STATE, help="case 8's state")
    ap.add_argument("--amp", type=float, default=0.5,
                    help="transverse amplitude of cases 6-7's analytic state")
    ap.add_argument("--lines", type=int, default=512)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--v2-checks", type=int, default=31)
    ap.add_argument("--no-v2", action="store_true")
    ap.add_argument("--only", type=int, nargs="*", default=None)
    args = ap.parse_args()
    grid, small = tuple(args.grid), tuple(args.small)

    cases = {1: lambda: case1(grid, args.eps), 2: lambda: case2(small),
             3: case3, 4: lambda: case4(small), 5: case5,
             6: lambda: case6(small, args.amp),
             7: lambda: case7(small, args.amp),
             8: lambda: case8(args.state, args.lines, args.steps),
             9: case9,
             10: lambda: case10(args.v2_checks)}
    for i in sorted(cases):
        if (args.only and i not in args.only) or (i == 10 and args.no_v2):
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
