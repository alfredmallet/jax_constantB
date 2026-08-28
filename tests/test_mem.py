#!/usr/bin/env python3
"""Acceptance tests for the memory refactor of the mu-form solver (SPEC_mem),
in the style of tests/test_v2.py: a plain script, no pytest, PASS/FAIL per
check with the measured numbers printed, nonzero exit on any failure.

    python3 tests/test_mem.py               # ~3-4 min CPU
    python3 tests/test_mem.py --only 1 2    # selected cases
    python3 tests/test_mem.py --grid 64     # cheaper memory/timing rung

CASES
  1  the SPEC_mem section-R literal reference runs (s0 / s1 / pinned / freeze)
  2  the separable Grid1D reproduces the full multiplier arrays it replaced,
     Nyquist split included
  3  the freeze scatter: the empty (0,3) case traces, and a frozen bin (with
     its kz = 0 Hermitian partner) is held
  4  memory_analysis at 96^3: grid-array persistent + argument bytes, and the
     phase-B temp gate
  5  wall time at 96^3 (PRINTED, not asserted -- see below)
  6  gn(donate=True): same answer, and the donated jax buffer is consumed
  7  the packing isometry (phase B): round trip, inner product, kz = 0 mirror
  8  CG equivalence (phase B): the packed CG is the real-space CG

WHY CASE 1 IS LITERAL.  The refactor rebuilds the mask, k^2, the Sobolev
weights, 1/k^2 and the Leray projector as fused broadcasts of 1D axis factors
instead of reading precomputed O(N^3) grids.  That is an arithmetic identity,
not an approximation, so the only honest test is that converged states come
back unchanged.  They are not bitwise equal -- XLA fuses the new expressions
differently, so the last bits of each CG iterate differ and the difference is
amplified by ~600 CG iterations near the 1e-15 residual floor -- hence the
relative 1e-9 window on the converged observables and the +-5% window on the
CG count, both far tighter than any semantic change could hide in.

WHY CASE 5 DOES NOT ASSERT.  The speed gate (SPEC_mem G3, <= 1.03x) compares
against a baseline measured on the SAME machine in the SAME session with the
PRE-refactor code, which no longer exists in the tree.  A number baked in here
would drift with the host, so the time is printed and the comparison is the
integrator's, from its own before/after logs.

WHY CASE 8 DOES NOT DEMAND AN IDENTICAL CG COUNT EVERYWHERE.  Packing is an
isometry, so the packed CG is the real-space CG in exact arithmetic -- but the
two accumulate different last bits, and the CG stops at ||r||^2 below
1e-26 ||r0||^2 -- the residual floor itself.  A solve that takes ~600
iterations to reach that floor crosses it one or two iterations apart
(measured: 580 vs 579, 412 vs 408); a solve that reaches it in ~100 crosses it
on the same iteration.  So the
count is gated EXACTLY on the short case and within the +-5% of SPEC_mem G2 on
the long one, while the residual SEQUENCE -- where round-off has not yet had
hundreds of iterations to amplify -- is gated at 1e-10 relative.
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

import constantB                                     # noqa: F401  (x64 first)
import jax
import jax.numpy as jnp

from constantB.solver_mu import (MuSolver, _gn_solve_mu, _mask, _wm2, _w2,
                                 _PW, _pack, _unpack, _precond_vec, _minv,
                                 _normal_op, _normal_op_p, _tq)
from constantB.spectral import (rfft_wavenumbers, dealias_mask_rfft,
                                axis_wavenumbers_1d, numpy_wavenumbers,
                                numpy_dif, rfft3, trunc, trunc3)

RESULTS = []


# ---------------------------------------------------------------------------
# harness
# ---------------------------------------------------------------------------

def report(name, ok, msg):
    RESULTS.append((name, bool(ok)))
    print(f"  {name:40s} {msg}   {'PASS' if ok else 'FAIL'}")


def rel(a, b):
    return abs(a - b) / max(abs(a), abs(b), 1e-300)


def maxgrad(B):
    """max_ij |d_j B_i| -- host numpy spectral derivatives, so the measurement
    is independent of whatever the solver carries."""
    B = np.asarray(B)
    K = numpy_wavenumbers(B.shape[1:])
    return max(float(np.abs(numpy_dif(B[i], j, K)).max())
               for i in range(3) for j in range(3))


# ---------------------------------------------------------------------------
# 1. the section-R reference runs
# ---------------------------------------------------------------------------

# Seed construction is EXACT (SPEC_mem section R): numpy's Generator is
# platform-deterministic, so these four runs are reproducible bit for bit up
# to the solver's own fusion choices.
R_SHAPE = (48, 48, 48)
R_BBAR = (-1.002818852690543e-04, 2.825848806652549e-04, 7.998737447366335e-01)
R_TARGETS = [3.895503616319754e-01, 3.639783550967165e-01]
R_CASES = [
    # tag, MuSolver kwargs, ci, maxgrad, tail_rms, tail_max
    ("s0", dict(smooth=0.0, fix_mean=True), 48,
     1.647872055250897e+01, 1.278353807886349e-01, 7.952766890689158e-01),
    ("s1", dict(smooth=1.0, fix_mean=True), 636,
     1.713486023469276e+01, 1.259333636931522e-01, 9.356704328424028e-01),
    ("pinned", dict(smooth=1.0, fix_mean=True, pins=[(14, 5, 10), (3, 7, 2)]),
     624,
     1.712697720148289e+01, 1.259309767726271e-01, 9.532851426591975e-01),
    ("freeze", dict(smooth=0.0, fix_mean=True, freeze=[(1, 0, 1), (0, 2, 0)]),
     48,
     1.653362113815959e+01, 1.278277190995687e-01, 7.995942054603421e-01),
]


def _r_seed():
    rng = np.random.default_rng(7)
    B0 = np.zeros((3,) + R_SHAPE)
    B0[2] = 0.8
    B0 += 0.12 * rng.standard_normal((3,) + R_SHAPE)
    return B0


def case1():
    print("== 1. SPEC_mem section-R literal reference runs (48^3) ==")
    B0 = _r_seed()
    for tag, kw, ci_ref, g_ref, trms_ref, tmax_ref in R_CASES:
        S = MuSolver(R_SHAPE, **kw)
        B = np.asarray(S.project(B0))
        tgt = None
        if S.m:
            e0 = S.pinned_energies(B)
            tgt = 1.1 * e0
            report(f"{tag}: 1.1 x entry == spec targets",
                   max(rel(float(a), b) for a, b in zip(tgt, R_TARGETS)) < 1e-12,
                   f"{[float(v) for v in tgt]}")
        if tag == "freeze":                      # (1,0,1): kz != 0, no partner
            h0 = np.fft.rfftn(B, axes=(1, 2, 3))[:, 1, 0, 1].copy()
        Bf, res, ci = S.gn(B, sweeps=10, cgit=2000, tol=1e-13,
                           **({} if tgt is None else dict(pin_targets=tgt)))
        Bf = np.asarray(Bf)
        g, (trms, tmax) = maxgrad(Bf), S.tail_norm(Bf)
        bb = Bf.mean(axis=(1, 2, 3))
        report(f"{tag}: converged", res < 1e-13, f"res {res:.2e}")
        report(f"{tag}: ci within 5%", rel(ci, ci_ref) < 0.05,
               f"{ci} vs {ci_ref}")
        report(f"{tag}: maxgrad", rel(g, g_ref) < 1e-9,
               f"rel {rel(g, g_ref):.2e}  ({g:.12f})")
        report(f"{tag}: tail_rms", rel(trms, trms_ref) < 1e-9,
               f"rel {rel(trms, trms_ref):.2e}  ({trms:.12e})")
        report(f"{tag}: tail_max", rel(tmax, tmax_ref) < 1e-9,
               f"rel {rel(tmax, tmax_ref):.2e}  ({tmax:.12e})")
        db = float(np.abs(bb - np.asarray(R_BBAR)).max())
        report(f"{tag}: Bbar (fix_mean)", db < 1e-12, f"max|dBbar| {db:.2e}")
        if S.m:
            pe = S.pin_error(Bf, tgt)
            report(f"{tag}: pin_error", pe < 1e-12, f"{pe:.2e}")
        if tag == "freeze":
            h1 = np.fft.rfftn(Bf, axes=(1, 2, 3))[:, 1, 0, 1]
            d = float(np.abs(h1 - h0).max())
            report(f"{tag}: frozen bin (1,0,1) held", d < 1e-11, f"{d:.2e}")


# ---------------------------------------------------------------------------
# 2. Grid1D == the arrays it replaced
# ---------------------------------------------------------------------------

def case2(shapes):
    print("== 2. Grid1D reproduces the full multiplier arrays ==")
    for shape in shapes:
        for s in (0.0, 1.0, 2.5):
            S = MuSolver(shape, smooth=s)
            g = S.grid1d
            kd = jnp.stack(jnp.broadcast_arrays(g.kdx, g.kdy, g.kdz))
            kt = jnp.stack(jnp.broadcast_arrays(g.ktx, g.kty, g.ktz))
            k2t = (1.0 + g.ktx ** 2 + g.kty ** 2 + g.ktz ** 2)
            checks = [
                ("KR (derivative k)", kd, S.KR),
                ("true k", kt, rfft_wavenumbers(shape)),
                ("K2r", k2t - 1.0, S.K2r),
                ("maskR", _mask(g), S.maskR),
                ("invK2", _invk2(g), S.invK2),
                ("Wm2r", _wm2(g), S.Wm2r),
                ("W2r", _w2(g), S.W2r),
            ]
            worst = max(float(jnp.abs(a - b).max()) for _n, a, b in checks)
            report(f"{shape} s={s}: multipliers", worst == 0.0,
                   f"max|delta| {worst:.1e} over "
                   + ", ".join(n for n, _a, _b in checks))
        # The Nyquist split (spectral.py module docstring): derivative axes
        # are ZERO on |k| = N/2, the weights' axes are not.  Crossing them
        # changes div residuals by orders of magnitude (test_v2 case 4).
        nyq = [float(jnp.abs(a).max())
               for a in axis_wavenumbers_1d(shape, zero_nyquist=True)]
        tru = [float(jnp.abs(a).max()) for a in axis_wavenumbers_1d(shape)]
        ok = all(nd < tt if n % 2 == 0 else nd == tt
                 for n, nd, tt in zip(shape, nyq, tru))
        report(f"{shape}: Nyquist zeroed in kd only", ok,
               f"max|kd| {nyq} vs max|kt| {tru}")


def _invk2(g):
    k2 = g.kdx ** 2 + g.kdy ** 2 + g.kdz ** 2
    return jnp.where(k2 > 0, 1.0 / jnp.where(k2 > 0, k2, 1.0), 0.0)


# ---------------------------------------------------------------------------
# 3. the freeze scatter
# ---------------------------------------------------------------------------

def case3(shape):
    print("== 3. freeze as a scatter (empty case included) ==")
    S = MuSolver(shape)                          # no freeze, no fix_mean
    report("unfrozen solver carries a (0,3) index",
           tuple(np.asarray(S.freeze_idx).shape) == (0, 3)
           and np.asarray(S.freeze_idx).dtype == np.int32,
           f"{tuple(np.asarray(S.freeze_idx).shape)} "
           f"{np.asarray(S.freeze_idx).dtype}")
    rng = np.random.default_rng(5)
    v = jnp.asarray(rng.normal(size=(3,) + tuple(shape)))
    out = np.asarray(jax.jit(_PW)(v, S.grid1d, S.freeze_idx))
    report("empty scatter traces and runs", np.all(np.isfinite(out)),
           f"max|PW v| {np.abs(out).max():.3e}")

    # (1,0,0) has kz = 0, so its Hermitian partner (-1,0,0) lives in the SAME
    # rfft half-array and must be zeroed with it (module docstring: 1e-3 leak).
    Sf = MuSolver(shape, fix_mean=True, freeze=[(1, 0, 0), (0, 1, 1)])
    idx = {tuple(int(v) for v in b) for b in np.asarray(Sf.freeze_idx)}
    want = {(0, 0, 0), (1, 0, 0), (shape[0] - 1, 0, 0), (0, 1, 1)}
    report("freeze bins incl. kz=0 partner", idx == want,
           f"{sorted(idx)}")
    fr = np.asarray(Sf.freezeR)                  # the compatibility array
    got = {tuple(int(v) for v in b) for b in zip(*np.nonzero(fr == 0.0))}
    report("freeze_idx == zeros of freezeR", got == idx, f"{len(got)} bins")
    PWv = np.asarray(jax.jit(_PW)(v, Sf.grid1d, Sf.freeze_idx))
    h = np.fft.rfftn(PWv, axes=(1, 2, 3))
    worst = max(float(np.abs(h[:, b[0], b[1], b[2]]).max()) for b in want)
    report("PW output has no frozen content", worst < 1e-12, f"{worst:.2e}")


# ---------------------------------------------------------------------------
# 4-5. memory and speed at the gate resolution
# ---------------------------------------------------------------------------

def _gate_state(N):
    shape = (N, N, N)
    S = MuSolver(shape, smooth=1.0, fix_mean=True)
    rng = np.random.default_rng(3)
    B0 = np.zeros((3,) + shape)
    B0[2] = 0.9
    B0 += 0.05 * rng.standard_normal((3,) + shape)
    return S, jnp.asarray(np.asarray(S.project(B0))), float(np.prod(shape))


# Phase-A temp at 96^3 (this file's own case 4, measured on the integrated
# phase-A tree, jax 0.10.0 CPU x64).  SPEC_mem G4 asks phase B for at least
# 25 B/pt off it and no growth anywhere else.  A compiler upgrade can move
# this number; it is a gate on OUR code, so re-measure before relaxing it.
A_TEMP = 130.01
B_TEMP_MAX = A_TEMP - 25.0


def case4(N):
    print(f"== 4. memory_analysis at {N}^3 ==")
    S, B, pt = _gate_state(N)
    band = sum(np.asarray(a).nbytes for a in S.band)
    persistent = sum(np.asarray(a).nbytes for a in S.grid1d) \
        + np.asarray(S.freeze_idx).nbytes + band
    args = (B, S.grid1d, S.freeze_idx, S.band, 2, 30, 1e-30)
    ma = _gn_solve_mu.lower(*args, verbose=False).compile().memory_analysis()
    grid_args = ma.argument_size_in_bytes - np.asarray(B).nbytes
    nb = _pack(rfft3(B[0]), S.band).size
    print(f"  persistent solver arrays  {persistent / pt:8.4f} B/pt "
          f"({persistent} bytes total, of which Band {band})")
    print(f"  argument                  {ma.argument_size_in_bytes / pt:8.4f}"
          f" B/pt  (of which B {np.asarray(B).nbytes / pt:.1f})")
    print(f"  output                    {ma.output_size_in_bytes / pt:8.2f} B/pt")
    print(f"  temp                      {ma.temp_size_in_bytes / pt:8.2f} B/pt"
          f"  (phase A {A_TEMP})")
    print(f"  packed CG vector          {nb} doubles = {nb * 8 / pt:.4f} B/pt"
          f"  ({100.0 * nb * 8 / (8 * pt):.1f}% of a real field)")
    grid = (persistent + grid_args) / pt
    report("grid arrays <= 2 B/pt", grid <= 2.0,
           f"{grid:.4f} B/pt persistent+argument (was ~69 = 36.8 + 32.7)")
    temp = ma.temp_size_in_bytes / pt
    report("temp down >= 25 B/pt vs phase A", temp <= B_TEMP_MAX,
           f"{temp:.2f} vs {A_TEMP} B/pt (delta {temp - A_TEMP:+.2f})")
    # The output is B plus the two reported scalars, as in phase A.
    report("output is still B + 2 scalars",
           ma.output_size_in_bytes <= 24 * pt + 64,
           f"{ma.output_size_in_bytes} B "
           f"({ma.output_size_in_bytes / pt:.4f} B/pt)")


def case6(shape):
    print("== 6. gn(donate=True) ==")
    rng = np.random.default_rng(9)
    B0 = np.zeros((3,) + tuple(shape))
    B0[2] = 1.0
    B0 += 0.3 * rng.standard_normal((3,) + tuple(shape))
    for tag, kw, tgt_from in (("v2", dict(smooth=1.0, fix_mean=True), None),
                              ("pinned", dict(smooth=1.0, fix_mean=True,
                                              pins=[(1, 0, 1)]), "entry")):
        S = MuSolver(shape, **kw)
        B = np.asarray(S.project(B0))
        kwargs = {} if tgt_from is None else \
            dict(pin_targets=S.pinned_energies(B))
        Bp, rp, cp = S.gn(np.array(B), sweeps=6, cgit=400, tol=1e-13, **kwargs)
        Bd, rd, cd = S.gn(np.array(B), sweeps=6, cgit=400, tol=1e-13,
                          donate=True, **kwargs)
        d = float(np.abs(np.asarray(Bp) - np.asarray(Bd)).max())
        report(f"{tag}: donate=True is bitwise identical",
               d == 0.0 and rp == rd and cp == cd,
               f"max|dB| {d:.1e}  res {rp:.2e}/{rd:.2e}  cg {cp}/{cd}")
        # A numpy input is always safe; a jax input handed with donate=True is
        # consumed, and any later use of it must raise (solver_mu.gn docstring).
        Bj = jnp.asarray(B)
        S.gn(Bj, sweeps=1, cgit=10, tol=1e-13, donate=True, **kwargs)
        try:
            float(jnp.abs(Bj).max())
            gone = False
        except Exception:
            gone = True
        report(f"{tag}: donated jax input invalidated", gone,
               "later use raises" if gone else "still readable")


def case5(N, reps):
    print(f"== 5. wall time at {N}^3 (printed, not asserted) ==")
    S, B, _pt = _gate_state(N)
    args = (B, S.grid1d, S.freeze_idx, S.band, 2, 30, 1e-30)
    jax.block_until_ready(_gn_solve_mu(*args, verbose=False))
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        jax.block_until_ready(_gn_solve_mu(*args, verbose=False))
        ts.append(time.perf_counter() - t0)
    print(f"  2 sweeps x 30 cgit: median {sorted(ts)[len(ts) // 2]:.3f} s "
          f"of {reps}  [" + " ".join(f"{t:.3f}" for t in ts) + "]")


# ---------------------------------------------------------------------------
# 7. the packing isometry
# ---------------------------------------------------------------------------

def _n_canonical(shape):
    """Independent count of the canonical packed coordinates: every in-band
    rfft bin with kz > 0 gives 2 (Re, Im), the kz = 0 plane gives one of each
    Hermitian pair (2 each), DC gives 1.  Counted off the mask array, not off
    the Band's own geometry."""
    m = np.asarray(dealias_mask_rfft(shape)) > 0
    n_plane = int(m[:, :, 0].sum())
    return 2 * int(m[:, :, 1:].sum()) + (n_plane - 1) + 1


def case7(shapes):
    print("== 7. the packing isometry (phase B) ==")
    for shape in shapes:
        S = MuSolver(shape)
        bd, g = S.band, S.grid1d
        rng = np.random.default_rng(17)
        n = _pack(rfft3(jnp.zeros(shape)), bd).size
        report(f"{shape}: packed length", n == _n_canonical(shape),
               f"{n} vs {_n_canonical(shape)} canonical coordinates "
               f"({100.0 * n / np.prod(shape):.1f}% of N^3)")
        worst_rt, worst_ip, worst_dc, worst_pl = 0.0, 0.0, 0.0, 0.0
        for _ in range(4):
            f = jnp.asarray(rng.normal(size=shape))
            h = jnp.asarray(rng.normal(size=shape))
            Tf = trunc(f, _mask(g), True)
            Th = trunc(h, _mask(g), True)
            vf, vh = _pack(rfft3(f), bd), _pack(rfft3(h), bd)
            worst_rt = max(worst_rt,
                           float(jnp.abs(_unpack(vf, bd, shape) - Tf).max()))
            worst_ip = max(worst_ip, rel(float((vf * vh).sum()),
                                         float((Tf * Th).sum())))
            worst_dc = max(worst_dc, abs(float(rfft3(f)[0, 0, 0].imag)))
            # The kz = 0 mirror trap (Band docstring): unpacking writes the
            # conjugate partner back, so the kz = 0 plane of the round trip
            # keeps its FULL content -- dropping the mirror halves it.
            a = np.asarray(rfft3(_unpack(vf, bd, shape))[:, :, 0])
            b = np.asarray((_mask(g) * rfft3(Tf))[:, :, 0])
            worst_pl = max(worst_pl, float(np.abs(a - b).max()))
        report(f"{shape}: unpack(pack(f)) == T f", worst_rt < 1e-15,
               f"max|delta| {worst_rt:.2e}")
        report(f"{shape}: <pack f, pack h> == sum_x Tf.Th", worst_ip < 1e-13,
               f"worst rel {worst_ip:.2e}")
        report(f"{shape}: kz=0 plane survives the round trip",
               worst_pl < 1e-12,
               f"max|delta| {worst_pl:.2e} (halved if the mirror is dropped)")
        report(f"{shape}: DC bin is real", worst_dc < 1e-12,
               f"max|Im F(0,0,0)| {worst_dc:.2e}")


# ---------------------------------------------------------------------------
# 8. the packed CG is the real-space CG
# ---------------------------------------------------------------------------

# First-sweep CG residual norms ||r_i||, i = 0..6, of the PHASE-A real-space
# solve on the case below (captured from the phase-A tree before the packing
# landed: git 402695a + phase A working tree, CPU x64, jax 0.10.0), and its
# per-sweep CG counts.  See the module docstring on why the counts of a
# ~600-iteration solve are gated at 5% and not exactly.
A_CG_NORMS = [17.737719503381243, 17.243706665253264, 17.43428328351975,
              20.06518643956033, 32.32089976901376, 59.16957726827933,
              85.58874817757733]
A_CG_COUNTS = [580, 412, 357, 317]
# The same for a case whose CG reaches the residual floor in ~100 iterations,
# where the two arithmetics still stop on the SAME iteration.
A_SHORT_COUNTS = [136, 75, 69, 70]


def _cg_norms(S, B, n, packed):
    """||r_i|| of the first `n` CG iterations of the first sweep, run in the
    packed coordinates or in real space, from the same start.

    The real-space branch is the phase-A recursion built from the module's own
    `_minv` / `_normal_op`, which the solve no longer calls: the two branches
    are the SAME algorithm in two coordinate systems, so this is an algebraic
    identity check (SPEC_mem B3.3), not a cross-method parity run.
    """
    g, fidx, bd = S.grid1d, S.freeze_idx, S.band
    Bc = trunc3(jnp.asarray(B), _mask(g), True)
    q = _tq(Bc, g)
    if packed:
        w = _precond_vec(g, bd)
        r = -_pack(rfft3(q), bd)
        A = lambda p: _normal_op_p(p, Bc, g, fidx, bd)
        M = lambda x: w * x
    else:
        r = -q
        A = lambda p: _normal_op(p, Bc, g, fidx)
        M = lambda x: _minv(x, g)
    z = M(r)
    rz, p = (r * z).sum(), z
    out = [float(jnp.sqrt((r * r).sum()))]
    for _ in range(n):
        Ap = A(p)
        r = r - (rz / (p * Ap).sum()) * Ap
        z = M(r)
        rz2 = (r * z).sum()
        p, rz = z + (rz2 / rz) * p, rz2
        out.append(float(jnp.sqrt((r * r).sum())))
    return out


def case8():
    print("== 8. the packed CG is the real-space CG (phase B) ==")
    shape = (32, 32, 32)
    S = MuSolver(shape, smooth=1.0, fix_mean=True)
    rng = np.random.default_rng(11)
    B0 = np.zeros((3,) + shape)
    B0[2] = 0.9
    B0 += 0.08 * rng.standard_normal((3,) + shape)
    B = np.asarray(S.project(B0))
    npk = _cg_norms(S, B, 6, True)
    nre = _cg_norms(S, B, 6, False)
    w1 = max(rel(a, b) for a, b in zip(npk, nre))
    w2 = max(rel(a, b) for a, b in zip(npk, A_CG_NORMS))
    report("32^3 s=1: ||r_i|| vs the real-space recursion", w1 < 1e-10,
           f"worst rel {w1:.2e} over 7 iterations")
    report("32^3 s=1: ||r_i|| vs the phase-A literals", w2 < 1e-10,
           f"worst rel {w2:.2e} over 7 iterations")
    ci = [S.gn(np.array(B), sweeps=k, cgit=2000, tol=1e-13)[2]
          for k in range(1, 5)]
    worst = max(rel(a, b) for a, b in zip(ci, A_CG_COUNTS))
    report("32^3 s=1: per-sweep ci within 5% of phase A", worst < 0.05,
           f"{ci} vs {A_CG_COUNTS}")

    # The short case: same iteration count, exactly.
    shape = (16, 16, 32)
    Ss = MuSolver(shape, smooth=0.0, fix_mean=True)
    rng = np.random.default_rng(5)
    B0 = np.zeros((3,) + shape)
    B0[2] = 0.9
    B0 += 0.3 * rng.standard_normal((3,) + shape)
    Bs = np.asarray(Ss.project(B0))
    cis = [Ss.gn(np.array(Bs), sweeps=k, cgit=2000, tol=1e-13)[2]
           for k in range(1, 5)]
    report("(16,16,32) s=0: per-sweep ci IDENTICAL to phase A",
           cis == A_SHORT_COUNTS, f"{cis} vs {A_SHORT_COUNTS}")


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, default=96,
                    help="cube side for the memory/speed cases")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--only", type=int, nargs="*", default=None)
    args = ap.parse_args()

    cases = {1: case1,
             2: lambda: case2([(16, 16, 32), (24, 24, 24), (16, 24, 32)]),
             3: lambda: case3((16, 16, 32)),
             4: lambda: case4(args.grid),
             5: lambda: case5(args.grid, args.reps),
             6: lambda: case6((16, 16, 32)),
             7: lambda: case7([(32, 32, 32), (32, 48, 64)]),
             8: case8}
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
