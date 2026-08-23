# DESIGN — constantB package

Numerical design rationale for the `constantB/` package. The code says *what*;
this file says *why*, so the hard-won rules don't have to live in 90-line
docstrings. The frozen numpy reference (`../constantB_tools.py`, project root)
remains the spec; `legacy/` holds the pre-rewrite single-file versions.

## The problem and the solver

Find B on the periodic box with div B = 0 and |B| = 1 pointwise, continued in
amplitude eps from a 1D arc-polarised carrier plus a 3D seed. The system is
underdetermined (2 constraints, 3 unknowns/point), so each Gauss–Newton sweep
takes the **minimum-norm** correction: solve (J Jᵀ)(λ, μ) = −F by matrix-free
CG, update B += W⁻² Jᵀ(λ, μ). With the Sobolev weight W⁻² = (1+k²)⁻ˢ (s =
`smooth`) the correction minimises ‖(1+k²)^(s/2) dB‖₂ instead of L² — the
smoothest correction cancelling the linearised residual. Weighted vs plain
paths reaching different-roughness members at equal amplitude is the paper's
central interpretive point.

## Resolution honesty (do not regress)

1. **Galerkin mode is the honest instrument.** `dealias=True` truncates to the
   strict band |kᵢ| < Nᵢ/3 (STRICT: with the inclusive cutoff, products of two
   edge modes alias exactly onto the band edge and Orszag exactness dies —
   hence the `- 1e-12` guard in `spectral._axis_mask`). All nonlinearities are
   quadratic, so retained-band equations are exact; `Solver.tail_norm` = the
   true continuum tail of |B|²−1 (power-preserving alias folding) = the honest
   unresolved burden, measured on the same grid.
2. **Collocation residuals are blind to aliasing.** In `dealias=False` mode
   the only honest check is zero-pad refinement (`refine` command).
3. **Inexact GN in dealias mode — deliberate, empirically forced.** The
   residual driven to zero is the projected Galerkin residual, but the CG
   step model (J, Jᵀ) stays PLAIN collocation. Truncating the operators makes
   J Jᵀ nearly singular along band-edge directions (the adjoint's output falls
   into the discarded band); observed consequence was a ~1e4× residual
   amplification limit cycle. B is kept in-band by truncating it at `gn` entry
   and after every update, not by truncating operators.
4. **Tail-vs-N ladders must ascend** from a state's native resolution
   (descending lets the solver hop branch members — observed at eps=0.98).
5. **Expect linear (~2×/sweep) convergence below the tail scale** in dealias
   mode (inexact Newton). Set `--res-ok` ≈ 10⁻² × `gal_tail_rms`; polishing
   the retained band below the honest tail floor wastes sweeps.

## JAX / GPU architecture

- **One XLA program per solve.** `solver._gn_solve` is a single `jax.jit`
  containing the GN sweep loop and the CG loop as `lax.while_loop`s: the whole
  multi-sweep solve runs on-device (Kaggle P100) with no per-iteration python
  dispatch. `donate_argnums=(0,)` recycles B's buffer.
- **Static vs traced.** Code-path flags (`pcg`, `weighted`, `verbose`,
  `dealias`) are STATIC python bools → at most a handful of compiled variants
  per grid shape, never traced conditionals. Numeric knobs (`sweeps`, `cgit`,
  `tol`) are traced scalars → changing them does not recompile. The CG
  iteration count is threaded through the loop carry because the quest
  drivers consume it (step-size heuristic, conditioning/fold proxy).
- **Float64 is mandatory** (GN tol 1e-10, CG relative stop 1e-26); enabled in
  `constantB/__init__` before anything touches jax. jax-metal has no x64 —
  Mac GPUs are not a target; P100 fp64 throughput is 1:2, fine.
- **Host/device split.** Seed construction (`seeds`), series diagnostics
  (`series`), state I/O, one-shot diagnostics and plotting stay host numpy:
  cold paths, small dense complex LUs (spotty GPU support), matplotlib.
  Arrays cross to jax only at the solver boundary. `zero_pad` is un-jitted on
  purpose: a different static output shape per call would recompile each time.

## Real-FFT hot path (the rewrite's performance change)

All fields are real, so the solver uses rfftn/irfftn over the last three axes,
with 3-component fields transformed in one batched call. Budget per CG
iteration: **8 real transforms unweighted / 11 weighted (+2 pcg)** — the
weighted adjoint's spectral form is reused for the div, and grad/div are
batched — versus 12/18 full complex transforms in the reference formulation.
Measured ~1.9× wall-clock on CPU at identical work; semantics identical
(parity: `tests/parity_check.py`).

**The Nyquist trap** (cost half a day once; don't rediscover it): the
reference computes derivatives as `real(ifftn(1j·k·fftn(f)))`. On even grids
`1j·k` makes the self-conjugate Nyquist planes anti-Hermitian, so their
contribution is purely imaginary and `real()` silently discards it — the
reference derivative operator has Nyquist output ZERO. An rfft round trip
does *not* discard the x/y-axis Nyquist planes. Therefore the derivative
wavenumbers (`Solver.KR`) have Nyquist bins zeroed
(`rfft_wavenumbers(shape, zero_nyquist=True)`), while k²-only quantities
(CG preconditioner, Sobolev weight) keep true Nyquist values, matching the
reference exactly. Symptom if regressed: div residual of a state with
near-Nyquist content wrong by orders of magnitude (mlstate_fine.npz:
8e-2 vs the true 4e-5).

## Preconditioner

`pcg=True` divides the λ-block residual by (k²+1) spectrally (λ-block of
J Jᵀ ≈ −∇²+…); essential at large N (unpreconditioned condition number grows
like k_max²) and for cold/localized starts. μ-block identity.

## Parity policy

Validate numpy ↔ jax ↔ package **at convergence** (converged residual,
maxgrad, tail norms) — never iterate-by-iterate; round-off reordering (rfft
vs fft) makes iterate paths diverge harmlessly. `tests/parity_check.py` runs
all three implementations: seeds bitwise; converged scalars to ~1e-8 rel or
below physical floors; state-file residuals/tails to 1e-8 rel. State files
(.npz) and CSV schemas are byte-compatible with the reference tool; dealias
and collocation runs are NOT mutually resumable (different residual meaning,
different CSV schema).
