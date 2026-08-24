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

## v2: the carrier-free, mu-only stack (2026-08)

New modules `potential.py`, `solver_mu.py`, `seeds_free.py`, `spectra.py`,
`descent.py` (experimental), driver `grow.py`; algorithm note
`docs/algorithm.tex`; binding build spec `docs/SPEC_v2.md`. The carrier
stack above is frozen: it reproduces the paper.

### The Schur / vector-potential reduction

Eliminating lambda analytically from the (lambda, mu) normal equations gives
the same minimum-norm step in ONE scalar unknown:

    dB = W^-2 P (mu B),   with   T B . W^-2 P T3 (mu B) = -T q,

P = Leray projector (equivalently: dB = curl dA with the Coulomb-gauge
minimum-norm vector potential). Measured consequences:
  - div B is PRESERVED to round-off by every step, but a pre-existing div
    error is NOT removed: `MuSolver.project` once at continuation entry
    (grow.py does this; gn() deliberately does not).
  - 6 real transforms per CG iteration (+2 pcg) vs 8-11 (+2) coupled.
  - the correct preconditioner is diagonal: T (1+k^2)^s. The legacy
    lambda-block preconditioner does nothing for the mu Schur complement
    (symbol (1+k^2)^-s sin^2 theta(k,B): the sin^2 factor is intrinsic --
    the field-aligned near-null directions of spectral rigidity -- the
    weight factor is what the preconditioner removes). CG converges to
    1e-13 in ~120-370 iterations where the coupled system exhausted every
    budget (cg=599/1499 on every legacy quest CSV row).

### REVISION of rules 2 and 5b (empirical, 2026-08-23)

Rule 2 above ("do not project the operators -- limit cycle") holds for the
COUPLED (lambda, mu) system with budget-truncated CG and stays in force for
the legacy solver. In the mu-only form the EXACT Galerkin operator is
correct and stable: quadratic convergence (5.0e-3 -> 1.3e-5 -> 8.3e-11 ->
8.2e-16 at 64^2x128, eps~1; the exact identity
Tq(B+d) = T(|d|^2)/2 -- docs/algorithm.tex, Thm 4.3, numerically exact to
round-off -- predicts contraction constant ~1/2 when ||d|| ~ ||Tq||;
localised pushes carry a larger prefactor, ~250 for the blob case, without
breaking quadratic order) with converged CG each sweep. Rule 5b's
linear-convergence budgeting (res-ok ~ 1e-2 x tail, salvage sweeps) is
obsolete in v2: the retained band converges to round-off cheaply, the
honest floor is tail_norm alone, and grow.py has NO salvage branch --
a step that misses --res-ok after `sweeps` is genuinely bad.

CAVEAT on legacy parity: the mu-form step solves min ||W dB|| subject to
the GALERKIN linearised constraints, the legacy step uses the collocation
step model against the Galerkin target. Both converge, to NEARBY BUT
DISTINCT members of the solution manifold (max|dB| ~ 5e-4 at s=0; the
quadratic constraint admits zero-residual displacements ~ sqrt(2 res)).
Measured observable differences at the test grid: maxgrad ~4e-3 (s=0) /
~5e-2 (s=1), tail_norm ~6-7e-2 -- REAL differences between the two
algorithms' members, not noise. Consequently tests/test_v2.py cases 1-2
are SMOKE-LEVEL consistency checks (their thresholds must sit above the
legitimate 7% tail discrepancy); the correctness of the mu solver's tails
is carried by the tight invariants of cases 3 (quadratic floor) and 6
(tail_norm == 2x-grid exact tail), not by legacy parity.

### Tail honesty, restated one-sidedly

The discarded-band content of q on the working grid is an ALIAS-FOLDED
image of the true continuum tail: true tail >= measured / sqrt(8) (up to 2
fold partners per axis; coherent cancellation not excluded). The exact
check is the q-spectrum on the 2x zero-padded grid (`spectra.q_spectrum` --
exact because in-band B makes supp q < the padded Nyquist). The old
"power-preserving" phrasing in the legacy docstrings overstates this;
measured fold ratio on test fields is ~1.003 (benign), but v2 documentation
and thresholds use the one-sided form.

### Carrier-free continuation and pins

`grow.py` starts from a uniform mean field plus an arbitrary div-free seed
b = trunc3(curl a) (`seeds_free`): blob, random-phase (coefficients keyed
per integer wavevector -- grid-independent, so ladder rungs rebuild the
IDENTICAL continuum seed to round-off; kmax must sit strictly inside the
grid0 band, enforced), or file. NOTE: blob seeds are NOT band-limited (chi
is analytic), so a blob rebuilt on a finer rung drifts by ~1e-3 at 16->24
and ~1e-14 at 48->72: for strict same-continuum-object ladders use
random/file seeds, or start blob runs at a grid where the drift is below
the tail floor. Mean field: free by default; `fix_mean` freezes the
k=0 bin -- guarded, since |Bbar| = 1 with |B| = 1 forces B == Bbar
(Cauchy-Schwarz), and the k=0 Jacobian row vanishes AT a uniform state, so
fixed-mean starts converge only linearly at first (raise --sweeps, not
--de).

3D pins: a single scalar amplitude pin cannot prevent degeneration to a 1D
circularly polarised state; nonzero pinned energies at THREE non-coplanar
wavevectors exclude 1D and planar states outright (a line through the
origin meets at most one, a plane at most two). v1 = `freeze` masks the
correction off those bins (Hermitian partners on the kz=0 rfft plane
included -- masking one partner of a conjugate pair leaks, measured 1e-3).
Freezing is exactly the constrained min-norm step (SPD survives the
projections; algorithm.tex Prop 3.2). NEAR-UNIFORM TRAP: at B ~ Bbar,
B.dB ~ Bbar.dB, so a residual bin at a frozen k is uncancellable --
grow.py activates pins only at eps >= --freeze-after (default 0.2).
Runtime monitor: k-space inertia eigenvalue ratios lam21/lam31
(rank 1 = 1D, 2 = planar) on every CSV row. v2 pins (bordered energy
rows, phase/polarisation free) live with the experimental smoothest-state
SQP descent in `descent.py`.

### No donation in v2

`Solver.gn`'s donate_argnums deletes the caller's jax array (silent
footgun; legacy drivers survive only because they pass numpy). v2 never
donates; the copy cost is negligible against a multi-sweep solve.
