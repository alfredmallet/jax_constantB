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

## v3: energy pins and the Poincare map (2026-08-24)

New: bordered energy-pin rows in `solver_mu.py`, `poincare.py` +
`poincare_map.py`, `--pin-top` in `grow.py`, `tests/test_v3.py`; binding
spec `docs/SPEC_v3.md`, algorithm note section 7.2.

### Energy pins beat frozen coefficients

v1 `freeze` holds a bin's VALUE. That is the wrong instrument for growth: the
held bin cannot participate in cancelling the residual its own presence
creates, so the solver buys feasibility with a spectral cascade. Measured on
one seed and path (36^2x72, random key 1, three top modes):

    v1 freeze,     eps 0.269:  maxgrad 2.45   Galerkin tail 1.2e-3
    unpinned,      eps 0.269:  maxgrad 0.80   tail 1.7e-6
    energy pins,   eps 0.290:  maxgrad 0.87   tail 3.8e-6   pin rel err 8e-14
    energy pins,   eps 0.356:  maxgrad 1.11   tail 8.3e-6

i.e. the pins cost essentially nothing while the freeze costs three orders of
magnitude of tail. (maxgrad here is grow.py's max |grad B|_F; test_v2's
max_ij |d_j B_i| is ~1.4x smaller on these states.) The energy row
e_j(B) = 0.25 sum_x |G_j|^2 with G_j = 2 P_{k_j} B leaves the PHASE free,
which is the whole difference: the mode keeps its prescribed amplitude and
the solver still gets to choose where to put it. Three non-coplanar pins with
nonzero energy exclude 1D and planar states outright (algorithm.tex
Lemma 7.2), so "genuinely 3D" becomes provable rather than hopeful.

### SUM units are load-bearing

e_j is a SUM over grid points, not a volume mean:

    e_j = sum_x |P_j B|^2 = vol * < |P_j B|^2 >,     vol = Nx*Ny*Nz.

That is the same inner product the CG uses (`.sum()`), so the pin rows, the
right-hand side and the nu-block preconditioner are mutually consistent and
the bordered operator is exactly symmetric (tests/test_v3.py case 2:
1e-14 relative defect). Mixing a volume mean into any one of them was the
prototype's ONE bug and it does not announce itself as a units error -- it
appears as pins that converge slowly or not at all. Two consequences the
drivers must respect:

  - targets handed to `gn(pin_targets=...)` are in SUM units;
  - they are GRID DEPENDENT. `grow.py` recomputes c_j(eps) = eps^2 e_j(seed)
    from the rebuilt seed on every ladder rung and never carries a schedule
    across a refinement -- carrying one from 16^2x32 to 24^2x48 would be
    wrong by vol_new/vol_old = 3.375. Both sides of the logged ratio
    e_j/c_j scale with vol, so `pin_err` itself is grid independent
    (measured 1e-13 on both sides of a refinement; test_v3 case 9).

### No activation threshold, and no --pin-after

`--freeze-top` needs `--freeze-after` (default 0.2) because near the uniform
start B.dB ~ Bbar.dB, so a residual bin at a frozen k is uncancellable and the
first steps are infeasible. An energy row has no such trap: it is one scalar
equation whose target starts at zero and grows as eps^2, satisfied by the
push itself to leading order. Pinned growth from eps = 0 was validated
(8 steps, pin residual 1e-13 throughout), so no `--pin-after` exists. The
bordered rows are re-linearised every sweep, which is what makes the pin
residual converge quadratically with the q-residual rather than lagging it.

`--freeze-top` stays, documented as deprecated for growth, because it is how
the cascade above was measured and it is the cheaper instrument when a
coefficient (not an energy) is genuinely what one wants held.

### CSV schema change

`grow.py` rows gain a trailing `pin_err` column (max_j |e_j/c_j - 1|, 0.0
without pins). v2 CSVs are therefore not appendable; the existing
fresh-state-implies-fresh-CSV guard already refuses to mix them.

### descent.py rebuilt on the solver's border

The experimental SQP descent no longer carries its own bordered machinery. It
imports the solver's operator, pin selectors and preconditioner and solves the
SAME bordered system with the SQP right-hand side (Tq + 1, 2 e_j), then
retracts with a PINNED `gn` whose targets are the entry energies -- so the
pinned energies come back exactly (1e-13) instead of to the O(alpha^2)
accuracy of the tangency, and there is exactly one definition of which bins a
pin owns.

### Poincare puncture maps

`poincare.trace` integrates dr/ds = B/|B| in ARC LENGTH by RK4, vmapped over
lines and `lax.scan`ned over steps, with the field trilinearly interpolated on
a `refine`x spectrally zero-padded grid. Positions stay unwrapped, as in
legacy `fieldlines.trace_lines` (untouched). Two error scales, both worth
stating on any figure:

  - interpolation, O((dx/refine)^2) -- THE dominant error, and the only one
    worth spending on: a 2x pad costs 8x memory and divides the error by 4;
  - integration. RK4's formal O(h^4) is NOT observed and cannot be: the
    trilinear interpolant is C^0, with a kink at every cell face, so a step
    crossing a face carries O(h^2) local error and the measured order is
    0.6-1.2 (test_v3 case 7). At h = 0.02 the integrator error is already
    ~100x below the interpolation bound, so accuracy is bought with `refine`,
    never with smaller h.

Memory is the design constraint: 512 lines x 300 transits is ~1.5e5 steps, so
`trace_punctures` carries the position through chunked scans, extracting each
chunk's crossings on the host and dropping the positions (`trace`, which
returns the whole trajectory, is for tests and short runs). Crossings are
located by cubic interpolation in arc length through the four straddling
samples -- linear interpolation would floor the accuracy at O(h^2) and become
the bottleneck.

Validation of the tracer is analytic where possible: for B = (a cos z,
a sin z, c) the lines integrate in closed form and every puncture map is the
IDENTITY, so the measured drift is pure error (1e-7, some 1e4 below the
interpolation bound, because for that field the interpolated chord's
DIRECTION error is third order and `bdir` normalises). On a real state the
check is the drift theorem: a unit-speed volume-preserving flow has
<dr/ds> = Bbar, so lines seeded uniformly in the VOLUME advance along Bbar by
|Bbar| per unit arc -- measured 0.842987 vs 0.843098 (1.3e-4) on the s = 0,
eps = 10.05 state. NOTE the projection: that state's mean field is tilted
12.7 degrees (Bbar_x = -0.186), so the z-advance is Bbar_z = 0.8224, NOT
|Bbar|; testing dz/ds against |Bbar| fails by 2.5% for a perfectly correct
tracer.
