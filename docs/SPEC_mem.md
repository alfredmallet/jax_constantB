# SPEC_mem — memory refactor of the mu-form solver (2026-08-28)

Goal: cut device memory of `_gn_solve_mu` / `_gn_solve_pinned` so a single
A100-40GB reaches ~620–660³, WITHOUT changing any semantics and WITHOUT
losing speed. Motivating baseline (96³, s=1, fix_mean, CPU x64, measured
2026-08-28 via `.lower().compile().memory_analysis()`):

    persistent solver arrays : 36.8 B/pt   (KR 12, K2r/maskR/invK2/Wm2r/W2r/freezeR ~4 each)
    jit temp                 : 154.0 B/pt
    argument (B) + output    : 48.0 B/pt   (grid-array args alias the persistent buffers)
    timing, 2 sweeps x 30 cgit, 96³ CPU: median 2.911 s

## Hard gates (apply to EVERY phase; a phase that fails any gate is rejected)

G1. `python3 tests/test_v2.py` (31 checks) and `python3 tests/test_v3.py`
    (32 checks; 2 of the file's cases self-skip on any tree that already
    carries `pins=`) pass, unmodified except where this spec explicitly
    says.
G2. Literal reference numbers (section R below) reproduce: converged
    residual reached (<1e-13), maxgrad / tail_rms / tail_max / Bbar match
    to RELATIVE 1e-9, CG iteration count `ci` within ±5% of reference.
G3. SPEED: median-of-3 wall time of `_gn_solve_mu` with (sweeps=2, cgit=30,
    tol=1e-30) at 96³ on the SAME machine ≤ 1.03 × the pre-phase baseline.
    Re-measure the baseline in the same session (thermal drift). A phase
    slower than 3% is rejected outright (A. Mallet, 2026-08-28).
G4. MEMORY: report `memory_analysis()` numbers at 96³ before/after.
    Phase A target: grid-array persistent+argument bytes ≤ 2 B/pt (from
    ~37). Phase B target: temp reduced by ≥ 25 B/pt. Temp must not increase
    in any phase.
G5. Float64 throughout; no mixed precision; strict `|k| < N/3` band
    unchanged; exact-Galerkin semantics unchanged; `solver.py` (legacy)
    untouched; state_io / CSV schemas / driver CLI unchanged.

## R. Literal reference numbers (generated 2026-08-28, current main @402695a,
##    CPU x64, jax 0.10.0)

Seed construction (EXACT — numpy Generator is platform-deterministic):

    shape = (48, 48, 48)
    rng = np.random.default_rng(7)
    B0 = np.zeros((3,)+shape); B0[2] = 0.8
    B0 += 0.12*rng.standard_normal((3,)+shape)
    # per case: B = np.asarray(S.project(B0)); S.gn(B, sweeps=10, cgit=2000, tol=1e-13)

maxgrad = max over the 9 |dif(B_i, axis)| grids; tail = S.tail_norm(B).

| case | config | res (reached) | ci | maxgrad | tail_rms | tail_max |
|---|---|---|---|---|---|---|
| s0 | smooth=0, fix_mean | 2.44e-15 | 48 | 1.647872055250897e+01 | 1.278353807886349e-01 | 7.952766890689158e-01 |
| s1 | smooth=1, fix_mean | 6.34e-16 | 636 | 1.713486023469276e+01 | 1.259333636931522e-01 | 9.356704328424028e-01 |
| pinned | smooth=1, fix_mean, pins=[(14,5,10),(3,7,2)], targets=1.1×entry = [3.895503616319754e-01, 3.639783550967165e-01] | 6.09e-16 | 624 | 1.712697720148289e+01 | 1.259309767726271e-01 | 9.532851426591975e-01 |
| freeze | smooth=0, fix_mean, freeze=[(1,0,1),(0,2,0)] | 3.78e-15 | 48 | 1.653362113815959e+01 | 1.278277190995687e-01 | 7.995942054603421e-01 |

All cases: Bbar = (-1.002818852690543e-04, 2.825848806652549e-04,
7.998737447366335e-01) (matches to 1e-12 abs across cases; fix_mean holds it).
Pinned case additionally: pin_error(Bf, targets) < 1e-12 (reference 3.5e-14).
Freeze case additionally: frozen-bin (1,0,1) rfft value drift < 1e-11
(reference 9.5e-13).

Write these checks as a NEW plain-script test `tests/test_mem.py` in the
style of test_v2.py (PASS/FAIL lines, nonzero exit on failure), created in
Phase A and kept green thereafter. Include in it the memory/speed
measurements (printed, with the memory gate asserted and the speed gate
asserted against a baseline time measured by running the OLD code path is
not possible post-refactor — so print speed and leave the 3% comparison to
the integrator's before/after logs).

## Phase A — separable multipliers, scatter freeze, donation

### A1. 1D axis arrays replace full-grid multiplier arrays
`spectral.py` gains:

    def axis_wavenumbers_1d(shape, zero_nyquist=False):
        """Three float64 arrays shaped (Nx,1,1), (1,Ny,1), (1,1,Nz//2+1):
        the rfft-layout wavenumbers as broadcastable 1D factors. Same
        Nyquist-zeroing rule as rfft_wavenumbers."""

    def axis_masks_1d(shape):
        """Three float64 {0,1} arrays, same shapes: the strict |k| < N/3
        axis masks (reuse _axis_mask)."""

The solver passes a single pytree of small arrays into the jitted programs
(a NamedTuple, e.g. `Grid1D`), containing:

    kdx, kdy, kdz   # derivative wavenumbers, Nyquist-ZEROED (as KR today)
    ktx, kty, ktz   # true wavenumbers incl. Nyquist (as K2r's basis today)
    mx,  my,  mz    # strict 2/3 axis masks
    s               # Sobolev exponent as a traced float64 scalar

In-kernel derived quantities (all fused broadcasts, computed per apply):

    mask   = mx*my*mz                       # only ever as a multiplier
    k2t    = ktx**2 + kty**2 + ktz**2
    W2     = (1.0 + k2t)**s                 # preconditioner weight
    Wm2    = (1.0 + k2t)**(-s)              # step weight
    k2d    = kdx**2 + kdy**2 + kdz**2
    invK2  = where(k2d > 0, 1/where(k2d > 0, k2d, 1), 0)   # as potential.inv_k2
    Leray:   vh - [kdx,kdy,kdz] * ((kdx*vh[0]+kdy*vh[1]+kdz*vh[2]) * invK2)

CRITICAL: keep the derivative-vs-true wavenumber split exactly as today
(the Nyquist trap, module docstring of spectral.py): Leray/derivatives use
kd*, weights use kt*. Getting these crossed changes div residuals by orders
of magnitude — there is a test for it in test_v2 case 4.

If G3 shows the per-apply `**s` pow costing > 1% (unlikely — it is O(N³/2)
elementwise vs O(N³ log N) FFT), fall back to precomputing ONLY W2 and Wm2
as full arrays (8 B/pt) and note it; everything else stays 1D.

### A2. Freeze as scatter
Replace the full `freezeR` multiplier with static index arrays. The
constructor keeps `_freeze_mask`'s VALIDATION logic (via `_bins`,
Hermitian-partner rule included — see the measured 1e-3 leak in the module
docstring) but stores only:

    self.freeze_idx = jnp.asarray([[i,j,k], ...], dtype=int32)  # (n_frozen, 3), may be (0,3)

In `_PW`, after the mask multiply:

    vh = vh.at[:, fi, fj, fk].set(0.0)     # fi/fj/fk = freeze_idx columns

The (0,3) empty case must trace and run cleanly (test it). `fix_mean`
freezes (0,0,0) as today.

### A3. Donation entry
`MuSolver.gn(..., donate=False)` — when True, dispatch to twins of the two
jitted programs wrapped with `donate_argnums` on the B argument (create the
wrapped jits ONCE at module scope; do not re-wrap per call). Docstring must
say: a caller-held jax array passed with donate=True is invalidated; numpy
inputs are always safe (jnp.asarray copies to device). Update `grow.py` to
pass donate=True in its gn calls — it holds the fallback as HOST numpy
(`Bprev = np.array(B)`), so donation is safe there. Do NOT change tests to
donate.

### A4. Compatibility
`MuSolver.KR/K2r/maskR/invK2/Wm2r/W2r/freezeR` become LAZY cached
properties (full arrays built on first access) so external users and
diagnostics keep working; the hot path must not touch them. Audit and fix
all in-repo callers of the changed private function signatures (`_PW`,
`_minv`, `_tq`, `_normal_op`, `_pin_*`, `_project`, `_residual_mu`,
`descent.py`, `diagnostics.py`, `series.py`, whatever grep finds). The
standalone `_*_jit` helpers at solver_mu.py:364 must keep their public
behavior.

## Phase B — packed retained-band CG (SEPARATE builder, lands after A)

Motivation: the CG state vectors (mu, r, z, p) are real N³ fields (8 B/pt
each) whose content lives entirely in the retained band (~15% of rfft
bins); the preconditioner `_minv` burns 2 FFTs per CG iteration to apply a
diagonal. Move the CG to packed band coordinates: state shrinks ~3×, trunc
becomes a no-op, `_minv` becomes an elementwise multiply, and the per-CG-
iteration FFT count drops from 10 scalar-equivalents to 8 → this phase
should be FASTER, not just smaller.

### B1. The packing isometry
Canonical in-band bin set of the rfft layout (host-precomputed index
arrays, static per solver):

  - all in-band bins with kz > 0 (in-band excludes the Nyquist plane, so
    no kz = Nyq subtlety): weight w = 2, both Re and Im packed;
  - kz = 0 plane: the rfft array stores BOTH Hermitian partners
    (kx,ky,0)/(-kx,-ky,0) explicitly. Pack only the canonical half
    (ky > 0) ∪ (ky = 0 ∧ kx > 0), weight w = 2 (Re and Im);
  - DC (0,0,0): weight 1, Re only (assert Im ≈ 0 in tests, do not pack).

Packed vector of a real scalar field f with rfft transform F:

    v = concat over canonical bins b of sqrt(w_b / vol) * [Re F_b, (Im F_b)]
    vol = Nx*Ny*Nz

Then <v, v> = sum_x (T f)² EXACTLY (Parseval; this is the same SUM inner
product the CG and the pin borders use — SUM-units consistency is
load-bearing, see solver_mu docstring). Unpacking scatters the canonical
bins AND their kz = 0 mirror partners (conjugated) into a zeroed rfft
array — the mirror-partner scatter is the same trap the freeze mask
documents; forgetting it halves kz=0-plane content after irfft.

### B2. What changes
- CG state (mu, r, z, p) and rhs live packed. Preconditioner: elementwise
  (1+k2t)^s per packed bin (precompute the packed k2t vector — it is
  O(n_band), not O(N³)).
- Operator per iteration: unpack mu → irfft (1) → mu·B (real, pointwise)
  → `_PW` body (rfft3: 3, diagonals+Leray+freeze scatter, irfft3: 3) →
  ·B sum (pointwise) → rfft (1) → gather canonical bins → pack. The final
  band mask is implicit in the gather. Freeze scatter stays in `_PW`.
- Sweep body: q and the update B ← T(B + PW(mu B)) unchanged in real
  space; r0 = pack(−T q) via one rfft + gather.
- `_gn_solve_pinned`: pack the field block (r1/z1/p1) identically; the nu
  block, borders (A_nu row = sum_x G_i · D, real-space) and the pin
  re-linearisation are UNCHANGED. G stays real-space (out of scope).
- Residual reported (`res_now`) stays max|T q| in REAL space — do not
  replace it with a packed norm (different norm, drivers calibrate on it).

### B3. Phase-B-specific tests (add to tests/test_mem.py)
1. Pack/unpack round trip: for random real f, unpack(pack(f)) == T f to
   1e-15 max-abs (32³ and 32·48·64 anisotropic — test non-cubic!).
2. Inner-product identity: <pack(f), pack(g)> == sum_x (T f)(T g) to
   rel 1e-13, several random pairs.
3. CG-equivalence (developer test, exact-arithmetic identity): on 32³,
   s=1, from the same start, the first 5 CG residual norms of the packed
   solve match the Phase-A solve to rel 1e-10, and total ci per sweep is
   IDENTICAL. (This is an algebraic-equivalence check of the SAME
   algorithm, not an iterate-by-iterate cross-method parity — the
   CLAUDE.md rule forbids the latter, not this.)
4. All section-R gates.

## Phase C — conditional (integrator decides from post-B memory_analysis)
If batched (3,·) FFT liveness still dominates temp, evaluate sequential
per-component transforms behind a static solver flag, gated by G3. Do not
build unless the numbers justify it.

## Build order & roles
1. Builder 1 (Opus): Phase A + tests/test_mem.py. Runs gates G1–G5 itself,
   prints before/after memory_analysis + timing in its report.
2. Integrator (Fable, main session): re-runs all gates independently,
   reviews the diff, fixes or bounces.
3. Builder 2 (Opus): Phase B on the integrated tree. Same protocol.
4. Integrator: gates again; decide Phase C from the numbers.
5. Fresh Fable agent: adversarial review of the full diff with EXECUTED
   reproducers (its own scripts, not the builders' claims): parity numbers,
   the kz=0 mirror trap, empty-freeze/no-pin/anisotropic grids, donation
   safety in grow.py, speed.
6. Fix findings, final gate run, commit.
