# SPEC v3 — bordered energy pins + Poincaré puncture map (2026-08-24)

Binding spec for the v3 additions. Ground rules and conventions of
docs/SPEC_v2.md §0–§1 apply unchanged (float64, rfft layout, strict 2/3 mask,
Nyquist-zeroed derivative wavenumbers, one jitted program per solve, no
donation, host-numpy cold paths, builders touch ONLY their assigned files).
The VALIDATED numerical prototype for the pins is
`/private/tmp/claude-501/-Users-alfy-aw-papers/0a4f64f6-19e2-414d-afb4-6edf91b0180d/scratchpad/pins/pin_proto.py`
— its operator, unit convention, and preconditioner are the spec; deviations
need overseer approval.

## A. Energy pins in `constantB/solver_mu.py` (builder A; EDITS this file)

### Math (validated 2026-08-24)
Pin constraints e_j(B) = 0.25·Σ_x |G_j|² (SUM units — same inner product as
the CG; this consistency is load-bearing, a volume-mean/sum mismatch was the
one prototype bug), where G_j = 2·P_{k_j}B is the conjugate-pair component
field at integer wavevector k_j (rfft bin + Hermitian partner when k_z=0;
k_z<0 triples map to the conjugate representative). Exactly de_j = Σ G_j·dB.
Unknowns (μ, ν∈R^m); update dB = PW(μB + Σ_j ν_j G_j) with PW the existing
W⁻²·mask·Leray(·freeze) map; bordered normal operator via
D = PW(μB + Σν_j G_j):  A_μ = T(B·D),  A_ν_i = Σ_x G_i·D.
Symmetric/SPD on the band subspace (all CG iterates stay in band: rhs, A
output, and preconditioner are band-projected). RHS = (−Tq, c_j − e_j).
Preconditioner: μ-block T·(1+k²)^s (unchanged); ν-block divide by
d_j = Σ G_j·PW(G_j) (compute once per sweep; guard max(d_j, 1e-300)).
G_j and e_j are recomputed at every sweep (re-linearisation ⇒ quadratic
convergence of the pin residual, measured).

### API
- `MuSolver(shape, smooth=0.0, fix_mean=False, freeze=(), pins=())` —
  `pins`: iterable of integer triples, static; build a `(m, …)` stacked
  rfft-layout selector array at init (validate in-band as for freeze;
  ValueError otherwise; `pins` and `freeze` may coexist — freeze bins must
  not overlap pin bins, ValueError if they do). m=0 ⇒ identical compiled
  program behaviour to v2 (no border arrays traced through).
- `gn(B, sweeps=8, cgit=800, tol=1e-12, verbose=False, pin_targets=None)` —
  `pin_targets`: length-m array in SUM units, TRACED (changing targets must
  not recompile). Convergence test: max(res_q, max_j |e_j − c_j|/scale) with
  scale = max(c_j, 1) — document. Return unchanged (B, res, ci) with res the
  q-residual; expose `pin_error(B, targets)` helper.
- `pinned_energies(B)` → length-m numpy array (SUM units).
- Module docstring: document the SUM-units convention prominently and the
  relation e_sum = vol·⟨|P_j B|²⟩.

### Tests (into tests/test_v3.py, builder C, but builder A must sanity-run
equivalents in the scratchpad before finishing)
1. m=0 regression: gn output bit-identical (≤1e-15) to the pre-edit solver
   on the standard 32²×64 case (KEEP a copy of the old module in the
   scratchpad to diff against, or compare against tests/test_v2.py's
   recorded behaviour — v2's 31 checks must still pass unchanged).
2. Operator symmetry: ⟨x, Ay⟩ = ⟨Ax, y⟩ to 1e-13 for random band-limited
   (μ,ν) pairs, with pins alone, freeze alone, and both.
3. Pin conservation + quadratic convergence, LITERAL reference (prototype,
   36²×72, random_seed kmax=4 key=1, pins = top_modes(b,3) =
   [(3,-2,0),(-3,-2,1),(-4,0,0)], schedule c_j = ε²·e_j(seed), de schedule
   [.03,.03,.039,.039,.0507,.0507,.0507,.06591]):
   at ε=0.290: maxgrad ≈ 0.87, gtail_rms ≈ 3.8e-6, q-res ≤ 1e-14,
   pin rel err ≤ 1e-12. MUST be far from the v1-freeze cascade reference
   (maxgrad 2.45, gtail 1.2e-3 at ε=0.269) and near the unpinned control
   (0.80, 1.7e-6 at 0.269). Tolerances: maxgrad within [0.75, 1.0],
   gtail < 1e-5 at that step.
4. kz=0 pin Hermitian partner handled (energy of (3,-2,0) invariant under
   building the solver with (-3,2,0) instead).

## B. `constantB/poincare.py` + root driver `poincare_map.py` (builder B)

### Tracer (jax)
- `trace(B, seeds, h=0.02, n_steps, refine=2)` — RK4 along the unit field
  direction, trilinear interpolation on the `refine`× zero-padded grid
  (build interpolant arrays once, host-side zero_pad then device;
  document interp error O((Δx/refine)²)). `lax.scan` over steps, vmapped
  over lines; float64. Positions UNWRAPPED (continuous), like legacy
  `fieldlines.trace_lines` (which remains untouched).
- `punctures(traj, z0=np.pi)` — upward crossings of z ≡ z0 (mod 2π):
  detect sign change of (z − plane) between steps, linear interpolation to
  the plane, return per-line ragged list / padded array of (x, y) mod 2π
  and crossing indices. Handle grazing/duplicate crossings (a step that
  lands within 1e-12 of the plane counts once).
- `rotation_number(punct_xy, center)` — average angular advance per
  puncture about `center`, with unwrapped angle differences.
- `poincare_map.py` CLI: `--state X.npz --lines 512 --transits 300
  --z0 pi --seed-mode ring|uniform|reversed --out fig_poincare.png`
  → figure (puncture scatter, coloured per line, equal-aspect, both a
  full-plane panel and a zoom on the structure) + companion .npz of
  punctures + a printed summary (lines, punctures/line, mean z-advance
  per transit vs |B̄|, transverse dispersion per transit, chaotic fraction
  by a simple nearest-neighbour-spread criterion — document the criterion).

### Tests (builder B sanity-runs; formalised by builder C)
5. Uniform B=ẑ: punctures are fixed points to interp tolerance.
6. Analytic 1D circularly polarised state B = (s cos(A sin z + φ0),
   s sin(...), c)-type or simpler B = (a cos z, a sin z, c)/|·|: compare
   puncture drift against direct high-accuracy ODE integration (scipy or
   fine-h reference) — agreement to the documented interp error.
7. h-refinement: puncture positions converge O(h⁴); refine=1 vs 2 differ
   by less than the O((Δx)²) estimate.
8. Mean z-advance per unit arclength over all lines ≈ |B̄| (the drift
   theorem) on the s=0 ε=10.05 state (available at
   scratchpad/kag2/state.npz) to 1%.

## C. Integration (builder C): grow.py, descent.py, tests/test_v3.py, docs

- grow.py: `--pin-top m` (v2 energy pins; the old `--freeze-top` REMAINS,
  documented as deprecated for growth, with a pointer to the cascade
  finding). Pins from `top_modes(seed, m)`; schedule c_j(ε) = ε²·e_j(seed)
  computed from the CURRENT grid's seed each rung (grid-independent for
  random seeds; recompute after refinement); store pins + e_j(seed) in
  meta; resume restores. CSV gains `pin_err` (max_j |e_j/c_j − 1|) — extend
  CSV_FIELDS and bump the stale-CSV guard note. `--pin-after` NOT needed
  (validated from ε=0) — do not add it.
- descent.py: replace the private bordered machinery with the solver's
  pins (`MuSolver(..., pins=...)` + `gn(pin_targets=e_current)`), keeping
  `sqp_step`'s API; the freeze/fix_mean guard stays.
- tests/test_v3.py: cases 1–8 above plus a mini pinned grow.run
  end-to-end (16²×32, kmax=3, pin-top 3, 4 steps: pin_err column < 1e-10)
  — all ≤5 min CPU, tempfile I/O, PASS/FAIL prints, nonzero exit on fail.
  Also RUN tests/test_v2.py and require 31/31 (regression gate).
- docs deltas: DESIGN.md v3 subsection (energy pins: the SUM-units rule,
  the cascade contrast numbers, why no pin-after; poincare: tracer design,
  interp error); algorithm.tex §7.2 updated from "proposed" to
  "implemented" with the measured cascade-vs-pins table (builder C edits
  ONLY §7.2 and the abstract's pin sentence).

## Reference numbers (for all builders; measured this session)
- v1 freeze cascade at ε=0.269 (36²×72): maxgrad 2.45, gtail 1.2e-3,
  inertia 0.62/0.46.
- Unpinned control: ε=0.269 → maxgrad 0.80, gtail 1.7e-6; ε=0.335 → 1.02,
  4.2e-6.
- Energy-pin prototype: ε=0.290 → 0.87, 3.8e-6, pin rel 8e-14; ε=0.356 →
  1.11, 8.3e-6.
- s=0 endpoint (96²×192, ε=11.73): maxgrad 18.02, tail 3.1e-4/1.5e-2.
- s=1 endpoint (54²×108, ε=13.04): maxgrad 5.36, tail 4.5e-5/8.7e-4.
- Existing states for tests: scratchpad/kag2/state.npz (s=0, ε=10.05,
  80²×162), scratchpad/kagS/growS1.npz (s=1, ε=13.04).


## Post-build addenda (overseer, 2026-08-24)
- Test 7's O(h^4) requirement is RETIRED: the C^0 trilinear interpolant caps
  the observed order in h at ~1 (builder C measurement, reviewer-confirmed
  order ~3 for the cubic crossing on exact samples); accuracy is bought with
  `refine` (second order, measured). The shipped test asserts convergence and
  order != 4. Cubic (not linear) crossing interpolation is RATIFIED.
- Review fixes applied: M1 (independent fine-grid schedule check in test 9),
  M2/m3 (pin_targets validated >= 0; absolute-regime stop-test trap
  documented), m4/m5/m6 (poincare docstrings), m7 (descent relative energy
  floor), m8 (bitwise reference pinned to f2f2891), m10 (CSV schema-drift
  warning). m11 (tangential touch counts as one puncture) accepted as
  documented behaviour.
