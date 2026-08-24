# SPEC v2 — carrier-free constant-|B| stack (2026-08-23)

Binding specification for the v2 modules. The overseer integrates; builders
implement EXACTLY these APIs. Read DESIGN.md first for the numerical
rationale of the existing stack. The validated algorithmic prototype is
`/Users/alfy/aw_papers/mu_solver_proto.py` — its numerics are the spec for
the solver core; deviations require overseer approval.

## 0. Ground rules (non-negotiable)

- Python + JAX, float64 REQUIRED. Never touch
  `jax.config` yourself: every module gets x64 via `from . import` chain
  (package `__init__` runs first). Root-level scripts must
  `import constantB` (or a submodule) before any other jax use.
- All new solver code: rfft layout over the last three axes, batched
  transforms for stacked (3, ...) fields, reuse helpers from
  `constantB/spectral.py` (`rfft3`, `irfft3`, `rfft_wavenumbers`,
  `dealias_mask_rfft`, `zero_pad`). Do NOT reimplement them.
- Strict 2/3 mask everywhere (`dealias_mask_rfft`); v2 is Galerkin-only.
- Derivative wavenumbers = `rfft_wavenumbers(shape, zero_nyquist=True)`
  (the Nyquist trap, DESIGN.md); k²-only quantities (weights,
  preconditioner) use `zero_nyquist=False`.
- Hot paths: ONE jitted program per solve; loop with `lax.while_loop`;
  code-path flags are STATIC python bools; numeric knobs traced. NO
  `donate_argnums` anywhere in v2 (footgun: deleted caller arrays).
- Cold paths (seeds, diagnostics, plotting, I/O): host numpy; jax only at
  the solver boundary. matplotlib imported lazily inside functions, Agg.
- Style: match the repo — top docstring states the *why*, functions have
  focused docstrings, modules ≤ ~300 lines, no dead code, no `print` in
  library modules (drivers may print).
- Do NOT edit: `constantB/__init__.py`, `constantB/solver.py`,
  `constantB/spectral.py`, `constantB/seeds.py`, `DESIGN.md`, `README.md`,
  legacy/, existing drivers. Each builder touches ONLY its assigned files.
- Data files (*.npz, *.csv, *.png) stay gitignored; never write them into
  the repo except under tests' tmp dirs.

## 1. Shared conventions

- Grids `shape = (Nx, Ny, Nz)`, periodic box [0, 2π)³, integer wavenumbers.
- Fields stacked `(3, Nx, Ny, Nz)` real float64. Spectral: rfftn layout
  `(…, Nz//2+1)`.
- "T" = strict 2/3 band projector; "P" = div-free (Leray) projector, k=0
  untouched; "W⁻²" = `(1+k²)^(-s)` Fourier multiplier, `s = smooth`.
- q = (|B|²−1)/2; residual reported = `max|T q|` (the exact Galerkin
  residual); div is monitored, not solved for (see §3).
- rfft Hermitian-partner rule: a bin (kx, ky, kz) with kz=0 (the only
  in-band case; Nz/2 is outside the band) has its conjugate partner
  (−kx, −ky, 0) stored in the SAME half-array. Any per-bin mask (freeze,
  pins) MUST zero/select both partners, else irfftn silently symmetrises
  and the mask leaks.

## 2. `constantB/potential.py` (builder A)

Jit-traceable building blocks (not decorated; called from jitted code) plus
small jitted public wrappers, pattern of `spectral.py`.

```python
def curl(F, KR):            # (3,...) real -> (3,...) real, spectral curl, one batched rfft pair
def divfree_project(F, KR): # Leray projector P; k=0 bins untouched (mean preserved)
def inv_curl(B, KR):        # A with curl A = B - mean(B), Coulomb gauge: Â = i k×B̂/k²; k=0 -> 0
def div(F, KR):             # scalar div, for monitoring
```
All take/return real fields; KR is the caller's derivative wavenumber grid.
`inv_curl(curl(A))` round-trips Coulomb-gauge in-band A to 1e-13.
`curl(inv_curl(B)) + mean(B) == B` for div-free B to 1e-13.

## 3. `constantB/solver_mu.py` (builder A)

```python
class MuSolver:
    def __init__(self, shape, smooth=0.0, fix_mean=False, freeze=()):
```
- `freeze`: iterable of integer triples (kx, ky, kz). Build a `freezeR`
  multiplier array in rfft layout: 1 everywhere, 0 at each frozen bin AND
  its Hermitian partner when kz == 0 (see §1). `fix_mean=True` adds
  (0,0,0). Frozen bins must lie inside the retained band (validate,
  `ValueError` otherwise). If ≥3 non-mean freeze triples are given and ALL
  selections of 3 are coplanar, warn (not error) — helper
  `noncoplanar(freeze) -> bool` exposed for callers.
- Internal operator (from the prototype, EXACTLY — this is the validated
  numerics):
  - `PW(v)`: `vh = rfft3(v); vh = freezeR*mask*(vh − KR·((KR·vh).sum(0)·invK2))`;
    return `irfft3(Wm2*vh)`. `invK2` from the Nyquist-zeroed KR with the
    k=0 (and zeroed-Nyquist) bins mapped to 0.
  - Normal operator `A(mu) = T((B*PW(mu[None]*B)).sum(0))` — exact
    Galerkin Jacobian; quadratic Newton convergence is expected and tested.
  - Preconditioner `Minv(r) = irfft3(W2*mask*rfft3(r))`, `W2=(1+k²)^s`.
  - CG: standard PCG on (mu), stop at `‖r‖² < 1e-26·‖r0‖²` or `cgit`.
  - Sweep: `B ← T(B + PW(mu[None]*B))`.
- Public methods:
  - `gn(B, sweeps=8, cgit=800, tol=1e-12, verbose=False)` → `(B, res, ci)`
    — ONE jitted program (sweep loop + CG loop as `lax.while_loop`), static
    args only for `verbose` (+ shape-derived constants closed over).
    Semantics as legacy `_gn_solve`: entry-converged sweep is a no-op
    preserving previous CG count; returned res recomputed after final
    update if not converged; `res = max|Tq|`. NO donation. B is truncated
    (`T3`) at entry. NOTE: `gn` does NOT project div — callers use
    `project` once (document loudly in the docstring).
  - `project(B)` → div-free + in-band B, mean preserved (uses
    `divfree_project` then trunc; NOT frozen-masked — it is a state
    cleanup, not a step).
  - `residual(B)` → `(div, Tq)` fields (jitted helper).
  - `tail_norm(B)` → same contract as legacy `Solver.tail_norm`.
  - `.shape, .smooth, .KR, .K2r, .maskR, .freezeR` attributes; `.K`
    lazy full-layout wavenumbers (copy the legacy property).
- Also module-level `noncoplanar(triples)`: True iff some 3 of the triples
  have `abs(det) > 0` (integer arithmetic).

## 4. `constantB/seeds_free.py` (builder B)

Host numpy; jax only via the passed solver's helpers. Every seed is
`b = S.trunc3(curl a)` normalised to `max|b| = 1`. Returns numpy.

```python
def from_potential(a, S)                 # a: (3,...) real numpy -> (b, meta_fragment)
def blob(S, w=(0.8,0.8,0.8), kz=1, center=(π,π,π))   # current geometry via from_potential
def random_seed(S, kmax=4, slope=0.0, key=0)
def make_seed(meta, S)                   # rebuild from state metadata on S.shape
def top_modes(b, m=3)                    # m dominant |b̂| integer triples (kz>=0 rep), for --freeze-top
```
- `random_seed`: coefficients for every integer triple 0<|k|≤kmax drawn
  deterministically from `np.random.default_rng(hash((key, kx, ky, kz)) &
  0xffffffff)` (grid-INDEPENDENT: same key ⇒ same continuum field on any
  grid — this is what makes ladder refinement produce the same seed), with
  amplitude ∝ |k|^(−slope), assembled as a complex potential â(k) with
  Hermitian symmetry, then curl. Test: seed built on (24,24,48) equals
  `zero_pad` of seed built on (16,16,32) wherever both resolve it (the
  curl+trunc of a kmax≤4 potential is band-limited on both).
- `make_seed(meta, S)`: dispatch on `meta['seed_kind']` ∈
  {'blob','random','file'}; 'file' seeds store the full potential array in
  the metadata (`meta['seed_a']`), zero-padded on rebuild.
- Curl via full-fft on host numpy is fine here (cold path) — reuse the
  pattern of legacy `blob_seed`.

## 5. `constantB/spectra.py` (builder B)

Host numpy; input B numpy or jax, cast at entry.

```python
def shell_spectrum(F)                    # (k, E): E(k)=Σ_shell |F̂|², shells k=round(|k|); F (3,...) or (...,)
def axis_spectra(F)                      # dict kx/ky/kz -> 1D summed spectra
def q_spectrum(B)                        # EXACT spectrum of q on the 2x zero-padded grid (alias-free
                                         # because in-band B ⇒ supp q̂ ⊂ |k|<2N/3 < Nyquist of 2N)
def kspace_inertia(B)                    # 3x3 T = Σ_{k≠0}|B̂_k|² k̂k̂ᵀ of B−B̄; returns sorted eigenvalues
                                         # λ1≥λ2≥λ3; ratios λ2/λ1, λ3/λ1: 1D→(0,0), planar→(x,0), 3D→both>0
def local_slope(k, E, k1, k2)            # -d lnE/d lnk fit on [k1,k2]
def exp_kappa(k, E, k1, k2)              # κ from lnE ≈ a − 2κk on [k1,k2] (analyticity strip)
def plot_spectra(states, out, labels=None)  # overlay figure: loglog E(k) | semilogy E(k) | loglog q-spectrum
                                         # with band-edge N/3 marked; sequential one-hue colors by ladder order
```
`states` = list of (B, eps) or filenames (use `load_state`). Figure style:
match `blob_quest.make_plot` (Agg, dpi 200, grids alpha .3, small fonts).

## 6. `grow.py` (repo root, builder C)

Carrier-free continuation driver; thin, style of `blob_quest.py`.

CLI: `--state grow.npz --csv grow.csv --grid0 24 24 48 --grid-max 96 96 192
--Bbar 0 0 1 --seed blob|random|file --w .8 .8 .8 --kz 1 --kmax 4 --slope 0
--key 0 --seed-file a.npz --eps-max 3.0 --de 0.03 --de-min 1e-4 --de-max 0.08
--sweeps 8 --cgit 800 --res-ok 1e-9 --gtail-max 1e-3 --edge-max 1e-5
--smooth 0.0 --fix-mean --freeze-top 0 --snap-de 0.25 --max-seconds 1e9`

Loop (per step):
1. init: `B = B̄` uniform (`--fix-mean` refused if |B̄| ≥ 1−1e-9 with a
   clear message: growth infeasible); build seed once via seeds_free,
   record `seed_kind` + params (+ key) in meta; if `--freeze-top m>0`,
   freeze `top_modes(seed, m)` (+ mean if `--fix-mean`), print the triples
   and the `noncoplanar` verdict.
2. push `B + de*seed` (seed rebuilt on current grid via `make_seed`),
   `S.gn(...)`; NO salvage branch (quadratic Newton: if res > res-ok after
   `sweeps`, the step is genuinely bad) — reject: restore previous B,
   halve de; de-underflow → "FOLD CANDIDATE" stop, same wording spirit as
   amplitude_quest.
3. accept: diagnostics row → CSV: eps, de, grid, res, cg, minutes,
   maxgrad, Bbar, maxdefl, vol_rev, gal_tail_rms/max, edge (retained-band
   edge tail as in `quest_diagnostics` dealias branch), fluct_rms
   (rms|B−B̄|), drift (1−|B̄|), lam21, lam31 (inertia ratios). Reuse
   `quest_diagnostics` pieces where they fit or inline (it needs a carrier
   dict — do NOT import carrier; compute directly).
4. adapt: refine (ascending ×1.5 ladder capped at --grid-max, zero_pad,
   re-polish with `sweeps+4`, check res-ok after polish and REPORT if not
   met) when `gal_tail_rms > --gtail-max` OR band-edge tail > `--edge-max`;
   top-grid exhaustion → "SHARPENING" stop. Regrow de ×1.3 after 2 clean
   accepts (cap --de-max).
5. snapshots every `--snap-de`; `--plot` mode: spectra ladder figure via
   `spectra.plot_spectra` over snapshots + quest curves (maxgrad, drift,
   gal_tail vs eps).
6. resumable: state npz stores B, eps, meta (incl. seed spec, smooth,
   freeze list); resume must NOT need CLI seed params again. Fresh CSV
   header logic as in blob_quest.

## 7. `constantB/descent.py` (builder C; EXPERIMENTAL, mark as such)

Smoothest-state SQP step at fixed pinned-mode energies:
`d = argmin ½‖W(B+d)‖² s.t. T(B·d) = −Tq, e_j-rows` ⇒ same CG machinery
with RHS `JB − F` and a bordered (μ, ν∈R^m) system; energy rows
`e_j = |B̂(k_j)|²` (conjugate-pair sum), gradients `g_j = 2 P_{k_j}B`
(real fields, one mode pair each, W⁻²-diagonal). Public:
```python
def sqp_step(S, B, pins, alpha=0.3)   # damped step towards min ‖WB‖ on the manifold; returns B'
def pinned_energies(B, pins)
```
Implement straightforwardly (may be un-jitted python CG loop over jitted
matvecs); correctness over speed; unit test: on a small converged state,
one damped step + `S.gn` retraction decreases Σ(1+k²)^s|B̂|² while
`pinned_energies` change < 1e-8 rel and res returns below 1e-10.

## 8. `tests/test_v2.py` (builder C)

Script style like `tests/parity_check.py` (main() + asserts + prints, no
pytest dependency), CPU-runnable in ≤ ~5 min. Cases:
1. PARITY s=0: 32²×64, standard blob push on uniform ẑ (eps 0.3): legacy
   `Solver((32,32,64), dealias=True).gn` (pcg, cgit 2000, sweeps 12) vs
   `MuSolver.gn` from the same start: converged `max|ΔB| < 1e-9`;
   tail_norm rel diff < 1e-10.
2. PARITY s=1 (loose): both converge; maxgrad & tail_norm rel diff < 1e-3.
3. Quadratic convergence: per-sweep residuals from `gn(..., verbose)` or
   repeated 1-sweep calls: ratios log-log superlinear (assert res_3 <
   1e-12 for the eps=0.3 case).
4. Div preservation: |div| < 1e-12 after every sweep from a projected
   start; and `project` removes an injected gradient-noise div error.
5. Freeze: frozen bins (incl. a kz=0 pin and its partner) change < 1e-14
   across a gn call; `fix_mean` holds B̄ to 1e-14. `noncoplanar` truth
   table on known triples.
6. Orszag/tail honesty: `tail_norm` of an in-band random field equals the
   q-tail measured on a 2× zero-padded grid to 1e-12 (power folding).
7. random_seed grid-independence (§4).
8. Mini end-to-end: grow.py loop imported (refactor loop body into a
   callable `run(args)` so the test can drive ~4 accepted steps on
   16²×32 with random seed, then one refinement to 24²×48; assert res-ok
   met and CSV rows written to a tmp dir).

## 9. Integration (overseer only)

`__init__` exports, DESIGN.md v2 section (incl. revision of rules 2/5b:
exact-Galerkin operators are CORRECT in μ-form — the limit cycle was a
(λ,μ)+truncated-CG artifact; rule retained for the legacy solver),
README layout, Kaggle notebook, LaTeX compile check.
