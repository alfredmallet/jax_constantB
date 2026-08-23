"""Minimum-norm Gauss-Newton solver for the constant-|B| system

    F(B) = ( div B ,  (|B|^2 - 1)/2 ) = 0.

THE ALGORITHM (unchanged from the numpy reference; see DESIGN.md for the
full rationale).  Each GN sweep solves the normal equations of the
*underdetermined* linearisation by matrix-free CG,

    (J J^T) (lam, mu) = -F(B),      dB = W^-2 J^T (lam, mu),

which is the minimum-norm (in L2, or in the Sobolev norm ||(1+k^2)^(s/2)dB||
when weighted) correction that cancels the linearised residual.  J maps a
vector field d to (div d, B.d); J^T maps scalars (lam, mu) to -grad lam + mu B.

DEALIAS (Galerkin 2/3-rule) MODE is *inexact* Gauss-Newton by design:
  - the RESIDUAL driven to zero is the exact projected Galerkin residual
    (quadratic entry band-truncated);
  - the CG step model uses the PLAIN collocation J / J^T -- truncating them
    makes J J^T nearly singular along band-edge directions (empirically: a
    ~1e4x residual amplification limit cycle).  Do not "fix" this.
  - B itself is kept in the retained band by truncation at gn() entry and
    after every update.

GPU EXECUTION.  `_gn_solve` is one jit program: the GN sweep loop and the CG
loop are both lax.while_loop, so the entire multi-sweep solve runs on-device
with no per-iteration python dispatch.  All flags that select code paths
(pcg / weighted / verbose / dealias) are STATIC jit arguments -- python bools
resolved at trace time, at most a handful of compiled variants per grid
shape.  `donate_argnums=(0,)` recycles B's device buffer.

FFT BUDGET (real transforms per CG iteration): 8 unweighted / 11 weighted
(+2 with pcg), all rfft-based and component-batched -- about a third of the
full-complex-FFT count of the numpy reference, with identical semantics.
"""
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

from .spectral import (rfft3, irfft3, rfft_wavenumbers, wavenumbers,
                       dealias_mask_rfft, dif, trunc, trunc3)

# Relative CG stopping criterion: |r.z| below this fraction of its initial
# value.  Effectively "run CG to round-off"; requires float64.
_CG_RTOL = 1e-26


# ---------------------------------------------------------------------------
# Residual and normal-equations operators.
# Naming: *_h = spectral (rfft) representation; KR/maskR/Wm2r live in the
# rfft layout.  `dealias` and `weighted` are plain python bools everywhere
# (jit static): the `if`s below are resolved at trace time.
# ---------------------------------------------------------------------------

def _residual(B, KR, maskR, dealias):
    """F(B) = (div B, (|B|^2-1)/2); quadratic entry band-projected in dealias
    mode (exact Galerkin residual), pointwise (aliasing-blind) otherwise."""
    shape = B.shape[1:]
    div = irfft3(1j * (KR * rfft3(B)).sum(0), shape)
    quad = trunc(0.5 * ((B ** 2).sum(0) - 1.0), maskR, dealias)
    return div, quad


def _apply_JT(lam, mu, B, KR, Wm2r, weighted):
    """J^T (lam, mu) = W^-2 (-grad lam + mu B).

    Returns (JT, JT_h) with JT_h the spectral form when it comes for free
    (weighted mode), so the caller's div can skip a forward transform;
    JT_h is None in unweighted mode.
    """
    shape = lam.shape
    grad = irfft3((1j * KR) * rfft3(lam), shape)      # batched: (3,...) at once
    raw = mu * B - grad
    if weighted:
        raw_h = Wm2r * rfft3(raw)
        return irfft3(raw_h, shape), raw_h
    return raw, None


def _apply_J(d, B, KR, d_h=None):
    """J d = (div d, B.d).  Pass d_h (spectral d) if already available."""
    shape = d.shape[1:]
    if d_h is None:
        d_h = rfft3(d)
    div = irfft3(1j * (KR * d_h).sum(0), shape)
    return div, (B * d).sum(0)


def _precondition(r1, r2, K2r):
    """Spectral preconditioner: lam-block divided by (k^2+1); mu-block
    identity.  (The lam-block of J J^T is approximately -Laplacian + 1.)"""
    return irfft3(rfft3(r1) / (K2r + 1.0), r1.shape), r2


# ---------------------------------------------------------------------------
# The compiled Gauss-Newton solve
# ---------------------------------------------------------------------------

@partial(jax.jit,
         static_argnames=("pcg", "weighted", "verbose", "dealias"),
         donate_argnums=(0,))
def _gn_solve(B, KR, K2r, Wm2r, maskR, sweeps, cgit, tol,
              pcg, weighted, verbose, dealias):
    """`sweeps` Gauss-Newton sweeps, each one matrix-free CG solve of
    (J J^T)(lam, mu) = -F followed by the minimum-norm update
    B <- trunc(B + W^-2 J^T (lam, mu)).

    Semantics preserved exactly from the reference implementation:
      - a sweep whose entry residual is already < tol is a no-op, and the CG
        count from the previous sweep is preserved (early-`return` parity);
      - if all `sweeps` run without converging, the returned residual is
        recomputed AFTER the final update;
      - returns (B, final max-norm residual, CG iterations of the last sweep
        that ran CG).  The CG count is a conditioning/fold proxy consumed by
        the quest drivers -- it is threaded through the loop carry, not
        dropped.
    """

    def sweep_cond(carry):
        _, k, _, _, done = carry
        return jnp.logical_and(k < sweeps, jnp.logical_not(done))

    def sweep_body(carry):
        Bc, k, _res_prev, ci_prev, _done_prev = carry
        r1, r2 = _residual(Bc, KR, maskR, dealias)
        res_now = jnp.maximum(jnp.abs(r1).max(), jnp.abs(r2).max())
        if verbose:
            jax.debug.print("    gn sweep {k}: residual {res:.2e}",
                            k=k, res=res_now)
        converged = res_now < tol

        def do_cg(_):
            # CG on the normal equations, unknowns (lam, mu), rhs -(r1, r2).
            lam0 = jnp.zeros_like(r1)
            mu0 = jnp.zeros_like(r2)
            R1_0, R2_0 = -r1, -r2
            Z1_0, Z2_0 = (_precondition(R1_0, R2_0, K2r) if pcg
                          else (R1_0, R2_0))
            rs_0 = (R1_0 * Z1_0).sum() + (R2_0 * Z2_0).sum()
            rs0_abs = jnp.abs(rs_0)

            def cg_cond(state):
                *_, n_done, stop = state
                return jnp.logical_and(n_done < cgit, jnp.logical_not(stop))

            def cg_body(state):
                lam, mu, R1, R2, Z1, Z2, p1, p2, rs, n_done, _stop = state
                JTp, JTp_h = _apply_JT(p1, p2, Bc, KR, Wm2r, weighted)
                A1, A2 = _apply_J(JTp, Bc, KR, d_h=JTp_h)
                al = rs / ((p1 * A1).sum() + (p2 * A2).sum())
                lam, mu = lam + al * p1, mu + al * p2
                R1n, R2n = R1 - al * A1, R2 - al * A2
                Z1n, Z2n = (_precondition(R1n, R2n, K2r) if pcg
                            else (R1n, R2n))
                rs2 = (R1n * Z1n).sum() + (R2n * Z2n).sum()
                stop = jnp.abs(rs2) < _CG_RTOL * rs0_abs
                beta = rs2 / rs
                p1n, p2n = Z1n + beta * p1, Z2n + beta * p2
                return (lam, mu, R1n, R2n, Z1n, Z2n, p1n, p2n, rs2,
                        n_done + 1, stop)

            init = (lam0, mu0, R1_0, R2_0, Z1_0, Z2_0, Z1_0, Z2_0, rs_0,
                    jnp.array(0), jnp.array(False))
            lam, mu, *_, n_done, _stop = lax.while_loop(cg_cond, cg_body, init)
            dB, _ = _apply_JT(lam, mu, Bc, KR, Wm2r, weighted)
            Bn = trunc3(Bc + dB, maskR, dealias)
            return Bn, jnp.maximum(n_done - 1, 0)

        def keep(_):
            return Bc, ci_prev

        Bn, ci_new = lax.cond(converged, keep, do_cg, operand=None)
        return Bn, k + 1, res_now, ci_new, converged

    B0 = trunc3(B, maskR, dealias)   # dealias: keep B in the retained band
    init = (B0, jnp.array(0), jnp.asarray(jnp.inf), jnp.array(0),
            jnp.array(False))
    Bf, _kf, res_last, cif, donef = lax.while_loop(sweep_cond, sweep_body,
                                                   init)

    r1f, r2f = _residual(Bf, KR, maskR, dealias)
    res_final = jnp.where(donef, res_last,
                          jnp.maximum(jnp.abs(r1f).max(), jnp.abs(r2f).max()))
    return Bf, res_final, cif


# Standalone jitted entry points for the small public helpers (the solver's
# own call graph is already inside _gn_solve's jit).
_dif_jit = jax.jit(dif, static_argnums=(1,))
_trunc_jit = jax.jit(trunc, static_argnums=(2,))
_trunc3_jit = jax.jit(trunc3, static_argnums=(2,))
_residual_jit = jax.jit(_residual, static_argnums=(3,))


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

class Solver:
    """Minimum-norm Gauss-Newton solver on a fixed grid.

    Parameters
    ----------
    shape   : (Nx, Ny, Nz) grid.
    dealias : False -> plain collocation (pointwise residual, blind to
              aliasing: cross-check with zero-pad refinement).
              True  -> Galerkin 2/3-rule (strict |k_i| < N_i/3 band): the
              retained-band equations are EXACT for this quadratic system
              (Orszag), and `tail_norm` measures the honest unresolved
              burden on the same grid.
    smooth  : Sobolev exponent s >= 0.  s > 0 makes gn() take the correction
              of minimum ||(1+k^2)^(s/2) dB||_2 instead of minimum L2 norm --
              the smoothest correction cancelling the linearised residual.

    All jit-relevant flags derived from these are static booleans: one
    compiled solver variant per (shape, dealias, weighted, pcg, verbose)
    combination.
    """

    def __init__(self, shape, dealias=False, smooth=0.0):
        self.shape = tuple(shape)
        self.dealias = bool(dealias)
        self.smooth = float(smooth)
        # Derivative wavenumbers: Nyquist bins zeroed to reproduce the
        # reference full-FFT real() semantics (see rfft_wavenumbers).
        self.KR = rfft_wavenumbers(self.shape, zero_nyquist=True)
        # k^2 for preconditioner/Sobolev weight: TRUE squares incl. Nyquist.
        self.K2r = (rfft_wavenumbers(self.shape) ** 2).sum(0)
        # Concrete mask array even when dealias=False (never read then), so
        # both modes present jit with consistent argument shapes/dtypes.
        self.maskR = (dealias_mask_rfft(self.shape) if self.dealias
                      else jnp.ones(self.K2r.shape))
        self._K = None                                  # full layout, lazy

    # -- public grid attributes -------------------------------------------
    @property
    def K(self):
        """Full-layout (3, Nx, Ny, Nz) wavenumbers (for seed construction
        etc.; built on first use)."""
        if self._K is None:
            self._K = wavenumbers(self.shape)
        return self._K

    @property
    def K2(self):
        return (self.K ** 2).sum(0)

    # -- weighting hooks ----------------------------------------------------
    @property
    def _weighted(self):
        return self.smooth > 0

    @property
    def _Wm2r(self):
        """Spectral multiplier W^-2 = (1+k^2)^-s in rfft layout."""
        if self._weighted:
            return (1.0 + self.K2r) ** (-self.smooth)
        return jnp.ones_like(self.K2r)

    # -- small public operations -------------------------------------------
    def dif(self, f, axis):
        """Spectral partial derivative of a real scalar field."""
        return _dif_jit(jnp.asarray(f), axis, self.KR)

    def trunc(self, f):
        """Project a scalar field onto the retained band (identity if
        dealias=False)."""
        return _trunc_jit(jnp.asarray(f), self.maskR, self.dealias)

    def trunc3(self, F):
        """Project a stacked (3, ...) field onto the retained band."""
        return _trunc3_jit(jnp.asarray(F), self.maskR, self.dealias)

    def residual(self, B):
        """F(B) = (div B, (|B|^2-1)/2); see class docstring for what the
        quadratic entry means in each mode."""
        return _residual_jit(jnp.asarray(B), self.KR, self.maskR, self.dealias)

    def tail_norm(self, B):
        """(rms, max) of q = (|B|^2-1)/2 outside the retained band.  For
        retained-band B this equals (by power-preserving alias folding) the
        TRUE continuum spectral tail of the constraint violation: the honest
        single-grid convergence diagnostic."""
        B = jnp.asarray(B)
        q = 0.5 * ((B ** 2).sum(0) - 1.0)
        qt = q - self.trunc(q)
        return float(jnp.sqrt((qt ** 2).mean())), float(jnp.abs(qt).max())

    # -- the solve ----------------------------------------------------------
    def gn(self, B, sweeps=6, cgit=500, tol=1e-10, verbose=False, pcg=False):
        """Run the compiled Gauss-Newton solve (see `_gn_solve`).

        Returns (B, final max-norm residual, CG iteration count of the last
        sweep that ran CG).  B stays a jax array; save_state converts.
        """
        Bf, res, ci = _gn_solve(jnp.asarray(B), self.KR, self.K2r, self._Wm2r,
                                self.maskR, int(sweeps), int(cgit),
                                float(tol), bool(pcg), bool(self._weighted),
                                bool(verbose), self.dealias)
        return Bf, float(res), int(ci)


class WeightedSolver(Solver):
    """Backward-compatible alias: WeightedSolver(shape, smooth, dealias) ==
    Solver(shape, dealias=dealias, smooth=smooth)."""

    def __init__(self, shape, smooth=0.0, dealias=False):
        super().__init__(shape, dealias=dealias, smooth=smooth)
