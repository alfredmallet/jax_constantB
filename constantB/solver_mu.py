"""Carrier-free Gauss-Newton solver in mu-form: the Schur complement of
`solver.Solver` in which the divergence constraint is solved once and for all
by construction instead of once per CG iteration.

THE STEP.  Search directions are restricted to div-free, in-band, unfrozen,
Sobolev-weighted fields parameterised by a single scalar mu,

    dB = PW(mu B),   PW = W^-2 . freeze . T . P    (P = Leray, T = 2/3 band),

so the only remaining equation is the quadratic one.  Its Galerkin
linearisation is the scalar normal equation

    A(mu) = T( B . PW(mu B) ) = -T q,     q = (|B|^2 - 1)/2,

solved matrix-free by CG preconditioned with M^-1 = T (1+k^2)^s (the inverse
of the weight the step carries).  The update is B <- T(B + PW(mu B)).

WHY THE EXACT GALERKIN OPERATOR IS SAFE HERE (cf. DESIGN.md rule 3).  The
legacy (lam, mu) solver must keep its J / J^T in plain collocation: projecting
them leaves the lam-block of J J^T with near-null band-edge directions and the
solve enters a ~1e4x residual amplification limit cycle.  That pathology is
the DIVERGENCE block's; with lam eliminated it is gone.  A(mu) above is the
exact Jacobian of the exact Galerkin residual, so Newton convergence here is
QUADRATIC (about 1e-1 -> 1e-3 -> 1e-7 -> 1e-14 in practice), not the linear
~2x/sweep of the legacy dealias mode.

DIV IS PRESERVED, NOT SOLVED FOR.  Every direction is Leray-projected, so
div B is invariant to round-off under `gn` -- and an initial div error is
invariant too.  Callers project ONCE at the entry of a continuation
(`project`); `gn` deliberately does not, so that a state's divergence is a
property the driver controls rather than something silently rewritten
mid-solve.

FREEZING.  `freeze` pins chosen Fourier bins of B: the step multiplier zeroes
them, so they keep their entry values through every sweep (a carrier-free way
to hold a mode's amplitude while the rest of the field relaxes).  Frozen bins
must lie in the retained band, and a kz = 0 bin must be zeroed together with
its Hermitian partner (-kx, -ky, 0), which lives in the SAME rfft half-array:
zero only one and irfftn symmetrises the pair, leaking half the update back
into the mode that was supposed to be frozen (measured: 1e-3 instead of
1e-17).

GPU EXECUTION.  `_gn_solve_mu` is ONE jit program (sweep loop and CG loop both
lax.while_loop), `verbose` its only static argument; grid-derived arrays are
ordinary traced arguments, so re-instantiating a solver does not recompile.
NO donate_argnums in v2: donation deletes the caller's array, and the drivers
keep the previous B to fall back on when a step is rejected.
"""
from functools import partial
import itertools
import warnings

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

from .spectral import (rfft3, irfft3, rfft_wavenumbers, wavenumbers,
                       dealias_mask_rfft, dif, trunc, trunc3)
from .potential import div, divfree_project, inv_k2

# Relative CG stop: ||r||^2 below this fraction of ||r0||^2, i.e. "run CG to
# round-off" -- requires float64 (see DESIGN.md).
_CG_RTOL = 1e-26


# ---------------------------------------------------------------------------
# Operators (jit-traceable building blocks, called from _gn_solve_mu)
# ---------------------------------------------------------------------------

def _PW(v, KR, maskR, freezeR, Wm2r, invK2):
    """W^-2 . freeze . T . P applied to a real (3, ...) field.

    The admissible-direction operator: Leray-project, drop the frozen bins and
    everything outside the retained band, then apply the Sobolev weight
    W^-2 = (1+k^2)^-s.  Two batched transforms.
    """
    vh = rfft3(v)
    vh = freezeR * maskR * (vh - KR * ((KR * vh).sum(0) * invK2)[None])
    return irfft3(Wm2r * vh, v.shape[1:])


def _minv(r, maskR, W2r):
    """Preconditioner M^-1 = T (1+k^2)^s (inverts the step's weight)."""
    return irfft3(W2r * maskR * rfft3(r), r.shape)


def _tq(B, maskR):
    """The exact Galerkin residual T q, q = (|B|^2 - 1)/2."""
    return trunc(0.5 * ((B ** 2).sum(0) - 1.0), maskR, True)


def _normal_op(mu, B, KR, maskR, freezeR, Wm2r, invK2):
    """A(mu) = T( B . PW(mu B) ): the exact Galerkin Jacobian in mu."""
    return trunc((B * _PW(mu[None] * B, KR, maskR, freezeR, Wm2r, invK2)).sum(0),
                 maskR, True)


def _project(B, KR, maskR):
    """Div-free + in-band cleanup of a state; the mean is preserved."""
    return trunc3(divfree_project(B, KR), maskR, True)


def _residual_mu(B, KR, maskR):
    """(div B, T q) -- div is monitored, T q is what `gn` drives to zero."""
    return div(B, KR), _tq(B, maskR)


# ---------------------------------------------------------------------------
# The compiled Gauss-Newton solve
# ---------------------------------------------------------------------------

@partial(jax.jit, static_argnames=("verbose",))
def _gn_solve_mu(B, KR, maskR, freezeR, Wm2r, W2r, invK2,
                 sweeps, cgit, tol, verbose):
    """`sweeps` mu-form Gauss-Newton sweeps, each one preconditioned-CG solve
    of A(mu) = -T q followed by B <- T(B + PW(mu B)).

    Semantics follow legacy `_gn_solve`:
      - a sweep whose entry residual is already < tol is a no-op, and the CG
        count of the previous sweep is preserved;
      - if all `sweeps` run without converging, the returned residual is
        recomputed AFTER the final update;
      - returns (B, max|T q|, CG iterations of the last sweep that ran CG) --
        the CG count is a conditioning/fold proxy the drivers consume.
    """

    def sweep_cond(carry):
        _, k, _, _, done = carry
        return jnp.logical_and(k < sweeps, jnp.logical_not(done))

    def sweep_body(carry):
        Bc, k, _res_prev, ci_prev, _done_prev = carry
        q = _tq(Bc, maskR)
        res_now = jnp.abs(q).max()
        if verbose:
            jax.debug.print("    mu sweep {k}: residual {res:.2e}",
                            k=k, res=res_now)
        converged = res_now < tol

        def do_cg(_):
            mu0 = jnp.zeros_like(q)
            r0 = -q
            z0 = _minv(r0, maskR, W2r)
            rz0 = (r0 * z0).sum()
            rr0 = (r0 * r0).sum()

            def cg_cond(state):
                _, r, _, _, n_done = state
                return jnp.logical_and(n_done < cgit,
                                       (r * r).sum() > _CG_RTOL * rr0)

            def cg_body(state):
                mu, r, p, rz, n_done = state
                Ap = _normal_op(p, Bc, KR, maskR, freezeR, Wm2r, invK2)
                al = rz / (p * Ap).sum()
                mu, r = mu + al * p, r - al * Ap
                z = _minv(r, maskR, W2r)
                rz2 = (r * z).sum()
                return mu, r, z + (rz2 / rz) * p, rz2, n_done + 1

            init = (mu0, r0, z0, rz0, jnp.array(0))
            mu, *_, n_done = lax.while_loop(cg_cond, cg_body, init)
            dB = _PW(mu[None] * Bc, KR, maskR, freezeR, Wm2r, invK2)
            return trunc3(Bc + dB, maskR, True), n_done

        def keep(_):
            return Bc, ci_prev

        Bn, ci_new = lax.cond(converged, keep, do_cg, operand=None)
        return Bn, k + 1, res_now, ci_new, converged

    B0 = trunc3(B, maskR, True)              # keep B in the retained band
    init = (B0, jnp.array(0), jnp.asarray(jnp.inf), jnp.array(0),
            jnp.array(False))
    Bf, _kf, res_last, cif, donef = lax.while_loop(sweep_cond, sweep_body,
                                                   init)
    res_final = jnp.where(donef, res_last, jnp.abs(_tq(Bf, maskR)).max())
    return Bf, res_final, cif


# Standalone jitted entry points for the small public helpers.
_dif_jit = jax.jit(dif, static_argnums=(1,))
_trunc_jit = jax.jit(trunc, static_argnums=(2,))
_trunc3_jit = jax.jit(trunc3, static_argnums=(2,))
_project_jit = jax.jit(_project)
_residual_jit = jax.jit(_residual_mu)


# ---------------------------------------------------------------------------
# Freeze-mask construction (host numpy: cold path, integer bookkeeping)
# ---------------------------------------------------------------------------

def noncoplanar(triples):
    """True iff some three of `triples` span R^3.

    Pure integer determinant test -- no float arithmetic, so the verdict is
    exact.  Three coplanar seed modes generate a field whose nonlinear
    interactions stay in their plane: the usual way a "3D" run turns out to be
    2.5D (paper section 7).  Drivers print this verdict for --freeze-top.
    """
    ts = [tuple(int(v) for v in t) for t in triples]
    for a, b, c in itertools.combinations(ts, 3):
        det = (a[0] * (b[1] * c[2] - b[2] * c[1])
               - a[1] * (b[0] * c[2] - b[2] * c[0])
               + a[2] * (b[0] * c[1] - b[1] * c[0]))
        if det:
            return True
    return False


def _canonical(triple):
    """rfft representative of an integer triple: kz >= 0, conjugating if not
    (the bin (kx, ky, -kz) is not stored; its partner carries the mode)."""
    kx, ky, kz = (int(v) for v in triple)
    return (-kx, -ky, -kz) if kz < 0 else (kx, ky, kz)


def _freeze_mask(shape, triples):
    """Multiplier array (rfft layout): 1 everywhere, 0 on the frozen bins and,
    for kz = 0 bins, on their Hermitian partners (-kx, -ky, 0) as well.

    Raises ValueError for a triple outside the retained band -- freezing a bin
    the band projector kills anyway is always a user error (usually a stale
    mode list from a coarser grid).
    """
    nx, ny, nz = shape
    maskR = np.asarray(dealias_mask_rfft(shape))
    freezeR = np.ones(maskR.shape)
    for t in triples:
        kx, ky, kz = t
        if not (-(nx // 2) <= kx <= (nx - 1) // 2
                and -(ny // 2) <= ky <= (ny - 1) // 2
                and 0 <= kz <= nz // 2):
            raise ValueError(f"freeze triple {t} is not representable on grid "
                             f"{shape}")
        if maskR[kx, ky, kz] == 0:            # negative kx/ky wrap correctly
            raise ValueError(f"freeze triple {t} lies outside the retained "
                             f"band |k_i| < N_i/3 of grid {shape}")
        freezeR[kx, ky, kz] = 0.0
        if kz == 0:                           # Hermitian partner, same array
            freezeR[-kx, -ky, 0] = 0.0
    return freezeR


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

class MuSolver:
    """Carrier-free Galerkin Gauss-Newton solver on a fixed grid.

    Parameters
    ----------
    shape    : (Nx, Ny, Nz) grid.  Strict 2/3 band |k_i| < N_i/3 always
               (v2 is Galerkin-only: the residual is the honest one).
    smooth   : Sobolev exponent s >= 0.  The step minimises
               ||(1+k^2)^(s/2) dB||_2, so s > 0 selects the smooth branch and
               s = 0 the L2 min-norm (rough) branch.
    fix_mean : freeze the k = 0 bin, holding B_bar exactly.
    freeze   : iterable of integer triples (kx, ky, kz) whose Fourier bins are
               held at their entry values (see module docstring).  Triples
               with kz < 0 are mapped to their conjugate representative.

    A warning (not an error) is issued if three or more non-mean freeze
    triples are given and every selection of three is coplanar -- see
    `noncoplanar`.
    """

    def __init__(self, shape, smooth=0.0, fix_mean=False, freeze=()):
        self.shape = tuple(int(n) for n in shape)
        self.smooth = float(smooth)
        self.fix_mean = bool(fix_mean)
        self.dealias = True                   # v2 is Galerkin-only
        # Derivative wavenumbers: Nyquist bins zeroed (the Nyquist trap).
        self.KR = rfft_wavenumbers(self.shape, zero_nyquist=True)
        # k^2 for the weight/preconditioner: TRUE squares including Nyquist.
        self.K2r = (rfft_wavenumbers(self.shape) ** 2).sum(0)
        self.maskR = dealias_mask_rfft(self.shape)
        self.invK2 = inv_k2(self.KR)
        self.Wm2r = (1.0 + self.K2r) ** (-self.smooth)
        self.W2r = (1.0 + self.K2r) ** self.smooth

        pinned = [_canonical(t) for t in freeze]
        if len(pinned) >= 3 and not noncoplanar(pinned):
            warnings.warn("all frozen mode triples are coplanar: the field "
                          "they generate cannot be genuinely 3D", stacklevel=2)
        if self.fix_mean and (0, 0, 0) not in pinned:
            pinned.append((0, 0, 0))
        self.freeze = tuple(pinned)
        self.freezeR = jnp.asarray(_freeze_mask(self.shape, self.freeze))
        self._K = None                                  # full layout, lazy

    # -- public grid attributes ---------------------------------------------
    @property
    def K(self):
        """Full-layout (3, Nx, Ny, Nz) wavenumbers (seed construction etc.;
        built on first use)."""
        if self._K is None:
            self._K = wavenumbers(self.shape)
        return self._K

    @property
    def K2(self):
        return (self.K ** 2).sum(0)

    # -- small public operations --------------------------------------------
    def dif(self, f, axis):
        """Spectral partial derivative of a real scalar field."""
        return _dif_jit(jnp.asarray(f), axis, self.KR)

    def trunc(self, f):
        """Project a scalar field onto the retained band."""
        return _trunc_jit(jnp.asarray(f), self.maskR, True)

    def trunc3(self, F):
        """Project a stacked (3, ...) field onto the retained band."""
        return _trunc3_jit(jnp.asarray(F), self.maskR, True)

    def project(self, B):
        """State cleanup: div-free (mean preserved) and in-band.

        Call this ONCE on the initial state of a continuation -- `gn`
        preserves div B but cannot remove it.  Deliberately NOT freeze-masked:
        this is a repair of the state, not a step along the manifold, and a
        frozen bin with a divergence error in it is still an error.
        """
        return _project_jit(jnp.asarray(B), self.KR, self.maskR)

    def residual(self, B):
        """(div B, T q): the monitored divergence and the exact Galerkin
        constraint residual whose max-norm `gn` reports."""
        return _residual_jit(jnp.asarray(B), self.KR, self.maskR)

    def tail_norm(self, B):
        """(rms, max) of q = (|B|^2-1)/2 outside the retained band.  For
        in-band B this is the ALIAS-FOLDED image of the true continuum tail
        of the constraint violation: a one-sided bound (true tail >=
        measured/sqrt(8); up to 2 fold partners per axis, coherent
        cancellation not excluded) -- the honest single-grid convergence
        diagnostic.  The exact check is spectra.q_spectrum (2x zero-padded,
        alias-free).  See DESIGN.md "Tail honesty, restated one-sidedly"."""
        B = jnp.asarray(B)
        q = 0.5 * ((B ** 2).sum(0) - 1.0)
        qt = q - self.trunc(q)
        return float(jnp.sqrt((qt ** 2).mean())), float(jnp.abs(qt).max())

    # -- the solve -----------------------------------------------------------
    def gn(self, B, sweeps=8, cgit=800, tol=1e-12, verbose=False):
        """Run the compiled mu-form Gauss-Newton solve (see `_gn_solve_mu`).

        Returns (B, max|T q|, CG iterations of the last sweep that ran CG).
        B is truncated to the retained band at entry and stays a jax array.

        DIV IS NOT SOLVED FOR.  The step is div-free by construction, so
        whatever divergence the input carries comes back out unchanged --
        neither amplified nor removed.  Project the state ONCE at the start of
        a continuation (`project`); do not expect `gn` to launder it.
        """
        Bf, res, ci = _gn_solve_mu(jnp.asarray(B), self.KR, self.maskR,
                                   self.freezeR, self.Wm2r, self.W2r,
                                   self.invK2, int(sweeps), int(cgit),
                                   float(tol), bool(verbose))
        return Bf, float(res), int(ci)
