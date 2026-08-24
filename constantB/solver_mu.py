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

ENERGY PINS (v3).  `freeze` holds a bin's VALUE, which turns out to be the
wrong instrument for growth: the held bin cannot participate in cancelling
the residual it creates, so the solver pays for it with a cascade (measured,
36^2x72: maxgrad 2.45 and gal_tail 1.2e-3 at eps = 0.269, against 0.80 and
1.7e-6 unpinned).  A pin instead holds the bin's ENERGY, leaving its phase
free, by appending one scalar equation per pin to the GN system:

    e_j(B) = c_j,   e_j(B) = 0.25 sum_x |G_j|^2,   G_j = 2 P_{k_j} B,

with P_{k_j} the projector onto the conjugate pair at integer wavevector k_j
(the rfft bin plus its Hermitian partner when k_z = 0).  The same run pinned
on the seed's top three modes reaches maxgrad 0.874 / tail 3.9e-6 at
eps = 0.290, against 0.866 / 2.4e-6 for the UNPINNED control at that same
eps: the pins cost essentially nothing, the freeze cost a cascade.

SUM UNITS, LOAD-BEARING.  e_j is a SUM over grid points, not a volume mean:

    e_j = 0.25 sum_x |G_j(x)|^2 = sum_x |P_j B|^2 = vol * < |P_j B|^2 >,

vol = Nx*Ny*Nz.  This is the SAME inner product the CG uses (`.sum()`), so
the bordered rows, the right-hand side and the preconditioner are mutually
consistent and the bordered operator is exactly symmetric.  Mixing a mean
into any one of them (the prototype's only bug) breaks the symmetry and the
pins converge slowly or not at all.  Targets `pin_targets` are therefore in
SUM units too; `pinned_energies` reports them in the same units.

THE BORDERED STEP.  Unknowns (mu, nu in R^m); the direction gains one term
per pin, dB = PW(mu B + sum_j nu_j G_j), and since de_j = sum_x G_j . dB
exactly, the normal system is the symmetric bordered one

    A_mu = T( B . D ),   A_nu_i = sum_x G_i . D,   rhs = (-T q, c_j - e_j),

with D = PW(mu B + sum_j nu_j G_j) built ONCE per operator application.

SPD on the band subspace (rhs, operator image and preconditioner are all
band-projected, so every CG iterate stays in band).  The mu-block
preconditioner is unchanged; the nu-block divides by d_j = sum_x G_j.PW(G_j).
G_j, e_j and d_j are RE-LINEARISED every sweep, which is what makes the pin
residual converge quadratically (measured: to 1e-13 relative in 3-4 sweeps).

GPU EXECUTION.  `_gn_solve_mu` is ONE jit program (sweep loop and CG loop both
lax.while_loop), `verbose` its only static argument; grid-derived arrays are
ordinary traced arguments, so re-instantiating a solver does not recompile.
`_gn_solve_pinned` is its bordered twin, selected by a STATIC flag (m > 0) so
that an unpinned solver traces the v2 program bit-for-bit -- no border arrays
appear in it.  `pin_targets` is a traced argument: changing the targets along
a continuation does not recompile.  NO donate_argnums in v2/v3: donation
deletes the caller's array, and the drivers keep the previous B to fall back
on when a step is rejected.
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


def _pin_fields(B, pinR):
    """G_j = 2 P_{k_j} B, stacked (m, 3, ...) -- one batched transform pair.

    `pinR` is the (m, ...) rfft-layout selector built by `_pin_masks`; the
    factor 2 makes de_j = sum_x G_j . dB exact (see the module docstring).
    """
    return irfft3(2.0 * pinR[:, None] * rfft3(B)[None], B.shape[1:])


def _pin_energies(G):
    """e_j = 0.25 sum_x |G_j|^2 from the pin fields -- SUM units, the CG's own
    inner product (module docstring: this consistency is load-bearing)."""
    return 0.25 * (G ** 2).sum(axis=(1, 2, 3, 4))


def _pin_dir(mu, nu, B, G, KR, maskR, freezeR, Wm2r, invK2):
    """The admissible direction D = PW(mu B + sum_j nu_j G_j)."""
    V = mu[None] * B + (nu[:, None, None, None, None] * G).sum(0)
    return _PW(V, KR, maskR, freezeR, Wm2r, invK2)


def _bordered_op(mu, nu, B, G, KR, maskR, freezeR, Wm2r, invK2):
    """The bordered normal operator (A_mu, A_nu) at (mu, nu), one PW.

    A_mu = T(B . D) extends `_normal_op`; A_nu_i = sum_x G_i . D is the pin
    row.  Symmetric because both borders are the SAME sum inner product.
    """
    D = _pin_dir(mu, nu, B, G, KR, maskR, freezeR, Wm2r, invK2)
    return (trunc((B * D).sum(0), maskR, True),
            (G * D[None]).sum(axis=(1, 2, 3, 4)))


def _pin_precond(G, KR, maskR, freezeR, Wm2r, invK2):
    """nu-block preconditioner diagonal d_j = sum_x G_j . PW(G_j).

    The diagonal of the pin-pin block of the bordered operator; recomputed
    once per sweep.  Floored at 1e-300 so a pin whose bin the freeze mask
    kills cannot divide by zero (the constructor rejects that case anyway).
    """
    PG = jax.vmap(_PW, in_axes=(0, None, None, None, None, None))(
        G, KR, maskR, freezeR, Wm2r, invK2)
    return jnp.maximum((G * PG).sum(axis=(1, 2, 3, 4)), 1e-300)


def _pin_rel_error(e, targets):
    """max_j |e_j - c_j| / max(c_j, 1) -- `gn`'s pin stopping test.

    The max(c_j, 1) scale is relative for the large targets that matter and
    absolute for targets near zero, where a relative test is meaningless
    (a pin at c_j = 0 asks for a mode to be absent, not for 12 digits of it).
    """
    return (jnp.abs(e - targets) / jnp.maximum(targets, 1.0)).max()


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


@partial(jax.jit, static_argnames=("verbose",))
def _gn_solve_pinned(B, targets, KR, maskR, freezeR, pinR, Wm2r, W2r, invK2,
                     sweeps, cgit, tol, verbose):
    """`_gn_solve_mu` with m bordered energy-pin rows appended (m = len(pinR)).

    Each sweep RE-LINEARISES the pins -- G_j, e_j and the nu-block
    preconditioner diagonal are rebuilt from the current B -- then runs one
    preconditioned CG on the bordered normal system

        [ A_mu   A_mu,nu ] (mu )   ( -T q      )
        [ A_nu,mu  A_nu  ] (nu ) = ( c_j - e_j )

    and takes the full step B <- T(B + PW(mu B + sum_j nu_j G_j)).

    Semantics match `_gn_solve_mu`, with the stopping test widened to
    max(max|T q|, max_j |e_j - c_j|/max(c_j, 1)) < tol so that a sweep is a
    no-op only when BOTH the quadratic constraint and the pins are converged.
    The returned residual is still the q-residual alone (`pin_error` reports
    the other half).
    """

    def sweep_cond(carry):
        _, k, _, _, done = carry
        return jnp.logical_and(k < sweeps, jnp.logical_not(done))

    def sweep_body(carry):
        Bc, k, _res_prev, ci_prev, _done_prev = carry
        q = _tq(Bc, maskR)
        res_now = jnp.abs(q).max()
        G = _pin_fields(Bc, pinR)                  # re-linearisation
        e = _pin_energies(G)
        pin_now = _pin_rel_error(e, targets)
        if verbose:
            jax.debug.print("    mu sweep {k}: residual {res:.2e}  "
                            "pin {pin:.2e}", k=k, res=res_now, pin=pin_now)
        converged = jnp.maximum(res_now, pin_now) < tol

        def do_cg(_):
            dnu = _pin_precond(G, KR, maskR, freezeR, Wm2r, invK2)
            r10, r20 = -q, targets - e
            z10 = _minv(r10, maskR, W2r)
            z20 = r20 / dnu
            rz0 = (r10 * z10).sum() + (r20 * z20).sum()
            rr0 = (r10 * r10).sum() + (r20 * r20).sum()

            def cg_cond(state):
                _, _, r1, r2, _, _, _, n_done = state
                return jnp.logical_and(
                    n_done < cgit,
                    (r1 * r1).sum() + (r2 * r2).sum() > _CG_RTOL * rr0)

            def cg_body(state):
                mu, nu, r1, r2, p1, p2, rz, n_done = state
                A1, A2 = _bordered_op(p1, p2, Bc, G, KR, maskR, freezeR,
                                      Wm2r, invK2)
                al = rz / ((p1 * A1).sum() + (p2 * A2).sum())
                mu, nu = mu + al * p1, nu + al * p2
                r1, r2 = r1 - al * A1, r2 - al * A2
                z1, z2 = _minv(r1, maskR, W2r), r2 / dnu
                rz2 = (r1 * z1).sum() + (r2 * z2).sum()
                return (mu, nu, r1, r2, z1 + (rz2 / rz) * p1,
                        z2 + (rz2 / rz) * p2, rz2, n_done + 1)

            init = (jnp.zeros_like(q), jnp.zeros_like(targets), r10, r20,
                    z10, z20, rz0, jnp.array(0))
            mu, nu, *_, n_done = lax.while_loop(cg_cond, cg_body, init)
            dB = _pin_dir(mu, nu, Bc, G, KR, maskR, freezeR, Wm2r, invK2)
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
_pin_energies_jit = jax.jit(
    lambda B, pinR: _pin_energies(_pin_fields(B, pinR)))


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


def _bins(shape, triple, maskR, what):
    """The rfft bin indices a canonical triple occupies: itself, plus its
    Hermitian partner (-kx, -ky, 0) when kz = 0 (they share the half-array,
    see the module docstring).  Validates representability and in-band-ness.

    Raises ValueError for a triple outside the retained band -- masking a bin
    the band projector kills anyway is always a user error (usually a stale
    mode list from a coarser grid).
    """
    nx, ny, nz = shape
    kx, ky, kz = triple
    if not (-(nx // 2) <= kx <= (nx - 1) // 2
            and -(ny // 2) <= ky <= (ny - 1) // 2
            and 0 <= kz <= nz // 2):
        raise ValueError(f"{what} triple {triple} is not representable on "
                         f"grid {shape}")
    if maskR[kx, ky, kz] == 0:                # negative kx/ky wrap correctly
        raise ValueError(f"{what} triple {triple} lies outside the retained "
                         f"band |k_i| < N_i/3 of grid {shape}")
    b = [(kx % nx, ky % ny, kz)]
    if kz == 0:
        b.append(((-kx) % nx, (-ky) % ny, 0))
    return b


def _freeze_mask(shape, triples):
    """Multiplier array (rfft layout): 1 everywhere, 0 on the frozen bins and,
    for kz = 0 bins, on their Hermitian partners (-kx, -ky, 0) as well."""
    maskR = np.asarray(dealias_mask_rfft(shape))
    freezeR = np.ones(maskR.shape)
    for t in triples:
        for b in _bins(shape, t, maskR, "freeze"):
            freezeR[b] = 0.0
    return freezeR


def _pin_masks(shape, triples, frozen):
    """Stacked (m, ...) rfft-layout selector: row j is 1 exactly on pin j's
    bin and (for kz = 0) its Hermitian partner, 0 elsewhere.

    `frozen` is the set of bins the freeze mask kills.  A pin sharing a bin
    with a freeze is rejected: the freeze zeroes that bin in every admissible
    direction, so its energy row would be identically zero -- a singular
    border, not a constraint.  Two pins on the same conjugate pair are
    rejected for the same reason (duplicate rows).
    """
    maskR = np.asarray(dealias_mask_rfft(shape))
    pinR = np.zeros((len(triples),) + maskR.shape)
    seen = {}
    for j, t in enumerate(triples):
        for b in _bins(shape, t, maskR, "pin"):
            if b in frozen:
                raise ValueError(f"pin triple {t} occupies rfft bin {b}, "
                                 "which is also frozen (fix_mean freezes "
                                 "(0,0,0)); a frozen bin's energy cannot be "
                                 "steered")
            if b in seen and seen[b] != j:
                raise ValueError(f"pin triples {triples[seen[b]]} and {t} "
                                 f"select the same rfft bin {b} (Hermitian "
                                 "partners are the same mode): duplicate pin")
            seen[b] = j
            pinR[(j,) + b] = 1.0
    return pinR


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
    pins     : iterable of integer triples whose modal ENERGY is held at a
               target instead, phase free (module docstring).  Static: the
               selector array is built here; only the targets passed to `gn`
               are traced.  Validated like `freeze`; a pin may not share a
               bin with a freeze (nor with another pin).  `pins = ()` is the
               v2 solver exactly -- no border arrays are traced at all.

    A warning (not an error) is issued if three or more non-mean freeze
    triples are given and every selection of three is coplanar -- see
    `noncoplanar`.
    """

    def __init__(self, shape, smooth=0.0, fix_mean=False, freeze=(), pins=()):
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
        freezeR = _freeze_mask(self.shape, self.freeze)
        self.freezeR = jnp.asarray(freezeR)

        self.pins = tuple(_canonical(t) for t in pins)
        self.m = len(self.pins)
        frozen = {tuple(int(v) for v in b)
                  for b in zip(*np.nonzero(freezeR == 0.0))}
        self.pinR = jnp.asarray(_pin_masks(self.shape, self.pins, frozen))
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

    # -- energy pins ---------------------------------------------------------
    def pinned_energies(self, B):
        """Pin energies e_j = 0.25 sum_x |2 P_j B|^2 of a state, as a length-m
        numpy array.

        SUM UNITS (module docstring): e_j = vol * <|P_j B|^2>, vol = Nx*Ny*Nz.
        Targets handed to `gn` must be in these units -- a volume mean is
        smaller by vol and the pins would be driven to ~0.
        """
        if self.m == 0:
            return np.zeros(0)
        return np.asarray(_pin_energies_jit(jnp.asarray(B), self.pinR))

    def pin_error(self, B, targets, relative=False):
        """How far the pins are from their targets, as a single float.

        Default: `gn`'s own stopping quantity max_j |e_j - c_j|/max(c_j, 1)
        (relative where it is meaningful, absolute for targets near zero).
        `relative=True` gives the plain max_j |e_j/c_j - 1| that the drivers
        log, with zero targets falling back to the absolute error.
        """
        e = self.pinned_energies(B)
        c = np.asarray(targets, float).reshape(-1)
        if e.size != c.size:
            raise ValueError(f"pin_targets has length {c.size}, expected "
                             f"{self.m}")
        if c.size == 0:
            return 0.0
        scale = np.where(c != 0.0, np.abs(c), 1.0) if relative \
            else np.maximum(c, 1.0)
        return float((np.abs(e - c) / scale).max())

    # -- the solve -----------------------------------------------------------
    def gn(self, B, sweeps=8, cgit=800, tol=1e-12, verbose=False,
           pin_targets=None):
        """Run the compiled mu-form Gauss-Newton solve (see `_gn_solve_mu`,
        or `_gn_solve_pinned` when the solver carries energy pins).

        Returns (B, max|T q|, CG iterations of the last sweep that ran CG).
        B is truncated to the retained band at entry and stays a jax array.
        The reported residual is the q-residual in both cases; query the pins
        with `pin_error`.

        PIN STOP-TEST TRAP (review M2): the pin convergence metric is
        |e_j - c_j| / max(c_j, 1) -- ABSOLUTE for targets below 1 in SUM
        units.  Production targets are >> 1 (SUM units scale with grid
        volume); if you pin a weak mode with c_j << 1, gn can report
        convergence at a large RELATIVE pin error -- check
        pin_error(B, targets, relative=True) yourself in that regime.

        `pin_targets` is a length-m array of pin energies in SUM UNITS
        (`pinned_energies`); it is TRACED, so sliding the targets along a
        continuation never recompiles.  None means "hold the entry energies".
        With no pins the argument must be absent or empty, and the compiled
        program is v2's, untouched.  The stopping test then widens to
        max(max|T q|, max_j |e_j - c_j|/max(c_j, 1)) < tol.

        DIV IS NOT SOLVED FOR.  The step is div-free by construction, so
        whatever divergence the input carries comes back out unchanged --
        neither amplified nor removed.  Project the state ONCE at the start of
        a continuation (`project`); do not expect `gn` to launder it.
        """
        B = jnp.asarray(B)
        if self.m == 0:
            if pin_targets is not None and np.size(pin_targets) > 0:
                raise ValueError("pin_targets given to a solver built without "
                                 "pins")
            Bf, res, ci = _gn_solve_mu(B, self.KR, self.maskR, self.freezeR,
                                       self.Wm2r, self.W2r, self.invK2,
                                       int(sweeps), int(cgit), float(tol),
                                       bool(verbose))
            return Bf, float(res), int(ci)

        if pin_targets is None:               # hold whatever we came in with
            targets = _pin_energies_jit(trunc3(B, self.maskR, True), self.pinR)
        else:
            pt = np.asarray(pin_targets, float).reshape(-1)
            if not np.all(np.isfinite(pt)) or (pt < 0).any():
                raise ValueError(
                    "pin_targets must be finite and non-negative (energies "
                    "cannot be negative; review m3): got %r" % (pin_targets,))
            targets = jnp.asarray(pt)
            if targets.shape != (self.m,):
                raise ValueError(f"pin_targets has length {targets.size}, "
                                 f"expected {self.m}")
        Bf, res, ci = _gn_solve_pinned(B, targets, self.KR, self.maskR,
                                       self.freezeR, self.pinR, self.Wm2r,
                                       self.W2r, self.invK2, int(sweeps),
                                       int(cgit), float(tol), bool(verbose))
        return Bf, float(res), int(ci)
