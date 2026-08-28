"""EXPERIMENTAL (v2/v3) -- move a converged state towards the SMOOTHEST member
of its family while holding chosen mode energies fixed.

WHY.  The rough/smooth dichotomy of the paper is path-selected: at equal
amplitude the L2 minimum-norm path lands on a rough state and the
Sobolev-weighted path on a smooth one.  That is evidence about paths, not
about what the manifold contains.  This module asks the question directly:
starting from a state, slide ALONG the constraint manifold (never off it)
downhill in the Sobolev roughness R(B) = 1/2 sum (1+k^2)^s |B_k|^2, holding a
few mode energies pinned so the state cannot simply deflate back to the
uniform solution.  Where that descent stalls is a property of the manifold.

THE STEP (SQP).  Minimise 1/2 ||W(B+d)||^2 subject to the linearised
constraint T(B.d) = -Tq and to zero change of each pinned energy
e_j = 0.25 sum_x |G_j|^2, G_j = 2 P_{k_j} B (SUM units, solver_mu's
convention), whose variation is exactly de_j = sum_x G_j.d.  Stationarity
gives d = -B + PW(mu B + sum_j nu_j G_j) with (mu, nu) solving

    [ A_mu    A_mu,nu ] (mu)   ( T q + 1 )
    [ A_nu,mu  A_nu   ] (nu) = ( 2 e_j   ) ,

i.e. THE SOLVER'S OWN bordered normal operator (solver_mu `_bordered_op`,
docs/algorithm.tex section 7.2, eq. 28) with the SQP right-hand side in
place of the Newton one: the quadratic row is T(B.B) - Tq = T(q+1) and
the pin row is sum_x G_j.B = 2 e_j.  Everything here -- the operator, the
pin selector arrays, the nu-block preconditioner diag d_j = sum_x G_j.PW(G_j) --
is IMPORTED from solver_mu rather than rebuilt here, so there is exactly one
definition of which bins a pin owns and of the inner product they use.  (v2's
private copies of that machinery are gone; they were a second source of
truth and they used a different, volume-mean unit convention.)

Deliberately a python CG loop over jitted matvecs: this is an exploratory
instrument called a few times per state, and readability beats device
residency here.  Damping alpha is the caller's: the SQP step is the full
constrained minimiser of the quadratic model, i.e. a large extrapolation, and
only its direction is trustworthy.  The retraction back onto the manifold is
the solver itself -- a PINNED `gn` whose targets are the entry energies, so
the pins come back exactly rather than merely to the O(alpha^2) accuracy of
the tangency.
"""
import numpy as np
import jax
import jax.numpy as jnp

from .solver_mu import (MuSolver, _minv, _tq, _bordered_op, _pin_dir,
                        _pin_energies, _pin_fields, _pin_precond)

_CGIT = 400
_CG_RTOL = 1e-24

# The solver's building blocks, jitted for use from the python CG loop.
_tq_jit = jax.jit(_tq)
_minv_jit = jax.jit(_minv)
_pin_fields_jit = jax.jit(_pin_fields)
_pin_precond_jit = jax.jit(_pin_precond)
_pin_dir_jit = jax.jit(_pin_dir)
_bordered_jit = jax.jit(_bordered_op)


def pinned_energies(B, pins):
    """Energy of each pinned conjugate mode pair of `B`, host numpy.

    SUM UNITS (solver_mu's module docstring): e_j = sum_x |P_j B|^2
    = vol * <|P_j B|^2>.  Values are therefore grid dependent -- compare them
    across resolutions only after rescaling by vol, and rebuild any schedule
    of targets on the grid it will be used on.
    """
    return MuSolver(np.shape(B)[1:], pins=pins).pinned_energies(B)


def _dot(a, b):
    return float((a * b).sum())


def sqp_step(S, B, pins, alpha=0.3):
    """One damped roughness-descent step along the constraint manifold.

    `S` is a MuSolver (its smooth= sets the norm being minimised), `pins` a
    list of integer triples whose mode energies are held fixed.  Returns the
    retracted state B' = Sp.gn(B + alpha*d, pin_targets=e(B)), where Sp is S's
    pinned twin.  The caller checks `pinned_energies` and the roughness to
    judge the step.
    """
    if getattr(S, "freeze", ()) or getattr(S, "fix_mean", False):
        raise ValueError(
            "sqp_step: S carries freeze/fix_mean pins, but the -B term of "
            "the SQP direction is not freeze-masked, so the step would drag "
            "every frozen coefficient toward zero at O(alpha) and the gn "
            "retraction would lock that in (violating the pin invariant "
            "silently). Pass an unfrozen MuSolver and pin modes via `pins` "
            "(the bordered energy rows) instead.")
    pins = [tuple(int(v) for v in k) for k in pins]
    m = len(pins)
    if m == 0:
        raise ValueError(
            "sqp_step without pins: the unconstrained descent has a global "
            "attractor (the 1D circularly polarised wave of lowest wavenumber)"
            " and answers nothing -- algorithm.tex, Remark 'pins are "
            "mandatory here'.")
    # S's pinned twin: same grid and same weight, plus the energy rows.  Its
    # arrays ARE the operator's arrays, so the tangent solve below and the
    # retraction cannot disagree about a pin.
    Sp = MuSolver(S.shape, smooth=S.smooth, pins=pins)
    op = (Sp.grid1d, Sp.freeze_idx)      # the solver's own kernel arguments
    B = jnp.asarray(Sp.trunc3(B))

    G = _pin_fields_jit(B, Sp.pinR)                    # (m, 3, ...)
    e = np.asarray(_pin_energies(G))
    e_floor = 1e-10 * float(np.asarray((jnp.asarray(B) ** 2).sum()))
    if float(e.min()) <= e_floor:
        j = int(np.argmin(e))
        raise ValueError(
            f"pinned mode {pins[j]} carries energy {float(e.min()):.2e} <= "
            f"1e-10 x field energy ({e_floor:.2e}): a (near-)empty energy row "
            "is a (near-)singular border (review m7)")
    dnu = _pin_precond_jit(G, *op)                     # nu-block precond diag

    # rhs = J B - F = ( T q + 1, 2 e ) -- the SQP right-hand side.
    r_mu, r_nu = _tq_jit(B, Sp.grid1d) + 1.0, jnp.asarray(2.0 * e)
    mu, nu = jnp.zeros_like(r_mu), jnp.zeros(m)
    z_mu, z_nu = _minv_jit(r_mu, Sp.grid1d), r_nu / dnu
    p_mu, p_nu = z_mu, z_nu
    rz = _dot(r_mu, z_mu) + _dot(r_nu, z_nu)
    rr0 = _dot(r_mu, r_mu) + _dot(r_nu, r_nu)
    for _ in range(_CGIT):
        a_mu, a_nu = _bordered_jit(p_mu, p_nu, B, G, *op)
        den = _dot(p_mu, a_mu) + _dot(p_nu, a_nu)
        if den == 0.0:
            break
        al = rz / den
        mu, nu = mu + al * p_mu, nu + al * p_nu
        r_mu, r_nu = r_mu - al * a_mu, r_nu - al * a_nu
        if _dot(r_mu, r_mu) + _dot(r_nu, r_nu) < _CG_RTOL * rr0:
            break
        z_mu, z_nu = _minv_jit(r_mu, Sp.grid1d), r_nu / dnu
        rz2 = _dot(r_mu, z_mu) + _dot(r_nu, z_nu)
        p_mu, p_nu = z_mu + (rz2 / rz) * p_mu, z_nu + (rz2 / rz) * p_nu
        rz = rz2

    d = -B + _pin_dir_jit(mu, nu, B, G, *op)
    Bp, _res, _ci = Sp.gn(B + alpha * d, sweeps=8, cgit=800, tol=1e-12,
                          pin_targets=e)
    return Bp
