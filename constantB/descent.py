"""EXPERIMENTAL (v2) -- move a converged state towards the SMOOTHEST member of
its family while holding chosen mode energies fixed.

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
e_j = |B_hat(k_j)|^2 (conjugate-pair sum), whose gradient as a real field is
g_j = 2 P_{k_j} B.  Stationarity gives

    d = -(B - W^-2 J^T m),      (J W^-2 J^T) m = J B - F,

with unknowns m = (mu field, nu in R^m) and the BORDERED normal operator

    [ T(B . PW(mu B)) + sum_j nu_j T(B . PW(g_j)) ]      rhs  T((|B|^2+1)/2)
    [ <g_i, PW(mu B)> + sum_j nu_j <g_i, PW(g_j)> ]      rhs  <g_i, B>

where PW is the freeze-masked, W^-2-weighted divergence-free band projector of
solver_mu (rebuilt here from the solver's public spectral attributes, so the
two stay consistent by construction).  The mu-block is exactly the solver's
normal operator, so the same preconditioner (multiply by (1+k^2)^s in the
retained band) works, and d is a genuine descent direction: dR.d =
-||W^-1(G - J^T m)||^2 <= 0, and J d = 0 to CG accuracy.

Deliberately a python CG loop over jitted matvecs: this is an exploratory
instrument called a few times per state, and readability beats device
residency here.  Damping alpha is the caller's: the SQP step is the full
constrained minimiser of the quadratic model, i.e. a large extrapolation, and
only its direction is trustworthy.  The retraction back onto the manifold is
the solver itself, S.gn.
"""
import numpy as np
import jax
import jax.numpy as jnp

from .spectral import rfft3, irfft3

_CGIT = 400
_CG_RTOL = 1e-24


# ---------------------------------------------------------------------------
# pinned modes
# ---------------------------------------------------------------------------

def _pin_index(k, shape):
    """rfft-layout indices holding the conjugate pair of the integer triple k.

    A bin with kz = 0 (or kz = Nz/2) stores its partner (-kx, -ky, kz) in the
    SAME half array, and a per-bin selection that catches only one of them
    leaks through irfftn's symmetrisation -- so both are returned (SPEC v2
    section 1).
    """
    kx, ky, kz = (int(v) for v in k)
    if kz < 0:
        kx, ky, kz = -kx, -ky, -kz
    Nx, Ny, Nz = shape
    if kz > Nz // 2:
        raise ValueError(f"pinned mode {tuple(k)} is off the rfft grid")
    idx = [(kx % Nx, ky % Ny, kz)]
    partner = ((-kx) % Nx, (-ky) % Ny, kz)
    if (kz == 0 or (Nz % 2 == 0 and kz == Nz // 2)) and partner != idx[0]:
        idx.append(partner)
    return idx


def _pair_field(B, k):
    """The real field carrying ONLY the +-k Fourier pair of B."""
    shape = B.shape[1:]
    Bh = rfft3(B)
    sel = jnp.zeros(Bh.shape[1:], dtype=Bh.dtype)
    for i in _pin_index(k, shape):
        sel = sel.at[i].set(1.0)
    return irfft3(sel[None] * Bh, shape)


def pinned_energies(B, pins):
    """Energy of each pinned conjugate mode pair, e_j = 2 |B_hat(k_j)|^2 summed
    over components, with grid-size-normalised coefficients so the values are
    comparable across resolutions.  Host numpy out."""
    B = jnp.asarray(B)
    shape = B.shape[1:]
    Bh = np.asarray(rfft3(B)) / float(np.prod(shape))
    out = []
    for k in pins:
        i0 = _pin_index(k, shape)[0]
        out.append(2.0 * float((np.abs(Bh[:, i0[0], i0[1], i0[2]]) ** 2).sum()))
    return np.array(out)


# ---------------------------------------------------------------------------
# operators (jitted once per grid shape; called from the python CG loop)
# ---------------------------------------------------------------------------

@jax.jit
def _T(f, maskR):
    """Strict 2/3 band projector on a real scalar field."""
    return irfft3(maskR * rfft3(f), f.shape[-3:])


@jax.jit
def _PW(v, KR, maskR, freezeR, Wm2):
    """W^-2 * freeze * T * (divergence-free projection) of a (3,...) field --
    the solver_mu step operator, rebuilt from public attributes."""
    K2d = (KR ** 2).sum(0)
    invK2 = jnp.where(K2d > 0, 1.0 / jnp.where(K2d > 0, K2d, 1.0), 0.0)
    vh = rfft3(v)
    vh = freezeR * maskR * (vh - KR * ((KR * vh).sum(0) * invK2)[None])
    return irfft3(Wm2 * vh, v.shape[1:])


@jax.jit
def _amu(mu, B, KR, maskR, freezeR, Wm2):
    """The solver's normal operator A(mu) = T(B . PW(mu B))."""
    return _T((B * _PW(mu[None] * B, KR, maskR, freezeR, Wm2)).sum(0), maskR)


@jax.jit
def _minv(r, maskR, W2):
    """CG preconditioner of the mu block (identity on the nu block)."""
    return irfft3(W2 * maskR * rfft3(r), r.shape)


def _dot(a, b):
    return float((a * b).sum())


# ---------------------------------------------------------------------------
# the step
# ---------------------------------------------------------------------------

def sqp_step(S, B, pins, alpha=0.3):
    """One damped roughness-descent step along the constraint manifold.

    `S` is a MuSolver (its smooth= sets the norm being minimised), `pins` a
    list of integer triples whose mode energies are held fixed to first order.
    Returns the retracted state B' = S.gn(B + alpha*d); the caller checks
    `pinned_energies` and the roughness to judge the step.
    """
    if getattr(S, "freeze", ()) or getattr(S, "fix_mean", False):
        raise ValueError(
            "sqp_step: S carries freeze/fix_mean pins, but the -B term of "
            "the SQP direction is not freeze-masked, so the step would drag "
            "every frozen coefficient toward zero at O(alpha) and the gn "
            "retraction would lock that in (violating the pin invariant "
            "silently). Pass an unfrozen MuSolver and pin modes via `pins` "
            "(the bordered energy rows) instead.")
    B = jnp.asarray(B)
    KR, maskR, freezeR = S.KR, S.maskR, S.freezeR
    W2, Wm2 = (1.0 + S.K2r) ** S.smooth, (1.0 + S.K2r) ** (-S.smooth)
    pins = [tuple(int(v) for v in k) for k in pins]
    m = len(pins)

    # Pin rows.  Each g_j is normalised: scaling row j of a SYMMETRIC bordered
    # system scales its column and its rhs entry identically, so d is
    # unchanged (only nu_j rescales) -- free conditioning.
    g = []
    for k in pins:
        gj = 2.0 * _pair_field(B, k)
        nrm = float(jnp.sqrt((gj * gj).sum()))
        if nrm <= 0.0:
            raise ValueError(f"pinned mode {k} carries no energy in B")
        g.append(gj / nrm)
    h = [_PW(gj, KR, maskR, freezeR, Wm2) for gj in g]
    c = [_T((B * hj).sum(0), maskR) for hj in h]          # = J W^-2 J^T e_j
    Gm = np.array([[_dot(g[i], h[j]) for j in range(m)] for i in range(m)]
                  ).reshape(m, m)

    def matvec(p_mu, p_nu):
        out_mu = _amu(p_mu, B, KR, maskR, freezeR, Wm2)
        for j in range(m):
            out_mu = out_mu + p_nu[j] * c[j]
        out_nu = np.array([_dot(p_mu, c[i]) for i in range(m)]).reshape(m)
        return out_mu, out_nu + Gm @ p_nu

    # rhs = J B - F = ( T((|B|^2+1)/2), <g_i, B> )
    r_mu = _T(0.5 * ((B ** 2).sum(0) + 1.0), maskR)
    r_nu = np.array([_dot(gj, B) for gj in g]).reshape(m)
    mu, nu = jnp.zeros_like(r_mu), np.zeros(m)
    z_mu, z_nu = _minv(r_mu, maskR, W2), r_nu
    p_mu, p_nu = z_mu, z_nu
    rz = _dot(r_mu, z_mu) + float(r_nu @ z_nu)
    rr0 = _dot(r_mu, r_mu) + float(r_nu @ r_nu)
    for _ in range(_CGIT):
        a_mu, a_nu = matvec(p_mu, p_nu)
        den = _dot(p_mu, a_mu) + float(p_nu @ a_nu)
        if den == 0.0:
            break
        al = rz / den
        mu, nu = mu + al * p_mu, nu + al * p_nu
        r_mu, r_nu = r_mu - al * a_mu, r_nu - al * a_nu
        if _dot(r_mu, r_mu) + float(r_nu @ r_nu) < _CG_RTOL * rr0:
            break
        z_mu, z_nu = _minv(r_mu, maskR, W2), r_nu
        rz2 = _dot(r_mu, z_mu) + float(r_nu @ z_nu)
        p_mu, p_nu = z_mu + (rz2 / rz) * p_mu, z_nu + (rz2 / rz) * p_nu
        rz = rz2

    d = -B + _PW(mu[None] * B, KR, maskR, freezeR, Wm2)
    for j in range(m):
        d = d + nu[j] * h[j]
    Bp, _res, _ci = S.gn(B + alpha * d, sweeps=8, cgit=800, tol=1e-12)
    return Bp
