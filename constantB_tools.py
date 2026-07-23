#!/usr/bin/env python3
"""
================================================================================
constantB_tools.py  (JAX port)

JAX port of the original numpy/scipy constantB_tools.py. This is a COPY, not
an edit of the original -- see ../constantB_tools.py for the reference
implementation and the full mathematical description (this docstring only
documents what differs in the port).

WHAT WAS PORTED TO JAX, AND WHY:

  The only genuinely hot path in this toolkit is the minimum-norm Gauss-Newton
  solve: `Solver.gn`, which runs several GN sweeps, each requiring hundreds of
  matrix-free CG iterations against the spectral operators J, J^T. This is
  called repeatedly (by amplitude_quest.py's continuation loop and
  resolution_check.py's per-grid polish) at grids up to 96x96x192. THAT is
  what got jit-compiled: `_dif`/`_residual` and the whole GN-sweep + CG solve
  (`_gn_jit`) are rewritten as jax.lax.while_loop bodies, compiling the entire
  multi-sweep, multi-hundred-iteration solve into a single XLA program instead
  of a slow Python loop that re-enters dispatch every iteration.

WHAT WAS DELIBERATELY *NOT* PORTED, AND WHY:

  - carrier / seed_mode / build_seed (Sec. 1) stay in host numpy + scipy.
    seed_mode's periodic-ODE solve is a dense COMPLEX LU factorisation of a
    small (Nz x Nz) matrix, done once per seed -- a cold path with nothing to
    gain from jit, and jax's complex-dtype LU support is spottier than real
    dtypes (especially on GPU). The resulting seed array is converted to a
    jax array only at the boundary where it meets the (jitted) solver.

  - series / domb_sykes / pade_poles (Sec. 5) stay in host numpy + scipy for
    the same reason (per-order dense complex LU, one-shot diagnostic run with
    small Nord) plus their per-order dict structure with data-dependent skips
    ("if not any(...): continue") is not a sane jit target.

  - zero_pad (grid refinement) is left un-jitted: it's called once per
    refinement step with a *different* static output shape each time, so
    jitting it would just force a recompile on every call for no benefit. It
    is rewritten in terms of jnp arrays with functional `.at[...].set(...)`
    (jax arrays are immutable) but otherwise unchanged.

  - diagnose / plot_cuts / plot_3d (Secs. 4, 6) stay in host numpy: they run
    once per command invocation, and mixing jax/numpy inside matplotlib calls
    buys nothing. State arrays are cast to numpy at the top of each of these
    functions.

PRECISION. jax.config.update("jax_enable_x64", True) is set at import time
and is NOT optional: the GN convergence tolerance is 1e-10 and the CG
stopping criterion is 1e-26 (relative); jax's float32 default would never
reach either.

STATE FILE COMPATIBILITY. save_state/load_state still go through
np.savez/np.load on host numpy arrays (jax arrays aren't natively picklable
into npz), so .npz state files are byte-identical in format to the original
tool and interchangeable between the two (e.g. mlstate_fine.npz can be loaded
here, and files written here can be loaded by the original).

PARITY. The CG iteration count returned by `gn` is preserved end-to-end
(threaded through the lax.while_loop carry) because it is not decoration:
amplitude_quest.py uses it for the step-size growth heuristic and
resolution_check.py logs it as a conditioning/fold proxy, exactly as in the
original.  `pcg` and the Sobolev-weighting flag are resolved as *static*
arguments at trace time (not traced values), per design review.

DEALIAS (Galerkin 2/3-rule) MODE. `Solver(shape, dealias=True)` mirrors the
numpy reference's Galerkin formulation exactly (see the reference's
`Solver.__init__`/`gn` docstrings for the mathematics): a per-shape strict
retained-band mask (`dealias_mask`, |k_i| < N_i/3 strict, NOT <=) is built
once in host numpy and closed over as a plain jax array; `dealias` itself is
resolved as a STATIC (trace-time) python bool everywhere it reaches jit, so
there are at most two compiled variants of `_residual`/`_gn_jit` per grid
shape, never a traced conditional. `trunc`/`trunc3` (one fftn + mask
multiply + ifftn) project onto that band; `_residual`'s quadratic entry is
the projected constraint trunc(0.5*(|B|^2-1)); and, per the reference's
explicit design decision, the CG normal-equations operators `_Jop`/
`_JT_core` stay the PLAIN COLLOCATION linearisation/adjoint even in dealias
mode (truncating them makes JJ^T nearly singular along band-edge
directions -- see `_JT_core`'s docstring). B itself is kept in the retained
band by truncating it at `gn` entry and after every CG update, not by
truncating the operators. `Solver.tail_norm` reports the discarded-band
content of |B|^2-1, the single-grid honest-convergence diagnostic that
(for retained-band B) equals the true continuum spectral tail.

Dependencies: numpy, scipy, jax; matplotlib for plots; scikit-image
(optional) for the 3D reversal isosurface.
================================================================================
"""
import argparse
import itertools
from functools import partial

import numpy as np
from scipy.linalg import lu_factor, lu_solve

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import lax

TWOPI = 2 * np.pi

# =============================================================================
# 1. Carrier wave and linearised seed  -- HOST numpy/scipy, unchanged from the
#    original.  Deliberately not jax: cold path, dense complex LU (see module
#    docstring).
# =============================================================================

def carrier(A, c, Nz):
    """The 1D arc-polarised carrier B0(z) and its tangent frame.

    Returns a dict with: z grid, psi(z), s, B0 (3,Nz), the orthonormal frame
    e1, e2 tangent to the unit sphere along B0 (so B0.e1 = B0.e2 = 0), and a
    dense spectral d/dz matrix Dz used by the seed ODE solver.
    """
    z = np.linspace(0, TWOPI, Nz, endpoint=False)
    s = np.sqrt(1 - c * c)
    psi = A * np.sin(z)
    B0 = np.array([s * np.cos(psi), s * np.sin(psi), c * np.ones(Nz)])
    e1 = np.array([-np.sin(psi), np.cos(psi), np.zeros(Nz)])
    e2 = np.array([-c * np.cos(psi), -c * np.sin(psi), s * np.ones(Nz)])  # = B0 x e1
    kz = np.fft.fftfreq(Nz, d=TWOPI / Nz) * TWOPI
    Dz = np.real(np.fft.ifft(1j * kz[:, None] * np.fft.fft(np.eye(Nz), axis=0), axis=0))
    return dict(A=A, c=c, s=s, z=z, psi=psi, B0=B0, e1=e1, e2=e2, Dz=Dz, kz=kz)


def seed_mode(car, kx, ky, amp, prof=(1.0, 0.7)):
    """First-order (linearised) deformation of the carrier for one transverse
    mode k_perp=(kx,ky):  b1 = Re{ [u e1 + v e2] exp(i k_perp.x_perp) }.

    u(z) = amp*(prof[0]*cos z + prof[1]) is the FREE profile; v(z) solves the
    linearised divergence constraint, the periodic ODE
        s v' - i c kappa cos(chi-psi) v = -i kappa sin(chi-psi) u,
    by dense LU on the spectral collocation matrix (host scipy -- cold path,
    complex dense LU, see module docstring). Returns the complex polarisation
    vector w(z) = u e1 + v e2, shape (3,Nz).
    """
    kap, chi = np.hypot(kx, ky), np.arctan2(ky, kx)
    cpz, spz = np.cos(chi - car['psi']), np.sin(chi - car['psi'])
    u = amp * (prof[0] * np.cos(car['z']) + prof[1])
    v = lu_solve(lu_factor(car['s'] * car['Dz'] - 1j * car['c'] * kap * np.diag(cpz)),
                 -1j * kap * spz * u)
    return u * car['e1'] + v * car['e2']


def build_seed(car, modes, shape, prof=(1.0, 0.7)):
    """Real seed field on the grid `shape`=(Nx,Ny,Nz) from a list of modes
    [(kx,ky,amp), ...].  Non-collinear modes make the seed spectrally 3D.
    Returns a plain numpy array; converted to jax only at the solver
    boundary."""
    Nx, Ny, Nz = shape
    x = np.linspace(0, TWOPI, Nx, endpoint=False)
    y = np.linspace(0, TWOPI, Ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing='ij')
    out = np.zeros((3, Nx, Ny, Nz))
    for (kx, ky, amp) in modes:
        w = seed_mode(car, kx, ky, amp, prof)
        out += 2 * np.real(w[:, None, None, :] * np.exp(1j * (kx * X + ky * Y))[None, :, :, None])
    return out

# =============================================================================
# 2. Spectral operators and the jit-compiled minimum-norm Gauss-Newton solver
#    (JAX).  This is the actual hot path in the toolkit.
# =============================================================================

def wavenumbers(shape):
    """Meshed spectral wavenumbers (KX,KY,KZ) for a periodic box [0,2pi)^3,
    stacked into a single (3,Nx,Ny,Nz) jax array."""
    ks = [jnp.fft.fftfreq(n, d=TWOPI / n) * TWOPI for n in shape]
    return jnp.stack(jnp.meshgrid(*ks, indexing='ij'))


def numpy_wavenumbers(shape):
    """Host-numpy twin of `wavenumbers`, used only by the (un-jitted)
    diagnostic/plotting functions in Secs. 4 and 6."""
    ks = [np.fft.fftfreq(n, d=TWOPI / n) * TWOPI for n in shape]
    return np.meshgrid(*ks, indexing='ij')


def numpy_dif(f, axis, K):
    """Host-numpy spectral derivative, used only by diagnose/plot_* (which
    run once per invocation and gain nothing from jit)."""
    return np.real(np.fft.ifftn(1j * K[axis] * np.fft.fftn(f)))


def dealias_mask(shape):
    """STRICT 2/3-rule retained-band mask: |k_i| < N_i/3 (strict inequality).
    Mirrors the numpy reference exactly -- see its `Solver.__init__` comment:
    with the INCLUSIVE cutoff |k| <= N/3, products of two boundary modes
    alias exactly onto the retained-band edge and exactness of the
    retained-band projection of a quadratic nonlinearity is lost. Built once
    per Solver (a shape-only, host-numpy computation with nothing to gain
    from jax) and closed over as a plain jax array by the jit-compiled
    trunc/residual/gn functions below -- it is NEVER a traced argument whose
    static-ness would force a retrace; only the boolean `dealias` flag is
    static."""
    ms = [np.abs(np.fft.fftfreq(n, d=TWOPI / n) * TWOPI) < n / 3 - 1e-12
          for n in shape]
    mask = (ms[0][:, None, None] * ms[1][None, :, None]
            * ms[2][None, None, :]).astype(float)
    return jnp.asarray(mask)


def _trunc(f, mask, dealias):
    """Project a scalar field onto the retained 2/3 band (identity if
    dealias=False). `dealias` is a plain python bool here (never a traced
    value -- callers always pass it as a jit static_arg), so this branch is
    resolved once at trace time, producing at most two compiled variants
    (dealias True/False) per grid shape rather than a traced conditional."""
    if not dealias:
        return f
    return jnp.real(jnp.fft.ifftn(mask * jnp.fft.fftn(f)))


def _trunc3(F, mask, dealias):
    """Project a 3-component field onto the retained band, componentwise."""
    if not dealias:
        return F
    return jnp.stack([_trunc(F[i], mask, dealias) for i in range(3)])


@partial(jax.jit, static_argnums=(2,))
def _trunc_jit(f, mask, dealias):
    """Standalone jit entry point for `Solver.trunc`/`tail_norm`, used
    outside the (already-jitted) `_residual`/`_gn_jit` call graphs."""
    return _trunc(f, mask, dealias)


@partial(jax.jit, static_argnums=(2,))
def _trunc3_jit(F, mask, dealias):
    """Standalone jit entry point for `Solver.trunc3`."""
    return _trunc3(F, mask, dealias)


@partial(jax.jit, static_argnums=(1,))
def _dif(f, axis, K):
    """Spectral partial derivative of a real scalar field (jit-compiled)."""
    return jnp.real(jnp.fft.ifftn(1j * K[axis] * jnp.fft.fftn(f)))


@partial(jax.jit, static_argnums=(3,))
def _residual(B, K, mask, dealias):
    """F(B) = (div B, (|B|^2-1)/2).  In dealias mode the quadratic entry is
    the PROJECTED constraint r2 = trunc(0.5*(|B|^2-1)) (Galerkin residual);
    the div entry is unchanged (automatically band-limited when B is). With
    dealias=False this is the plain pointwise-collocation residual, blind to
    aliasing -- see `refine` for the honest check in that mode."""
    div = _dif(B[0], 0, K) + _dif(B[1], 1, K) + _dif(B[2], 2, K)
    quad = 0.5 * ((B ** 2).sum(0) - 1)
    quad = _trunc(quad, mask, dealias)
    return div, quad


def _Jop(B, d, K):
    """The linearisation J.  NOTE (dealias mode): deliberately the PLAIN
    collocation operator, never truncated, even when dealias=True -- see
    `_JT_core` for why."""
    div = _dif(d[0], 0, K) + _dif(d[1], 1, K) + _dif(d[2], 2, K)
    return div, (B * d).sum(0)


def _JT_core(B, lam, mu, K, Wm2, weighted):
    """Adjoint J^T(lam,mu); if `weighted` (a STATIC python bool resolved at
    trace time) applies the Sobolev weight Wm2 = (1+k^2)^-s used by
    amplitude_quest.py's WeightedSolver.

    NOTE (dealias mode): this stays the PLAIN collocation adjoint -- no band
    truncation inside it, mirroring the numpy reference's `Solver._JT`
    exactly, and deliberately so.  Design rationale (do not "improve" this):
    the CG normal-equations solve uses J and J^T as a matched pair to build
    JJ^T; if both were replaced by the exactly-projected (truncated)
    operators, JJ^T becomes nearly singular along band-edge directions (the
    adjoint's output falls almost entirely into the discarded band there),
    which amplifies a converged residual by ~1e4x into a limit cycle -- this
    was hit and diagnosed empirically before landing on the current scheme.
    The scheme actually used is INEXACT Gauss-Newton: the plain collocation
    operator is the STEP MODEL (used only inside the CG solve), while the
    RESIDUAL being driven to zero is the exact projected Galerkin residual
    (see `_residual`). Consistency of B with the retained band is enforced
    separately, by truncating B itself (`_trunc3`) at gn entry and after
    every update -- see `_gn_jit`."""
    raw = jnp.stack([-_dif(lam, i, K) + mu * B[i] for i in range(3)])
    if weighted:
        raw = jnp.stack([jnp.real(jnp.fft.ifftn(Wm2 * jnp.fft.fftn(f))) for f in raw])
    return raw


def _precond(r1, r2, K2):
    """Spectral CG preconditioner: divide the lambda-block residual by
    (k^2+1) in Fourier space (the mu-block is left as identity)."""
    return jnp.real(jnp.fft.ifftn(jnp.fft.fftn(r1) / (K2 + 1.0))), r2


@partial(jax.jit, static_argnames=('pcg', 'weighted', 'verbose', 'dealias'),
         donate_argnums=(0,))
def _gn_jit(B, K, K2, Wm2, mask, sweeps, cgit, tol, pcg, weighted, verbose, dealias):
    """Jit-compiled Gauss-Newton iteration: `sweeps` GN sweeps, each solving
    (JJ^T)(lam,mu) = -F by CG (matrix-free) and applying the minimum-norm
    update B += J^T(lam,mu).  Both the outer sweep loop and the inner CG loop
    are lax.while_loop so the whole multi-sweep solve is one XLA program.

    Mirrors the original's early-exit and post-loop residual semantics
    exactly:
      - if the residual is already below `tol` at the top of a sweep, that
        sweep is a no-op (B unchanged) and the CG count from the previous
        sweep is preserved (matching the original's `used` staying at its
        last value across an early `return`);
      - if the loop runs for all `sweeps` without ever converging, the
        residual returned is recomputed AFTER the final update (matching the
        original's trailing `r1, r2 = self.residual(B)` after the for-loop).

    DEALIAS MODE (`dealias`, a STATIC python bool resolved at trace time --
    at most two compiled variants per grid shape, never a traced branch):
    mirrors the numpy reference's `Solver.gn` exactly --
      - B is truncated to the retained band ONCE, at entry, before the sweep
        loop starts (`B = trunc3(B)`);
      - after every CG update, B is re-truncated (`B = trunc3(B + JT(...))`);
      - the linearisation/adjoint (`_Jop`/`_JT_core`) used INSIDE the CG
        solve are deliberately left as the plain collocation operators (no
        truncation) -- see `_JT_core`'s docstring for why. Only the RESIDUAL
        (`_residual`, called at the top of each sweep and after the final
        update) is the exact projected Galerkin residual.

    Returns (B, final max-norm residual, CG iteration count of the last
    sweep that actually ran the CG solve) -- the CG count is a conditioning
    proxy used verbatim by amplitude_quest.py and resolution_check.py, so it
    is threaded through the while_loop carry rather than dropped.
    """

    def sweep_cond(carry):
        _, k, _, _, done = carry
        return jnp.logical_and(k < sweeps, jnp.logical_not(done))

    def sweep_body(carry):
        Bc, k, _res_prev, ci_prev, _done_prev = carry
        r1, r2 = _residual(Bc, K, mask, dealias)
        res_now = jnp.maximum(jnp.abs(r1).max(), jnp.abs(r2).max())
        if verbose:
            jax.debug.print("    gn sweep {k}: residual {res:.2e}", k=k, res=res_now)
        converged = res_now < tol

        def do_cg(_):
            lam0 = jnp.zeros_like(r1)
            mu0 = jnp.zeros_like(r2)
            R1_0, R2_0 = -r1, -r2
            if pcg:
                Z1_0, Z2_0 = _precond(R1_0, R2_0, K2)
            else:
                Z1_0, Z2_0 = R1_0, R2_0
            p1_0, p2_0 = Z1_0, Z2_0
            rs_0 = (R1_0 * Z1_0).sum() + (R2_0 * Z2_0).sum()
            rs0_abs = jnp.abs(rs_0)

            def cg_cond(state):
                *_, n_done, stop = state
                return jnp.logical_and(n_done < cgit, jnp.logical_not(stop))

            def cg_body(state):
                lam, mu, R1, R2, Z1, Z2, p1, p2, rs, n_done, _stop = state
                JTp = _JT_core(Bc, p1, p2, K, Wm2, weighted)
                A1, A2 = _Jop(Bc, JTp, K)
                denom = (p1 * A1).sum() + (p2 * A2).sum()
                al = rs / denom
                lam = lam + al * p1
                mu = mu + al * p2
                R1n = R1 - al * A1
                R2n = R2 - al * A2
                if pcg:
                    Z1n, Z2n = _precond(R1n, R2n, K2)
                else:
                    Z1n, Z2n = R1n, R2n
                rs2 = (R1n * Z1n).sum() + (R2n * Z2n).sum()
                new_stop = jnp.abs(rs2) < (1e-26 * rs0_abs)
                beta = rs2 / rs
                p1n = Z1n + beta * p1
                p2n = Z2n + beta * p2
                return (lam, mu, R1n, R2n, Z1n, Z2n, p1n, p2n, rs2, n_done + 1, new_stop)

            init = (lam0, mu0, R1_0, R2_0, Z1_0, Z2_0, p1_0, p2_0, rs_0,
                     jnp.array(0), jnp.array(False))
            lam, mu, _R1, _R2, _Z1, _Z2, _p1, _p2, _rs, n_done, _stop = \
                lax.while_loop(cg_cond, cg_body, init)
            dB = _JT_core(Bc, lam, mu, K, Wm2, weighted)
            Bn = _trunc3(Bc + dB, mask, dealias)
            ci_new = jnp.maximum(n_done - 1, 0)
            return Bn, ci_new

        def keep(_):
            return Bc, ci_prev

        Bn, ci_new = lax.cond(converged, keep, do_cg, operand=None)
        return Bn, k + 1, res_now, ci_new, converged

    B0 = _trunc3(B, mask, dealias)   # dealias mode: keep B in the retained band
    init = (B0, jnp.array(0), jnp.asarray(jnp.inf), jnp.array(0), jnp.array(False))
    Bf, _kf, res_last, cif, donef = lax.while_loop(sweep_cond, sweep_body, init)

    r1f, r2f = _residual(Bf, K, mask, dealias)
    res_true_final = jnp.maximum(jnp.abs(r1f).max(), jnp.abs(r2f).max())
    final_res = jnp.where(donef, res_last, res_true_final)
    return Bf, final_res, cif


class Solver:
    """Component-free minimum-norm Gauss-Newton solver on a fixed grid
    (JAX-backed: residual/J/JT/CG/GN-sweep loop are jit-compiled, see
    `_gn_jit`).  Unweighted by default; amplitude_quest.py's WeightedSolver
    subclasses this and only overrides the `_Wm2`/`_weighted` hooks.

    dealias=True switches from collocation to a GALERKIN (2/3-rule)
    formulation: all fields are truncated to the retained band |k_i| <
    N_i/3 (strict), and since every nonlinearity here is quadratic,
    Orszag's argument makes the retained-band projection of every product
    EXACT on the unpadded grid -- no aliasing corruption anywhere in the
    solve. See `dealias_mask`, `_trunc`/`_trunc3`, `_residual`, `_gn_jit`,
    and `tail_norm` below for exactly what that means; mirrors the numpy
    reference's `Solver(shape, dealias=True)` line for line. `dealias` is
    resolved as a STATIC (trace-time) argument everywhere it reaches jit --
    at most two compiled variants per grid shape."""

    def __init__(self, shape, dealias=False):
        self.shape = tuple(shape)
        self.K = wavenumbers(self.shape)
        self.K2 = sum(k ** 2 for k in self.K)
        self.dealias = bool(dealias)
        # mask is always a concrete jax array of the right shape (even when
        # dealias=False, where its value is simply never read inside the
        # `if not dealias` branches of `_trunc`/`_trunc3`) so that Solver
        # instances of the same grid shape but different `dealias` always
        # present jit with argument of consistent shape/dtype.
        self.mask = dealias_mask(self.shape) if self.dealias else jnp.ones(self.shape)

    @property
    def _Wm2(self):
        return jnp.ones_like(self.K2)

    @property
    def _weighted(self):
        return False

    def dif(self, f, axis):
        """Spectral partial derivative of a real scalar field."""
        return _dif(jnp.asarray(f), axis, self.K)

    def trunc(self, f):
        """Project a scalar field onto the retained 2/3 band (identity if
        dealias=False)."""
        return _trunc_jit(jnp.asarray(f), self.mask, self.dealias)

    def trunc3(self, F):
        """Project a 3-component field onto the retained band, componentwise
        (identity if dealias=False)."""
        return _trunc3_jit(jnp.asarray(F), self.mask, self.dealias)

    def tail_norm(self, B):
        """Discarded-band content of |B|^2-1: for 2/3-truncated B this
        equals (by power-preserving alias folding) the TRUE spectral tail
        of the continuum constraint violation -- the honest single-grid
        convergence diagnostic. Mirrors the numpy reference's
        `Solver.tail_norm` line for line. Meaningful in either mode, but
        only equals the TRUE continuum tail when B is itself retained-band
        (i.e. when this Solver has dealias=True and B came out of its
        `gn`)."""
        B = jnp.asarray(B)
        q = 0.5 * ((B ** 2).sum(0) - 1)
        qt = q - self.trunc(q)
        return float(jnp.sqrt((qt ** 2).mean())), float(jnp.abs(qt).max())

    def residual(self, B):
        """F(B) = (div B, (|B|^2-1)/2).  With dealias=False the second entry
        is evaluated pointwise and is blind to aliasing -- see `refine` for
        the honest check. With dealias=True the second entry is the
        PROJECTED (exact Galerkin) constraint r2 = trunc(0.5*(|B|^2-1))."""
        return _residual(jnp.asarray(B), self.K, self.mask, self.dealias)

    def gn(self, B, sweeps=6, cgit=500, tol=1e-10, verbose=False, pcg=False):
        """Gauss-Newton iteration; see `_gn_jit` for the compiled
        implementation and the exact early-exit/CG-count semantics preserved
        from the original. Returns (B, final max-norm residual, CG iteration
        count) with B as a jax array (converted back to numpy at
        save_state)."""
        B = jnp.asarray(B)
        Bf, res, ci = _gn_jit(B, self.K, self.K2, self._Wm2, self.mask,
                              int(sweeps), int(cgit), float(tol),
                              bool(pcg), bool(self._weighted), bool(verbose),
                              self.dealias)
        return Bf, float(res), int(ci)


def zero_pad(f, new_shape):
    """Spectral interpolation of a real field to a finer grid (exact for the
    retained trigonometric modes).  Deliberately left UN-jitted: called once
    per grid-refinement step with a different static output shape each time,
    so jit would just force a recompile every call for no benefit.  Uses
    functional `.at[...].set(...)` since jax arrays are immutable."""
    f = jnp.asarray(f)
    F = jnp.fft.fftn(f)
    G = jnp.zeros(new_shape, dtype=jnp.complex128)
    n = f.shape
    sl_old = [(slice(0, ni // 2), slice(ni - ni // 2, ni)) for ni in n]
    sl_new = [(slice(0, ni // 2), slice(Ni - ni // 2, Ni)) for ni, Ni in zip(n, new_shape)]
    for choice in itertools.product((0, 1), repeat=3):
        idx_new = tuple(sl_new[d][choice[d]] for d in range(3))
        idx_old = tuple(sl_old[d][choice[d]] for d in range(3))
        G = G.at[idx_new].set(F[idx_old])
    return jnp.real(jnp.fft.ifftn(G)) * (np.prod(new_shape) / np.prod(n))

# =============================================================================
# 3. State files -- host numpy, unchanged format (npz interchangeable with
#    the original tool's state files).
# =============================================================================

def save_state(fn, B, eps, meta):
    np.savez(fn, B=np.asarray(B), eps=eps, **meta)

def load_state(fn):
    st = np.load(fn, allow_pickle=True)
    meta = {k: st[k] for k in st.files if k not in ('B', 'eps')}
    return st['B'], float(st['eps']), meta

def meta_args(args):
    return dict(A=args.A, c=args.c, modes=np.array(args.modes, float),
                prof=np.array(args.prof, float))

def rebuild_seed(meta, shape):
    car = carrier(float(meta['A']), float(meta['c']), shape[2])
    modes = [tuple(m) for m in np.atleast_2d(meta['modes'])]
    return build_seed(car, modes, shape, prof=tuple(np.array(meta['prof']).ravel())), car

# =============================================================================
# 4. Diagnostics -- HOST numpy (one-shot per invocation; no benefit from
#    jax).  State arrays are cast to numpy at entry.
# =============================================================================

def diagnose(B, eps, meta, full=True):
    """Physical and numerical health report of a state.  Deflection is
    measured from the volume-mean field direction (observational convention);
    Z = (1-cos Theta)/2; 'switchback' = deflection > 90 deg."""
    B = np.asarray(B)
    shape = B.shape[1:]
    K = numpy_wavenumbers(shape)
    def dif(f, axis):
        return np.real(np.fft.ifftn(1j * K[axis] * np.fft.fftn(f)))
    r1 = dif(B[0], 0) + dif(B[1], 1) + dif(B[2], 2)
    r2 = 0.5 * ((B ** 2).sum(0) - 1)
    nrm = np.sqrt((B ** 2).sum(0))
    car = carrier(float(meta['A']), float(meta['c']), shape[2])
    Bbar = B.mean(axis=(1, 2, 3)); nb = np.linalg.norm(Bbar)
    cosM = np.clip((B * Bbar[:, None, None, None]).sum(0) / (nrm * nb), -1, 1)
    defl = np.degrees(np.arccos(cosM)); Z = 0.5 * (1 - cosM)
    print(f"state: grid {shape}, eps={eps:.3f}, A={float(meta['A'])}, c={float(meta['c'])}")
    print(f"  residuals (this grid): div {np.abs(r1).max():.2e}, |B|^2-1 {np.abs(2*r2).max():.2e}")
    print(f"  | |B|-1 | max {np.abs(nrm-1).max():.2e};  |Bbar| {nb:.3f}")
    print(f"  modulation max|B-B0| {np.sqrt(((B-car['B0'][:,None,None,:])**2).sum(0)).max():.3f}")
    print(f"  deflection from mean: max {defl.max():.1f} deg; vol>90deg {100*(defl>90).mean():.2f}%")
    print(f"  Z: max {Z.max():.3f}, mean {Z.mean():.3f}")
    if full:
        g2 = sum(dif(B[i], j) ** 2 for i in range(3) for j in range(3))
        sh = (defl > 80) & (defl < 100)
        print(f"  max|gradB|_F {np.sqrt(g2.max()):.2f};"
              f" shell(80-100deg) grad2/mean {g2[sh].mean()/g2.mean():.2f}" if sh.any() else
              "  (no 80-100deg shell present)")
        bh = np.abs(np.fft.fftn(B - car['B0'][:, None, None, :], axes=(1, 2, 3))) ** 2
        tails = []
        for ax, N in ((1, shape[0]), (2, shape[1]), (3, shape[2])):
            E = bh.sum(axis=tuple(i for i in range(4) if i != ax))[:N // 2]
            tails.append(E[-3:].max() / E.max())
        print(f"  spectral tails (fraction of peak): kx {tails[0]:.1e}, ky {tails[1]:.1e}, kz {tails[2]:.1e}")
        print("  [tails > ~1e-4 mean the state is under-resolved: refine!]")
    return defl

# =============================================================================
# 5. Perturbation series on the harmonic ladder (dealiased) -- HOST
#    numpy/scipy, unchanged from the original (dense complex LU per order,
#    one-shot diagnostic; see module docstring).
# =============================================================================

def series(A, c, kap, chi, Nord=16, Nz=256, prof=(0.3, 0.2), bw=(10, 6)):
    """Order-by-order expansion for a single transverse wavevector
    (kappa, chi).  Order n lives on ladder modes j*k_perp, |j|<=n; the sphere
    constraint prescribes the B0-component f_n from lower orders (a ladder
    convolution) and the divergence constraint is a periodic ODE per mode
    (LU-factorised once per j).  A per-order spectral filter with cutoff
    bw[0]+bw[1]*n suppresses round-off amplified by the d/dz in the source
    (verified filter- and resolution-independent).  Returns (norms a_n,
    scalar coefficient series, rotation number omega)."""
    car = carrier(A, c, Nz); s, z, psi = car['s'], car['z'], car['psi']
    cpz, spz = np.cos(chi - psi), np.sin(chi - psi)
    kz = car['kz']
    dz = lambda f: np.fft.ifft(1j * kz * np.fft.fft(f))
    filt = lambda f, n: np.fft.ifft(np.where(np.abs(kz) > bw[0] + bw[1] * n, 0,
                                             np.fft.fft(f)))
    LU = {j: lu_factor(s * car['Dz'] - 1j * c * (j * kap) * np.diag(cpz))
          for j in range(1, Nord + 1)}
    J = Nord + 2; idx = lambda j: j + J
    zero = np.zeros((2 * J + 1, Nz), complex)
    U, V, Fc = {1: zero.copy()}, {1: zero.copy()}, {1: zero.copy()}
    u = prof[0] * np.cos(z) + prof[1]
    U[1][idx(1)] = u
    V[1][idx(1)] = lu_solve(LU[1], -1j * kap * spz * u)
    U[1][idx(-1)], V[1][idx(-1)] = np.conj(u), np.conj(V[1][idx(1)])
    a = [max(np.abs(U[1]).max(), np.abs(V[1]).max())]
    scal = [V[1][idx(1)][0]]
    for n in range(2, Nord + 1):
        f = zero.copy()
        for m in range(1, n):
            for j1 in range(-m, m + 1):
                w = (U[m][idx(j1)], V[m][idx(j1)], Fc[m][idx(j1)])
                if not any(np.abs(t).max() for t in w):
                    continue
                for j2 in range(-(n - m), n - m + 1):
                    f[idx(j1 + j2)] += -0.5 * (w[0] * U[n-m][idx(j2)]
                                               + w[1] * V[n-m][idx(j2)]
                                               + w[2] * Fc[n-m][idx(j2)])
        for j in range(-n, n + 1):
            f[idx(j)] = filt(f[idx(j)], n)
        U[n], V[n], Fc[n] = zero.copy(), zero.copy(), f
        V[n][idx(0)] = -(c / s) * f[idx(0)]
        for j in range(1, n + 1):
            rhs = -(1j * (j * kap) * s * cpz * f[idx(j)] + c * dz(f[idx(j)]))
            V[n][idx(j)] = filt(lu_solve(LU[j], rhs), n)
            V[n][idx(-j)] = np.conj(V[n][idx(j)])
        a.append(max(np.abs(V[n]).max(), np.abs(Fc[n]).max()))
        scal.append(V[n][idx(1)][0])
    om = (c * kap / s) * np.cos(chi) * np.trapezoid(np.cos(psi), z) / TWOPI
    return np.array(a), np.array(scal), om


def domb_sykes(a):
    """Estimate 1/radius from the coefficient norms via a Domb-Sykes fit
    (ratios vs 1/n, last 6 points)."""
    r = a[1:] / a[:-1]; ns = np.arange(2, len(a) + 1)
    p = np.polyfit(1.0 / ns[-6:], r[-6:], 1)
    return p[1], r


def pade_poles(cn, L=None):
    """Poles of the [L/M] Pade approximant of the scalar coefficient series
    (least-squares denominator); nearest pole ~ radius and location of the
    limiting singularity in the complex eps-plane."""
    N = len(cn); L = (N - 1) // 2 if L is None else L; M = N - 1 - L
    C = np.array([[cn[L+i-j] if 0 <= L+i-j < N else 0 for j in range(1, M+1)]
                  for i in range(1, M+1)], complex)
    b = np.linalg.lstsq(C, -np.array([cn[L+i] for i in range(1, M+1)], complex),
                        rcond=None)[0]
    return np.roots(np.r_[b[::-1], 1.0])

# =============================================================================
# 6. Plotting -- HOST numpy, unchanged from the original.
# =============================================================================

def plot_cuts(B, meta, out):
    """Three verification panels: (a) mean-field-aligned component and |B|
    along z through the deepest reversal; (b) Cartesian components along the
    same cut (spacecraft-style); (c) deflection map in the (x,y) plane through
    the reversal with the 90-degree contour."""
    B = np.asarray(B)
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    shape = B.shape[1:]
    z = np.linspace(0, TWOPI, shape[2], endpoint=False)
    x = np.linspace(0, TWOPI, shape[0], endpoint=False)
    Bbar = B.mean(axis=(1, 2, 3)); bb = Bbar / np.linalg.norm(Bbar)
    nrm = np.sqrt((B ** 2).sum(0))
    Bpar = (B * bb[:, None, None, None]).sum(0)
    defl = np.degrees(np.arccos(np.clip(Bpar / nrm, -1, 1)))
    ix, iy, iz = np.unravel_index(np.argmax(defl), defl.shape)
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
    ax[0].plot(z, Bpar[ix, iy, :], 'C0', lw=1.5, label=r'$\mathbf{B}\cdot\hat{\bar{\mathbf{B}}}$')
    ax[0].plot(z, nrm[ix, iy, :], 'k--', lw=1, label=r'$|\mathbf{B}|$')
    ax[0].axhline(0, color='gray', lw=0.6); ax[0].axvline(z[iz], color='r', lw=0.6, ls=':')
    ax[0].set_xlabel('z'); ax[0].legend(fontsize=8)
    ax[0].set_title('cut along z through deepest reversal', fontsize=9)
    for i, lab in enumerate(('$B_x$', '$B_y$', '$B_z$')):
        ax[1].plot(z, B[i, ix, iy, :], f'C{i}', lw=1.2, label=lab)
    ax[1].plot(z, nrm[ix, iy, :], 'k--', lw=1, label='$|B|$')
    ax[1].set_xlabel('z'); ax[1].legend(fontsize=8, ncol=2)
    ax[1].set_title('components along same cut', fontsize=9)
    im = ax[2].pcolormesh(x, x, defl[:, :, iz].T, shading='auto', cmap='RdBu_r')
    ax[2].contour(x, x, defl[:, :, iz].T, levels=[90], colors='k', linewidths=1.2)
    plt.colorbar(im, ax=ax[2], label='deflection (deg)')
    ax[2].plot(x[ix], x[iy], 'k+', ms=10); ax[2].set_xlabel('x'); ax[2].set_ylabel('y')
    ax[2].set_title(f'slice z={z[iz]:.2f}; black: 90$^\\circ$', fontsize=9)
    plt.tight_layout(); plt.savefig(out, dpi=200)
    print(f"wrote {out}")


def plot_3d(B, meta, out):
    """3D visualisation: box faces coloured by the mean-field-aligned
    component (left) and, if scikit-image is available, the reversal
    isosurface B.Bbar-hat = 0 (right)."""
    B = np.asarray(B)
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import cm
    shape = B.shape[1:]; L = TWOPI
    x = np.linspace(0, L, shape[0], endpoint=False)
    z = np.linspace(0, L, shape[2], endpoint=False)
    Bbar = B.mean(axis=(1, 2, 3)); bb = Bbar / np.linalg.norm(Bbar)
    Bpar = (B * bb[:, None, None, None]).sum(0)
    norm = plt.matplotlib.colors.TwoSlopeNorm(vmin=Bpar.min(), vcenter=0, vmax=Bpar.max())
    cmap = cm.RdBu_r
    edges = ([[0,L],[L,L],[L,L]],[[0,L],[0,0],[L,L]],[[0,0],[0,L],[L,L]],
             [[L,L],[0,L],[L,L]],[[L,L],[L,L],[0,L]],[[L,L],[0,0],[0,L]],
             [[0,0],[L,L],[0,L]],[[0,L],[L,L],[0,0]],[[L,L],[0,L],[0,0]])
    fig = plt.figure(figsize=(11, 4.6))
    ax = fig.add_subplot(121, projection='3d')
    Xf, Yf = np.meshgrid(x, x, indexing='ij'); Xz, Zz = np.meshgrid(x, z, indexing='ij')
    ax.plot_surface(Xf, Yf, np.full_like(Xf, L), facecolors=cmap(norm(Bpar[:, :, -1])),
                    shade=False, rstride=1, cstride=1)
    ax.plot_surface(Xz, np.full_like(Xz, L), Zz, facecolors=cmap(norm(Bpar[:, -1, :])),
                    shade=False, rstride=1, cstride=2)
    ax.plot_surface(np.full_like(Xz, L), Xz, Zz, facecolors=cmap(norm(Bpar[-1, :, :])),
                    shade=False, rstride=1, cstride=2)
    for e in edges: ax.plot(*e, 'k', lw=0.8, zorder=10)
    ax.set_xlim(0, L); ax.set_ylim(0, L); ax.set_zlim(0, L)
    ax.view_init(elev=28, azim=42); ax.set_axis_off()
    ax.set_title(r'$\mathbf{B}\cdot\hat{\bar{\mathbf{B}}}$ on box faces', fontsize=10)
    m = cm.ScalarMappable(norm=norm, cmap=cmap); m.set_array([])
    fig.colorbar(m, ax=ax, shrink=0.6, pad=0.02)
    ax2 = fig.add_subplot(122, projection='3d')
    try:
        from skimage import measure
        verts, faces, _, _ = measure.marching_cubes(
            Bpar, level=0.0, spacing=(L/shape[0], L/shape[1], L/shape[2]))
        ax2.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2],
                         color='crimson', alpha=0.55, lw=0)
        ax2.set_title(r'reversal surfaces $\mathbf{B}\cdot\hat{\bar{\mathbf{B}}}=0$', fontsize=10)
    except ImportError:
        idx = np.argwhere(Bpar < 0)
        ax2.scatter(idx[:, 0]*L/shape[0], idx[:, 1]*L/shape[1], idx[:, 2]*L/shape[2],
                    s=2, c='crimson', alpha=0.4)
        ax2.set_title('reversal region (scatter; install scikit-image for isosurface)',
                      fontsize=9)
    for e in edges: ax2.plot(*e, 'k', lw=0.8, zorder=10)
    ax2.set_xlim(0, L); ax2.set_ylim(0, L); ax2.set_zlim(0, L)
    ax2.view_init(elev=28, azim=42); ax2.set_axis_off()
    plt.tight_layout(); plt.savefig(out, dpi=200)
    print(f"wrote {out}")

# =============================================================================
# 7. Command-line interface
# =============================================================================

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('cmd', choices=['series', 'init', 'cont', 'refine',
                                   'polish', 'diagnose', 'plotcuts', 'plot3d'])
    p.add_argument('--A', type=float, default=1.2, help='carrier arc amplitude (rad)')
    p.add_argument('--c', type=float, default=0.2, help='carrier mean-aligned component')
    p.add_argument('--grid', type=int, nargs=3, default=[32, 32, 64],
                   help='Nx Ny Nz (init: working grid; refine: target grid)')
    p.add_argument('--modes', type=float, nargs=3, action='append',
                   help='kx ky amp (repeatable); default (1,1,.15) and (1,-1,.15)')
    p.add_argument('--prof', type=float, nargs=2, default=[1.0, 0.7],
                   help='free profile u(z) = amp*(a cos z + b)')
    p.add_argument('--eps', type=float, default=0.02, help='initial seed amplitude')
    p.add_argument('--de', type=float, default=0.02, help='continuation step')
    p.add_argument('--steps', type=int, default=1, help='continuation steps')
    p.add_argument('--sweeps', type=int, default=4, help='GN sweeps (polish)')
    p.add_argument('--cgit', type=int, default=500, help='CG budget per sweep')
    p.add_argument('--dealias', action='store_true',
                   help='Galerkin 2/3-rule solve: alias-free retained-band '
                        'equations; honest tail measured on the same grid')
    p.add_argument('--pcg', action='store_true',
                   help='preconditioned CG (recommended for cold or localized starts)')
    p.add_argument('--state', default='state.npz', help='state file')
    p.add_argument('--out', default='fig.png', help='output figure file')
    # series-only options
    p.add_argument('--kap', type=float, default=np.sqrt(2.0))
    p.add_argument('--chi', type=float, default=np.pi/4)
    p.add_argument('--Nord', type=int, default=16)
    p.add_argument('--Nz', type=int, default=256)
    args = p.parse_args()
    if args.modes is None:
        args.modes = [[1, 1, 0.15], [1, -1, 0.15]]

    if args.cmd == 'series':
        a, scal, om = series(args.A, args.c, args.kap, args.chi,
                             Nord=args.Nord, Nz=args.Nz)
        invr, r = domb_sykes(a)
        poles = pade_poles(scal)
        print(f"omega = {om:+.5f}   (divisors 2|sin(pi j omega)|)")
        print("a_n ratios:", " ".join(f"{x:.2f}" for x in r))
        print(f"Domb-Sykes 1/rho = {invr:.2f}  =>  rho ~ {1/invr:.4g}")
        print(f"nearest Pade pole: {poles[np.argmin(np.abs(poles))]:.4g}")

    elif args.cmd == 'init':
        car = carrier(args.A, args.c, args.grid[2])
        seed = build_seed(car, [tuple(m) for m in args.modes], args.grid,
                          prof=tuple(args.prof))
        B = car['B0'][:, None, None, :] + args.eps * seed
        B, res, ci = Solver(args.grid, dealias=args.dealias).gn(B, sweeps=6, cgit=args.cgit, pcg=args.pcg)
        print(f"init eps={args.eps}: residual {res:.1e} (cg {ci})")
        save_state(args.state, B, args.eps, meta_args(args))

    elif args.cmd == 'cont':
        B, eps, meta = load_state(args.state)
        shape = B.shape[1:]
        seed, car = rebuild_seed(meta, shape)
        S = Solver(shape, dealias=args.dealias)
        for k in range(args.steps):
            eps += args.de
            B = B + args.de * seed
            B, res, ci = S.gn(B, sweeps=4, cgit=args.cgit, pcg=args.pcg)
            g = max(np.abs(S.dif(B[i], j)).max() for i in range(3) for j in range(3))
            print(f"eps={eps:.2f}: res={res:.1e} maxgrad={g:.2f} cg={ci}")
        save_state(args.state, B, eps, meta)

    elif args.cmd == 'refine':
        B, eps, meta = load_state(args.state)
        Bf = jnp.stack([zero_pad(B[i], tuple(args.grid)) for i in range(3)])
        r1, r2 = Solver(args.grid).residual(Bf)
        print(f"HONEST residual on {tuple(args.grid)}: div {np.abs(r1).max():.2e}, "
              f"|B|^2-1 {np.abs(2*r2).max():.2e}   [this is the number that counts]")
        save_state(args.state, Bf, eps, meta)

    elif args.cmd == 'polish':
        B, eps, meta = load_state(args.state)
        B, res, ci = Solver(B.shape[1:], dealias=args.dealias).gn(B, sweeps=args.sweeps,
                                            cgit=args.cgit, verbose=True, pcg=args.pcg)
        print(f"polished: residual {res:.2e}")
        save_state(args.state, B, eps, meta)

    elif args.cmd == 'diagnose':
        B, eps, meta = load_state(args.state)
        diagnose(B, eps, meta)

    elif args.cmd == 'plotcuts':
        B, eps, meta = load_state(args.state)
        plot_cuts(B, meta, args.out)

    elif args.cmd == 'plot3d':
        B, eps, meta = load_state(args.state)
        plot_3d(B, meta, args.out)


if __name__ == '__main__':
    main()
