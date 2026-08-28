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
lax.while_loop), `verbose` its only static argument.  The grid enters as
`Grid1D`, a pytree of BROADCASTABLE 1D axis arrays (O(N) each, not O(N^3)):
the mask, k^2, the Sobolev weights, 1/k^2 and the Leray projector are rebuilt
as fused broadcasts inside every kernel that needs them, so the only O(N^3)
buffers the program touches are the field itself and its transforms.  The
frozen bins travel as a static (n, 3) index array and are applied by a scatter
rather than by a full multiplier grid.  Everything stays a TRACED argument, so
re-instantiating a solver -- or sliding `smooth`, which is a traced scalar --
does not recompile.  `_gn_solve_pinned` is its bordered twin, selected by a
STATIC flag (m > 0) so that an unpinned solver traces the v2 program
bit-for-bit -- no border arrays appear in it.  `pin_targets` is traced too.

THE CG RUNS IN PACKED BAND COORDINATES.  mu is band limited, so its N^3 real
samples carry only ~0.28 N^3 degrees of freedom.  `Band` (below) is the
isometry onto them: the CG state (mu, r, z, p), the right-hand side and the
preconditioner are packed vectors of that length, and <v, w> is the SAME sum
inner product the real-space CG used, exactly (Parseval), so the pin borders
stay consistent with it and the algorithm is unchanged -- iterate for iterate,
in exact arithmetic.  Two things follow: the operator's band projection is the
packing slice itself (no mask multiply, no transform pair), and the
preconditioner (1+k^2)^s is an elementwise multiply by a packed weight vector
built ONCE per sweep instead of a transform pair per iteration.  Per CG
iteration that is 8 scalar transforms instead of 10; measured at 96^3, jit
temp 130.0 -> 98.9 B/pt and 0.82x the wall time.  What stays in real space is
what the drivers calibrate on: q and the reported residual max|T q|, the
update B <- T(B + PW(mu B)), the pin fields G_j and the nu-block borders.

DONATION IS OPT-IN.  `gn(..., donate=True)` dispatches to donate_argnums twins
of the two programs, so B's device buffer is reused for the output instead of
being allocated afresh.  It is not the default because donation INVALIDATES a
caller-held jax array, and the drivers keep the previous B to fall back on
when a step is rejected; a caller whose fallback is host numpy (grow.py's
`Bprev = np.array(B)`) can donate safely, and a numpy input is always safe
because `jnp.asarray` copies it to the device first.
"""
import itertools
import warnings
from typing import NamedTuple

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

from .spectral import (rfft3, irfft3, rfft_wavenumbers, wavenumbers,
                       axis_wavenumbers_1d, axis_masks_1d,
                       dealias_mask_rfft, trunc, trunc3)
from .potential import inv_k2

# Relative CG stop: ||r||^2 below this fraction of ||r0||^2, i.e. "run CG to
# round-off" -- requires float64 (see DESIGN.md).
_CG_RTOL = 1e-26


# ---------------------------------------------------------------------------
# The separable grid description
# ---------------------------------------------------------------------------

class Grid1D(NamedTuple):
    """The grid as broadcastable 1D axis factors, plus the Sobolev exponent.

    Shapes are (Nx,1,1), (1,Ny,1), (1,1,Nz//2+1) throughout, so every derived
    quantity below is an outer product XLA fuses into its consumer instead of
    a materialised O(N^3) multiplier grid.

    THE NYQUIST SPLIT IS LOAD-BEARING (spectral.py module docstring): kd* are
    the DERIVATIVE wavenumbers, with the |k| = N/2 bins zeroed, and drive the
    Leray projector, 1/k^2 and every derivative; kt* are the TRUE wavenumbers,
    Nyquist included, and drive k^2 and the Sobolev weights alone.  Crossing
    the two changes divergence residuals by orders of magnitude.
    """
    kdx: jnp.ndarray
    kdy: jnp.ndarray
    kdz: jnp.ndarray
    ktx: jnp.ndarray
    kty: jnp.ndarray
    ktz: jnp.ndarray
    mx: jnp.ndarray
    my: jnp.ndarray
    mz: jnp.ndarray
    s: jnp.ndarray


def _grid1d(shape, smooth):
    """Build a `Grid1D` for `shape` with Sobolev exponent `smooth`."""
    kd = axis_wavenumbers_1d(shape, zero_nyquist=True)
    kt = axis_wavenumbers_1d(shape)
    m = axis_masks_1d(shape)
    return Grid1D(*kd, *kt, *m, jnp.asarray(float(smooth), dtype=jnp.float64))


def _mask(g):
    """The strict 2/3 retained-band mask as a broadcast product."""
    return g.mx * g.my * g.mz


def _wm2(g):
    """Step weight W^-2 = (1+k^2)^-s, TRUE k^2 (Nyquist included)."""
    return (1.0 + g.ktx ** 2 + g.kty ** 2 + g.ktz ** 2) ** (-g.s)


def _w2(g):
    """Preconditioner weight W^2 = (1+k^2)^s, TRUE k^2."""
    return (1.0 + g.ktx ** 2 + g.kty ** 2 + g.ktz ** 2) ** g.s


def _leray_h(vh, g):
    """Leray projection of an rfft-layout (3, ...) spectrum, DERIVATIVE k.

    1/k^2 is zero wherever the derivative k^2 vanishes -- k = 0 and the zeroed
    Nyquist bins -- so the projector is a no-op on exactly the bins the
    derivative operator cannot see (potential.py module docstring), and the
    mean of the field survives.
    """
    k2 = g.kdx ** 2 + g.kdy ** 2 + g.kdz ** 2
    ik2 = jnp.where(k2 > 0, 1.0 / jnp.where(k2 > 0, k2, 1.0), 0.0)
    d = (g.kdx * vh[0] + g.kdy * vh[1] + g.kdz * vh[2]) * ik2
    return vh - jnp.stack([g.kdx * d, g.kdy * d, g.kdz * d])


# ---------------------------------------------------------------------------
# The packed retained band (the CG's coordinates)
# ---------------------------------------------------------------------------

class Band(NamedTuple):
    """The retained band as packed real coordinates: three axis k^2 factors.

    A real band-limited scalar field is determined by the in-band bins of its
    rfft spectrum, and the retained band |k_i| < N_i/3 is a symmetric BOX --
    |kx| <= ax, |ky| <= ay, 0 <= kz <= az -- so in CENTRED axis order (kx from
    -ax to ax, likewise ky) it is a contiguous slab, reachable by slicing and
    rebuildable by concatenation.  No index arrays, no gather, no scatter.
    The canonical bins are

      - the kz > 0 slab, weight w = 2 (the conjugate partner is not stored;
        in-band kz never reaches the Nyquist plane, so w = 1 cannot occur);
      - the kz = 0 plane, where the rfft array stores BOTH Hermitian partners
        explicitly.  In centred row-major order the partner of entry n is
        entry (nbx*nby - 1 - n), so the canonical half is exactly the tail
        n > (nbx*nby-1)/2 -- lexicographic-positive (kx > 0, or kx = 0 and
        ky > 0) -- weight w = 2, and UNPACKING MUST WRITE THE CONJUGATE BACK
        into the reversed head.  Dropping that mirror halves the kz = 0
        content of the field: the same trap the freeze mask documents (module
        docstring, measured 1e-3 leak);
      - DC, the centre entry n = (nbx*nby-1)/2, weight 1, real part only (a
        real field's DC bin has no imaginary part; the mode is its own
        Hermitian partner).

    Packed vector of a real scalar field f with rfft spectrum F:

        v = concat_b sqrt(w_b / vol) * [Re F_b, Im F_b],  vol = Nx*Ny*Nz,

    laid out as [Re slab, Im slab, Re plane-half, Im plane-half, Re DC].  Then

        <v, v'> = sum_x (T f)(T f')   EXACTLY

    (Parseval), which is the sum inner product the CG and the pin borders use
    -- SUM-units consistency is load-bearing (module docstring).

    The band geometry is carried by the SHAPES of the three k^2 factors:
    nbx = 2 ax + 1, nby = 2 ay + 1, and az, all static at trace time.
    """
    k2x: jnp.ndarray           # (nbx,1,1) TRUE kx^2, centred order
    k2y: jnp.ndarray           # (1,nby,1) TRUE ky^2, centred order
    k2z: jnp.ndarray           # (1,1,az)  TRUE kz^2, kz = 1..az
    s1: jnp.ndarray            # sqrt(1/vol)  (DC)
    s2: jnp.ndarray            # sqrt(2/vol)  (everything else)


def _band(shape):
    """Build the `Band` of `shape` (host numpy: cold path).

    The band is read off `axis_masks_1d` -- the same axis masks the solver
    multiplies by -- so the packed coordinates and `_mask` describe the same
    retained band by construction, and the symmetric-contiguous form the
    slicing assumes is ASSERTED here rather than trusted.
    """
    nx, ny, nz = shape
    ms = [np.asarray(m).ravel() > 0 for m in axis_masks_1d(shape)]
    kt = [np.asarray(k).ravel() for k in axis_wavenumbers_1d(shape)]

    def width(m, k):
        a = int(round(float(np.abs(k[m]).max()))) if m.any() else 0
        if not np.array_equal(m, np.abs(k) < a + 0.5):
            raise ValueError(f"retained band on grid {shape} is not the "
                             "contiguous symmetric run the packing assumes")
        return a

    ax, ay, az = (width(m, k) for m, k in zip(ms, kt))
    cx = np.r_[nx - ax:nx, 0:ax + 1]              # centred x: kx = -ax..ax
    cy = np.r_[ny - ay:ny, 0:ay + 1]
    vol = float(nx * ny * nz)
    f64 = jnp.float64
    return Band(jnp.asarray(kt[0][cx] ** 2, dtype=f64)[:, None, None],
                jnp.asarray(kt[1][cy] ** 2, dtype=f64)[None, :, None],
                jnp.asarray(kt[2][1:az + 1] ** 2, dtype=f64)[None, None, :],
                jnp.asarray(vol ** -0.5, dtype=f64),
                jnp.asarray((2.0 / vol) ** 0.5, dtype=f64))


def _pack(Fh, bd):
    """Packed band coordinates of an rfft-layout scalar spectrum.

    The slices read in-band bins only, so this IS the band projection: a
    caller holding an unmasked spectrum needs no `_mask` multiply first.
    """
    nx, ny = Fh.shape[0], Fh.shape[1]
    ax, ay = (bd.k2x.shape[0] - 1) // 2, (bd.k2y.shape[1] - 1) // 2
    az = bd.k2z.shape[2]
    C = jnp.concatenate([Fh[nx - ax:], Fh[:ax + 1]], axis=0)
    C = jnp.concatenate([C[:, ny - ay:], C[:, :ay + 1]], axis=1)
    P = C[:, :, 0].reshape(-1)                    # kz = 0, centred row-major
    h = (P.size - 1) // 2                         # DC sits at the centre
    C = C[:, :, 1:az + 1]
    return jnp.concatenate([(bd.s2 * C.real).reshape(-1),
                            (bd.s2 * C.imag).reshape(-1),
                            bd.s2 * P[h + 1:].real, bd.s2 * P[h + 1:].imag,
                            (bd.s1 * P[h].real).reshape(1)])


def _unpack(v, bd, shape):
    """The real band-limited field a packed vector stands for (`_pack`'s left
    inverse: `_unpack(_pack(rfft3(f))) == T f`).

    The reversed-conjugate head of the kz = 0 plane is not optional -- see
    `Band`.
    """
    nx, ny, nzh = shape[0], shape[1], shape[2] // 2 + 1
    nbx, nby, az = bd.k2x.shape[0], bd.k2y.shape[1], bd.k2z.shape[2]
    ax, ay = (nbx - 1) // 2, (nby - 1) // 2
    nc, h = nbx * nby * az, (nbx * nby - 1) // 2
    C = (v[:nc] + 1j * v[nc:2 * nc]).reshape(nbx, nby, az) / bd.s2
    P = (v[2 * nc:2 * nc + h] + 1j * v[2 * nc + h:2 * nc + 2 * h]) / bd.s2
    dc = (v[-1] / bd.s1).astype(C.dtype).reshape(1)
    plane = jnp.concatenate([jnp.conj(P[::-1]), dc, P]).reshape(nbx, nby, 1)
    F = jnp.concatenate([plane, C,
                         jnp.zeros((nbx, nby, nzh - 1 - az), C.dtype)], axis=2)
    F = jnp.concatenate([F[:, ay:], jnp.zeros((nbx, ny - nby, nzh), C.dtype),
                         F[:, :ay]], axis=1)
    F = jnp.concatenate([F[ax:], jnp.zeros((nx - nbx, ny, nzh), C.dtype),
                         F[:ax]], axis=0)
    return irfft3(F, shape)


def _precond_vec(g, bd):
    """The packed preconditioner M^-1 = (1+k^2)^s, TRUE k^2, as a vector.

    Built ONCE per sweep and closed over by the CG body: the exponent is a
    traced scalar, so `**s` is a real pow, and an O(n_band) pow per sweep is
    free where an O(n_band) pow per iteration would not be.  Re and Im of a
    bin carry the same weight, hence the repeated blocks; DC has k^2 = 0.
    """
    w = ((1.0 + bd.k2x + bd.k2y + bd.k2z) ** g.s).reshape(-1)
    plane = ((1.0 + bd.k2x + bd.k2y) ** g.s).reshape(-1)
    p = plane[(plane.size - 1) // 2 + 1:]         # the canonical half
    return jnp.concatenate([w, w, p, p, jnp.ones(1, dtype=w.dtype)])


def _freeze_h(vh, fidx):
    """Zero the frozen rfft bins of a (..., Nx, Ny, Nz//2+1) spectrum.

    `fidx` is the static (n, 3) index array built by `_freeze_indices`; it
    already carries the kz = 0 Hermitian partners, and n = 0 traces to a
    no-op scatter.
    """
    return vh.at[..., fidx[:, 0], fidx[:, 1], fidx[:, 2]].set(0.0)


# ---------------------------------------------------------------------------
# Operators (jit-traceable building blocks, called from _gn_solve_mu)
# ---------------------------------------------------------------------------

def _PW(v, g, fidx):
    """W^-2 . freeze . T . P applied to a real (3, ...) field.

    The admissible-direction operator: Leray-project, drop the frozen bins and
    everything outside the retained band, then apply the Sobolev weight
    W^-2 = (1+k^2)^-s.  Two batched transforms.

    Mask, weight and freeze are diagonal and commute, so they are applied in
    the order that fuses: the two multipliers ride the Leray expression in one
    elementwise pass and the scatter lands last, on the buffer the inverse
    transform is about to read.
    """
    vh = (_mask(g) * _wm2(g)) * _leray_h(rfft3(v), g)
    return irfft3(_freeze_h(vh, fidx), v.shape[1:])


def _minv(r, g):
    """Preconditioner M^-1 = T (1+k^2)^s (inverts the step's weight)."""
    return irfft3(_w2(g) * _mask(g) * rfft3(r), r.shape)


def _tq(B, g):
    """The exact Galerkin residual T q, q = (|B|^2 - 1)/2."""
    return trunc(0.5 * ((B ** 2).sum(0) - 1.0), _mask(g), True)


def _normal_op(mu, B, g, fidx):
    """A(mu) = T( B . PW(mu B) ): the exact Galerkin Jacobian in mu.

    Real-space form, kept as the definition of the operator (`descent.py` and
    the equivalence tests read it); the CG itself runs `_normal_op_p`.
    """
    return trunc((B * _PW(mu[None] * B, g, fidx)).sum(0), _mask(g), True)


def _normal_op_p(v, B, g, fidx, bd):
    """`_normal_op` in packed band coordinates: v -> pack(A(unpack(v))).

    The trailing `trunc` of the real-space form is gone, not skipped: `_pack`
    reads in-band bins only, so the band projection is the packing slice.
    Eight scalar transforms (unpack 1, PW 6, pack 1).
    """
    D = _PW(_unpack(v, bd, B.shape[1:])[None] * B, g, fidx)
    return _pack(rfft3((B * D).sum(0)), bd)


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


def _pin_dir(mu, nu, B, G, g, fidx):
    """The admissible direction D = PW(mu B + sum_j nu_j G_j)."""
    V = mu[None] * B + (nu[:, None, None, None, None] * G).sum(0)
    return _PW(V, g, fidx)


def _bordered_op(mu, nu, B, G, g, fidx):
    """The bordered normal operator (A_mu, A_nu) at (mu, nu), one PW.

    A_mu = T(B . D) extends `_normal_op`; A_nu_i = sum_x G_i . D is the pin
    row.  Symmetric because both borders are the SAME sum inner product.
    """
    D = _pin_dir(mu, nu, B, G, g, fidx)
    return (trunc((B * D).sum(0), _mask(g), True),
            (G * D[None]).sum(axis=(1, 2, 3, 4)))


def _bordered_op_p(v, nu, B, G, g, fidx, bd):
    """`_bordered_op` with the FIELD block packed: (pack(A_mu), A_nu).

    Only the mu row moves: the nu row is sum_x G_i . D, a real-space sum over
    the m pin fields, and packing it would buy nothing (m is 1 or 2).
    """
    D = _pin_dir(_unpack(v, bd, B.shape[1:]), nu, B, G, g, fidx)
    return (_pack(rfft3((B * D).sum(0)), bd),
            (G * D[None]).sum(axis=(1, 2, 3, 4)))


def _pin_precond(G, g, fidx):
    """nu-block preconditioner diagonal d_j = sum_x G_j . PW(G_j).

    The diagonal of the pin-pin block of the bordered operator; recomputed
    once per sweep.  Floored at 1e-300 so a pin whose bin the freeze mask
    kills cannot divide by zero (the constructor rejects that case anyway).
    """
    PG = jax.vmap(_PW, in_axes=(0, None, None))(G, g, fidx)
    return jnp.maximum((G * PG).sum(axis=(1, 2, 3, 4)), 1e-300)


def _pin_rel_error(e, targets):
    """max_j |e_j - c_j| / max(c_j, 1) -- `gn`'s pin stopping test.

    The max(c_j, 1) scale is relative for the large targets that matter and
    absolute for targets near zero, where a relative test is meaningless
    (a pin at c_j = 0 asks for a mode to be absent, not for 12 digits of it).
    """
    return (jnp.abs(e - targets) / jnp.maximum(targets, 1.0)).max()


def _dif_1d(f, axis, g):
    """Spectral d/dx_axis of a real scalar field; `axis` is a jit static."""
    return irfft3(1j * (g.kdx, g.kdy, g.kdz)[axis] * rfft3(f), f.shape)


def _div_1d(F, g):
    """Spectral divergence of a real (3, ...) field (monitoring diagnostic)."""
    Fh = rfft3(F)
    return irfft3(1j * (g.kdx * Fh[0] + g.kdy * Fh[1] + g.kdz * Fh[2]),
                  F.shape[1:])


def _project(B, g):
    """Div-free + in-band cleanup of a state; the mean is preserved."""
    return trunc3(irfft3(_leray_h(rfft3(B), g), B.shape[1:]), _mask(g), True)


def _residual_mu(B, g):
    """(div B, T q) -- div is monitored, T q is what `gn` drives to zero."""
    return _div_1d(B, g), _tq(B, g)


# ---------------------------------------------------------------------------
# The compiled Gauss-Newton solve
# ---------------------------------------------------------------------------

def _gn_program_mu(B, g, fidx, bd, sweeps, cgit, tol, verbose):
    """`sweeps` mu-form Gauss-Newton sweeps, each one preconditioned-CG solve
    of A(mu) = -T q followed by B <- T(B + PW(mu B)).

    The CG runs in packed band coordinates (`Band`); everything outside it --
    q, the reported residual, the update -- stays in real space.

    Semantics follow legacy `_gn_solve`:
      - a sweep whose entry residual is already < tol is a no-op, and the CG
        count of the previous sweep is preserved;
      - if all `sweeps` run without converging, the returned residual is
        recomputed AFTER the final update;
      - returns (B, max|T q|, CG iterations of the last sweep that ran CG) --
        the CG count is a conditioning/fold proxy the drivers consume.
    """
    shape = B.shape[1:]

    def sweep_cond(carry):
        _, k, _, _, done = carry
        return jnp.logical_and(k < sweeps, jnp.logical_not(done))

    def sweep_body(carry):
        Bc, k, _res_prev, ci_prev, _done_prev = carry
        # One forward transform serves both faces of the residual: T q in real
        # space, which the drivers calibrate on, and its packed form -T q.
        qh = rfft3(0.5 * ((Bc ** 2).sum(0) - 1.0))
        q = irfft3(_mask(g) * qh, shape)
        res_now = jnp.abs(q).max()
        if verbose:
            jax.debug.print("    mu sweep {k}: residual {res:.2e}",
                            k=k, res=res_now)
        converged = res_now < tol

        def do_cg(_):
            w = _precond_vec(g, bd)               # M^-1, one pow per sweep
            r0 = -_pack(qh, bd)
            z0 = w * r0
            rz0 = (r0 * z0).sum()
            rr0 = (r0 * r0).sum()

            def cg_cond(state):
                _, r, _, _, n_done = state
                return jnp.logical_and(n_done < cgit,
                                       (r * r).sum() > _CG_RTOL * rr0)

            def cg_body(state):
                mu, r, p, rz, n_done = state
                Ap = _normal_op_p(p, Bc, g, fidx, bd)
                al = rz / (p * Ap).sum()
                mu, r = mu + al * p, r - al * Ap
                z = w * r
                rz2 = (r * z).sum()
                return mu, r, z + (rz2 / rz) * p, rz2, n_done + 1

            init = (jnp.zeros_like(r0), r0, z0, rz0, jnp.array(0))
            mu, *_, n_done = lax.while_loop(cg_cond, cg_body, init)
            dB = _PW(_unpack(mu, bd, shape)[None] * Bc, g, fidx)
            return trunc3(Bc + dB, _mask(g), True), n_done

        def keep(_):
            return Bc, ci_prev

        Bn, ci_new = lax.cond(converged, keep, do_cg, operand=None)
        return Bn, k + 1, res_now, ci_new, converged

    B0 = trunc3(B, _mask(g), True)           # keep B in the retained band
    init = (B0, jnp.array(0), jnp.asarray(jnp.inf), jnp.array(0),
            jnp.array(False))
    Bf, _kf, res_last, cif, donef = lax.while_loop(sweep_cond, sweep_body,
                                                   init)
    res_final = jnp.where(donef, res_last, jnp.abs(_tq(Bf, g)).max())
    return Bf, res_final, cif


def _gn_program_pinned(B, targets, g, fidx, pinR, bd, sweeps, cgit, tol,
                       verbose):
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

    Only the FIELD block of the CG is packed (`Band`): the nu block is m
    scalars, its border rows are real-space sums over the pin fields, and its
    preconditioner is a diagonal of m numbers.
    """
    shape = B.shape[1:]

    def sweep_cond(carry):
        _, k, _, _, done = carry
        return jnp.logical_and(k < sweeps, jnp.logical_not(done))

    def sweep_body(carry):
        Bc, k, _res_prev, ci_prev, _done_prev = carry
        qh = rfft3(0.5 * ((Bc ** 2).sum(0) - 1.0))
        q = irfft3(_mask(g) * qh, shape)
        res_now = jnp.abs(q).max()
        G = _pin_fields(Bc, pinR)                  # re-linearisation
        e = _pin_energies(G)
        pin_now = _pin_rel_error(e, targets)
        if verbose:
            jax.debug.print("    mu sweep {k}: residual {res:.2e}  "
                            "pin {pin:.2e}", k=k, res=res_now, pin=pin_now)
        converged = jnp.maximum(res_now, pin_now) < tol

        def do_cg(_):
            w = _precond_vec(g, bd)               # M^-1, one pow per sweep
            dnu = _pin_precond(G, g, fidx)
            r10, r20 = -_pack(qh, bd), targets - e
            z10 = w * r10
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
                A1, A2 = _bordered_op_p(p1, p2, Bc, G, g, fidx, bd)
                al = rz / ((p1 * A1).sum() + (p2 * A2).sum())
                mu, nu = mu + al * p1, nu + al * p2
                r1, r2 = r1 - al * A1, r2 - al * A2
                z1, z2 = w * r1, r2 / dnu
                rz2 = (r1 * z1).sum() + (r2 * z2).sum()
                return (mu, nu, r1, r2, z1 + (rz2 / rz) * p1,
                        z2 + (rz2 / rz) * p2, rz2, n_done + 1)

            init = (jnp.zeros_like(r10), jnp.zeros_like(targets), r10, r20,
                    z10, z20, rz0, jnp.array(0))
            mu, nu, *_, n_done = lax.while_loop(cg_cond, cg_body, init)
            dB = _pin_dir(_unpack(mu, bd, shape), nu, Bc, G, g, fidx)
            return trunc3(Bc + dB, _mask(g), True), n_done

        def keep(_):
            return Bc, ci_prev

        Bn, ci_new = lax.cond(converged, keep, do_cg, operand=None)
        return Bn, k + 1, res_now, ci_new, converged

    B0 = trunc3(B, _mask(g), True)           # keep B in the retained band
    init = (B0, jnp.array(0), jnp.asarray(jnp.inf), jnp.array(0),
            jnp.array(False))
    Bf, _kf, res_last, cif, donef = lax.while_loop(sweep_cond, sweep_body,
                                                   init)
    res_final = jnp.where(donef, res_last, jnp.abs(_tq(Bf, g)).max())
    return Bf, res_final, cif


# The four compiled programs.  Each pair is the SAME traced function; the
# `_d` twin only adds donation of B (argument 0), so the two share nothing but
# must be created once here -- wrapping per call would recompile every time.
_gn_solve_mu = jax.jit(_gn_program_mu, static_argnames=("verbose",))
_gn_solve_mu_d = jax.jit(_gn_program_mu, static_argnames=("verbose",),
                         donate_argnums=(0,))
_gn_solve_pinned = jax.jit(_gn_program_pinned, static_argnames=("verbose",))
_gn_solve_pinned_d = jax.jit(_gn_program_pinned, static_argnames=("verbose",),
                             donate_argnums=(0,))

# Standalone jitted entry points for the small public helpers.
_dif_jit = jax.jit(_dif_1d, static_argnums=(1,))
_trunc_jit = jax.jit(lambda f, g: trunc(f, _mask(g), True))
_trunc3_jit = jax.jit(lambda F, g: trunc3(F, _mask(g), True))
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


def _freeze_indices(shape, triples):
    """The frozen rfft bins as a static (n, 3) int32 index array.

    Validation is `_bins`': representable, in band, and kz = 0 triples carry
    their Hermitian partner (-kx, -ky, 0), which the solver's scatter then
    zeroes together with the bin itself.  Duplicates are dropped so that the
    array doubles as the definitive frozen-bin set for `_pin_masks`.  n = 0 is
    a legitimate (0, 3) array: the scatter traces to a no-op.
    """
    maskR = np.asarray(dealias_mask_rfft(shape))
    bins = []
    for t in triples:
        for b in _bins(shape, t, maskR, "freeze"):
            if b not in bins:
                bins.append(b)
    return np.asarray(bins, dtype=np.int32).reshape(len(bins), 3)


def _freeze_mask(shape, triples):
    """Multiplier array (rfft layout): 1 everywhere, 0 on the frozen bins and,
    for kz = 0 bins, on their Hermitian partners (-kx, -ky, 0) as well.

    Compatibility only -- the solver itself carries `_freeze_indices`.
    """
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
        # The grid as 1D axis factors (module docstring): kd* derivative
        # wavenumbers with Nyquist zeroed, kt* true wavenumbers, m* the strict
        # 2/3 axis masks, s the Sobolev exponent as a traced scalar.
        self.grid1d = _grid1d(self.shape, self.smooth)
        # The CG's coordinates: the retained band as packed real degrees of
        # freedom (three 1D k^2 factors, O(N) -- see `Band`).
        self.band = _band(self.shape)

        pinned = [_canonical(t) for t in freeze]
        if len(pinned) >= 3 and not noncoplanar(pinned):
            warnings.warn("all frozen mode triples are coplanar: the field "
                          "they generate cannot be genuinely 3D", stacklevel=2)
        if self.fix_mean and (0, 0, 0) not in pinned:
            pinned.append((0, 0, 0))
        self.freeze = tuple(pinned)
        fidx = _freeze_indices(self.shape, self.freeze)
        self.freeze_idx = jnp.asarray(fidx)

        self.pins = tuple(_canonical(t) for t in pins)
        self.m = len(self.pins)
        frozen = {tuple(int(v) for v in b) for b in fidx}
        self.pinR = jnp.asarray(_pin_masks(self.shape, self.pins, frozen))
        self._K = None                                  # full layout, lazy
        self._full = {}                       # compatibility grids, lazy

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

    # -- compatibility: the full-grid multiplier arrays ----------------------
    # The solve itself never touches these -- it carries `grid1d` and rebuilds
    # each of them as a fused broadcast wherever it is needed.  They are kept,
    # built on first access and cached, because diagnostics, scripts and tests
    # read them; touching one costs O(N^3) device memory for the solver's
    # lifetime, which is exactly what the hot path was refactored to avoid.
    def _cached(self, name, build):
        if name not in self._full:
            self._full[name] = build()
        return self._full[name]

    @property
    def KR(self):
        """Derivative wavenumbers, rfft layout, Nyquist bins zeroed."""
        return self._cached(
            "KR", lambda: rfft_wavenumbers(self.shape, zero_nyquist=True))

    @property
    def K2r(self):
        """TRUE k^2 (Nyquist included): the weights' basis, not the
        derivative's."""
        return self._cached(
            "K2r", lambda: (rfft_wavenumbers(self.shape) ** 2).sum(0))

    @property
    def maskR(self):
        """Strict |k_i| < N_i/3 retained-band mask, rfft layout."""
        return self._cached("maskR", lambda: dealias_mask_rfft(self.shape))

    @property
    def invK2(self):
        """1/k^2, zero where the DERIVATIVE k^2 vanishes."""
        return self._cached("invK2", lambda: inv_k2(self.KR))

    @property
    def Wm2r(self):
        """Step weight (1+k^2)^-s."""
        return self._cached("Wm2r", lambda: (1.0 + self.K2r) ** (-self.smooth))

    @property
    def W2r(self):
        """Preconditioner weight (1+k^2)^s."""
        return self._cached("W2r", lambda: (1.0 + self.K2r) ** self.smooth)

    @property
    def freezeR(self):
        """Freeze multiplier: 0 on the frozen bins (partners included), 1
        elsewhere -- the array form of `freeze_idx`."""
        return self._cached(
            "freezeR",
            lambda: jnp.asarray(_freeze_mask(self.shape, self.freeze)))

    # -- small public operations --------------------------------------------
    def dif(self, f, axis):
        """Spectral partial derivative of a real scalar field."""
        return _dif_jit(jnp.asarray(f), axis, self.grid1d)

    def trunc(self, f):
        """Project a scalar field onto the retained band."""
        return _trunc_jit(jnp.asarray(f), self.grid1d)

    def trunc3(self, F):
        """Project a stacked (3, ...) field onto the retained band."""
        return _trunc3_jit(jnp.asarray(F), self.grid1d)

    def project(self, B):
        """State cleanup: div-free (mean preserved) and in-band.

        Call this ONCE on the initial state of a continuation -- `gn`
        preserves div B but cannot remove it.  Deliberately NOT freeze-masked:
        this is a repair of the state, not a step along the manifold, and a
        frozen bin with a divergence error in it is still an error.
        """
        return _project_jit(jnp.asarray(B), self.grid1d)

    def residual(self, B):
        """(div B, T q): the monitored divergence and the exact Galerkin
        constraint residual whose max-norm `gn` reports."""
        return _residual_jit(jnp.asarray(B), self.grid1d)

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
           pin_targets=None, donate=False):
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

        `donate=True` hands B's device buffer to the compiled program, which
        writes the result into it: one fewer N^3 field allocated per call.  It
        INVALIDATES a caller-held jax array -- any later use of it raises.  A
        HOST NUMPY input is always safe (`jnp.asarray` copies it to the device
        first, and the copy is what gets donated), so a driver whose rejection
        fallback is numpy (grow.py) can donate unconditionally; a driver
        holding the previous state as a jax array must not.
        """
        B = jnp.asarray(B)
        if self.m == 0:
            if pin_targets is not None and np.size(pin_targets) > 0:
                raise ValueError("pin_targets given to a solver built without "
                                 "pins")
            prog = _gn_solve_mu_d if donate else _gn_solve_mu
            Bf, res, ci = prog(B, self.grid1d, self.freeze_idx, self.band,
                               int(sweeps), int(cgit), float(tol),
                               bool(verbose))
            return Bf, float(res), int(ci)

        if pin_targets is None:               # hold whatever we came in with
            targets = _pin_energies_jit(self.trunc3(B), self.pinR)
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
        prog = _gn_solve_pinned_d if donate else _gn_solve_pinned
        Bf, res, ci = prog(B, targets, self.grid1d, self.freeze_idx, self.pinR,
                           self.band, int(sweeps), int(cgit), float(tol),
                           bool(verbose))
        return Bf, float(res), int(ci)
