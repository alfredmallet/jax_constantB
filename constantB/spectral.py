"""Spectral machinery: wavenumber grids, real-FFT helpers, the strict 2/3-rule
dealias mask, spectral derivatives, and zero-pad grid refinement.

All periodic fields live on [0, 2pi)^3, so wavenumbers are integers.

REAL FFTs.  Every field here is real, so the hot path uses rfftn/irfftn over
the last three axes: half the work and memory of the complex transforms used
by the numpy reference implementation.  This is safe because
  - the x/y axes of an rfftn keep the full fftfreq layout (identical k's);
  - on the half z-axis the only difference is the Nyquist bin's sign
    convention (+N/2 vs -N/2), and for both the derivative (1j*k*F -> purely
    imaginary Nyquist bin, symmetrised away by the real inverse transform)
    and the mask / Sobolev weight (both depend on |k| or k^2 only) the two
    conventions give pointwise-identical results.
Helpers that take stacked (3, Nx, Ny, Nz) fields transform all three
components in ONE batched call.

Results agree with the full-FFT numpy reference to round-off; parity is
validated at convergence (see DESIGN.md).
"""
import itertools

import numpy as np
import jax.numpy as jnp

TWOPI = 2 * np.pi

# Guard band for the strict 2/3-rule inequality |k| < N/3: keeps float
# comparison honest when N is a multiple of 3 (see dealias_mask).
_STRICT = 1e-12


# ---------------------------------------------------------------------------
# Wavenumber grids
# ---------------------------------------------------------------------------

def _freqs(n):
    """Integer wavenumbers of an n-point axis, full FFT layout."""
    return np.fft.fftfreq(n, d=TWOPI / n) * TWOPI


def _rfreqs(n):
    """Integer wavenumbers of an n-point axis, real-FFT (half) layout."""
    return np.fft.rfftfreq(n, d=TWOPI / n) * TWOPI


def wavenumbers(shape):
    """(3, Nx, Ny, Nz) stacked wavenumber grids, full FFT layout (jax).

    Public / compatibility helper (e.g. for spectrally curl-ing a seed with
    full fftn).  The solver's internals use `rfft_wavenumbers`.
    """
    ks = [_freqs(n) for n in shape]
    return jnp.stack(jnp.meshgrid(*[jnp.asarray(k) for k in ks], indexing="ij"))


def _axis_k(n, freqs, zero_nyquist):
    """One axis of the rfft wavenumber layout, Nyquist optionally zeroed."""
    k = freqs.copy()
    if zero_nyquist and n % 2 == 0:
        k[np.abs(np.abs(k) - n / 2) < _STRICT] = 0.0
    return k


def _rfft_axes(shape, zero_nyquist=False):
    """The three 1D rfft-layout wavenumber axes (host numpy)."""
    return [_axis_k(shape[0], _freqs(shape[0]), zero_nyquist),
            _axis_k(shape[1], _freqs(shape[1]), zero_nyquist),
            _axis_k(shape[2], _rfreqs(shape[2]), zero_nyquist)]


def rfft_wavenumbers(shape, zero_nyquist=False):
    """(3, Nx, Ny, Nz//2+1) stacked wavenumber grids, rfftn layout (jax).

    zero_nyquist=True zeroes the |k| = N/2 bins (even N).  REQUIRED for the
    derivative operator: the full-FFT reference computes derivatives as
    real(ifftn(1j*k*fftn(f))), and 1j*k makes the self-conjugate Nyquist
    planes anti-Hermitian, so their contribution is purely imaginary and the
    real() DISCARDS it.  An rfft round trip does not discard the x/y-axis
    Nyquist planes, so to reproduce the reference operator exactly the
    derivative multiplier must be zero there.  (Verified: without this, the
    div residual of a state with near-Nyquist content is wrong by orders of
    magnitude.)  Quantities that depend on k^2 only -- the CG preconditioner
    and the Sobolev weight -- keep the true Nyquist values (matching the
    reference), so they use zero_nyquist=False.
    """
    ks = _rfft_axes(shape, zero_nyquist)
    return jnp.stack(jnp.meshgrid(*[jnp.asarray(k) for k in ks], indexing="ij"))


def axis_wavenumbers_1d(shape, zero_nyquist=False):
    """Three float64 arrays shaped (Nx,1,1), (1,Ny,1), (1,1,Nz//2+1): the
    rfft-layout wavenumbers as broadcastable 1D factors.

    Same Nyquist-zeroing rule as `rfft_wavenumbers`, of which this is the
    separable form: `jnp.stack(jnp.broadcast_arrays(*axis_wavenumbers_1d(...)))`
    reproduces it bin for bin.  O(N) storage instead of O(N^3), so the solver
    carries these and fuses the outer products into the kernels that use them.
    """
    ks = _rfft_axes(shape, zero_nyquist)
    return (jnp.asarray(ks[0])[:, None, None],
            jnp.asarray(ks[1])[None, :, None],
            jnp.asarray(ks[2])[None, None, :])


def axis_masks_1d(shape):
    """Three float64 {0,1} arrays, same shapes as `axis_wavenumbers_1d`: the
    strict |k| < N/3 axis masks whose product is `dealias_mask_rfft`."""
    ms = [_axis_mask(shape[0], _freqs(shape[0])),
          _axis_mask(shape[1], _freqs(shape[1])),
          _axis_mask(shape[2], _rfreqs(shape[2]))]
    return (jnp.asarray(ms[0].astype(float))[:, None, None],
            jnp.asarray(ms[1].astype(float))[None, :, None],
            jnp.asarray(ms[2].astype(float))[None, None, :])


def numpy_wavenumbers(shape):
    """Host-numpy meshed wavenumbers (full layout), for one-shot diagnostics
    and plotting that never touch jax."""
    return np.meshgrid(*[_freqs(n) for n in shape], indexing="ij")


# ---------------------------------------------------------------------------
# Real-FFT helpers (batched over any leading axes)
# ---------------------------------------------------------------------------

def rfft3(f):
    """rfftn over the last three axes; leading axes (if any) are batched."""
    return jnp.fft.rfftn(f, axes=(-3, -2, -1))


def irfft3(F, shape):
    """Inverse of rfft3; `shape` is the (Nx, Ny, Nz) of the real output."""
    return jnp.fft.irfftn(F, s=shape, axes=(-3, -2, -1))


# ---------------------------------------------------------------------------
# The strict 2/3-rule mask
# ---------------------------------------------------------------------------

def _axis_mask(n, freqs):
    """Retained-band indicator |k| < n/3, STRICTLY.

    Strictness is load-bearing: with the inclusive cutoff |k| <= n/3, products
    of two modes ON the cutoff alias exactly onto the retained-band edge, and
    Orszag's exactness argument for quadratic nonlinearities fails.  See
    DESIGN.md ("Resolution honesty").
    """
    return np.abs(freqs) < n / 3 - _STRICT


def _outer3(ms):
    """Separable 3D mask from three 1D axis masks."""
    return (ms[0][:, None, None] * ms[1][None, :, None]
            * ms[2][None, None, :]).astype(float)


def dealias_mask(shape):
    """Full-layout retained-band mask (compatibility helper)."""
    return jnp.asarray(_outer3([_axis_mask(n, _freqs(n)) for n in shape]))


def dealias_mask_rfft(shape):
    """rfftn-layout retained-band mask, as used by the solver."""
    return jnp.asarray(_outer3([_axis_mask(shape[0], _freqs(shape[0])),
                                _axis_mask(shape[1], _freqs(shape[1])),
                                _axis_mask(shape[2], _rfreqs(shape[2]))]))


# ---------------------------------------------------------------------------
# Derivatives and band projection (jit-traceable building blocks; these are
# called from inside already-jitted solver code, so they are not decorated)
# ---------------------------------------------------------------------------

def dif(f, axis, KR):
    """Spectral d/dx_axis of a real scalar field (rfft round trip)."""
    return irfft3(1j * KR[axis] * rfft3(f), f.shape)


def trunc(f, maskR, dealias):
    """Project a real scalar field onto the retained band.

    `dealias` must be a plain python bool (a jit static argument in every
    caller), so this `if` is resolved at trace time -- at most two compiled
    variants per grid shape, never a traced conditional.
    """
    if not dealias:
        return f
    return irfft3(maskR * rfft3(f), f.shape)


def trunc3(F, maskR, dealias):
    """Project a stacked (3, ...) field onto the retained band (one batched
    transform pair, not three)."""
    if not dealias:
        return F
    return irfft3(maskR * rfft3(F), F.shape[1:])


# ---------------------------------------------------------------------------
# Host-numpy twins (for diagnostics/plotting that never touch jax)
# ---------------------------------------------------------------------------

def numpy_dif(f, axis, K):
    """Host-numpy spectral derivative; K from numpy_wavenumbers."""
    return np.real(np.fft.ifftn(1j * K[axis] * np.fft.fftn(f)))


# ---------------------------------------------------------------------------
# Grid refinement
# ---------------------------------------------------------------------------

def zero_pad(f, new_shape):
    """Spectral interpolation of a real field to a finer grid (exact for the
    retained trigonometric modes).

    Deliberately NOT jitted: it is called once per refinement step with a
    different static output shape each time, so jit would recompile on every
    call for no benefit.  Uses the full complex FFT for simplicity -- this is
    a cold path.
    """
    f = jnp.asarray(f)
    F = jnp.fft.fftn(f)
    G = jnp.zeros(new_shape, dtype=jnp.complex128)
    n = f.shape
    sl_old = [(slice(0, ni // 2), slice(ni - ni // 2, ni)) for ni in n]
    sl_new = [(slice(0, ni // 2), slice(Ni - ni // 2, Ni))
              for ni, Ni in zip(n, new_shape)]
    for choice in itertools.product((0, 1), repeat=3):
        idx_new = tuple(sl_new[d][choice[d]] for d in range(3))
        idx_old = tuple(sl_old[d][choice[d]] for d in range(3))
        G = G.at[idx_new].set(F[idx_old])
    return jnp.real(jnp.fft.ifftn(G)) * (np.prod(new_shape) / np.prod(n))
