"""Vector-potential building blocks for the carrier-free (v2) stack: spectral
curl, its Coulomb-gauge inverse, the Leray (div-free) projector, and the
divergence.

WHY THIS MODULE EXISTS.  The mu-form Gauss-Newton step (`solver_mu`) moves B
along W^-2 P (mu B) -- a div-free direction by construction -- so it PRESERVES
div B exactly but can never REMOVE a pre-existing divergence error.  v2
therefore represents states the way the constraint wants them: a uniform mean
plus a curl.  Seeds are built as `curl a`, states are projected once at the
entry of a continuation, and after that the solver keeps div at round-off for
free.

CONVENTIONS.  All functions take and return REAL fields, stacked (3, ...) for
vectors, and the caller's derivative wavenumber grid `KR` in the rfft layout
(`rfft_wavenumbers(shape, zero_nyquist=True)` -- the Nyquist trap, DESIGN.md).
Because KR zeroes the Nyquist bins, 1/k^2 is taken to be ZERO wherever the
derivative k^2 vanishes (k = 0 and those Nyquist bins): every operator here is
then a no-op on exactly the bins the derivative operator cannot see, which is
what makes the round trips below exact.

    inv_curl(curl(A)) == A               for in-band Coulomb-gauge A
    curl(inv_curl(B)) + mean(B) == B     for in-band div-free B

(both to ~1e-13).  The four operators are undecorated so they can be called
from inside already-jitted solver code, in the style of `spectral.py`; the
`*_jit` aliases at the bottom are the standalone entry points for cold-path
callers (seeds, diagnostics).
"""
import jax
import jax.numpy as jnp

from .spectral import rfft3, irfft3


# ---------------------------------------------------------------------------
# Small spectral helpers
# ---------------------------------------------------------------------------

def _cross_h(K, Fh):
    """k x F in a spectral layout (component axis first, batched over bins)."""
    return jnp.stack([K[1] * Fh[2] - K[2] * Fh[1],
                      K[2] * Fh[0] - K[0] * Fh[2],
                      K[0] * Fh[1] - K[1] * Fh[0]])


def inv_k2(KR):
    """1/k^2 in the rfft layout, ZERO wherever the derivative k^2 vanishes.

    That is the k = 0 bin and -- since KR carries zeroed Nyquist planes -- the
    Nyquist bins as well.  The nested `where` keeps the division away from the
    zero bins so no NaN is produced (or differentiated through).
    """
    k2 = (KR ** 2).sum(0)
    return jnp.where(k2 > 0, 1.0 / jnp.where(k2 > 0, k2, 1.0), 0.0)


# ---------------------------------------------------------------------------
# The operators (jit-traceable; not decorated)
# ---------------------------------------------------------------------------

def curl(F, KR):
    """Spectral curl of a real (3, ...) field: one batched rfft pair."""
    return irfft3(1j * _cross_h(KR, rfft3(F)), F.shape[1:])


def div(F, KR):
    """Spectral divergence of a real (3, ...) field (monitoring diagnostic)."""
    return irfft3(1j * (KR * rfft3(F)).sum(0), F.shape[1:])


def divfree_project(F, KR):
    """Leray projector P F = F - grad div^-1 div F, spectrally
    F_hat - k (k.F_hat)/k^2.

    The k = 0 bins are untouched (1/k^2 -> 0 there), so the MEAN of F survives
    the projection -- essential: the mean field B_bar is the continuation's
    anchor, not a divergence error.
    """
    Fh = rfft3(F)
    Fh = Fh - KR * ((KR * Fh).sum(0) * inv_k2(KR))[None]
    return irfft3(Fh, F.shape[1:])


def inv_curl(B, KR):
    """Coulomb-gauge vector potential A with curl A = B - mean(B).

    A_hat = i k x B_hat / k^2 (k = 0 -> 0), which is div-free by construction
    and is the minimum-norm potential.  Any solenoidal part of B is recovered
    exactly; a non-solenoidal part is silently dropped (curl A is div-free), so
    this doubles as the "represent B as a potential" half of `project`.
    """
    return irfft3(1j * _cross_h(KR, rfft3(B)) * inv_k2(KR), B.shape[1:])


# ---------------------------------------------------------------------------
# Standalone jitted entry points for cold-path callers
# ---------------------------------------------------------------------------

curl_jit = jax.jit(curl)
div_jit = jax.jit(div)
divfree_project_jit = jax.jit(divfree_project)
inv_curl_jit = jax.jit(inv_curl)
