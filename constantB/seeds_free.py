"""Carrier-free seeds: divergence-free perturbations built as b = curl a,
with enough metadata to rebuild the IDENTICAL continuum field on any grid.

Why a separate module from `seeds.py`: the v2 stack grows states from a
uniform mean field, so there is no 1D carrier and no linearised deformation
ODE -- only a vector potential and its curl.  Everything here is exactly
divergence-free before truncation (a spectral curl is), and truncation with
the strict 2/3 mask keeps it so on the retained band.

GRID INDEPENDENCE is the load-bearing property: a continuation run refines
its grid, and the seed pushed at eps must be the same continuum field before
and after the refinement, or the ladder silently changes the object being
continued.  Two mechanisms provide it:
  * the random seed's coefficients are keyed on the integer wavevector, not
    on an array index, so the same key gives the same continuum field on any
    grid (spec v2 section 4);
  * the normalisation constant is fixed once and RECORDED in the metadata,
    so rebuilds divide by the same number.  (Renormalising per grid would
    not be grid-independent: max|b| of a fixed band-limited field over grid
    points drifts by ~1% between grids, since the continuum maximum does not
    sit on a node.)  Hence max|b| = 1 exactly on the construction grid for
    file/blob seeds, and to ~1% for random seeds, whose constant is the max
    over a canonical dense evaluation grid determined by kmax alone.

Host numpy throughout (cold path): jax is touched only through the passed
solver's `S.trunc3` / `S.K`, and via `zero_pad` when a stored potential is
lifted to a finer grid.
"""
import numpy as np

from .spectral import TWOPI, numpy_wavenumbers, zero_pad

# Canonical evaluation grid for the random seed's normalisation: >= 8 points
# per shortest wavelength (16 per unit wavenumber), capped so the one-off
# cost stays trivial.  Depends on kmax ONLY -- never on the working grid.
_CANON_MIN, _CANON_MAX = 32, 192


# ---------------------------------------------------------------------------
# curl / normalisation plumbing
# ---------------------------------------------------------------------------

def _curl_hat(ah, K):
    """Real b = curl a from the full-layout spectral potential `ah`.

    Same expression (and the same full-FFT path) as legacy `blob_seed`, so
    blob geometry reproduces bit-for-bit.
    """
    KX, KY, KZ = K
    return np.stack([
        np.real(np.fft.ifftn(1j * (KY * ah[2] - KZ * ah[1]))),
        np.real(np.fft.ifftn(1j * (KZ * ah[0] - KX * ah[2]))),
        np.real(np.fft.ifftn(1j * (KX * ah[1] - KY * ah[0])))])


def _curl_trunc(ah, S):
    """Spectral curl of `ah` followed by the solver's band projector."""
    b = _curl_hat(ah, np.asarray(S.K))
    return np.asarray(S.trunc3(b), dtype=float)


def _normalise(b, norm):
    """(b / c, c): c = max|b| on this grid when `norm` is None, else `norm`."""
    c = float(np.abs(b).max()) if norm is None else float(norm)
    if not np.isfinite(c) or c <= 0:
        raise ValueError(f"seed normalisation constant is not positive: {c}")
    return b / c, c


def _grid(shape):
    """Meshed [0, 2pi)^3 coordinates on `shape`."""
    return np.meshgrid(*[np.linspace(0, TWOPI, n, endpoint=False)
                         for n in shape], indexing="ij")


# ---------------------------------------------------------------------------
# constructors:  (b, meta_fragment)
# ---------------------------------------------------------------------------

def from_potential(a, S, norm=None):
    """Seed from an explicit real vector potential `a` (3, Nx, Ny, Nz).

    b = T(curl a), normalised to max|b| = 1 (or by the given `norm`).  The
    potential itself goes into the metadata, so the seed can be rebuilt on a
    finer grid by zero-padding it.  Returns (b, meta_fragment).
    """
    a = np.asarray(a, dtype=float)
    if a.shape != (3,) + tuple(S.shape):
        raise ValueError(f"potential shape {a.shape} does not match solver "
                         f"grid {(3,) + tuple(S.shape)}")
    b, c = _normalise(_curl_trunc(np.fft.fftn(a, axes=(1, 2, 3)), S), norm)
    return b, dict(seed_kind="file", seed_a=a, seed_norm=c)


def _blob_potential(shape, w, kz, center):
    """psi = chi (cos kz z, sin kz z, 0) with chi the periodic bump of
    `seeds.blob_seed` -- a compact packet of transversally rotating field."""
    X, Y, Z = _grid(shape)
    chi = np.exp((np.cos(X - center[0]) - 1) / w[0] ** 2
                 + (np.cos(Y - center[1]) - 1) / w[1] ** 2
                 + (np.cos(Z - center[2]) - 1) / w[2] ** 2)
    return np.stack([chi * np.cos(kz * Z), chi * np.sin(kz * Z),
                     np.zeros(tuple(shape))])


def blob(S, w=(0.8, 0.8, 0.8), kz=1, center=(np.pi, np.pi, np.pi), norm=None):
    """Localised, divergence-free, transversally rotating blob (the Alfvenon
    geometry of `seeds.blob_seed`, same field to round-off).

    chi is analytic but NOT band-limited, so the blob is defined by its
    construction recipe rather than by finitely many coefficients: a rebuild
    on a finer grid re-evaluates chi there and divides by the STORED norm.
    Returns (b, meta_fragment).
    """
    a = _blob_potential(S.shape, w, kz, center)
    b, c = _normalise(_curl_trunc(np.fft.fftn(a, axes=(1, 2, 3)), S), norm)
    return b, dict(seed_kind="blob", seed_w=np.asarray(w, float),
                   seed_kz=int(kz), seed_center=np.asarray(center, float),
                   seed_norm=c)


def _rng_vector(key, k):
    """The deterministic complex 3-vector attached to integer triple `k`.

    Keyed on (key, kx, ky, kz) -- python's hash of an int tuple is stable
    across runs and interpreters (PYTHONHASHSEED randomises str/bytes only),
    so the drawn coefficient belongs to the WAVEVECTOR, not to an array slot.
    """
    rng = np.random.default_rng(hash((key,) + tuple(k)) & 0xffffffff)
    return rng.standard_normal(3) + 1j * rng.standard_normal(3)


def _random_coeffs(kmax, slope, key):
    """[(k, ahat_k)] over one member of each +-k pair with 0 < |k| <= kmax,
    amplitude ~ |k|^(-slope).  The partner is filled in by conjugation, which
    is what makes the assembled potential real."""
    out = []
    for kx in range(-kmax, kmax + 1):
        for ky in range(-kmax, kmax + 1):
            for kz in range(-kmax, kmax + 1):
                km = np.sqrt(kx * kx + ky * ky + kz * kz)
                if km < 0.5 or km > kmax + 1e-12:
                    continue
                if not (kz > 0 or (kz == 0 and (ky > 0 or
                                                (ky == 0 and kx > 0)))):
                    continue                      # partner of one already in
                out.append(((kx, ky, kz),
                            _rng_vector(key, (kx, ky, kz)) * km ** (-slope)))
    return out


def _assemble(coeffs, shape):
    """Hermitian full-layout spectral potential on `shape`.

    Coefficients are scaled by the number of grid points so that the ifftn
    reproduces sum_k ahat_k exp(i k.x) with GRID-INDEPENDENT amplitude.
    Modes the grid cannot represent (|k_i| >= N_i/2, incl. the self-conjugate
    Nyquist plane) are dropped -- they are outside the retained band anyway.
    """
    shape = tuple(shape)
    ah = np.zeros((3,) + shape, dtype=complex)
    scale = float(np.prod(shape))
    for k, c in coeffs:
        if any(abs(ki) >= n / 2 for ki, n in zip(k, shape)):
            continue
        pos = tuple(ki % n for ki, n in zip(k, shape))
        neg = tuple((-ki) % n for ki, n in zip(k, shape))
        ah[(slice(None),) + pos] = c * scale
        ah[(slice(None),) + neg] = np.conj(c) * scale
    return ah


def _random_norm(kmax, slope, key):
    """max|curl a| of the random potential on a canonical dense grid: a
    function of (kmax, slope, key) alone, hence grid-independent."""
    n = int(min(_CANON_MAX, max(_CANON_MIN, 16 * kmax)))
    shape = (n, n, n)
    b = _curl_hat(_assemble(_random_coeffs(kmax, slope, key), shape),
                  numpy_wavenumbers(shape))
    return float(np.abs(b).max())


def random_seed(S, kmax=4, slope=0.0, key=0, norm=None):
    """Band-limited random divergence-free seed: every integer triple with
    0 < |k| <= kmax carries a deterministic complex potential coefficient of
    amplitude ~ |k|^(-slope).

    The same `key` gives the same CONTINUUM field on every grid (test: the
    seed on (24,24,48) equals the zero-pad of the seed on (16,16,32) to
    round-off), which is what lets a ladder refine without changing the
    object being continued.  Consequently max|b| is ~1 rather than exactly 1:
    the normalisation constant is the maximum over a canonical dense grid
    fixed by kmax, recorded as meta['seed_norm'].  Returns (b, meta_fragment).

    kmax must lie strictly inside S's retained band: if the strict 2/3 mask
    truncated any recipe mode, the built field would no longer equal the
    recipe's continuum object, and every later ladder rung would rebuild a
    DIFFERENT seed (measured: 4% drift per rung) -- the silent
    object-corruption the grid-keyed construction exists to prevent.
    """
    if kmax >= min(S.shape) / 3.0 - 1e-12:
        raise ValueError(
            "random_seed: kmax=%d is not strictly inside the retained band "
            "|k| < min(N)/3 = %.2f of grid %s; the truncated seed would not "
            "rebuild identically on finer ladder rungs. Reduce kmax or "
            "enlarge --grid0." % (kmax, min(S.shape) / 3.0, tuple(S.shape)))
    if norm is None:
        norm = _random_norm(kmax, slope, key)
    ah = _assemble(_random_coeffs(kmax, slope, key), S.shape)
    b, c = _normalise(_curl_trunc(ah, S), norm)
    return b, dict(seed_kind="random", seed_kmax=int(kmax),
                   seed_slope=float(slope), seed_key=int(key), seed_norm=c)


# ---------------------------------------------------------------------------
# rebuild + mode inspection
# ---------------------------------------------------------------------------

def _item(meta, name):
    """Scalar out of a metadata entry (npz round-trips them as 0-d arrays)."""
    return np.asarray(meta[name]).ravel()[0]


def make_seed(meta, S):
    """Rebuild the seed recorded in a state's metadata on S.shape.

    Returns the field only (the metadata is already in hand).  The stored
    normalisation constant is reused, so the rebuilt seed is the same
    continuum field as the original -- not merely the same shape of field.
    """
    kind = str(_item(meta, "seed_kind"))
    norm = float(_item(meta, "seed_norm")) if "seed_norm" in meta else None
    if kind == "blob":
        b, _ = blob(S, w=tuple(np.asarray(meta["seed_w"], float).ravel()),
                    kz=int(_item(meta, "seed_kz")),
                    center=tuple(np.asarray(meta["seed_center"],
                                            float).ravel()), norm=norm)
        return b
    if kind == "random":
        b, _ = random_seed(S, kmax=int(_item(meta, "seed_kmax")),
                           slope=float(_item(meta, "seed_slope")),
                           key=int(_item(meta, "seed_key")), norm=norm)
        return b
    if kind == "file":
        a = np.asarray(meta["seed_a"], float)
        if a.shape[1:] != tuple(S.shape):
            a = np.stack([np.asarray(zero_pad(a[i], tuple(S.shape)))
                          for i in range(3)])
        b, _ = from_potential(a, S, norm=norm)
        return b
    raise ValueError(f"unknown seed_kind {kind!r} "
                     "(expected 'blob', 'random' or 'file')")


def top_modes(b, m=3):
    """The `m` dominant integer wavevectors of |bhat|^2, as (kx, ky, kz)
    triples with kz >= 0 (one per conjugate pair) -- the candidates for
    --freeze-top, which pins the seed's own modes so continuation cannot
    quietly rotate the state onto a smoother nearby branch member."""
    b = np.asarray(b)
    shape = b.shape[1:]
    P = (np.abs(np.fft.rfftn(b, axes=(1, 2, 3))) ** 2).sum(0)
    kx = np.fft.fftfreq(shape[0], d=TWOPI / shape[0]) * TWOPI
    ky = np.fft.fftfreq(shape[1], d=TWOPI / shape[1]) * TWOPI
    kz = np.fft.rfftfreq(shape[2], d=TWOPI / shape[2]) * TWOPI
    P[0, 0, 0] = -1.0                                   # never the mean
    out, seen = [], set()
    for flat in np.argsort(P, axis=None)[::-1]:
        i, j, l = np.unravel_index(flat, P.shape)
        if P[i, j, l] <= 0:
            break
        k = (int(round(kx[i])), int(round(ky[j])), int(round(kz[l])))
        if k in seen or (-k[0], -k[1], -k[2]) in seen:   # kz = 0 partners
            continue
        seen.add(k)
        out.append(k)
        if len(out) >= m:
            break
    return out
