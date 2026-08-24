"""Spectral diagnostics of a state: shell and axis spectra, the EXACT
spectrum of the constraint residual q, the k-space inertia tensor
(1D / planar / genuinely 3D), decay-rate fits, and the ladder overlay figure.

Two things make these numbers trustworthy, and both are the point of the
module:
  * Spectra are Fourier COEFFICIENT spectra -- every transform is divided by
    the number of grid points -- so the same continuum field gives the same
    E(k) on every grid and ladder rungs may be overlaid without rescaling.
  * q = (|B|^2 - 1)/2 is quadratic, so its spectrum on the working grid is
    aliased.  `q_spectrum` forms q on a 2x zero-padded grid instead: for
    in-band B the support of qhat is |k| < 2N/3, strictly inside the padded
    grid's Nyquist wavenumber, so the measured spectrum is exact -- this is
    the honest picture of the tail that `tail_norm` reports as one number.

Host numpy; jax arrays are cast at entry, matplotlib imported lazily (Agg).
"""
import numpy as np

from .spectral import numpy_wavenumbers, zero_pad


# ---------------------------------------------------------------------------
# spectra
# ---------------------------------------------------------------------------

def _coeffs(F):
    """|Fhat_k|^2 summed over components, in Fourier-COEFFICIENT normalisation
    (grid-independent).  `F` is (3, Nx, Ny, Nz) or (Nx, Ny, Nz)."""
    F = np.asarray(F)
    if F.ndim == 3:
        F = F[None]
    shape = F.shape[1:]
    Fh = np.fft.fftn(F, axes=(1, 2, 3)) / float(np.prod(shape))
    return (np.abs(Fh) ** 2).sum(0), shape


def shell_spectrum(F):
    """(k, E) with E(k) = sum over the shell round(|k'|) = k of |Fhat|^2.

    Shells are truncated at min(N)/2 so that no partially-sampled corner
    shell (which would fake a tail) enters the plot.
    """
    P, shape = _coeffs(F)
    K = numpy_wavenumbers(shape)
    idx = np.rint(np.sqrt(K[0] ** 2 + K[1] ** 2 + K[2] ** 2)).astype(int)
    E = np.bincount(idx.ravel(), weights=P.ravel())
    kcut = min(shape) // 2
    k = np.arange(min(len(E), kcut + 1))
    return k, E[:len(k)]


def axis_spectra(F):
    """{'kx': E(|kx|), 'ky': ..., 'kz': ...}: |Fhat|^2 summed over the other
    two axes and folded onto |k| >= 0.  The per-axis view is the one that
    matches the retained band, which is a box (|k_i| < N_i/3), not a ball."""
    P, shape = _coeffs(F)
    K = numpy_wavenumbers(shape)
    out = {}
    for ax, name in enumerate(("kx", "ky", "kz")):
        ki = np.rint(np.abs(K[ax])).astype(int)
        E = np.bincount(ki.ravel(), weights=P.ravel())
        out[name] = E[:shape[ax] // 2 + 1]
    return out


def q_spectrum(B):
    """(k, E) of q = (|B|^2 - 1)/2, formed on the 2x zero-padded grid.

    Alias-free by construction (see the module docstring), so the high-k end
    is the true forced tail of the constraint, not a folding artefact.
    """
    B = np.asarray(B)
    fine = tuple(2 * n for n in B.shape[1:])
    Bf = np.stack([np.asarray(zero_pad(B[i], fine)) for i in range(3)])
    return shell_spectrum(0.5 * ((Bf ** 2).sum(0) - 1.0))


# ---------------------------------------------------------------------------
# geometry of the spectral support
# ---------------------------------------------------------------------------

def kspace_inertia(B):
    """(eigenvalues, lam2/lam1, lam3/lam1) of T = sum_{k != 0} |Bhat_k|^2
    khat khat^T, normalised to unit trace and sorted descending.

    The dimensionality monitor: a 1D field (all power on one k-line) gives
    ratios (0, 0), a planar/2.5D field (x, 0), and a genuinely 3D field two
    positive ratios.  Section 7 of the paper makes 3D-ness the property that
    forces spectral tails, so it is worth watching every step.
    """
    P, shape = _coeffs(B)
    K = numpy_wavenumbers(shape)
    k2 = K[0] ** 2 + K[1] ** 2 + K[2] ** 2
    w = np.where(k2 > 0, P / np.where(k2 > 0, k2, 1.0), 0.0)   # P/|k|^2
    T = np.array([[(w * K[i] * K[j]).sum() for j in range(3)]
                  for i in range(3)])
    tr = np.trace(T)
    if tr <= 0:
        return np.zeros(3), 0.0, 0.0
    lam = np.sort(np.linalg.eigvalsh(T / tr))[::-1]
    return lam, float(lam[1] / lam[0]), float(lam[2] / lam[0])


# ---------------------------------------------------------------------------
# decay-rate fits
# ---------------------------------------------------------------------------

def _window(k, E, k1, k2):
    m = (k >= k1) & (k <= k2) & (E > 0) & (k > 0)
    if m.sum() < 2:
        raise ValueError(f"fit window [{k1}, {k2}] holds {int(m.sum())} "
                         "usable points (need >= 2)")
    return np.asarray(k)[m], np.asarray(E)[m]


def local_slope(k, E, k1, k2):
    """Algebraic decay exponent p in E ~ k^-p, fitted on [k1, k2]."""
    kk, EE = _window(k, E, k1, k2)
    return float(-np.polyfit(np.log(kk), np.log(EE), 1)[0])


def exp_kappa(k, E, k1, k2):
    """Analyticity-strip width kappa from ln E ~ a - 2 kappa k on [k1, k2]
    (E is an energy, hence the factor 2: |Fhat| ~ exp(-kappa k))."""
    kk, EE = _window(k, E, k1, k2)
    return float(-0.5 * np.polyfit(kk, np.log(EE), 1)[0])


# ---------------------------------------------------------------------------
# ladder figure
# ---------------------------------------------------------------------------

def _as_state(s):
    """(B, eps) from either a (B, eps) pair or a state filename."""
    if isinstance(s, str):
        from .state_io import load_state
        B, eps, _ = load_state(s)
        return np.asarray(B), float(eps)
    B, eps = s
    return np.asarray(B), float(eps)


def plot_spectra(states, out, labels=None):
    """Ladder overlay: log-log E(k) (algebraic decay), semi-log E(k)
    (exponential decay / analyticity strip), and the exact q-spectrum, with
    each state's isotropic band edge min(N)/3 marked.

    `states` is a list of (B, eps) pairs or state filenames, in ladder order;
    colours run through one sequential hue so the order is readable without
    consulting the legend.  Returns the output path.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    data = [_as_state(s) for s in states]
    cols = plt.cm.Blues(np.linspace(0.35, 0.95, max(len(data), 1)))
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
    for n, (B, eps) in enumerate(data):
        shape = B.shape[1:]
        lab = (labels[n] if labels is not None
               else f"{shape[0]}x{shape[1]}x{shape[2]}, "
                    f"$\\varepsilon$={eps:.2f}")
        k, E = shell_spectrum(B - B.mean(axis=(1, 2, 3), keepdims=True))
        kq, Eq = q_spectrum(B)
        edge = min(shape) / 3.0
        m, mq = (k > 0) & (E > 0), (kq > 0) & (Eq > 0)
        ax[0].loglog(k[m], E[m], color=cols[n], lw=1.2, label=lab)
        ax[1].semilogy(k[m], E[m], color=cols[n], lw=1.2)
        ax[2].loglog(kq[mq], Eq[mq], color=cols[n], lw=1.2)
        for a in ax:
            a.axvline(edge, color=cols[n], ls=':', lw=1)
    ax[0].set_xlabel('$k$'); ax[0].set_ylabel(r'$E_b(k)$')
    ax[0].set_title('fluctuation shell spectrum (log-log)', fontsize=9)
    ax[0].legend(fontsize=7)
    ax[1].set_xlabel('$k$')
    ax[1].set_title('same, semi-log (analyticity strip)', fontsize=9)
    ax[2].set_xlabel('$k$'); ax[2].set_ylabel(r'$E_q(k)$')
    ax[2].set_title('exact $q$ spectrum (2x padded); dotted: $N/3$',
                    fontsize=9)
    for a in ax:
        a.grid(alpha=0.3); a.tick_params(labelsize=8)
    plt.tight_layout(); plt.savefig(out, dpi=200); plt.close(fig)
    return out
