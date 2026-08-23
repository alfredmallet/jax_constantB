"""One-shot host-numpy diagnostics: physical health reports of a state, the
per-step quest diagnostics, the gradient-divergence verdict, and the blob
(localisation) measures.

All host numpy by design: each runs once per command/step, so there is
nothing to gain from jit.  State arrays are cast to numpy at entry.
"""
import numpy as np

from .spectral import TWOPI, numpy_wavenumbers, numpy_dif
from .seeds import carrier


# ---------------------------------------------------------------------------
# Full state health report
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Per-step quest diagnostics
# ---------------------------------------------------------------------------

def quest_diagnostics(S, B, car):
    """Per-continuation-step diagnostics for amplitude quests.  Accepts B as
    a jax or numpy array; `S` is the Solver, `car` the carrier dict."""
    B = np.asarray(B)
    nrm = np.sqrt((B ** 2).sum(0))
    Bbar = B.mean(axis=(1, 2, 3)); nb = np.linalg.norm(Bbar)
    cosM = np.clip((B * Bbar[:, None, None, None]).sum(0) / (nrm * nb), -1, 1)
    defl = np.degrees(np.arccos(cosM))
    K = numpy_wavenumbers(S.shape)
    g2 = sum(numpy_dif(B[i], j, K) ** 2 for i in range(3) for j in range(3))
    bh = np.abs(np.fft.fftn(B - car['B0'][:, None, None, :], axes=(1, 2, 3))) ** 2
    # Spectral-tail monitor. CRITICAL dealias distinction: in Galerkin mode the
    # field is hard-truncated at |k| < N/3, so its top-of-grid modes are
    # identically zero and the collocation criterion (content just below the
    # Nyquist wavenumber) can NEVER fire. Instead we measure (a) the field's
    # content at the RETAINED-BAND EDGE (how hard the solution presses against
    # its allowed band -- the direct analogue of the old criterion), and
    # (b) the forced Galerkin tail of the constraint, tail_norm(B), which is
    # the honest unresolved burden and gets its own threshold (--gtail-max).
    dealias = bool(getattr(S, 'dealias', False))
    tails = []
    for ax, N in ((1, S.shape[0]), (2, S.shape[1]), (3, S.shape[2])):
        kc = (N // 3) if dealias else (N // 2)     # band edge vs Nyquist
        E = bh.sum(axis=tuple(i for i in range(4) if i != ax))[:kc]
        tails.append(float(E[-3:].max() / E.max()))
    out = dict(maxgrad=float(np.sqrt(g2.max())), Bbar=float(nb),
               maxdefl=float(defl.max()), vol_rev=float((defl > 90).mean()),
               tail=max(tails))
    if dealias:
        grms, gmax = S.tail_norm(B)
        out['gal_tail_rms'], out['gal_tail_max'] = float(grms), float(gmax)
    else:
        out['gal_tail_rms'], out['gal_tail_max'] = float('nan'), float('nan')
    return out


def divergence_verdict(hist, window=6):
    """Fit Q = [d ln g/d eps]^{-1} over the trailing window; extrapolate.

    Q ~ constant  -> exponential gradient growth, no finite-eps blow-up;
    Q declining linearly to zero -> finite-eps blow-up candidate, with the
    linear extrapolation of Q to zero estimating eps*."""
    pts = [(h['eps'], h['maxgrad']) for h in hist][-window-1:]
    if len(pts) < 4:
        return "insufficient data"
    e = np.array([p[0] for p in pts]); g = np.log([p[1] for p in pts])
    rate = np.diff(g) / np.diff(e)                 # d ln g / d eps at midpoints
    em = 0.5 * (e[1:] + e[:-1]); Q = 1.0 / np.maximum(rate, 1e-12)
    sl, ic = np.polyfit(em, Q, 1)
    if sl >= -0.05 * abs(ic) / max(em[-1] - em[0], 1e-9):
        return f"Q~const ({Q[-1]:.2f}): exponential growth, no finite-eps blow-up detected"
    eps_star = -ic / sl
    return (f"Q declining (slope {sl:.2f}): finite-eps blow-up candidate, "
            f"extrapolated eps* ~ {eps_star:.2f}")


# ---------------------------------------------------------------------------
# Blob (localisation) diagnostics
# ---------------------------------------------------------------------------

def fwhm(profile, dx):
    """Periodic-aware full width at half max of a 1D profile."""
    p = np.asarray(profile) - np.asarray(profile).min()
    i0 = int(np.argmax(p)); p = np.roll(p, len(p)//2 - i0)   # centre the peak
    return float((p >= 0.5*p.max()).sum())*dx


def blob_diagnostics(S, B, chi_half):
    """Per-step diagnostics for the blob quest: standard quest measures plus
    Lpar/Lperp (FWHM of the parallel/transverse energy profiles), flat_par
    (min P / max P: 1 = flat tube, 0 = solitary blob), and loc_frac (fraction
    of |b|^2 inside the seed's half-max ellipsoid `chi_half`)."""
    b = B - B.mean(axis=(1, 2, 3), keepdims=True)
    Bbar = np.asarray(B).mean(axis=(1, 2, 3)); nb = np.linalg.norm(Bbar)
    nrm = np.sqrt((np.asarray(B)**2).sum(0))
    cosM = np.clip((np.asarray(B)*Bbar[:, None, None, None]).sum(0)/(nrm*nb), -1, 1)
    defl = np.degrees(np.arccos(cosM))
    g2 = sum(np.asarray(S.dif(B[i], j))**2 for i in range(3) for j in range(3))
    e = (np.asarray(b)**2).sum(0)
    P = e.mean(axis=(0, 1)); Q = e.mean(axis=(1, 2))
    n = S.shape
    tn = S.tail_norm(B)[0]                     # (rms, max) -> rms
    return dict(maxgrad=float(np.sqrt(g2.max())), Bbar=float(nb),
                maxdefl=float(defl.max()), vol_rev=float((defl > 90).mean()),
                gal_tail_rms=float(tn),
                Lpar=fwhm(P, TWOPI/n[2]), Lperp=fwhm(Q, TWOPI/n[0]),
                flat_par=float(P.min()/P.max()),
                loc_frac=float(e[chi_half].sum()/e.sum()))
