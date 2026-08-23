"""Perturbation series on the harmonic ladder (dealiased) and its
radius-of-convergence diagnostics (Domb-Sykes, Pade poles).

Host numpy/scipy by design: a dense complex LU per order in a one-shot
diagnostic run with small Nord -- a cold path with nothing to gain from jit,
and the per-order dict structure with data-dependent skips is not a sane jit
target.
"""
import numpy as np

from .spectral import TWOPI
from .seeds import carrier, lu_factor, lu_solve


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
