"""Seed construction: the 1D arc-polarised carrier, its linearised 3D
deformations, and the localised divergence-free blob.

All host numpy/scipy by design: these run once per continuation run (cold
path), and seed_mode's periodic-ODE solve is a small dense COMPLEX LU --
nothing to gain from jit, and complex LU support on GPU is spotty.  Arrays
cross into jax only at the solver boundary.

scipy is imported lazily so that load-state + Gauss-Newton workflows survive
environments with a broken numpy/scipy pairing (seen on Kaggle).
"""
import numpy as np

from .spectral import TWOPI

try:
    from scipy.linalg import lu_factor, lu_solve
except Exception as _scipy_err:                          # noqa: F841
    def _need_scipy(*a, **k):
        raise ImportError(
            "scipy.linalg unavailable (import failed: %r); it is required "
            "only for seed_mode/series. Fix the numpy/scipy pairing, e.g. "
            "pip install -U --force-reinstall numpy scipy, then restart "
            "the kernel." % (_scipy_err,))
    lu_factor = lu_solve = _need_scipy


def carrier(A, c, Nz):
    """The 1D arc-polarised carrier B0(z) and its tangent frame.

    B0 = (s cos(psi), s sin(psi), c) with psi = A sin z and s = sqrt(1-c^2):
    |B0| = 1 and div B0 = 0 identically.  Returns a dict with the z grid,
    psi, s, B0 (3, Nz), the orthonormal tangent frame e1, e2 (B0.e1 = B0.e2
    = 0, e2 = B0 x e1), and a dense spectral d/dz matrix Dz for the seed ODE.
    """
    z = np.linspace(0, TWOPI, Nz, endpoint=False)
    s = np.sqrt(1 - c * c)
    psi = A * np.sin(z)
    B0 = np.array([s * np.cos(psi), s * np.sin(psi), c * np.ones(Nz)])
    e1 = np.array([-np.sin(psi), np.cos(psi), np.zeros(Nz)])
    e2 = np.array([-c * np.cos(psi), -c * np.sin(psi), s * np.ones(Nz)])
    kz = np.fft.fftfreq(Nz, d=TWOPI / Nz) * TWOPI
    Dz = np.real(np.fft.ifft(1j * kz[:, None]
                             * np.fft.fft(np.eye(Nz), axis=0), axis=0))
    return dict(A=A, c=c, s=s, z=z, psi=psi, B0=B0, e1=e1, e2=e2, Dz=Dz,
                kz=kz)


def seed_mode(car, kx, ky, amp, prof=(1.0, 0.7)):
    """First-order (linearised) deformation of the carrier for one transverse
    mode k_perp = (kx, ky):  b1 = Re{ [u e1 + v e2] exp(i k_perp . x_perp) }.

    u(z) = amp*(prof[0] cos z + prof[1]) is the FREE profile; v(z) solves the
    linearised divergence constraint, the periodic ODE

        s v' - i c kappa cos(chi - psi) v = -i kappa sin(chi - psi) u,

    by dense LU on the spectral collocation matrix.  Returns the complex
    polarisation vector w(z) = u e1 + v e2, shape (3, Nz).
    """
    kap, chi = np.hypot(kx, ky), np.arctan2(ky, kx)
    cpz, spz = np.cos(chi - car["psi"]), np.sin(chi - car["psi"])
    u = amp * (prof[0] * np.cos(car["z"]) + prof[1])
    v = lu_solve(lu_factor(car["s"] * car["Dz"]
                           - 1j * car["c"] * kap * np.diag(cpz)),
                 -1j * kap * spz * u)
    return u * car["e1"] + v * car["e2"]


def build_seed(car, modes, shape, prof=(1.0, 0.7)):
    """Real seed field on `shape` = (Nx, Ny, Nz) from [(kx, ky, amp), ...].
    Non-collinear modes make the seed spectrally 3D.  Plain numpy out."""
    Nx, Ny, Nz = shape
    x = np.linspace(0, TWOPI, Nx, endpoint=False)
    y = np.linspace(0, TWOPI, Ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing="ij")
    out = np.zeros((3, Nx, Ny, Nz))
    for (kx, ky, amp) in modes:
        w = seed_mode(car, kx, ky, amp, prof)
        out += 2 * np.real(w[:, None, None, :]
                           * np.exp(1j * (kx * X + ky * Y))[None, :, :, None])
    return out


def blob_seed(S, w=(0.8, 0.8, 0.8), kz=1, center=(np.pi, np.pi, np.pi)):
    """Localised, divergence-free, transversally rotating blob seed
    (the Alfvenon geometry): b = curl(psi1 e_x + psi2 e_y) with

        psi1 = chi cos(kz z),  psi2 = chi sin(kz z),
        chi  = exp[ sum_i (cos(x_i - x0_i) - 1) / w_i^2 ]   (periodic bump),

    built spectrally (exactly div-free), truncated with S.trunc3, and
    normalised to max|b| = 1 so eps is the injected amplitude.
    Returns (b, chi).  `S` is a Solver (uses S.shape, S.K, S.trunc3).
    """
    n = S.shape
    x = [np.linspace(0, TWOPI, m, endpoint=False) for m in n]
    X, Y, Z = np.meshgrid(*x, indexing="ij")
    chi = np.exp((np.cos(X - center[0]) - 1) / w[0] ** 2
                 + (np.cos(Y - center[1]) - 1) / w[1] ** 2
                 + (np.cos(Z - center[2]) - 1) / w[2] ** 2)
    F = np.stack([chi * np.cos(kz * Z), chi * np.sin(kz * Z), np.zeros(n)])
    Fh = np.fft.fftn(F, axes=(1, 2, 3))
    KX, KY, KZ = np.asarray(S.K)
    b = np.stack([                                   # ik x F
        np.real(np.fft.ifftn(1j * (KY * Fh[2] - KZ * Fh[1]))),
        np.real(np.fft.ifftn(1j * (KZ * Fh[0] - KX * Fh[2]))),
        np.real(np.fft.ifftn(1j * (KX * Fh[1] - KY * Fh[0])))])
    b = np.asarray(S.trunc3(b))
    return b / np.abs(b).max(), chi
