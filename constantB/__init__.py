"""constantB -- tools for large-amplitude, 3D, constant-|B|, divergence-free
magnetic fields (JAX-backed).

The problem: find B(x, y, z) on the periodic box [0, 2pi)^3 with

    div B = 0   and   |B| = 1   (pointwise),

continued to large amplitude from a 1D arc-polarised carrier plus a genuinely
3D seed perturbation.  The workhorse is a minimum-norm Gauss-Newton solver
(`Solver.gn`), jit-compiled so the whole multi-sweep GN + CG solve runs as a
single XLA program (one GPU kernel graph on Kaggle's P100).

Module map (see DESIGN.md in the repo root for the numerical rationale):

    spectral     grids, wavenumbers, FFT helpers, 2/3-rule masks, zero_pad
    solver       residual, J / J^T, CG, the GN loop, Solver / WeightedSolver
    seeds        1D carrier, linearised seed modes, blob seed
    state_io     .npz state files (format-compatible with the numpy reference)
    series       perturbation series, Domb-Sykes, Pade diagnostics (host numpy)
    diagnostics  physical health reports, quest step diagnostics, verdicts
    plotting     verification cuts, 3D structure figures
    fieldlines   field-line tracing / topology classification (host numpy)

Float64 is REQUIRED (GN tolerance 1e-10, CG stop 1e-26 relative): x64 mode is
enabled here, before anything else touches jax.  Do not import jax-dependent
submodules before this package's __init__ has run.
"""
import jax

jax.config.update("jax_enable_x64", True)

from .spectral import (TWOPI, wavenumbers, rfft_wavenumbers, dealias_mask,
                       numpy_wavenumbers, numpy_dif, zero_pad)
from .solver import Solver, WeightedSolver
from .seeds import carrier, seed_mode, build_seed, blob_seed
from .state_io import save_state, load_state, meta_args, rebuild_seed
from .series import series, domb_sykes, pade_poles
from .diagnostics import diagnose, quest_diagnostics, divergence_verdict

__all__ = [
    "TWOPI", "wavenumbers", "rfft_wavenumbers", "dealias_mask",
    "numpy_wavenumbers", "numpy_dif", "zero_pad",
    "Solver", "WeightedSolver",
    "carrier", "seed_mode", "build_seed", "rebuild_seed", "blob_seed",
    "save_state", "load_state", "meta_args",
    "series", "domb_sykes", "pade_poles",
    "diagnose", "quest_diagnostics", "divergence_verdict",
]
