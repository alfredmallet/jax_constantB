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

v2, carrier-free stack (docs/SPEC_v2.md, docs/algorithm.tex):

    potential    curl / inverse curl / Leray projector (rfft building blocks)
    solver_mu    mu-only exact-Galerkin minimum-norm GN (the v2 hot path)
    seeds_free   carrier-free seeds: curl of arbitrary/blob/random potentials
    spectra      shell & axis spectra, exact q-spectrum, 3D-ness monitor
    descent      EXPERIMENTAL smoothest-state SQP descent (not re-exported)

Float64 is REQUIRED (GN tolerance 1e-10, CG stop 1e-26 relative): x64 mode is
enabled here, before anything else touches jax.  Do not import jax-dependent
submodules before this package's __init__ has run.
"""
import jax

jax.config.update("jax_enable_x64", True)

from .spectral import (TWOPI, wavenumbers, rfft_wavenumbers, dealias_mask,
                       numpy_wavenumbers, numpy_dif, zero_pad)
from .solver import Solver, WeightedSolver
from .solver_mu import MuSolver, noncoplanar
from .potential import curl, div, divfree_project, inv_curl
from .seeds_free import (from_potential, blob, random_seed, make_seed,
                         top_modes)
from .spectra import (shell_spectrum, axis_spectra, q_spectrum,
                      kspace_inertia, local_slope, exp_kappa, plot_spectra)
from .seeds import carrier, seed_mode, build_seed, blob_seed
from .state_io import save_state, load_state, meta_args, rebuild_seed
from .series import series, domb_sykes, pade_poles
from .diagnostics import diagnose, quest_diagnostics, divergence_verdict

__all__ = [
    "TWOPI", "wavenumbers", "rfft_wavenumbers", "dealias_mask",
    "numpy_wavenumbers", "numpy_dif", "zero_pad",
    "Solver", "WeightedSolver",
    "MuSolver", "noncoplanar",
    "curl", "div", "divfree_project", "inv_curl",
    "from_potential", "blob", "random_seed", "make_seed", "top_modes",
    "shell_spectrum", "axis_spectra", "q_spectrum", "kspace_inertia",
    "local_slope", "exp_kappa", "plot_spectra",
    # descent (EXPERIMENTAL) deliberately not re-exported: import
    # constantB.descent explicitly.
    "carrier", "seed_mode", "build_seed", "rebuild_seed", "blob_seed",
    "save_state", "load_state", "meta_args",
    "series", "domb_sykes", "pade_poles",
    "diagnose", "quest_diagnostics", "divergence_verdict",
]
