# jax_constantB

JAX code for generating large-amplitude, three-dimensional,
constant-field-strength (|B| = 1), divergence-free magnetic fields.

## Layout

    constantB/            the package (import this)
      spectral.py         grids, rfft helpers, strict 2/3-rule masks, zero_pad
      solver.py           jit-compiled minimum-norm Gauss-Newton (legacy hot path)
      potential.py        curl / inverse curl / Leray projector (rfft)
      solver_mu.py        mu-only exact-Galerkin minimum-norm GN (v2 hot path)
      seeds_free.py       carrier-free seeds: curl of arbitrary/blob/random potentials
      spectra.py          shell & axis spectra, exact q-spectrum, 3D-ness monitor
      descent.py          EXPERIMENTAL smoothest-state SQP descent + energy pins
      seeds.py            1D carrier, linearised 3D seed modes, blob seed
      state_io.py         .npz state files (compatible with the numpy reference)
      series.py           perturbation series, Domb-Sykes, Pade (host numpy)
      diagnostics.py      health reports, quest diagnostics, verdicts
      plotting.py         verification cuts, 3D structure figures
      fieldlines.py       field-line tracing / topology classification
    constantB_tools.py    CLI (init/cont/refine/polish/diagnose/plots/series)
    amplitude_quest.py    adaptive continuation quest (the paper's branches)
    resolution_check.py   convergence-with-resolution study
    blob_quest.py         localised-blob (Alfvenon-geometry) experiment
    fieldline_topology.py open/trapped field-line classification + figure
    branch_figures.py     paper figures from quest CSV/state files
    grow.py               carrier-free continuation driver (the v2 quest)
    kaggle_grow.ipynb     P100 driver notebook for grow.py
    tests/parity_check.py 3-way parity: numpy reference / legacy jax / package
    tests/test_v2.py      v2 gates: observable parity, invariants, end-to-end grow
    docs/SPEC_v2.md       binding v2 build spec
    docs/algorithm.tex    the v2 algorithm note (pdflatex)
    legacy/               pre-rewrite single-file versions (frozen)
    DESIGN.md             numerical design rationale -- READ THIS FIRST

The frozen numpy reference implementation lives in the project root
(`../constantB_tools.py`) and remains the spec; parity is validated at
convergence by `tests/parity_check.py` (see DESIGN.md for the policy).

## Requirements

numpy, scipy, jax (float64 REQUIRED -- enabled by `import constantB`;
jax-metal has no x64, Mac GPUs are not a target). Kaggle P100 is the main
compute target: the entire multi-sweep GN + CG solve compiles to a single
XLA program. matplotlib for plots; scikit-image optional (3D isosurface);
pandas for branch_figures.
