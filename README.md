# jax_constantB

JAX code for generating large-amplitude, three-dimensional,
constant-field-strength (|B| = 1), divergence-free magnetic fields.

## Layout

    constantB/            the package (import this)
      spectral.py         grids, rfft helpers, strict 2/3-rule masks, zero_pad
      solver.py           jit-compiled minimum-norm Gauss-Newton (the hot path)
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
    tests/parity_check.py 3-way parity: numpy reference / legacy jax / package
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
