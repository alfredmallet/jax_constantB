#!/usr/bin/env python3
"""
resolution_check.py (JAX port) -- convergence-with-resolution study of a
saved constantB_tools state (Squire & Mallet 2022 style: plot key quantities
vs grid size to demonstrate the solution is resolved and not an artifact).

This is a COPY of ../resolution_check.py, re-pointed at jax_constantB's
jax-backed Solver/zero_pad/carrier/load_state (see constantB_tools.py's
module docstring for what is and isn't jit-compiled). Logic is otherwise
unchanged.

For each grid in GRIDS the script:
  1. spectrally interpolates (zero-pads) the state to that grid -- exact for
     the retained modes, so the initial residual measured there is the HONEST
     (continuum) residual of the incoming solution;
  2. re-polishes with minimum-norm Gauss-Newton (jit-compiled GN/CG solve)
     until the residual converges (or the sweep budget is exhausted --
     increase SWEEPS/CGIT if the last grids have not plateaued);
  3. records: max | |B|-1 |, max |div B|, max |grad B| (Frobenius), max
     deflection from the mean field, and the reversal volume fraction.

A resolved solution shows: residuals decreasing (or floored) with N, and
gradient/deflection numbers CONVERGING to N-independent values. Growth of
max|grad B| with N that does not converge would be the sharpening signature
discussed in the paper (Sec. 7.6) -- interesting rather than merely bad.

Intended for the eps=0.98 switchback state:
    python3 resolution_check.py mlstate_fine.npz
Runtime warning: even with the jit-compiled solve this can take a while at
the largest default grid (96x96x192) -- the first call per grid pays a one-
time XLA compilation cost, then each GN/CG solve runs as a single compiled
program. Edit GRIDS/SWEEPS/CGIT to taste. Results are appended to
resolution_check.csv so the run can be interrupted and resumed.
"""
import sys, time, csv, os
import numpy as np
from constantB_tools import Solver, zero_pad, carrier, load_state, numpy_wavenumbers, numpy_dif

GRIDS  = [(48, 48, 96), (64, 64, 128), (96, 96, 192)]
PCG    = True        # spectral preconditioner: essential at large N, where the
                     # unpreconditioned JJ^T condition number grows like k_max^2
                     # and a fixed CG budget solves a shrinking fraction per sweep
SWEEPS = 30          # GN sweeps per grid (each sweep = one CG solve + update)
CGIT   = 400         # CG iterations per sweep
TARGET = 1e-8        # stop polishing early below this residual

def study(state_file):
    B0state, eps, meta = load_state(state_file)
    rows = []
    done = set()
    if os.path.exists("resolution_check.csv"):        # resume: skip finished grids
        with open("resolution_check.csv") as f:
            for r in csv.DictReader(f):
                done.add((int(r['Nx']), int(r['Ny']), int(r['Nz'])))
    for grid in GRIDS:
        if grid in done:
            print(f"[{grid}] already in resolution_check.csv -- skipping")
            continue
        t0 = time.time()
        B = np.stack([np.asarray(zero_pad(B0state[i], grid)) for i in range(3)])
        S = Solver(grid)
        r1, r2 = S.residual(B)
        print(f"[{grid}] honest incoming residual: div {np.abs(r1).max():.2e}, "
              f"|B|^2-1 {np.abs(2*r2).max():.2e}")
        B, res, ci = S.gn(B, sweeps=SWEEPS, cgit=CGIT, tol=TARGET, verbose=True, pcg=PCG)
        B = np.asarray(B)
        # diagnostics (host numpy; one-shot per grid, no benefit from jit)
        nrm = np.sqrt((B ** 2).sum(0))
        Bbar = B.mean(axis=(1, 2, 3)); nb = np.linalg.norm(Bbar)
        cosM = np.clip((B * Bbar[:, None, None, None]).sum(0) / (nrm * nb), -1, 1)
        defl = np.degrees(np.arccos(cosM))
        K = numpy_wavenumbers(grid)
        g2 = sum(numpy_dif(B[i], j, K) ** 2 for i in range(3) for j in range(3))
        row = dict(Nx=grid[0], Ny=grid[1], Nz=grid[2],
                   res_div=float(np.abs(np.asarray(S.residual(B)[0])).max()),
                   res_norm=float(np.abs(nrm-1).max()),
                   maxgrad=float(np.sqrt(g2.max())),
                   maxdefl=float(defl.max()),
                   vol_rev=float((defl > 90).mean()),
                   minutes=(time.time()-t0)/60)
        print(f"[{grid}] polished: | |B|-1 | {row['res_norm']:.2e}  "
              f"div {row['res_div']:.2e}  maxgrad {row['maxgrad']:.3f}  "
              f"maxdefl {row['maxdefl']:.2f}  vol>90 {100*row['vol_rev']:.2f}%  "
              f"({row['minutes']:.1f} min)")
        rows.append(row)
        new = not os.path.exists("resolution_check.csv")
        with open("resolution_check.csv", "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new: w.writeheader()
            w.writerow(row)
        np.savez(f"state_res{grid[0]}x{grid[2]}.npz", B=B, eps=eps, **meta)
    # plot
    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        N = [r['Nx'] for r in rows]
        fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
        ax[0].semilogy(N, [r['res_norm'] for r in rows], 'o-', label=r'max$||B|-1|$')
        ax[0].semilogy(N, [r['res_div'] for r in rows], 's-', label=r'max$|\nabla\cdot B|$')
        ax[0].set_xlabel(r'$N_x$'); ax[0].legend(); ax[0].set_title('constraint residuals')
        ax[1].plot(N, [r['maxgrad'] for r in rows], 'o-')
        ax[1].set_xlabel(r'$N_x$'); ax[1].set_title(r'max$|\nabla B|_F$')
        ax[2].plot(N, [r['maxdefl'] for r in rows], 'o-')
        ax[2].set_xlabel(r'$N_x$'); ax[2].set_title('max deflection (deg)')
        for a in ax: a.grid(alpha=0.3)
        plt.tight_layout(); plt.savefig('fig_resolution_check.png', dpi=200)
        print("wrote fig_resolution_check.png")
    except Exception as e:
        print("plotting skipped:", e)

if __name__ == "__main__":
    study(sys.argv[1] if len(sys.argv) > 1 else "mlstate_fine.npz")
