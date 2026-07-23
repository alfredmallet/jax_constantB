#!/usr/bin/env python3
"""
================================================================================
fieldline_topology.py

Field-line topology diagnostic for a saved constant-|B| state (numpy; new
code lives here per project policy even though this particular analysis is a
one-off diagnostic rather than a solver component -- there is nothing to gain
from JAX here since the hot loop is a handful of gather/interpolate calls on
O(500) points, not a differentiable optimisation).

QUESTION.  Do field lines of a large-amplitude constant-|B| state progress
systematically through the periodic box along the mean field Bbar (OPEN
topology), or do some of them circulate without net progress (CLOSED/TRAPPED,
flux-rope-like islands)?  This matters because the solar-wind electron strahl
is a passive tracer of field-line connectivity back to the corona: switchback
observations require OPEN topology straight through the deflected region
(Huang et al. 2026, Appendix A/B).

METHOD.
  1. Trace field lines by RK4 integration in ARC LENGTH: dr/ds = b(r), where
     b = B/|B| is the local unit tangent (so the parametrisation is unit
     speed and "arc length traversed" = number of steps * step size exactly,
     independent of how curvy the line is).
  2. b is evaluated by vectorized trilinear interpolation of the gridded B
     field, periodic-wrapped for the lookup; the *trajectory* itself is kept
     in unwrapped (absolute) coordinates so net displacement is meaningful.
  3. All seeds are integrated simultaneously -- the interpolation and RK4
     stages are numpy array ops over the whole seed population at once; this
     is the hot loop and is kept fully vectorized (no per-line python loop).
  4. Classification: for each line, the displacement along the mean-field
     direction Bbar_hat, d(s) = (r(s)-r(0)).Bbar_hat, is fit against s.
     OPEN:          |d(s_end)| > 5 box lengths AND d(s) is ~linear in s
                     (Pearson r^2 > 0.8 against a linear fit) -- systematic,
                     roughly constant-rate progress.
     TRAPPED:       |d(s_end)| < 1 box length over the *whole* traced arc --
                     no net progress despite >=40 box lengths of arc length.
     INTERMEDIATE:  everything else (e.g. long arc, moderate net progress,
                     or a "fast" |d|>5 that is not linear -- superdiffusive/
                     ballistic-but-erratic rather than confidently open).

CHUNKING / RESUME.  Because this tool may be invoked from an environment with
a hard wall-clock limit per call, the integrator supports resuming: pass
--checkpoint pointing at an .npz that stores the running trajectory array and
step count; rerunning the *same* command continues from where it left off
until --nsteps total steps are reached. On the hardware this was developed on
(numpy, single core) the full run (512 seeds x ~6300 RK4 steps, i.e. 40 box
lengths of arc length at a half-grid-cell step) takes a few seconds, so
--chunk-steps larger than --nsteps (the default) simply finishes in one call.

USAGE.
    python3 fieldline_topology.py --validate
    python3 fieldline_topology.py                      # full analysis + figure
    python3 fieldline_topology.py --chunk-steps 1000    # demo of resumability

Dependencies: numpy, matplotlib.
================================================================================
"""
import os
import sys
import time
import argparse

import numpy as np

# ---------------------------------------------------------------------------
# constantB_tools is the frozen numpy reference toolkit (see project
# CLAUDE.md).  A copy lives alongside this script in jax_constantB/, and the
# original lives in the project root next to the state file; make both
# importable so `--state` can point at either without extra setup.
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.join(_HERE, '..')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from constantB_tools import load_state, carrier as build_carrier  # noqa: E402

TWOPI = 2.0 * np.pi

DEFAULT_STATE = os.path.join(_HERE, '..', 'quest_smooth.npz')
DEFAULT_FIG = os.path.join(_HERE, '..', 'fig_topology.png')
DEFAULT_CKPT = os.path.join(_HERE, 'fieldline_checkpoint.npz')


# ===========================================================================
# 1. Vectorized trilinear interpolation with periodic wrap
# ===========================================================================

def grid_params(B):
    """Grid spacing/shape for the periodic box [0,2pi)^3 that B lives on."""
    _, Nx, Ny, Nz = B.shape
    return dict(Nx=Nx, Ny=Ny, Nz=Nz, dx=TWOPI / Nx, dy=TWOPI / Ny, dz=TWOPI / Nz)


def interp_B(pos, B, grid):
    """Trilinear interpolation of the vector field B at absolute positions.

    pos   : (M,3) array, ABSOLUTE (possibly unwrapped, i.e. outside [0,2pi))
            coordinates -- the trajectory itself stays unwrapped so that net
            progress through the periodic box is unambiguous.
    B     : (3,Nx,Ny,Nz) gridded field on the periodic box [0,2pi)^3.
    grid  : dict from grid_params(B).

    Returns (M,3) interpolated field vectors. Periodic wrap is applied only
    for the *lookup* (mod 2pi + periodic corner indices); it never touches
    `pos`. Fully vectorized over all M points at once -- this is the hot
    loop of the whole script, so no python-level loop over points.
    """
    Nx, Ny, Nz = grid['Nx'], grid['Ny'], grid['Nz']
    dx, dy, dz = grid['dx'], grid['dy'], grid['dz']
    x = np.mod(pos[:, 0], TWOPI)
    y = np.mod(pos[:, 1], TWOPI)
    z = np.mod(pos[:, 2], TWOPI)
    fx, fy, fz = x / dx, y / dy, z / dz
    i0 = np.floor(fx).astype(np.int64)
    j0 = np.floor(fy).astype(np.int64)
    k0 = np.floor(fz).astype(np.int64)
    tx, ty, tz = fx - i0, fy - j0, fz - k0
    i0 %= Nx
    j0 %= Ny
    k0 %= Nz
    i1 = (i0 + 1) % Nx
    j1 = (j0 + 1) % Ny
    k1 = (k0 + 1) % Nz
    out = np.zeros((pos.shape[0], 3))
    for ii, wx in ((i0, 1.0 - tx), (i1, tx)):
        for jj, wy in ((j0, 1.0 - ty), (j1, ty)):
            for kk, wz in ((k0, 1.0 - tz), (k1, tz)):
                w = (wx * wy * wz)[:, None]
                out += w * B[:, ii, jj, kk].T
    return out


def unit_dir(Bvec, floor_eps=1e-14):
    """b = B/|B|; also returns |B| (used for the interpolation sanity check
    -- along a genuine trace of a |B|=1 state this should stay close to 1;
    it need not be exactly 1 off-grid because trilinear interpolation of a
    curved unit vector field is not itself exactly unit-norm)."""
    nrm = np.sqrt((Bvec ** 2).sum(axis=1))
    safe = np.maximum(nrm, floor_eps)
    return Bvec / safe[:, None], nrm


# ===========================================================================
# 2. RK4 integrator in arc length: dr/ds = b(r)
# ===========================================================================

def rk4_step(pos, h, B, grid):
    k1, _ = unit_dir(interp_B(pos, B, grid))
    k2, _ = unit_dir(interp_B(pos + 0.5 * h * k1, B, grid))
    k3, _ = unit_dir(interp_B(pos + 0.5 * h * k2, B, grid))
    k4, _ = unit_dir(interp_B(pos + h * k3, B, grid))
    return pos + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def trace_lines(pos0, B, grid, h, nsteps, progress_every=0):
    """Integrate all seeds simultaneously for `nsteps` RK4 steps.

    Returns traj, shape (nsteps+1, M, 3): unwrapped positions at every step
    (step 0 = pos0). No sub-sampling here -- 512 seeds x ~6300 steps is only
    ~77MB, and the per-line classification / puncture plot both want full
    resolution.
    """
    M = pos0.shape[0]
    traj = np.empty((nsteps + 1, M, 3))
    traj[0] = pos0
    pos = pos0.copy()
    t0 = time.time()
    for n in range(1, nsteps + 1):
        pos = rk4_step(pos, h, B, grid)
        traj[n] = pos
        if progress_every and (n % progress_every == 0):
            print(f"    step {n}/{nsteps}  ({time.time() - t0:.2f}s elapsed)")
    return traj


# ===========================================================================
# 3. Checkpoint (chunk / resume) support
# ===========================================================================

def save_checkpoint(path, traj, seeds0, seed_kind, h, params):
    np.savez(path, traj=traj, seeds0=seeds0, seed_kind=seed_kind, h=h,
              n_uniform=params['n_uniform'], n_reversed=params['n_reversed'],
              seed_rng=params['seed_rng'])


def load_checkpoint(path):
    d = np.load(path, allow_pickle=True)
    traj = d['traj']
    seeds0 = d['seeds0']
    seed_kind = d['seed_kind']
    h = float(d['h'])
    params = dict(n_uniform=int(d['n_uniform']), n_reversed=int(d['n_reversed']),
                  seed_rng=int(d['seed_rng']))
    return traj, seeds0, seed_kind, h, params


def integrate_with_resume(B, grid, seeds0, seed_kind, h, nsteps_total,
                            chunk_steps, checkpoint_path, params, fresh=False):
    """Run RK4 tracing up to nsteps_total steps, checkpointing every
    `chunk_steps` steps so a hard-timeout caller can invoke this repeatedly
    (same args) to resume. Returns (traj, done) where done=True iff
    nsteps_total steps have been completed."""
    if (not fresh) and os.path.exists(checkpoint_path):
        traj_old, seeds0_ck, seed_kind_ck, h_ck, params_ck = load_checkpoint(checkpoint_path)
        if h_ck != h or params_ck['n_uniform'] != params['n_uniform'] or \
           params_ck['n_reversed'] != params['n_reversed'] or \
           params_ck['seed_rng'] != params['seed_rng']:
            print("  [checkpoint params mismatch current CLI args -- ignoring "
                  "checkpoint and starting fresh]")
        else:
            traj = traj_old
            seeds0 = seeds0_ck
            seed_kind = seed_kind_ck
            print(f"  resumed from checkpoint: {traj.shape[0]-1} steps already done "
                  f"({checkpoint_path})")
    else:
        traj = seeds0[None, :, :].copy()

    step_done = traj.shape[0] - 1
    while step_done < nsteps_total:
        n_this = min(chunk_steps, nsteps_total - step_done)
        t0 = time.time()
        extra = trace_lines(traj[-1], B, grid, h, n_this)
        traj = np.concatenate([traj, extra[1:]], axis=0)
        step_done += n_this
        save_checkpoint(checkpoint_path, traj, seeds0, seed_kind, h, params)
        print(f"  chunk: +{n_this} steps in {time.time()-t0:.2f}s -> "
              f"{step_done}/{nsteps_total} total steps done, checkpoint saved.")
    return traj, seed_kind, (step_done >= nsteps_total)


# ===========================================================================
# 4. Seeding
# ===========================================================================

def seed_uniform(n, rng):
    return rng.uniform(0.0, TWOPI, size=(n, 3))


def seed_in_reversed_region(n, B, Bbar_hat, grid, rng):
    """n seed points sampled uniformly inside grid cells where B.Bbar_hat<0
    (the reversed / switchback region), with a random jitter within each
    chosen cell so seeds aren't glued to the grid."""
    dotB = (B * Bbar_hat[:, None, None, None]).sum(axis=0)
    idx = np.argwhere(dotB < 0.0)  # (K,3) of (i,j,k)
    choice = rng.integers(0, idx.shape[0], size=n)
    ijk = idx[choice].astype(float)
    jitter = rng.uniform(0.0, 1.0, size=(n, 3))
    cellsize = np.array([grid['dx'], grid['dy'], grid['dz']])
    return (ijk + jitter) * cellsize[None, :]


# ===========================================================================
# 5. Classification
# ===========================================================================

def linear_r2(s, d):
    """Pearson r^2 of d against s (per line). s: (T,), d: (T,M) -> (M,)."""
    s = s - s.mean()
    d = d - d.mean(axis=0, keepdims=True)
    num = (s[:, None] * d).sum(axis=0) ** 2
    den = (s ** 2).sum() * (d ** 2).sum(axis=0)
    den = np.maximum(den, 1e-300)
    return num / den


def classify(traj, Bbar_hat, open_box=5.0, trapped_box=1.0, r2_min=0.8, h=None):
    """traj: (T,M,3) unwrapped positions. Returns dict of per-line arrays:
    d_end, s_end, drift_rate, r2, label (0=trapped,1=intermediate,2=open)."""
    T = traj.shape[0]
    s = np.arange(T) * h
    disp = (traj - traj[0][None]) @ Bbar_hat          # (T,M)
    d_end = disp[-1]
    s_end = s[-1]
    drift_rate = d_end / s_end
    r2 = linear_r2(s, disp)
    max_excursion = np.max(np.abs(disp), axis=0)

    label = np.full(traj.shape[1], 1, dtype=int)  # default INTERMEDIATE
    is_trapped = np.abs(d_end) < trapped_box * TWOPI
    is_open = (np.abs(d_end) > open_box * TWOPI) & (r2 > r2_min)
    label[is_trapped] = 0
    label[is_open] = 2   # open check applied after trapped so a >5-box but
                          # non-linear excursion isn't miscoded as trapped
    return dict(d_end=d_end, s_end=s_end, drift_rate=drift_rate, r2=r2,
                max_excursion=max_excursion, label=label, disp=disp, s=s)


LABEL_NAMES = {0: 'TRAPPED', 1: 'INTERMEDIATE', 2: 'OPEN'}
LABEL_COLORS = {0: 'crimson', 1: 'darkorange', 2: 'steelblue'}


def visited_reversed(traj, B, grid, Bbar_hat, stride=4):
    """For each line, did it ever pass through a cell with B.Bbar_hat<0?
    Subsamples the trajectory every `stride` steps (plenty for a yes/no flag
    given the step size is already a fraction of a grid cell) and does one
    big vectorized interpolation call."""
    sub = traj[::stride]                              # (Ts,M,3)
    Ts, M, _ = sub.shape
    flat = sub.reshape(Ts * M, 3)
    Bv = interp_B(flat, B, grid)
    dot = (Bv @ Bbar_hat).reshape(Ts, M)
    return (dot < 0.0).any(axis=0)


# ===========================================================================
# 6. Puncture plot: upward crossings of planes z = 0 (mod 2pi)
# ===========================================================================

def upward_crossings(traj_line):
    """traj_line: (T,3) unwrapped positions for ONE line. Returns (x,y) mod
    2pi at every upward crossing of z = integer multiple of 2pi, using linear
    interpolation between the two bracketing samples for sub-step accuracy.
    Handles the (rare, since h << 2pi) case of >1 crossing in a single step.
    """
    z = traj_line[:, 2]
    n0 = np.floor(z[:-1] / TWOPI).astype(np.int64)
    n1 = np.floor(z[1:] / TWOPI).astype(np.int64)
    xs, ys = [], []
    steps_up = np.where(n1 > n0)[0]
    for t in steps_up:
        for L in range(n0[t] + 1, n1[t] + 1):
            zc = L * TWOPI
            frac = (zc - z[t]) / (z[t + 1] - z[t])
            xc = traj_line[t, 0] + frac * (traj_line[t + 1, 0] - traj_line[t, 0])
            yc = traj_line[t, 1] + frac * (traj_line[t + 1, 1] - traj_line[t, 1])
            xs.append(xc % TWOPI)
            ys.append(yc % TWOPI)
    return np.array(xs), np.array(ys)


# ===========================================================================
# 7. Validation mode: analytic sanity checks
# ===========================================================================

def validate():
    ok = True

    # --- (1) uniform field B=(0,0,1): every line must be OPEN with drift=1 ---
    print("[validate] (1) uniform field B=(0,0,1) ...")
    Nx = Ny = Nz = 8
    Bu = np.zeros((3, Nx, Ny, Nz))
    Bu[2] = 1.0
    gridu = grid_params(Bu)
    rng = np.random.default_rng(0)
    seeds = seed_uniform(16, rng)
    h = 0.04
    nsteps = int(np.ceil(10 * TWOPI / h))
    traj = trace_lines(seeds, Bu, gridu, h, nsteps)
    Bbar_hat_u = np.array([0.0, 0.0, 1.0])
    res = classify(traj, Bbar_hat_u, h=h)
    print(f"    drift_rate: mean={res['drift_rate'].mean():.6f} "
          f"std={res['drift_rate'].std():.2e}  (expect 1.0 exactly)")
    print(f"    labels: {[LABEL_NAMES[l] for l in np.unique(res['label'])]}")
    pass1 = np.allclose(res['drift_rate'], 1.0, atol=1e-6) and np.all(res['label'] == 2)
    print(f"    -> {'PASS' if pass1 else 'FAIL'}")
    ok &= pass1

    # --- (2) analytic carrier B0(z), c=0.2, A=1.2 (matches quest_smooth.npz
    #         seed).  NOTE: since B0_z = c is CONSTANT (not just on average),
    #         z(s) = z0 + c*s exactly -- unit-speed motion in z is exactly
    #         linear regardless of x,y.  z spends "equal time" (equal arc
    #         length) at every z, so the long-s average of B0_x is simply its
    #         z-average, Bbar_x = s*<cos(A sin z)> = s*J0(A) (Bessel J0), NOT
    #         zero (a common trap: <cos(A sin z)> = J0(A) != 0 in general).
    #         So Bbar_c is tilted mostly toward x here, and the exact
    #         prediction for the drift rate along Bbar_hat works out to
    #         |Bbar_c| itself (displacement growth along Bbar_hat_x and
    #         Bbar_hat_z combine to reconstruct |Bbar_c|^2/|Bbar_c|) --
    #         this is a stronger analytic check than "drift=c" would have
    #         been, because it engages both components and matches the
    #         solar-wind-relevant flux-average identity used in the main
    #         analysis below. ---
    print("[validate] (2) analytic 1D carrier A=1.2, c=0.2 ...")
    Nzc = 144
    car = build_carrier(A=1.2, c=0.2, Nz=Nzc)
    B0 = car['B0']                                       # (3, Nzc)
    Bc = np.repeat(np.repeat(B0[:, None, None, :], 8, axis=1), 8, axis=2)  # (3,8,8,Nzc)
    gridc = grid_params(Bc)
    Bbar_c = Bc.mean(axis=(1, 2, 3))
    Bbar_hat_c = Bbar_c / np.linalg.norm(Bbar_c)
    print(f"    Bbar_c = {Bbar_c}  (Bbar_x = s*J0(A) != 0, NOT (0,0,c) -- see comment)")
    seeds = seed_uniform(16, rng)
    nsteps = int(np.ceil(40 * TWOPI / h))
    traj = trace_lines(seeds, Bc, gridc, h, nsteps)
    res = classify(traj, Bbar_hat_c, h=h)
    print(f"    drift_rate: mean={res['drift_rate'].mean():.5f} "
          f"std={res['drift_rate'].std():.2e}  (expect |Bbar_c|={np.linalg.norm(Bbar_c):.5f})")
    print(f"    all OPEN: {np.all(res['label'] == 2)}")
    pass2 = abs(res['drift_rate'].mean() - np.linalg.norm(Bbar_c)) < 0.01 and np.all(res['label'] == 2)
    print(f"    -> {'PASS' if pass2 else 'FAIL'}")
    ok &= pass2

    # --- (3) interpolation sanity on the REAL state: |B_interp| ~ 1 along a
    #         short trace (checks the trilinear interpolation of the actual
    #         gridded field, not just the analytic tests above) ---
    print("[validate] (3) |B| sanity on the real state (short trace) ...")
    B, eps, meta = load_state(DEFAULT_STATE)
    grid = grid_params(B)
    print(f"    loaded state: B.shape={B.shape}, |B| range on grid = "
          f"[{np.sqrt((B**2).sum(0)).min():.5f}, {np.sqrt((B**2).sum(0)).max():.5f}]")
    seeds = seed_uniform(20, rng)
    nsteps = 300
    pos = seeds.copy()
    nrm_all = []
    for _ in range(nsteps):
        Bv = interp_B(pos, B, grid)
        b, nrm = unit_dir(Bv)
        nrm_all.append(nrm)
        pos = pos + h * b   # simple Euler is fine for this sanity probe
    nrm_all = np.array(nrm_all)
    print(f"    |B_interp| along traces: min={nrm_all.min():.5f} "
          f"mean={nrm_all.mean():.5f} max={nrm_all.max():.5f}  (expect ~1; trilinear "
          f"interpolation of a curved unit-vector field is a chord, not the arc, so a "
          f"few-percent undershoot off-grid is expected and NOT a bug -- it must not "
          f"blow up or exceed ~1 by more than the grid's own |B|-1 residual)")
    pass3 = (nrm_all.min() > 0.90) and (nrm_all.max() < 1.02)
    print(f"    -> {'PASS' if pass3 else 'FAIL'}")
    ok &= pass3

    print(f"\n[validate] overall: {'ALL PASS' if ok else 'SOME CHECKS FAILED'}")
    return ok


# ===========================================================================
# 8. Main analysis
# ===========================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--validate', action='store_true',
                     help='run analytic correctness checks and exit')
    ap.add_argument('--state', default=DEFAULT_STATE)
    ap.add_argument('--fig-out', default=DEFAULT_FIG)
    ap.add_argument('--checkpoint', default=DEFAULT_CKPT)
    ap.add_argument('--fresh', action='store_true',
                     help='ignore any existing checkpoint, start over')
    ap.add_argument('--h', type=float, default=0.04, help='arc-length RK4 step')
    ap.add_argument('--arc-boxes', type=float, default=40.0,
                     help='total arc length per line, in units of 2pi (box length)')
    ap.add_argument('--n-uniform', type=int, default=256)
    ap.add_argument('--n-reversed', type=int, default=256)
    ap.add_argument('--chunk-steps', type=int, default=20000,
                     help='max RK4 steps per invocation before checkpointing; '
                          'defaults large enough to finish in one call')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--n-plot-lines-a', type=int, default=40)
    ap.add_argument('--n-plot-lines-b', type=int, default=20)
    args = ap.parse_args()

    if args.validate:
        ok = validate()
        sys.exit(0 if ok else 1)

    print(f"[topology] loading state: {args.state}")
    B, eps, meta = load_state(args.state)
    grid = grid_params(B)
    Nx, Ny, Nz = grid['Nx'], grid['Ny'], grid['Nz']
    print(f"    B.shape={B.shape}  eps={eps}")

    Bbar = B.mean(axis=(1, 2, 3))
    Bbar_hat = Bbar / np.linalg.norm(Bbar)
    print(f"    Bbar = {Bbar}, |Bbar| = {np.linalg.norm(Bbar):.6f}")

    dotB = (B * Bbar_hat[:, None, None, None]).sum(axis=0)
    frac_reversed = (dotB < 0.0).mean()
    print(f"    reversed-region volume fraction (B.Bbar_hat<0): {frac_reversed:.4f}")

    # --- flux-average sanity check ---
    flux_avg = dotB.mean()                                       # = Bbar.Bbar_hat = |Bbar| exactly
    mag = np.sqrt((B ** 2).sum(0))
    b_field = B / mag
    dotb = (b_field * Bbar_hat[:, None, None, None]).sum(axis=0)
    unit_dir_avg = dotb.mean()                                    # volume avg of unit tangent . Bbar_hat
    print(f"    sanity: mean(B.Bbar_hat) over grid       = {flux_avg:.6f}  "
          f"(must equal |Bbar|={np.linalg.norm(Bbar):.6f} exactly, by definition of Bbar)")
    print(f"            mean(b.Bbar_hat) over grid        = {unit_dir_avg:.6f}  "
          f"(unit-direction volume average -- NOT the same quantity as the flux "
          f"average above in general; here they nearly coincide because |B|~=1 "
          f"pointwise makes b~=B)")

    # --- seeds ---
    rng = np.random.default_rng(args.seed)
    seeds_u = seed_uniform(args.n_uniform, rng)
    seeds_r = seed_in_reversed_region(args.n_reversed, B, Bbar_hat, grid, rng)
    seeds0 = np.concatenate([seeds_u, seeds_r], axis=0)
    seed_kind = np.array(['uniform'] * args.n_uniform + ['reversed'] * args.n_reversed)

    nsteps_total = int(np.ceil(args.arc_boxes * TWOPI / args.h))
    print(f"    seeds: {args.n_uniform} uniform + {args.n_reversed} in-reversed-region "
          f"= {seeds0.shape[0]} total")
    print(f"    arc length target: {args.arc_boxes} box lengths = "
          f"{args.arc_boxes*TWOPI:.3f}  ->  nsteps={nsteps_total}, h={args.h}")

    params = dict(n_uniform=args.n_uniform, n_reversed=args.n_reversed, seed_rng=args.seed)
    traj, seed_kind, done = integrate_with_resume(
        B, grid, seeds0, seed_kind, args.h, nsteps_total, args.chunk_steps,
        args.checkpoint, params, fresh=args.fresh)

    if not done:
        print("\n[topology] PARTIAL RUN -- rerun the same command to continue "
              f"from the checkpoint at {args.checkpoint}.")
        sys.exit(0)

    print(f"\n[topology] integration complete: {traj.shape[0]-1} steps, "
          f"{traj.shape[1]} lines, arc length {(traj.shape[0]-1)*args.h:.2f}")

    # --- interpolation sanity along the real traces ---
    sub = traj[::50]
    Bv_sub = interp_B(sub.reshape(-1, 3), B, grid)
    nrm_sub = np.sqrt((Bv_sub ** 2).sum(axis=1))
    print(f"    |B_interp| along traces (subsampled): min={nrm_sub.min():.5f} "
          f"mean={nrm_sub.mean():.5f} max={nrm_sub.max():.5f}  (expect ~1)")

    # --- classify ---
    res = classify(traj, Bbar_hat, h=args.h)
    visited = visited_reversed(traj, B, grid, Bbar_hat)

    is_u = seed_kind == 'uniform'
    is_r = seed_kind == 'reversed'

    def report_population(name, mask):
        lab = res['label'][mask]
        n = mask.sum()
        counts = {LABEL_NAMES[k]: int((lab == k).sum()) for k in (0, 1, 2)}
        print(f"    {name:9s} (n={n}):  OPEN={counts['OPEN']:3d} "
              f"({100*counts['OPEN']/n:5.1f}%)   "
              f"TRAPPED={counts['TRAPPED']:3d} ({100*counts['TRAPPED']/n:5.1f}%)   "
              f"INTERMEDIATE={counts['INTERMEDIATE']:3d} ({100*counts['INTERMEDIATE']/n:5.1f}%)")
        return counts

    print("\n[topology] === CLASSIFICATION TABLE ===")
    counts_u = report_population('uniform', is_u)
    counts_r = report_population('reversed', is_r)
    counts_all = report_population('ALL', np.ones_like(is_u, dtype=bool))

    print("\n[topology] === DRIFT RATE ===")
    dr = res['drift_rate']
    print(f"    drift rate v_d = d_end/s_end over all {len(dr)} lines: "
          f"mean={dr.mean():.5f}  std={dr.std():.5f}  "
          f"min={dr.min():.5f}  max={dr.max():.5f}")
    print(f"    |Bbar| (theoretical flux-average reference) = {np.linalg.norm(Bbar):.5f}")
    print(f"    volume avg of b.Bbar_hat (ergodic-average reference) = {unit_dir_avg:.5f}")
    print(f"    seed-averaged drift rate vs |Bbar|: "
          f"{dr.mean():.5f} vs {np.linalg.norm(Bbar):.5f}  "
          f"(ratio {dr.mean()/np.linalg.norm(Bbar):.3f})")

    print("\n[topology] === REVERSED-REGION ESCAPE ===")
    for name, mask in (('uniform', is_u), ('reversed', is_r), ('ALL', np.ones_like(is_u, bool))):
        n = mask.sum()
        v = visited[mask]
        opn = res['label'][mask] == 2
        n_visited = int(v.sum())
        n_visited_and_open = int((v & opn).sum())
        frac = n_visited_and_open / max(n_visited, 1)
        print(f"    {name:9s}: {n_visited}/{n} lines visited the reversed region; "
              f"of those, {n_visited_and_open} ({100*frac:.1f}%) classified OPEN")

    # ---------------------------------------------------------------------
    # Figure
    # ---------------------------------------------------------------------
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 2, figsize=(10, 4), dpi=200)

    # panel (a): displacement along Bbar_hat vs arc length, sample of lines
    ax = axs[0]
    M = traj.shape[1]
    n_a = min(args.n_plot_lines_a, M)
    idx_a = np.linspace(0, M - 1, n_a).astype(int)
    s_boxes = res['s'] / TWOPI
    d_boxes = res['disp'] / TWOPI
    seen_labels = set()
    for i in idx_a:
        lab = res['label'][i]
        lbl = LABEL_NAMES[lab] if lab not in seen_labels else None
        seen_labels.add(lab)
        ax.plot(s_boxes, d_boxes[:, i], color=LABEL_COLORS[lab], alpha=0.6,
                 lw=0.8, label=lbl)
    ax.set_xlabel('arc length s [box lengths]')
    ax.set_ylabel(r'displacement along $\hat{B}_{bar}$ [box lengths]')
    ax.set_title('(a) displacement vs arc length')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc='best')

    # panel (b): puncture plot at upward z=0(mod 2pi) crossings
    ax = axs[1]
    n_b = min(args.n_plot_lines_b, M)
    idx_b = np.linspace(0, M - 1, n_b).astype(int)
    cmap = plt.get_cmap('tab20')
    for c, i in enumerate(idx_b):
        xc, yc = upward_crossings(traj[:, i, :])
        if len(xc):
            ax.scatter(xc, yc, s=6, color=cmap(c % 20), alpha=0.7)
    ax.set_xlim(0, TWOPI)
    ax.set_ylim(0, TWOPI)
    ax.set_xlabel('x mod 2pi')
    ax.set_ylabel('y mod 2pi')
    ax.set_title('(b) puncture plot: upward z=0 (mod 2pi) crossings')
    ax.grid(alpha=0.3)
    ax.set_aspect('equal')

    fig.tight_layout()
    fig.savefig(args.fig_out)
    print(f"\n[topology] figure written: {args.fig_out}")

    # ---------------------------------------------------------------------
    # Final summary
    # ---------------------------------------------------------------------
    open_frac_all = counts_all['OPEN'] / M
    print("\n[topology] === SUMMARY ===")
    print(f"    Overall OPEN fraction: {open_frac_all:.3f}  "
          f"({counts_all['OPEN']}/{M} lines)")
    print(f"    Overall TRAPPED fraction: {counts_all['TRAPPED']/M:.3f}  "
          f"({counts_all['TRAPPED']}/{M} lines)")
    verdict = "OPEN" if open_frac_all > 0.9 and counts_all['TRAPPED'] == 0 else \
              ("MOSTLY OPEN with some trapped/intermediate lines" if open_frac_all > 0.5
               else "topology is NOT predominantly open -- investigate further")
    print(f"    Verdict: {verdict}")


if __name__ == '__main__':
    main()
