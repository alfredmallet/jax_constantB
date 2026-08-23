"""Field-line tracing and topology classification for a saved constant-|B|
state (pure host numpy -- the hot loop is a handful of gather/interpolate
calls over O(500) points, not a differentiable optimisation).

QUESTION.  Do field lines progress systematically through the periodic box
along the mean field Bbar (OPEN topology), or do some circulate without net
progress (CLOSED/TRAPPED, flux-rope-like islands)?  The solar-wind electron
strahl is a passive tracer of field-line connectivity, so switchback
observations require OPEN topology straight through the deflected region.

METHOD.
  1. Trace field lines by RK4 integration in ARC LENGTH: dr/ds = b(r), where
     b = B/|B| is the local unit tangent (unit-speed parametrisation, so arc
     length traversed = steps * step size exactly).
  2. b is evaluated by vectorized trilinear interpolation of the gridded B
     field, periodic-wrapped for the lookup; the trajectory itself is kept in
     unwrapped (absolute) coordinates so net displacement is meaningful.
  3. All seeds are integrated simultaneously (fully vectorized numpy).
  4. Classification of the displacement d(s) = (r(s)-r(0)).Bbar_hat:
     OPEN:          |d(s_end)| > 5 box lengths AND d(s) ~linear in s
                    (Pearson r^2 > 0.8) -- systematic, constant-rate progress.
     TRAPPED:       |d(s_end)| < 1 box length over the whole traced arc.
     INTERMEDIATE:  everything else (e.g. fast but non-linear excursions).

Chunk/resume support (save_checkpoint / integrate_with_resume) lets a caller
with a hard wall-clock limit rerun the same command to continue.
"""
import os
import time

import numpy as np

from .spectral import TWOPI


# ---------------------------------------------------------------------------
# Vectorized trilinear interpolation with periodic wrap
# ---------------------------------------------------------------------------

def grid_params(B):
    """Grid spacing/shape for the periodic box [0,2pi)^3 that B lives on."""
    _, Nx, Ny, Nz = B.shape
    return dict(Nx=Nx, Ny=Ny, Nz=Nz, dx=TWOPI / Nx, dy=TWOPI / Ny, dz=TWOPI / Nz)


def interp_B(pos, B, grid):
    """Trilinear interpolation of the vector field B at absolute positions.

    pos   : (M,3) array, ABSOLUTE (possibly unwrapped) coordinates -- the
            trajectory stays unwrapped so net progress is unambiguous.
    B     : (3,Nx,Ny,Nz) gridded field on the periodic box [0,2pi)^3.
    grid  : dict from grid_params(B).

    Returns (M,3) interpolated field vectors.  Periodic wrap is applied only
    for the lookup (mod 2pi + periodic corner indices); it never touches
    `pos`.  Fully vectorized over all M points at once -- this is the hot
    loop, so no python-level loop over points.
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
    """b = B/|B|; also returns |B| (interpolation sanity check -- along a
    genuine trace of a |B|=1 state this should stay close to 1; it need not
    be exactly 1 off-grid because trilinear interpolation of a curved unit
    vector field is not itself exactly unit-norm)."""
    nrm = np.sqrt((Bvec ** 2).sum(axis=1))
    safe = np.maximum(nrm, floor_eps)
    return Bvec / safe[:, None], nrm


# ---------------------------------------------------------------------------
# RK4 integrator in arc length: dr/ds = b(r)
# ---------------------------------------------------------------------------

def rk4_step(pos, h, B, grid):
    k1, _ = unit_dir(interp_B(pos, B, grid))
    k2, _ = unit_dir(interp_B(pos + 0.5 * h * k1, B, grid))
    k3, _ = unit_dir(interp_B(pos + 0.5 * h * k2, B, grid))
    k4, _ = unit_dir(interp_B(pos + h * k3, B, grid))
    return pos + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def trace_lines(pos0, B, grid, h, nsteps, progress_every=0):
    """Integrate all seeds simultaneously for `nsteps` RK4 steps.

    Returns traj, shape (nsteps+1, M, 3): unwrapped positions at every step
    (step 0 = pos0).  No sub-sampling -- the per-line classification and the
    puncture plot both want full resolution.
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


# ---------------------------------------------------------------------------
# Checkpoint (chunk / resume) support
# ---------------------------------------------------------------------------

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
    (same args) to resume.  Returns (traj, seed_kind, done) where done=True
    iff nsteps_total steps have been completed."""
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


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Puncture plot: upward crossings of planes z = 0 (mod 2pi)
# ---------------------------------------------------------------------------

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
