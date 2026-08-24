"""Poincare puncture maps of the field lines of a constant-|B| state (jax).

WHY.  `fieldlines.py` answers the topology question (do lines progress along
Bbar, or circulate?) with a host-numpy tracer over O(500) lines.  The question
here is finer and needs an order of magnitude more lines and arc length: does
the deflected structure carry INVARIANT SURFACES -- nested tori whose punctures
through a fixed plane lie on closed curves -- or is the line flow chaotic?  A
constant-|B| field is a unit-speed volume-preserving flow, so its z = const
return map is an area-preserving map and the usual KAM picture applies:
regular lines puncture on 1D curves, chaotic lines fill 2D patches.

METHOD (and its two error scales, both documented because the answer is a
geometric classification and geometry is what interpolation error destroys):

  * dr/ds = b(r) = B/|B| integrated by RK4 in ARC LENGTH, so arc length
    traversed = steps * h exactly.  ACCURACY: RK4's formal O(h^4) does NOT
    survive the C^0 trilinear interpolant -- the observed order in h collapses
    to ~1 at working `refine` (tests/test_v3.py case 7; DESIGN.md v3) -- so
    accuracy is bought with `refine` (second order in dx/refine, measured
    ratios 16.0/4.0 per doubling), not with small h.  On smooth analytic
    fields at very large `refine` the h-convergence does look fourth order
    down to the interpolation floor; do not rely on that in production.
    h = 0.02 it is invisible.
  * b is evaluated by TRILINEAR interpolation of B on a `refine`x zero-padded
    grid, and THIS is what limits the map.  Trilinear interpolation is second
    order, so the direction error is O((dx/refine)^2), dx = 2pi/N; measured
    on an exact arc-polarised state (endpoint error over 8 units of arc
    length): 7.67e-5 at dx/refine = 0.049, 4.80e-6 at 0.0123, 1.20e-6 at
    0.0061 -- ratios 16.0 and 4.00, i.e. exactly second order, with prefactor
    0.032 for that smooth field.  On the maxgrad ~ 18 state at eps = 10.05
    the same rate holds with prefactor ~16: puncture positions after 5
    transits move by 0.100, 0.019, 8.0e-3, 3.2e-3 rad for refine = 1, 2, 3, 4
    against a refine = 6 reference.  Zero-padding is spectral (exact for the
    retained modes), so `refine` buys real accuracy -- at 8x the memory per
    doubling.  ALWAYS report the refine used beside a puncture figure: a
    "chaotic" patch narrower than the interpolation error is not a
    measurement.
  * Positions stay UNWRAPPED (continuous), like legacy
    `fieldlines.trace_lines`; the periodic box enters only through the
    floor-and-modulo grid lookup and through the final (x, y) mod 2pi of the
    punctures.

MEMORY.  512 lines x 300 transits is ~1.5e5 RK4 steps; a stored trajectory
would be gigabytes, so the scan CARRIES the position and the driver never
holds more than one chunk: `trace_punctures` runs the jitted scan over
`chunk` steps at a time, extracts that chunk's crossings on the host, drops
the positions, and restarts the next chunk from the carried endpoint (the
chunk's first sample is the previous chunk's last, so no crossing is counted
twice and none falls between chunks).  Peak = chunk * lines * 3 * 8 bytes
(~50 MB at the defaults) plus the padded field (~200 MB at 160^2x324); a
512-line, 300-transit run of the driver peaks at 1.4 GB RSS in 28 s on CPU.
`trace` returns a whole trajectory and is for tests and short runs only.
Chunking is EXACT: puncture counts and step indices are chunk-independent,
and positions match a single-shot trace except for crossings within ~2
steps of a chunk edge, where a one-sided cubic stencil shifts them by
<~1e-6 (well below the interpolation error); counts and indices are exact
at every chunk size.  For chunk >= ~1000 non-edge positions are bit-identical
trace (below that, crossings near a chunk edge use a one-sided cubic
stencil and shift by ~1e-5 -- still far under the interpolation error).

Float64 throughout (the package `__init__` enables x64 before jax is used).
"""
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

from .spectral import TWOPI, zero_pad

# A sample this close (in z) to the section plane counts as being ON it, and
# so contributes exactly one crossing -- to the step that arrives at it, never
# also to the step that leaves it.
_GRAZE = 1e-12


# ---------------------------------------------------------------------------
# the tracer
# ---------------------------------------------------------------------------

def fine_field(B, refine=2):
    """Interpolant array: B spectrally zero-padded `refine`x, laid out
    (Nx', Ny', Nz', 3) so one gather fetches a whole vector.

    Host-side and cold: called ONCE per trace, not per step (zero_pad is a
    full-FFT round trip and is deliberately un-jitted, see spectral.py).
    """
    B = np.asarray(B, dtype=float)
    refine = int(refine)
    if refine < 1:
        raise ValueError(f"refine must be >= 1, got {refine}")
    if refine == 1:
        Bf = B
    else:
        fine = tuple(refine * n for n in B.shape[1:])
        Bf = np.stack([np.asarray(zero_pad(B[i], fine)) for i in range(3)])
    return jnp.asarray(np.ascontiguousarray(np.moveaxis(Bf, 0, -1)))


@partial(jax.jit, static_argnums=(3,))
def _run(Bf, pos0, h, n_steps):
    """`n_steps` RK4 steps for every line: (final positions, all positions).

    ONE jitted program: `lax.scan` over the steps, `vmap` over the lines.
    Grid constants come from Bf.shape, which is static under jit, so nothing
    but the field, the seeds and h is traced.
    """
    N = Bf.shape[:3]
    inv_d = jnp.asarray([n / TWOPI for n in N])
    Nj = jnp.asarray(N)

    def bdir(p):
        f = p * inv_d
        i0 = jnp.floor(f).astype(jnp.int64)
        t = f - i0
        i0 = jnp.mod(i0, Nj)
        i1 = jnp.mod(i0 + 1, Nj)
        v = jnp.zeros(3)
        for ii, wx in ((i0[0], 1.0 - t[0]), (i1[0], t[0])):
            for jj, wy in ((i0[1], 1.0 - t[1]), (i1[1], t[1])):
                for kk, wz in ((i0[2], 1.0 - t[2]), (i1[2], t[2])):
                    v = v + (wx * wy * wz) * Bf[ii, jj, kk]
        return v / jnp.maximum(jnp.linalg.norm(v), 1e-14)

    def body(p, _):
        k1 = bdir(p)
        k2 = bdir(p + 0.5 * h * k1)
        k3 = bdir(p + 0.5 * h * k2)
        k4 = bdir(p + h * k3)
        pn = p + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return pn, pn

    fin, traj = jax.vmap(lambda p: lax.scan(body, p, None, length=n_steps))(pos0)
    return fin, jnp.swapaxes(traj, 0, 1)


def trace(B, seeds, h=0.02, n_steps=1000, refine=2):
    """RK4 field-line trace, returning the full unwrapped trajectory.

    B      : (3, Nx, Ny, Nz) real field on [0, 2pi)^3 (numpy or jax).
    seeds  : (M, 3) absolute start positions.
    h      : arc-length step (see ACCURACY in the module docstring:
             effective order in h is ~1 on the C^0 interpolant; buy
             accuracy with `refine`).
    refine : zero-pad factor for the interpolant; error O((dx/refine)^2).

    Returns (n_steps+1, M, 3) numpy, positions UNWRAPPED (index 0 = seeds).
    Allocates the whole trajectory -- use `trace_punctures` for long runs.
    """
    Bf = fine_field(B, refine)
    p0 = jnp.asarray(np.asarray(seeds, dtype=float))
    _, traj = _run(Bf, p0, float(h), int(n_steps))
    return np.concatenate([np.asarray(p0)[None], np.asarray(traj)], axis=0)


def trace_punctures(B, seeds, h=0.02, n_steps=1000, refine=2, z0=np.pi,
                    chunk=4096, progress=None):
    """Memory-bounded trace: only the punctures survive (see module docstring).

    `n_steps` is rounded UP to a whole number of chunks so that every chunk
    compiles the same program.  `progress`, if given, is called as
    progress(steps_done, n_steps) after each chunk.

    Returns (xy_list, idx_list, final_pos, n_steps_actual) with xy_list and
    idx_list as in `punctures` and indices counted from the seed sample.
    """
    Bf = fine_field(B, refine)
    pos = jnp.asarray(np.asarray(seeds, dtype=float))
    M = pos.shape[0]
    chunk = int(chunk)
    nchunk = max(1, -(-int(n_steps) // chunk))
    n_total = nchunk * chunk
    xs = [[] for _ in range(M)]
    ids = [[] for _ in range(M)]
    done = 0
    for _ in range(nchunk):
        fin, traj = _run(Bf, pos, float(h), chunk)
        seg = np.concatenate([np.asarray(pos)[None], np.asarray(traj)], axis=0)
        del traj
        xy_c, id_c = punctures(seg, z0)
        for m in range(M):
            if len(id_c[m]):
                xs[m].append(xy_c[m])
                ids[m].append(id_c[m] + done)
        done += chunk
        pos = fin
        if progress is not None:
            progress(done, n_total)
    empty_xy, empty_i = np.zeros((0, 2)), np.zeros(0, dtype=np.int64)
    xy_list = [np.concatenate(a) if a else empty_xy for a in xs]
    idx_list = [np.concatenate(a) if a else empty_i for a in ids]
    return xy_list, idx_list, np.asarray(pos), n_total


# ---------------------------------------------------------------------------
# punctures
# ---------------------------------------------------------------------------

def _lag4(u):
    """Cubic Lagrange basis and its derivative on the nodes u = 0, 1, 2, 3."""
    L = np.stack([-(u - 1) * (u - 2) * (u - 3) / 6.0,
                  u * (u - 2) * (u - 3) / 2.0,
                  -u * (u - 1) * (u - 3) / 2.0,
                  u * (u - 1) * (u - 2) / 6.0], axis=1)
    dL = np.stack([-(3 * u * u - 12 * u + 11) / 6.0,
                   (3 * u * u - 10 * u + 6) / 2.0,
                   -(3 * u * u - 8 * u + 3) / 2.0,
                   (3 * u * u - 6 * u + 2) / 6.0], axis=1)
    return L, dL


def punctures(traj, z0=np.pi):
    """Upward crossings of the planes z = z0 (mod 2pi).

    traj : (T, M, 3) unwrapped positions (as returned by `trace`).

    A crossing is detected from the change of the plane index
    n = floor((z - z0)/2pi) between consecutive samples; every level in
    (n_t, n_{t+1}] is emitted, so the rare step that clears more than one
    plane is handled.  Grazing: a sample within `_GRAZE` (1e-12) of a plane is
    snapped ONTO it, hence is the end of the arriving interval and the start
    of the leaving one -- counted exactly once.

    The crossing point itself is located by CUBIC interpolation in arc length
    through the four samples straddling it (Newton on z(u) = z_plane from the
    linear guess, then the same cubic for x and y).  This matters: samples are
    equally spaced in s, so the cubic keeps the puncture positions at the
    trajectory's own accuracy (measured order ~3 on exact samples) and
    at the order of the RK4 step, whereas the obvious linear interpolation to
    the plane would cap them at O(h^2) and be the accuracy bottleneck (checked
    -- linear floors the analytic-state drift error near 1e-6 at h = 0.005,
    cubic tracks the trajectory error).  Falls back to linear if the Newton
    iterate leaves its bracket or the trajectory is shorter than 4 samples.

    Returns (xy_list, idx_list): ragged per-line lists of (P_m, 2) puncture
    coordinates (mod 2pi) and (P_m,) integer step indices (the sample BEFORE
    each crossing), both in increasing arc length.  Downward crossings are
    deliberately dropped -- the map must be a return map of one orientation.
    """
    traj = np.asarray(traj, dtype=float)
    T, M = traj.shape[0], traj.shape[1]
    g = (traj[:, :, 2] - float(z0)) / TWOPI
    gr = np.round(g)
    g = np.where(np.abs(g - gr) < _GRAZE / TWOPI, gr, g)
    n = np.floor(g).astype(np.int64)
    dn = n[1:] - n[:-1]
    up = dn > 0
    ti, mi = np.nonzero(up)                       # t ascending within each m
    if ti.size == 0:
        return ([np.zeros((0, 2)) for _ in range(M)],
                [np.zeros(0, dtype=np.int64) for _ in range(M)])
    cnt = dn[ti, mi]
    tt = np.repeat(ti, cnt)
    mm = np.repeat(mi, cnt)
    off = np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    lev = np.repeat(n[ti, mi], cnt) + off + 1     # planes cleared by this step
    zc = float(z0) + TWOPI * lev
    z_a, z_b = traj[tt, mm, 2], traj[tt + 1, mm, 2]
    frac = np.clip((zc - z_a) / np.where(z_b == z_a, 1.0, z_b - z_a), 0.0, 1.0)
    xy = np.empty((tt.size, 2))
    if T >= 4:
        j = np.clip(tt - 1, 0, T - 4)             # 4-sample stencil, in range
        u0 = (tt - j) + frac
        Z = np.stack([traj[j + i, mm, 2] for i in range(4)], axis=1)
        u = u0.copy()
        for _ in range(3):
            L, dL = _lag4(u)
            df = (dL * Z).sum(1)
            u = u - np.where(np.abs(df) > 1e-30, ((L * Z).sum(1) - zc) / df, 0.0)
        lo = (tt - j).astype(float)
        u = np.where(np.isfinite(u) & (u >= lo - 1e-9) & (u <= lo + 1 + 1e-9),
                     u, u0)
        L, _ = _lag4(u)
        for c in (0, 1):
            X = np.stack([traj[j + i, mm, c] for i in range(4)], axis=1)
            xy[:, c] = np.mod((L * X).sum(1), TWOPI)
    else:
        for c in (0, 1):
            a, b = traj[tt, mm, c], traj[tt + 1, mm, c]
            xy[:, c] = np.mod(a + frac * (b - a), TWOPI)
    order = np.argsort(mm, kind="stable")         # group by line, keep t order
    xy, tt, mm = xy[order], tt[order], mm[order]
    cuts = np.cumsum(np.bincount(mm, minlength=M))[:-1]
    return list(np.split(xy, cuts)), list(np.split(tt, cuts))


def pad_punctures(xy_list):
    """(M, Pmax, 2) NaN-padded array + (M,) counts, for saving to .npz."""
    M = len(xy_list)
    P = max([len(a) for a in xy_list], default=0)
    out = np.full((M, P, 2), np.nan)
    cnt = np.zeros(M, dtype=np.int64)
    for m, a in enumerate(xy_list):
        out[m, :len(a)] = a
        cnt[m] = len(a)
    return out, cnt


def _wrap(d):
    """Signed periodic offset in (-pi, pi]."""
    return (d + np.pi) % TWOPI - np.pi


def rotation_number(punct_xy, center, turns=False):
    """Mean angular advance per puncture about `center`.

    punct_xy : (P, 2) for one line, or a list of such (one entry per line).
    Angles are taken from the PERIODIC offset to `center` (punctures are mod
    2pi, so a raw difference would jump a box width) and unwrapped along the
    sequence, so the result counts whole turns: it is
    (theta_last - theta_first)/(P-1), in radians unless `turns=True`, in
    which case it is divided by 2pi (the usual rotation-number units).
    NaN for a line with fewer than two punctures.
    """
    if isinstance(punct_xy, (list, tuple)):
        return np.array([rotation_number(p, center, turns) for p in punct_xy])
    xy = np.asarray(punct_xy, dtype=float)
    if xy.shape[0] < 2:
        return np.nan
    th = np.unwrap(np.arctan2(_wrap(xy[:, 1] - center[1]),
                              _wrap(xy[:, 0] - center[0])))
    adv = (th[-1] - th[0]) / (xy.shape[0] - 1)
    return float(adv / TWOPI if turns else adv)


# ---------------------------------------------------------------------------
# regular vs chaotic: the nearest-neighbour-spread criterion
# ---------------------------------------------------------------------------

def nn_spread(xy_list, min_punct=8):
    """Classify each line's puncture cloud as curve-like or area-filling.

    CRITERION (simple, scale-free, and P-aware; document it wherever the
    chaotic fraction is quoted).  For one line with P punctures let R be the
    RMS distance from the cloud's circular mean (periodic metric) and d the
    MEDIAN nearest-neighbour distance within the cloud.  Then

        P punctures spread along a closed invariant curve  =>  d ~ 2 pi R / P
        P punctures filling a disc of radius R (Poisson)   =>  d ~ R sqrt(pi/P) / 2

    These differ by a factor sqrt(P/pi)/4, so the geometric mean of the two is
    a threshold that needs no tuning: the line is called CHAOTIC when
    d > sqrt(d_curve * d_area).  Reported as rho = d / d_curve, with the
    threshold rho_crit = sqrt(0.25*sqrt(P/pi)); rho ~ 1 is a curve, rho ~
    rho_crit^2 an area.  CAVEATS: the discrimination only grows as P^(1/4),
    so short runs are indecisive; and an eccentric invariant curve is longer
    than 2 pi R, which inflates rho -- treat a marginal rho as unclassified,
    not as chaos.  Validated on synthetic clouds (100 each): closed curves
    are called chaotic 0% of the time at P = 30, 100 and 300 (median
    rho 0.35); uniformly filled discs 88% at P = 30 and 100% at P = 100 and
    300 (median rho 1.10, 1.96, 3.28).  On the eps = 10.05 state it isolates
    exactly the 13 island lines out of 512, and their rotation numbers rise
    monotonically outward (0.125 -> 0.155 turns/transit) as an island's
    twist profile must.

    Returns (flag, rho, rho_crit): bool / float / float arrays of length M,
    NaN and False for lines with fewer than `min_punct` punctures.
    """
    M = len(xy_list)
    flag = np.zeros(M, dtype=bool)
    rho = np.full(M, np.nan)
    crit = np.full(M, np.nan)
    for m, xy in enumerate(xy_list):
        P = len(xy)
        if P < min_punct:
            continue
        cx = np.angle(np.exp(1j * xy[:, 0]).mean())
        cy = np.angle(np.exp(1j * xy[:, 1]).mean())
        dx, dy = _wrap(xy[:, 0] - cx), _wrap(xy[:, 1] - cy)
        R = float(np.sqrt(np.mean(dx ** 2 + dy ** 2)))
        if R <= 0:
            continue
        sep = np.hypot(_wrap(dx[:, None] - dx[None, :]),
                       _wrap(dy[:, None] - dy[None, :]))
        np.fill_diagonal(sep, np.inf)
        d = float(np.median(sep.min(axis=1)))
        d_curve = TWOPI * R / P
        rho[m] = d / d_curve
        crit[m] = np.sqrt(0.25 * np.sqrt(P / np.pi))
        flag[m] = rho[m] > crit[m]
    return flag, rho, crit


def transverse_dispersion(xy_list):
    """(M,) median in-plane distance between CONSECUTIVE punctures of a line,
    periodic metric -- how far the return map moves a point per transit."""
    out = np.full(len(xy_list), np.nan)
    for m, xy in enumerate(xy_list):
        if len(xy) < 2:
            continue
        out[m] = float(np.median(np.hypot(_wrap(np.diff(xy[:, 0])),
                                          _wrap(np.diff(xy[:, 1])))))
    return out
