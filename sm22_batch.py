"""Batched SM22 2.5D growth: all seeds advance together in one vmapped RK4 step.

Same physics and outputs as sm22_grow2d.py (series_/snaps_ files per seed, same tags), but the
state is (S,3,N,N) so a GPU sees one large batched FFT/CG workload instead of S tiny ones.
Each seed keeps its own dt (CFL on its own umax) and its own save thresholds; finished seeds
are frozen (dt=0). The batched CG runs until every seed's CG has converged.

Usage: SM22_FP32=1 python sm22_batch.py --N 128 --seeds 8-57 --Amax 1.2 --out DIR
"""
import argparse, os, time
import numpy as np
from sm22_grow2d import FP32, build, seed_state, diagnostics, jax, jnp


def parse_seeds(s):
    if "-" in s:
        lo, hi = map(int, s.split("-")); return list(range(lo, hi + 1))
    return [int(x) for x in s.split(",")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=128)
    ap.add_argument("--seeds", default="8-57")
    ap.add_argument("--th", type=float, default=30.0)
    ap.add_argument("--A0", type=float, default=0.05)
    ap.add_argument("--Amax", type=float, default=1.2)
    ap.add_argument("--kcut", type=int, default=1)
    ap.add_argument("--dtmax", type=float, default=0.01)
    ap.add_argument("--cgit", type=int, default=300, help="CG iteration cap per solve")
    ap.add_argument("--snapmin", type=float, default=0.0, help="only store snapshots with A >= this")
    ap.add_argument("--out", default="sm22_out")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    th = np.deg2rad(a.th); B0 = 1.0; dx = 1.0 / a.N
    sfx = ("_fp32" if FP32 else "") + (f"_cg{a.cgit}" if a.cgit != 300 else "")
    tags = {s: f"N{a.N}_s{s}_th{int(a.th)}_k{a.kcut}{sfx}" for s in parse_seeds(a.seeds)}
    seeds = [s for s in tags if not os.path.exists(os.path.join(a.out, f"series_{tags[s]}.npz"))]
    if not seeds:
        print("all seeds done"); return
    S = len(seeds)
    ops = build(a.N, th, B0, cg_iters=a.cgit, cg_tol=1e-5 if FP32 else 1e-11)
    Bbar = ops["Bbar"]
    step = jax.jit(jax.vmap(ops["rk4"], in_axes=(0, 0, 0)))

    @jax.jit
    def cheap(B):
        Bsq = jnp.sum(B * B, axis=1)
        A = jnp.sqrt(jnp.mean(jnp.sum((B - Bbar[None, :, None, None])**2, axis=1), axis=(1, 2))) / B0
        m = jnp.mean(Bsq, axis=(1, 2))
        return A, jnp.sqrt(jnp.mean((Bsq - m[:, None, None])**2, axis=(1, 2))) / m

    B = jnp.asarray(np.stack([seed_state(a.N, th, B0, s, a.A0, a.kcut) for s in seeds]))
    phi = jnp.zeros((S, a.N, a.N))
    umax = np.zeros(S); nextA = np.full(S, a.A0); done = np.zeros(S, bool)
    rows = [[] for _ in range(S)]; snaps = [{} for _ in range(S)]
    T0 = time.time(); nstep = 0
    while not done.all():
        A, Berr = map(np.asarray, cheap(B))
        for i in np.nonzero(~done)[0]:
            if A[i] >= nextA[i] or A[i] >= a.Amax:
                Bi = np.asarray(B[i]); d = diagnostics(Bi, ops, B0); d.update(umax=float(umax[i]))
                rows[i].append(d)
                if d['A'] >= a.snapmin: snaps[i][f"A{d['A']:.3f}"] = Bi
                print(f"[{tags[seeds[i]]}] A={d['A']:.3f} Berr={d['Berr']:.2e} maxgrad={d['maxgrad']:.2f} "
                      f"Z={d['zfrac']:.3f} top|w|={d['top_absw']:.3f}/{d['all_absw']:.3f} "
                      f"step {nstep} ({time.time()-T0:.0f}s)", flush=True)
                nextA[i] *= 1.1
            if A[i] >= a.Amax or not np.isfinite(A[i]) or Berr[i] > 0.2:
                done[i] = True
                np.savez(os.path.join(a.out, f"series_{tags[seeds[i]]}.npz"),
                         **{k: np.array([r[k] for r in rows[i]]) for k in rows[i][0]})
                np.savez_compressed(os.path.join(a.out, f"snaps_{tags[seeds[i]]}.npz"), **snaps[i])
        if done.all():
            break
        dt = np.where(done, 0.0, np.minimum(a.dtmax, 0.3 * dx / (umax + 1e-12)))
        B, phi, its, res, um = step(B, phi, jnp.asarray(dt, B.dtype))
        umax = np.asarray(um); nstep += 1


if __name__ == "__main__":
    main()
