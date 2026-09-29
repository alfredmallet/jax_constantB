"""Timing + accuracy: taranis SM22 (branch sm22) vs the jax_constantB 2D reference (sm22_v2).

  python sm22_taranis_bench.py tar2d N seed OUTDIR [scheme]   # taranis dims=2, lands on TARGETS
  python sm22_taranis_bench.py ref2d N seed OUTDIR [K]        # sm22_v2.ref_run (fixed-K BiCGSTAB)
  python sm22_taranis_bench.py tar3d N OUTDIR [Amax]          # taranis z_spectral N^3, genuinely 3D IC

Precision: taranis reads TARANIS_PRECISION (32/64), the reference SM22_FP32 (1/0); the runner
sets both. Writes one JSON per job to OUTDIR: wall (excl. compile), steps, per-target
diagnostics (maxgrad, lp8, Berr; taranis also Berr_out, edge_max, Krylov its).
Parity is only meaningful AT CONVERGENCE (the codes discretize differently: 2/3-band
Galerkin vs Hou-Li collocation) -- compare ladders, never splice them.
"""
import json, os, sys, time
import numpy as np

TARGETS = [0.3, 0.5, 0.7, 0.9, 1.0]
TH = np.deg2rad(30.0)


def prec_tag():
    if sys.argv[1].startswith("tar"):
        return "fp" + os.environ.get("TARANIS_PRECISION", "32")
    return "fp32" if os.environ.get("SM22_FP32", "0") == "1" else "fp64"


def dump(out, name, d):
    os.makedirs(out, exist_ok=True)
    d["prec"] = prec_tag()
    import jax
    d["device"] = str(jax.devices()[0])
    with open(os.path.join(out, name + ".json"), "w") as f:
        json.dump(d, f, indent=1)
    print(json.dumps({k: v for k, v in d.items() if k != "rows"}), flush=True)
    for r in d.get("rows", []):
        print("   ", r, flush=True)


def ref2d(N, seed, out, K=25):
    import jax
    import sm22_v2
    from sm22_grow2d import seed_state
    B0 = seed_state(N, TH, 1.0, seed, 0.05, 1)
    ops = sm22_v2.build_v2(N, TH, K=K)
    rows, _, _, wall, n = sm22_v2.ref_run(ops, np.asarray(B0)[None], TARGETS)
    rows = [{k: r.get(k) for k in ("target", "A", "maxgrad", "lp8", "Berr")} for r in rows[0]]
    dump(out, f"ref2d_N{N}_s{seed}_K{K}_{prec_tag()}",
         dict(code="ref2d", N=N, seed=seed, K=K, wall=wall, steps=n, rows=rows))


def _taranis_grow(p, st, targets, scheme_name, max_steps=20000, wall_cap=None):
    import jax
    from taranis.run import init_aux
    from taranis.timestepping import get_scheme
    from taranis.physics import equation_registry, construct_rhs
    from taranis import diagnostics as D
    kg = __import__("taranis").setup_kgrids(p)
    rec = equation_registry["SM22"]; rhs = construct_rhs(rec)
    stepper, scheme = get_scheme(scheme_name)
    step = jax.jit(lambda s, a, dto: stepper(s, kg, p, rhs, rec.set_timestep_func, scheme, dto, aux=a))
    stepa = jax.jit(lambda s, a: stepper(s, kg, p, rhs, rec.set_timestep_func, scheme, aux=a))
    amp = jax.jit(lambda s: D.sm22.amplitude(s, p))
    diag = jax.jit(lambda s: D.sm22.diagnostics(s, kg, p))
    its = jax.jit(lambda s, a: rec.grad_func(s, kg, p, a).its)
    aux = init_aux(st, kg, p)
    jax.block_until_ready(stepa(st, aux)); jax.block_until_ready(step(st, aux, 0.01))
    jax.block_until_ready(diag(st)); jax.block_until_ready(its(st, aux))
    T0 = time.perf_counter(); tdiag = 0.0
    A = float(amp(st)); ti = 0; rows = []; n = 0; itl = []
    while ti < len(targets) and n < max_steps:
        s2, a2 = stepa(st, aux); n += 1
        A2 = float(amp(s2))
        while ti < len(targets) and A2 >= targets[ti]:
            t1 = time.perf_counter()
            dt = float(s2.t - st.t)
            sl, al = step(st, aux, dt * (targets[ti] - A) / (A2 - A))
            d = {k: float(v) for k, v in diag(sl).items()}
            rows.append(dict(target=targets[ti], its=int(its(sl, al)),
                             **{k: d[k] for k in ("A", "maxgrad", "lp8", "Berr", "Berr_out", "edge_max", "divB")}))
            print(f"  A={targets[ti]} step={n} {time.perf_counter()-T0:.1f}s {rows[-1]}", flush=True)
            ti += 1
            tdiag += time.perf_counter() - t1
        if not np.isfinite(A2):
            rows.append(dict(unstable=True, step=n)); break
        st, aux, A = s2, a2, A2
        if wall_cap and time.perf_counter() - T0 > wall_cap:
            rows.append(dict(wall_cap_hit=True, step=n, A=A)); break
    return rows, time.perf_counter() - T0 - tdiag, n


def tar2d(N, seed, out, scheme="lsrk54"):
    import jax.numpy as jnp
    import taranis
    from sm22_grow2d import seed_state
    B0 = seed_state(N, TH, 1.0, seed, 0.05, 1)
    p = taranis.Parameters(N, N, 1.0, 1.0, 0.5, eqtype="SM22", dims=2, eqpars=dict(maxiter=200))
    st = taranis.initialize(lambda x, y: jnp.asarray(B0)[:, None], p)
    rows, wall, n = _taranis_grow(p, st, TARGETS, scheme)
    dump(out, f"tar2d_N{N}_s{seed}_{scheme}_{prec_tag()}",
         dict(code="tar2d", N=N, seed=seed, scheme=scheme, wall=wall, steps=n, rows=rows))


def tar3d(N, out, Amax=0.75):
    import jax.numpy as jnp
    import taranis
    rng = np.random.default_rng(0)
    bh = np.array([np.cos(0.5)*np.cos(0.3), np.cos(0.5)*np.sin(0.3), np.sin(0.5)])
    modes = [((a, b, c), rng.normal(), rng.uniform(0, 2*np.pi)) for (a, b, c) in
             [(1,0,0),(0,1,0),(0,0,1),(1,1,0),(1,0,1),(0,1,1),(1,-1,0),(1,0,-1),(0,1,-1)]]

    def f3(x, y, z):   # dB = bhat x grad(chi): div-free and _|_ Bbar (|B|^2 error O(A^2))
        X, Y, Z = [2*np.pi*a for a in (x, y, z)]
        G = 0.0
        for (a, b, c), amp, ph in modes:
            G = G + amp*jnp.stack([k*(-jnp.sin(a*X + b*Y + c*Z + ph)) for k in (a, b, c)])
        dB = jnp.stack([bh[1]*G[2] - bh[2]*G[1], bh[2]*G[0] - bh[0]*G[2], bh[0]*G[1] - bh[1]*G[0]])
        dB = dB * (0.05 / jnp.sqrt(jnp.mean(jnp.sum(dB**2, 0))))
        return dB + jnp.asarray(bh)[:, None, None, None]

    p = taranis.Parameters(N, N, 1.0, 1.0, 0.5, eqtype="SM22", dims=3, nz=N, Lz=1.0, z_spectral=True,
                           eqpars=dict(maxiter=300))
    st = taranis.initialize(f3, p)
    tg = [a for a in (0.1, 0.2, 0.3, 0.5, 0.75, 1.0) if a <= Amax]
    rows, wall, n = _taranis_grow(p, st, tg, "lsrk54", wall_cap=2400)
    dump(out, f"tar3d_N{N}_{prec_tag()}", dict(code="tar3d", N=N, wall=wall, steps=n, rows=rows))


if __name__ == "__main__":
    m = sys.argv[1]
    if m == "ref2d":
        ref2d(int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], *(int(a) for a in sys.argv[5:6]))
    elif m == "tar2d":
        tar2d(int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], *sys.argv[5:6])
    elif m == "tar3d":
        tar3d(int(sys.argv[2]), sys.argv[3], *(float(a) for a in sys.argv[4:5]))
