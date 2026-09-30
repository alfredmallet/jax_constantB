"""Onset-amplitude scans A_c(p) over low-dimensional IC families, on taranis SM22 (branch sm22).

  python sm22_taranis_scan.py fam2d FAM TH_DEG N OUTDIR [NPSI]   # 2D two-mode family, psi grid
  python sm22_taranis_scan.py dense3d N OUTDIR                    # the round-1/2 9-mode 3D IC, dense A

fam2d: Ax = cos(psi) cos(2pi k1.x) + sin(psi) cos(2pi k2.x), psi in [0, pi/2] (NPSI points, ends
included); the relative phase is removable by translation for two modes, so psi (the amplitude
ratio) is the only shape parameter, plus the 2D field angle theta. dB is built exactly as
sm22_grow2d.seed_state (A0=0.05). FAM: "10_01" = (1,0)+(0,1), "10_11" = (1,0)+(1,1).

No A_c is decided here: each IC is grown through dense TARGETS and the raw rule-7 diagnostics
are stored (maxgrad, lp8, Berr_out, Berr_out_max, edge_*); A_c comes from the ASCENDING N ladder
offline (per-rung slopes + Berr_out/Berr_out_max drainage). Never splice with sm22_v2 ladders.
Guard: an IC stops at nonfinite A, Berr_out > GUARD_BOUT (unresolved at this N anyway), or its
step/wall cap; the stop reason is recorded.
"""
import json, os, sys, time
import numpy as np

A0 = 0.05
T2D = [round(0.3 + 0.05*i, 2) for i in range(35)]          # 0.30 ... 2.00
T3D = [0.1, 0.3, 0.5] + [round(0.75 + 0.025*i, 3) for i in range(11)]
GUARD_BOUT = 2e-2
FAMS = {"10_01": ((1, 0), (0, 1)), "10_11": ((1, 0), (1, 1))}


def seed2d(N, th, fam, psi):
    (a1, b1), (a2, b2) = FAMS[fam]
    x = np.arange(N) / N
    L, Z = np.meshgrid(x, x, indexing="ij")
    Ax = np.cos(psi)*np.cos(2*np.pi*(a1*L + b1*Z)) + np.sin(psi)*np.cos(2*np.pi*(a2*L + b2*Z))
    k = 2*np.pi*np.fft.fftfreq(N, 1.0/N)
    KL, KZ = np.meshgrid(k, k, indexing="ij")
    Ah = np.fft.fft2(Ax)
    Axl = np.real(np.fft.ifft2(1j*KL*Ah)); Axz = np.real(np.fft.ifft2(1j*KZ*Ah))
    s, c = np.sin(th), np.cos(th)
    dB = np.stack([-s*Axz, s*Axl, -c*Axz])
    dB *= A0 / np.sqrt(np.mean(np.sum(dB**2, axis=0)))
    return dB + np.array([c, 0.0, -s])[:, None, None]


class Grower:
    """jit once per Parameters; grow many ICs through targets (landing exactly on each)."""

    def __init__(self, p, scheme="lsrk54", extra=None):
        import jax
        import taranis
        from taranis.run import init_aux
        from taranis.timestepping import get_scheme
        from taranis.physics import equation_registry, construct_rhs
        from taranis import diagnostics as D
        self.p = p; kg = taranis.setup_kgrids(p); self.kg = kg
        rec = equation_registry["SM22"]; rhs = construct_rhs(rec)
        stepper, sch = get_scheme(scheme)
        self.step = jax.jit(lambda s, a, dto: stepper(s, kg, p, rhs, rec.set_timestep_func, sch, dto, aux=a))
        self.stepa = jax.jit(lambda s, a: stepper(s, kg, p, rhs, rec.set_timestep_func, sch, aux=a))
        self.amp = jax.jit(lambda s: D.sm22.amplitude(s, p))
        self.diag = jax.jit(lambda s: D.sm22.diagnostics(s, kg, p))
        self.its = jax.jit(lambda s, a: rec.grad_func(s, kg, p, a).its)
        self.init_aux = lambda s: init_aux(s, kg, p)
        self.extra = extra   # optional jitted state -> dict of extra (e.g. localization) outputs
        self.jax = jax

    def grow(self, st, targets, max_steps=20000, wall_cap=None):
        aux = self.init_aux(st)
        T0 = time.perf_counter(); tdiag = 0.0
        A = float(self.amp(st)); ti = 0; rows = []; n = 0; stop = "done"
        while ti < len(targets):
            if n >= max_steps:
                stop = "max_steps"; break
            s2, a2 = self.stepa(st, aux); n += 1
            A2 = float(self.amp(s2))
            if not np.isfinite(A2):
                stop = "nonfinite"; break
            while ti < len(targets) and A2 >= targets[ti]:
                t1 = time.perf_counter()
                dt = float(s2.t - st.t)
                sl, al = self.step(st, aux, dt*(targets[ti] - A)/(A2 - A))
                d = {k: float(v) for k, v in self.diag(sl).items()}
                d.update(target=targets[ti], its=int(self.its(sl, al)), step=n, t=float(sl.t))
                if self.extra is not None:
                    d.update(self.extra(sl))
                rows.append(d); ti += 1
                tdiag += time.perf_counter() - t1
                if d["Berr_out"] > GUARD_BOUT:
                    stop = "guard_Bout"; break
            if stop != "done":
                break
            st, aux, A = s2, a2, A2
            if wall_cap and time.perf_counter() - T0 > wall_cap:
                stop = "wall_cap"; break
        return dict(rows=rows, stop=stop, steps=n, wall=time.perf_counter() - T0 - tdiag, Alast=A)


def warm(g, st):
    # compile everything (step, landing step, diag, its) outside the timed loops
    aux = g.init_aux(st); j = g.jax
    j.block_until_ready(g.stepa(st, aux)); j.block_until_ready(g.step(st, aux, 1e-3))
    j.block_until_ready(g.diag(st)); j.block_until_ready(g.its(st, aux))
    if g.extra is not None:
        g.extra(st)


def fam2d(fam, th_deg, N, out, npsi=33):
    # sm22_grow2d is not imported here, but keep SM22_FP32 consistent anyway (see sm22_taranis_bench)
    os.environ["SM22_FP32"] = "0" if os.environ.get("TARANIS_PRECISION", "32") == "64" else "1"
    import jax.numpy as jnp
    import taranis
    th = np.deg2rad(th_deg)
    p = taranis.Parameters(N, N, 1.0, 1.0, 0.5, eqtype="SM22", dims=2, eqpars=dict(maxiter=200))
    g = Grower(p)
    mk = lambda psi: taranis.initialize(lambda x, y: jnp.asarray(seed2d(N, th, fam, psi))[:, None], p)
    warm(g, mk(0.3))
    tag = f"fam2d_{fam}_th{th_deg:g}_N{N}_fp{os.environ.get('TARANIS_PRECISION', '32')}"
    res = []
    for i, psi in enumerate(np.linspace(0, np.pi/2, npsi)):
        r = g.grow(mk(psi), T2D, max_steps=6000, wall_cap=240)
        r.update(psi=float(psi), i=i)
        res.append(r)
        last = r["rows"][-1] if r["rows"] else {}
        print(f"[{tag}] psi={psi:.4f} stop={r['stop']} steps={r['steps']} wall={r['wall']:.1f}s "
              f"Amax={last.get('A', float('nan')):.3f} mg={last.get('maxgrad', float('nan')):.2f} "
              f"Bout={last.get('Berr_out', float('nan')):.2e}", flush=True)
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, tag + ".json"), "w") as f:   # rewrite each IC: partial results survive
            json.dump(dict(fam=fam, th_deg=th_deg, N=N, A0=A0, targets=T2D, runs=res), f)


def f3_ic():
    # identical to sm22_taranis_bench.tar3d (rounds 1-2), so this ladder extends that one
    import jax.numpy as jnp
    rng = np.random.default_rng(0)
    bh = np.array([np.cos(0.5)*np.cos(0.3), np.cos(0.5)*np.sin(0.3), np.sin(0.5)])
    modes = [((a, b, c), rng.normal(), rng.uniform(0, 2*np.pi)) for (a, b, c) in
             [(1,0,0),(0,1,0),(0,0,1),(1,1,0),(1,0,1),(0,1,1),(1,-1,0),(1,0,-1),(0,1,-1)]]

    def f3(x, y, z):
        X, Y, Z = [2*np.pi*a for a in (x, y, z)]
        G = 0.0
        for (a, b, c), amp, ph in modes:
            G = G + amp*jnp.stack([k*(-jnp.sin(a*X + b*Y + c*Z + ph)) for k in (a, b, c)])
        dB = jnp.stack([bh[1]*G[2] - bh[2]*G[1], bh[2]*G[0] - bh[0]*G[2], bh[0]*G[1] - bh[1]*G[0]])
        dB = dB * (0.05 / jnp.sqrt(jnp.mean(jnp.sum(dB**2, 0))))
        return dB + jnp.asarray(bh)[:, None, None, None]
    return f3


def dense3d(N, out):
    import jax
    import jax.numpy as jnp
    import taranis
    from taranis import grids
    from taranis.physics import sm22 as P
    p = taranis.Parameters(N, N, 1.0, 1.0, 0.5, eqtype="SM22", dims=3, nz=N, Lz=1.0, z_spectral=True,
                           eqpars=dict(maxiter=300))
    kg = taranis.setup_kgrids(p)

    @jax.jit
    def loc(s):
        # localization of the gradient peak and of the out-of-band |B|^2 peak: argmax positions,
        # volume fractions above (max/2), and a 2D slice of |grad B|/|B| through the maxgrad point
        fk = s.fields
        B = jnp.stack([grids.ifft(fk[i], p) for i in range(3)])
        Bsq = jnp.sum(B*B, 0)
        kx, ky, kz = P._kvec(kg, p)
        G2 = sum(grids.ifft(1j*k*fk[i], p)**2 for i in range(3) for k in (kx, ky, kz)) / Bsq
        out_ = grids.ifft(jnp.where(kg.dealias, 0.0, 1.0)*grids.fft(Bsq, p), p) / jnp.mean(Bsq)
        ig = jnp.stack(jnp.unravel_index(jnp.argmax(G2), G2.shape))
        io = jnp.stack(jnp.unravel_index(jnp.argmax(jnp.abs(out_)), out_.shape))
        m = G2.max(); mo = jnp.abs(out_).max()
        sl = G2[ig[0]]
        return dict(ig=ig, io=io, fg=jnp.mean(G2 > m/4), fo=jnp.mean(jnp.abs(out_) > mo/2),
                    slice=jnp.sqrt(sl))

    def extra(s):
        d = loc(s)
        r = dict(igrad=[int(v) for v in d["ig"]], iout=[int(v) for v in d["io"]],
                 fgrad_half=float(d["fg"]), fout_half=float(d["fo"]))
        extra.slices.append(np.asarray(d["slice"], np.float32))
        return r
    extra.slices = []
    g = Grower(p, extra=extra)
    st = taranis.initialize(f3_ic(), p)
    warm(g, st); extra.slices.clear()
    r = g.grow(st, T3D, wall_cap=3000)
    tag = f"dense3d_N{N}_fp{os.environ.get('TARANIS_PRECISION', '32')}"
    os.makedirs(out, exist_ok=True)
    np.savez_compressed(os.path.join(out, tag + "_slices.npz"), slices=np.stack(extra.slices),
                        targets=np.array([row["target"] for row in r["rows"]]))
    with open(os.path.join(out, tag + ".json"), "w") as f:
        json.dump(dict(N=N, targets=T3D, **r), f)
    for row in r["rows"]:
        print(f"[{tag}] A={row['target']} mg={row['maxgrad']:.2f} Bout={row['Berr_out']:.2e} "
              f"Boutmax={row['Berr_out_max']:.2e} edge={row['edge_max']:.3f} fg={row['fgrad_half']:.2e}",
              flush=True)
    print(f"[{tag}] stop={r['stop']} steps={r['steps']} wall={r['wall']:.1f}s", flush=True)


if __name__ == "__main__":
    m = sys.argv[1]
    if m == "fam2d":
        fam2d(sys.argv[2], float(sys.argv[3]), int(sys.argv[4]), sys.argv[5], *(int(a) for a in sys.argv[6:7]))
    elif m == "dense3d":
        dense3d(int(sys.argv[2]), sys.argv[3])
