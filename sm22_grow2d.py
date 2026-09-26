"""SM22 growth rule (Squire & Mallet 2022, eqs 2.1-2.2) in 2.5D, pseudospectral JAX.

Exploratory instrument for the discontinuity-formation question
(../discontinuity_lower_bound_scratch.md, prediction P1): does steepening
sit where the ignorable component w = B.e_ign reverses ALONG in-plane field
lines (Z_bad = {w=0, |d_t w| > |d_n w|})?

Geometry as SM22 §4: variation in (l, z) on [0,1)^2, mean field B0 x-hat at
angle theta2d to l-hat. Components stored in the (l-hat, z-hat, e_ign) frame,
e_ign = (-sin th, cos th, 0), so Bbar = B0 (cos th, 0, -sin th).

Growth:  dB/dt = dB + curl(grad(phi) x B),  phi = least-squares solution of
         B.curl(grad(phi) x B) = -B.dB + const
(exactly SM22's B^2 lap_perp phi = -dB.B when |B|^2 is uniform), solved by
preconditioned CG on the normal equations (L^T L via jax.vjp).
Hou-Li exponential filter on the state; no other dissipation.

Usage: python sm22_grow2d.py --N 128 --seed 1 --Amax 5 --out DIR
"""
import argparse, os, time
import jax
# SM22_FP32=1 selects float32 (Kaggle T4); default float64 per project policy
FP32 = os.environ.get("SM22_FP32", "0") == "1"
jax.config.update("jax_enable_x64", not FP32)
import jax.numpy as jnp
import numpy as np


def make_ops(N):
    k = 2 * np.pi * np.fft.fftfreq(N, 1.0 / N)
    KL, KZ = np.meshgrid(k, k, indexing="ij")
    K2 = KL**2 + KZ**2
    kmax = np.abs(k).max()
    filt = np.exp(-36.0 * ((np.abs(KL) / kmax) ** 36 + (np.abs(KZ) / kmax) ** 36))
    return jnp.asarray(KL), jnp.asarray(KZ), jnp.asarray(K2), jnp.asarray(filt)


def build(N, th, B0, gamma=0.0, cg_iters=300, cg_tol=1e-11):
    KL, KZ, K2, filt = make_ops(N)
    Bbar = jnp.array([B0 * np.cos(th), 0.0, -B0 * np.sin(th)])

    def dl(f):
        return jnp.real(jnp.fft.ifft2(1j * KL * jnp.fft.fft2(f)))

    def dz(f):
        return jnp.real(jnp.fft.ifft2(1j * KZ * jnp.fft.fft2(f)))

    def curl_uxB(phi, B):
        # u = (phi_l, phi_z, 0);  E = u x B;  curl E with d/d(ign) = 0
        ul, uz = dl(phi), dz(phi)
        E1 = uz * B[2]
        E2 = -ul * B[2]
        E3 = ul * B[1] - uz * B[0]
        return jnp.stack([dz(E3), -dl(E3), dl(E2) - dz(E1)])

    def Lop(phi, B):
        return jnp.sum(B * curl_uxB(phi, B), axis=0)

    def demean(f):
        return f - jnp.mean(f)

    def solve_phi(B, phi0):
        dB = B - Bbar[:, None, None]
        Bsq = jnp.sum(B * B, axis=0)
        rhs = demean(-jnp.sum(B * dB, axis=0) - 0.5 * gamma * (Bsq - jnp.mean(Bsq)))
        A = lambda p: demean(Lop(p, B))
        _, vjp = jax.vjp(A, phi0)
        AtA = lambda p: vjp(A(p))[0]
        b = vjp(rhs)[0]
        b4 = jnp.mean(Bsq) ** 2
        Pk = jnp.where(K2 > 0, 1.0 / (b4 * K2**2 + 1e-30), 0.0)
        prec = lambda r: jnp.real(jnp.fft.ifft2(Pk * jnp.fft.fft2(r)))

        def body(c):
            x, r, z, p, rz, i = c
            Ap = AtA(p)
            a = rz / jnp.sum(p * Ap)
            x = x + a * p
            r = r - a * Ap
            z = prec(r)
            rzn = jnp.sum(r * z)
            p = z + (rzn / rz) * p
            return x, r, z, p, rzn, i + 1

        bn = jnp.sum(b * b)
        def cond(c):
            return (c[5] < cg_iters) & (jnp.sum(c[1] * c[1]) > cg_tol**2 * bn)

        r0 = b - AtA(phi0)
        z0 = prec(r0)
        x, r, _, _, _, it = jax.lax.while_loop(
            cond, body, (phi0, r0, z0, z0, jnp.sum(r0 * z0), 0))
        res = jnp.sqrt(jnp.mean((A(x) - rhs) ** 2) / (jnp.mean(rhs**2) + 1e-300))
        return x, it, res

    def rhs_fn(B, phi0):
        phi, it, res = solve_phi(B, phi0)
        dB = B - Bbar[:, None, None]
        return dB + curl_uxB(phi, B), phi, it, res

    def filt_state(B):
        return jnp.real(jnp.fft.ifft2(filt * jnp.fft.fft2(B)))

    @jax.jit
    def rk4(B, phi, dt):
        k1, phi, i1, r1 = rhs_fn(B, phi)
        k2, phi, i2, r2 = rhs_fn(B + 0.5 * dt * k1, phi)
        k3, phi, i3, r3 = rhs_fn(B + 0.5 * dt * k2, phi)
        k4, phi, i4, r4 = rhs_fn(B + dt * k3, phi)
        Bn = filt_state(B + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4))
        # keep the mean exactly
        Bn = Bn - jnp.mean(Bn, axis=(1, 2))[:, None, None] + Bbar[:, None, None]
        umax = jnp.max(jnp.sqrt(dl(phi) ** 2 + dz(phi) ** 2))
        return Bn, phi, jnp.maximum(jnp.maximum(i1, i2), jnp.maximum(i3, i4)), \
            jnp.maximum(jnp.maximum(r1, r2), jnp.maximum(r3, r4)), umax

    return dict(rk4=rk4, dl=jax.jit(dl), dz=jax.jit(dz), Bbar=Bbar, solve_phi=jax.jit(solve_phi))


def seed_state(N, th, B0, seed, A0, kcut=1):
    """Linear Alfvenic seed dB = curl(A_x x-hat), modes |k_l|,|k_z| <= kcut (units 2pi),
    then SM22-style: we do NOT pre-project; start at small amplitude where the |B|^2 error
    is O(A0^2) and let the growth's least-squares step keep it from growing."""
    rng = np.random.default_rng(seed)
    x = np.arange(N) / N
    L, Z = np.meshgrid(x, x, indexing="ij")
    Ax = np.zeros((N, N))
    for ml in range(-kcut, kcut + 1):
        for mz in range(-kcut, kcut + 1):
            if ml == 0 and mz == 0:
                continue
            a, p = rng.normal(), rng.uniform(0, 2 * np.pi)
            Ax += a * np.cos(2 * np.pi * (ml * L + mz * Z) + p)
    k = 2 * np.pi * np.fft.fftfreq(N, 1.0 / N)
    KL, KZ = np.meshgrid(k, k, indexing="ij")
    Ah = np.fft.fft2(Ax)
    Axl = np.real(np.fft.ifft2(1j * KL * Ah))
    Axz = np.real(np.fft.ifft2(1j * KZ * Ah))
    s, c = np.sin(th), np.cos(th)
    dB = np.stack([-s * Axz, s * Axl, -c * Axz])
    amp = np.sqrt(np.mean(np.sum(dB**2, axis=0))) / B0
    dB *= A0 / amp
    Bbar = np.array([B0 * c, 0.0, -B0 * s])
    return dB + Bbar[:, None, None]


def diagnostics(B, ops, B0):
    dl, dz = ops["dl"], ops["dz"]
    Bn = np.asarray(B)
    G = np.stack([np.stack([np.asarray(dl(Bn[i])), np.asarray(dz(Bn[i]))]) for i in range(3)])
    Bsq = np.sum(Bn**2, axis=0)
    Bmag = np.sqrt(Bsq)
    gradF = np.sqrt(np.sum(G**2, axis=(0, 1))) / Bmag
    dB = Bn - np.asarray(ops["Bbar"])[:, None, None]
    A = np.sqrt(np.mean(np.sum(dB**2, axis=0))) / B0
    # SM22 fig-5 metric
    sm = np.sum([np.abs(G[i, j]).max() / np.abs(Bn[i]).max() for i in range(3) for j in range(2)]) / 3
    # ignorable-component geometry
    w = Bn[2] / Bmag
    m = Bn[:2]
    mm = np.sqrt(np.sum(m**2, axis=0)) + 1e-300
    t = m / mm
    gw = G[2] / Bmag  # approx grad of w (Bmag ~ uniform)
    dtw = t[0] * gw[0] + t[1] * gw[1]
    dnw = -t[1] * gw[0] + t[0] * gw[1]
    Zmask = np.abs(w) < 0.03
    zfrac = Zmask.mean()
    zbad = (np.abs(dtw) > np.abs(dnw)) & Zmask
    zbad_frac = zbad.sum() / max(Zmask.sum(), 1)
    # where is the steepening? top-0.5% gradient voxels: their |w| and bad-cosine
    thr = np.quantile(gradF, 0.995)
    top = gradF >= thr
    cos_bad = np.abs(dtw) / (np.sqrt(dtw**2 + dnw**2) + 1e-300)
    # twist sigma = b.curl b (2.5D): curl B = (dz B3, -dl B3, dl B2 - dz B1)
    curl = np.stack([G[2, 1], -G[2, 0], G[1, 0] - G[0, 1]])
    sigma = np.sum(Bn * curl, axis=0) / Bsq
    return dict(A=A, Berr=np.sqrt(np.mean((Bsq - Bsq.mean())**2)) / Bsq.mean(),
                maxgrad=gradF.max(), sm22=sm, wmin=w.min(), wmax=w.max(), zfrac=zfrac,
                zbad_frac=zbad_frac, top_absw=np.median(np.abs(w[top])),
                all_absw=np.median(np.abs(w)), top_cosbad=np.median(cos_bad[top]),
                top_absig=np.median(np.abs(sigma[top])), all_absig=np.median(np.abs(sigma)))


def run_one(a, ops, seed):
    th = np.deg2rad(a.th); B0 = 1.0
    B = jnp.asarray(seed_state(a.N, th, B0, seed, a.A0, a.kcut))
    phi = jnp.zeros((a.N, a.N))
    dx = 1.0 / a.N
    t, nextA, rows, T0 = 0.0, a.A0, [], time.time()
    tag = f"N{a.N}_s{seed}_th{int(a.th)}_k{a.kcut}" + ("_fp32" if FP32 else "")
    snaps = {}
    umax = 0.0
    Bbar = ops["Bbar"]

    @jax.jit
    def cheap(B):  # amplitude and |B|^2 non-uniformity, on device
        Bsq = jnp.sum(B * B, axis=0)
        A = jnp.sqrt(jnp.mean(jnp.sum((B - Bbar[:, None, None])**2, axis=0))) / B0
        return A, jnp.sqrt(jnp.mean((Bsq - Bsq.mean())**2)) / Bsq.mean()

    while True:
        A, Berr = map(float, cheap(B))
        if A >= nextA or A >= a.Amax:
            d = diagnostics(B, ops, B0)
            d.update(t=t, umax=float(umax))
            rows.append(d)
            print(f"[{tag}] A={d['A']:.3f} Berr={d['Berr']:.2e} maxgrad={d['maxgrad']:.2f} "
                  f"wrange=({d['wmin']:.2f},{d['wmax']:.2f}) Z={d['zfrac']:.3f} "
                  f"Zbad={d['zbad_frac']:.2f} top|w|={d['top_absw']:.3f}/{d['all_absw']:.3f} "
                  f"topcos={d['top_cosbad']:.2f} top|sig|={d['top_absig']:.2f}/{d['all_absig']:.2f} "
                  f"({time.time()-T0:.0f}s)", flush=True)
            snaps[f"A{d['A']:.3f}"] = np.asarray(B)
            nextA *= 1.1
        if A >= a.Amax or not np.isfinite(A) or Berr > 0.2:
            break
        dt = min(a.dtmax, 0.3 * dx / (float(umax) + 1e-12))
        B, phi, its, res, umax = ops["rk4"](B, phi, dt)
        t += dt
    np.savez(os.path.join(a.out, f"series_{tag}.npz"),
             **{k: np.array([r[k] for r in rows]) for k in rows[0]})
    np.savez_compressed(os.path.join(a.out, f"snaps_{tag}.npz"), **snaps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=128)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--seeds", default=None, help="range 'a-b' or list 'a,b,c'; overrides --seed")
    ap.add_argument("--th", type=float, default=30.0)
    ap.add_argument("--A0", type=float, default=0.05)
    ap.add_argument("--Amax", type=float, default=5.0)
    ap.add_argument("--kcut", type=int, default=1)
    ap.add_argument("--gamma", type=float, default=0.0)
    ap.add_argument("--dtmax", type=float, default=0.01)
    ap.add_argument("--out", default="sm22_out")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ops = build(a.N, np.deg2rad(a.th), 1.0, gamma=a.gamma, cg_tol=1e-5 if FP32 else 1e-11)
    if a.seeds is None:
        seeds = [a.seed]
    elif "-" in a.seeds:
        lo, hi = map(int, a.seeds.split("-")); seeds = range(lo, hi + 1)
    else:
        seeds = [int(x) for x in a.seeds.split(",")]
    for s in seeds:
        tag = f"N{a.N}_s{s}_th{int(a.th)}_k{a.kcut}" + ("_fp32" if FP32 else "")
        if os.path.exists(os.path.join(a.out, f"series_{tag}.npz")):
            print(f"[{tag}] exists, skipping", flush=True); continue
        run_one(a, ops, s)


if __name__ == "__main__":
    main()
