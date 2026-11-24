"""Cold start from large-amplitude divergence-free noise, then an ascending resolution ladder at fixed state.

Paper 1, Sec. 5.  Initial condition: uniform mean 0.8 z plus the curl of a white-noise potential over 0 < |k| <= kmax,
scaled to a given rms.  It is projected to divergence-free once and handed to the mu-form Gauss-Newton solver with no
continuation.  The converged state is then zero-padded to each finer N^3 grid and re-converged (ascending only).
Each rung is saved, with max|grad B|, the out-of-band part of q (rms and max) and max||B|-1|.

    python cold_ladder.py --out ../data/coldstart_paper            # 64 -> 96 -> 128 -> 192 -> 256, s = 0
    python cold_ladder.py --out ../data/coldstart_scan --scan --rungs 64 96 128 --rms 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.7 --key 7 11 23 42
"""
import argparse, json, os, time
import numpy as np
from constantB.solver_mu import MuSolver
from constantB import zero_pad


def noise(shape, kmax, key, rms):
    rng = np.random.default_rng(key)
    K = np.meshgrid(*[np.fft.fftfreq(n)*n for n in shape], indexing="ij")
    ah = rng.standard_normal((3,) + shape) + 1j*rng.standard_normal((3,) + shape)
    km = np.sqrt(sum(k*k for k in K)); ah *= (km <= kmax) & (km > 0.5)
    b = np.stack([np.real(np.fft.ifftn(1j*(K[(i+1) % 3]*ah[(i+2) % 3] - K[(i+2) % 3]*ah[(i+1) % 3]))) for i in range(3)])
    b -= b.mean(axis=(1, 2, 3), keepdims=True)
    return b*(rms/np.sqrt((b**2).sum(0).mean()))


def measure(S, B):
    g = float(np.sqrt(sum(np.asarray(S.dif(B[i], j))**2 for i in range(3) for j in range(3)).max()))
    mB = np.sqrt((B**2).sum(0)); tr, tm = S.tail_norm(B)
    return dict(maxgrad=g, tail_rms=float(tr), tail_max=float(tm), max_dev=float(np.abs(mB - 1).max()),
                min_B=float(mB.min()), Bbar=float(np.linalg.norm(B.mean(axis=(1, 2, 3)))))


_solvers = {}
def solver(N, smooth):
    if (N, smooth) not in _solvers: _solvers[(N, smooth)] = MuSolver((N, N, N), smooth=smooth)
    return _solvers[(N, smooth)]


def ladder(rungs, rms, key, smooth=0.0, mean=0.8, kmax=8.0, sweeps=12, save=None, quiet=False):
    """Cold start on rungs[0]^3, then re-converge upward.  Returns one row of diagnostics per rung."""
    rows = []
    for n, N in enumerate(rungs):
        shape = (N, N, N); S = solver(N, smooth); t0 = time.time()
        if n == 0:
            B = np.zeros((3,) + shape); B[2] = mean; B = np.asarray(S.project(B + noise(shape, kmax, key, rms)))
            mB = np.sqrt((B**2).sum(0)); init = dict(min_B0=float(mB.min()), max_B0=float(mB.max()), rev0=float((B[2] < 0).mean()))
        else:
            B = np.asarray(S.project(np.stack([np.asarray(zero_pad(B[i], shape)) for i in range(3)])))
        hist = []
        for sw in range(sweeps):      # one sweep at a time, to record the residual history
            B, res, ci = S.gn(B, sweeps=1, cgit=1500, tol=1e-13); B = np.asarray(B); hist.append(float(res))
            if hist[-1] < 1e-13: break
        row = dict(N=N, rms=rms, key=key, res=hist[-1], sweeps=len(hist), hist=hist, minutes=(time.time() - t0)/60, **init, **measure(S, B))
        rows.append(row)
        if not quiet:
            print("N %4d: res %.1e (%d sweeps)  maxgrad %.2f  tail rms/max %.2e/%.2e  max||B|-1| %.2e  min|B| %.3f  |Bbar| %.4f  (%.1f min)"
                  % (N, row["res"], row["sweeps"], row["maxgrad"], row["tail_rms"], row["tail_max"], row["max_dev"], row["min_B"],
                     row["Bbar"], row["minutes"]), flush=True)
        if save: np.savez(save % N, B=B, **{k: v for k, v in row.items() if k != "hist"})
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True); p.add_argument("--rungs", type=int, nargs="+", default=[64, 96, 128, 192, 256])
    p.add_argument("--smooth", type=float, default=0.0); p.add_argument("--rms", type=float, nargs="+", default=[1.0])
    p.add_argument("--mean", type=float, default=0.8); p.add_argument("--kmax", type=float, default=8.0)
    p.add_argument("--key", type=int, nargs="+", default=[7]); p.add_argument("--sweeps", type=int, default=12)
    p.add_argument("--scan", action="store_true", help="loop over all rms x key; one json per key, states not saved")
    a = p.parse_args(); os.makedirs(a.out, exist_ok=True); kw = dict(smooth=a.smooth, mean=a.mean, kmax=a.kmax, sweeps=a.sweeps)
    if a.scan:
        for key in a.key:
            cases = []
            for rms in a.rms:
                rows = ladder(a.rungs, rms, key, quiet=True, **kw); cases.append(rows); r0, r1 = rows[0], rows[-1]
                print("rms %.2f key %2d: min|B0| %.3f  |Bbar| %.3f  maxgrad %.1f -> %.1f (slope %.2f)  tail max %.2f -> %.2f  min|B| %.3f  [%d sweeps, %.1f min]"
                      % (rms, key, r0["min_B0"], r1["Bbar"], r0["maxgrad"], r1["maxgrad"], np.log(r1["maxgrad"]/r0["maxgrad"])/np.log(r1["N"]/r0["N"]),
                         r0["tail_max"], r1["tail_max"], r1["min_B"], r0["sweeps"], sum(r["minutes"] for r in rows)), flush=True)
                json.dump(dict(args=vars(a), cases=cases), open(os.path.join(a.out, f"scan_s{a.smooth:g}_key{key}.json"), "w"), indent=1)
    else:
        tag = f"s{a.smooth:g}_rms{a.rms[0]:g}_key{a.key[0]}"
        rows = ladder(a.rungs, a.rms[0], a.key[0], save=os.path.join(a.out, f"cold_{tag}_N%d.npz"), **kw)
        json.dump(dict(args=vars(a), rows=rows), open(os.path.join(a.out, f"cold_{tag}_ladder.json"), "w"), indent=1)
        for r0, r1 in zip(rows, rows[1:]):
            print("slope %d->%d: %.3f" % (r0["N"], r1["N"], np.log(r1["maxgrad"]/r0["maxgrad"])/np.log(r1["N"]/r0["N"])))
