#!/usr/bin/env python3
"""
blob_quest.py -- grow a LOCALISED, divergence-free "blob" of rotating field on a
uniform mean field with the alias-free (Galerkin, 2/3-rule) minimum-norm
Gauss-Newton continuation, and record exactly what the constraint manifold does
with it.  Companion experiment to the Alfvenon of Huang et al. (2026), who grow
solitary structures by alternating projections but must relax exact
|B|-constancy; and to the collocation-mode blob experiment of the paper
(Sec. 7.5), which found that localisation is rejected and field-aligned tubes
form.  This script repeats that experiment with the honest instrument, so the
conclusion cannot be blamed on aliasing, and saves snapshots for plotting.
Thin driver over the `constantB` package (blob_seed, fwhm/blob_diagnostics).

    python3 blob_quest.py --grid 32 32 64 --eps-max 1.2       # continuation
    python3 blob_quest.py --plot                              # figure from snapshots

SEED.  b_seed = curl(psi1 e_x + psi2 e_y), built spectrally (exactly
div-free, then 2/3-truncated), with
    psi1 = chi(x) cos(kz z),  psi2 = chi(x) sin(kz z),
    chi  = exp[ (cos(x-x0)-1)/wx^2 + (cos(y-y0)-1)/wy^2 + (cos(z-z0)-1)/wz^2 ],
a periodic Gaussian-like bump: a compact packet of transversally rotating
field, localised in ALL THREE directions (the Alfvenon geometry), normalised
to max|b_seed| = 1 so that eps is the injected amplitude.

DIAGNOSTICS per accepted step (CSV):
  maxgrad, maxdefl, vol_rev, gal_tail_rms  -- as in amplitude_quest;
  Lpar, Lperp    -- FWHM of the parallel / transverse energy profiles
                    P(z) = <|b|^2>_{xy},  Q(x) = <|b|^2>_{yz}  (periodic-aware);
  flat_par       -- min P / max P: 1 = perfectly flat tube, 0 = solitary blob;
  loc_frac       -- fraction of |b|^2 inside the seed's half-max ellipsoid:
                    stays ~1 if the state remains blob-shaped, falls as the
                    injected energy is redistributed.
Snapshots blob_epsXX.npz are written every --snap-de of eps for --plot.
"""
import argparse, csv, os, sys, time
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from constantB import TWOPI, Solver, save_state, load_state
from constantB.seeds import blob_seed
from constantB.diagnostics import fwhm, blob_diagnostics


# ---------------------------------------------------------------------------
# continuation
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--state', default='blobstate.npz')
    p.add_argument('--csv', default='blob_quest.csv')
    p.add_argument('--grid', type=int, nargs=3, default=[32, 32, 64])
    p.add_argument('--w', type=float, nargs=3, default=[0.8, 0.8, 0.8])
    p.add_argument('--kz', type=int, default=1)
    p.add_argument('--eps-max', type=float, default=1.2)
    p.add_argument('--de', type=float, default=0.03)
    p.add_argument('--de-min', type=float, default=1e-4)
    p.add_argument('--sweeps', type=int, default=10)
    p.add_argument('--cgit', type=int, default=400)
    p.add_argument('--res-ok', type=float, default=1e-7)
    p.add_argument('--snap-de', type=float, default=0.2)
    p.add_argument('--max-seconds', type=float, default=1e9,
                   help='stop cleanly after this wall time (resumable)')
    p.add_argument('--plot', action='store_true')
    p.add_argument('--fig-out', default='fig_blobgrow_galerkin.png')
    args = p.parse_args()

    if args.plot:
        return make_plot(args)

    grid = tuple(args.grid)
    S = Solver(grid, dealias=True)
    seed, chi = blob_seed(S, tuple(args.w), args.kz)
    chi_half = chi >= 0.5*chi.max()          # the seed's half-max ellipsoid

    if os.path.exists(args.state):
        B, eps, _ = load_state(args.state)
        print(f"resuming: eps={eps:.3f}")
    else:
        B = np.zeros((3,)+grid); B[2] = 1.0  # uniform carrier z-hat
        eps = 0.0

    t0, de, snap_next = time.time(), args.de, (int(eps/args.snap_de)+1)*args.snap_de
    while eps < args.eps_max and time.time()-t0 < args.max_seconds:
        Bprev, epsprev = np.array(B), eps
        B = np.asarray(B) + de*seed; eps += de
        B, res, ci = S.gn(B, sweeps=args.sweeps, cgit=args.cgit,
                          tol=args.res_ok*1e-2, pcg=True)
        if res > args.res_ok:
            B, eps = Bprev, epsprev
            de /= 2
            print(f"  [reject res {res:.1e}; de -> {de:.4f}]")
            if de < args.de_min:
                print("STOP: de underflow (fold? -- inspect)"); break
            continue
        d = blob_diagnostics(S, B, chi_half)
        d.update(eps=eps, de=de, res=res, cg=ci)
        new = not os.path.exists(args.csv)
        with open(args.csv, 'a', newline='') as f:
            wtr = csv.DictWriter(f, fieldnames=list(d))
            if new: wtr.writeheader()
            wtr.writerow(d)
        print(f"eps={eps:.3f} res={res:.1e} maxdefl={d['maxdefl']:.1f} "
              f"Lpar={d['Lpar']:.2f} Lperp={d['Lperp']:.2f} "
              f"flat={d['flat_par']:.2f} loc={d['loc_frac']:.2f} "
              f"gtail={d['gal_tail_rms']:.1e}")
        save_state(args.state, np.asarray(B), eps, dict(w=np.array(args.w), kz=args.kz))
        if eps >= snap_next - 1e-9:
            save_state(f"blob_eps{eps:.2f}.npz", np.asarray(B), eps,
                       dict(w=np.array(args.w), kz=args.kz))
            snap_next += args.snap_de


# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------
def make_plot(args):
    import glob, matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    snaps = sorted(glob.glob("blob_eps*.npz"))
    rows = list(csv.DictReader(open(args.csv)))
    eps = np.array([float(r['eps']) for r in rows])

    nshow = min(4, len(snaps))
    pick = [snaps[int(i*(len(snaps)-1)/max(nshow-1, 1))] for i in range(nshow)]
    fig, ax = plt.subplots(3, nshow, figsize=(3.0*nshow, 8.2),
                           gridspec_kw=dict(height_ratios=[1, 1, 0.85]))
    ax = np.atleast_2d(ax)
    for j, fn in enumerate(pick):
        B, e, _ = load_state(fn)
        b = B - B.mean(axis=(1, 2, 3), keepdims=True)
        n = B.shape[1:]; iy = n[1]//2
        bperp = np.sqrt(b[0]**2 + b[1]**2)
        im0 = ax[0, j].imshow(bperp[:, iy, :].T, origin='lower', aspect='auto',
                              extent=[0, TWOPI, 0, TWOPI*n[2]/n[0]], cmap='magma')
        ax[0, j].set_title(f"$|b_\\perp|$,  $\\varepsilon={e:.2f}$", fontsize=9)
        plt.colorbar(im0, ax=ax[0, j], shrink=0.85)
        im1 = ax[1, j].imshow(B[2][:, iy, :].T, origin='lower', aspect='auto',
                              extent=[0, TWOPI, 0, TWOPI*n[2]/n[0]], cmap='RdBu_r')
        ax[1, j].set_title("$B_z$", fontsize=9)
        plt.colorbar(im1, ax=ax[1, j], shrink=0.85)
        for a in (ax[0, j], ax[1, j]):
            a.set_xlabel('$x$'); a.set_ylabel('$z$' if j == 0 else '')
        e2 = (b**2).sum(0)
        z = np.linspace(0, TWOPI, n[2], endpoint=False)
        ax[2, j].plot(z, e2.mean(axis=(0, 1))/max(e2.mean(axis=(0, 1)).max(), 1e-30))
        ax[2, j].set_ylim(0, 1.05); ax[2, j].set_xlabel('$z$')
        ax[2, j].set_title('parallel envelope $P(z)$', fontsize=9)
        ax[2, j].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.fig_out.replace('.png', '_cuts.png'), dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
    ax[0].plot(eps, [float(r['Lpar']) for r in rows], label=r'$L_\parallel$')
    ax[0].plot(eps, [float(r['Lperp']) for r in rows], label=r'$L_\perp$')
    ax[0].axhline(TWOPI, color='gray', ls=':', lw=1)
    ax[0].text(eps[0], TWOPI*1.01, 'box', fontsize=8, color='gray')
    ax[0].set_xlabel(r'$\varepsilon$'); ax[0].set_ylabel('FWHM')
    ax[0].legend(fontsize=8); ax[0].set_title('extent of $|b|^2$', fontsize=9)
    ax[1].plot(eps, [float(r['flat_par']) for r in rows], label='flat$_\\parallel$')
    ax[1].plot(eps, [float(r['loc_frac']) for r in rows], label='loc. frac.')
    ax[1].set_xlabel(r'$\varepsilon$'); ax[1].set_ylim(0, 1.05)
    ax[1].legend(fontsize=8)
    ax[1].set_title('tube-ness vs blob-ness', fontsize=9)
    ax[2].semilogy(eps, [float(r['gal_tail_rms']) for r in rows])
    ax[2].set_xlabel(r'$\varepsilon$'); ax[2].set_title('gal_tail_rms', fontsize=9)
    for a in ax: a.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.fig_out, dpi=200)
    print(f"wrote {args.fig_out} and {args.fig_out.replace('.png', '_cuts.png')}")


if __name__ == '__main__':
    main()
