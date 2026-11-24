"""Figures for the cold-start experiment of cold_ladder.py (paper 1, Sec. 5).

cold_ladder.pdf : (a) max|grad B| against N, (b) out-of-band part of q (max and rms) against N.
cold_sheet.pdf  : at one rung, through the point of largest gradient: (a) deflection angle theta from the mean field
                  and (b) |grad B| in the coordinate plane that best contains the sheet normal; (c) B and |B| along the
                  sheet normal (spectrally interpolated).  The normal is the direction of fastest variation of B there.

    python cold_figures.py --dir ../data/coldstart_paper --tag s0_rms1_key7 --N 256 --out ../figures/paper1
"""
import argparse, json, os
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.ticker, matplotlib.colors
from matplotlib.colors import LinearSegmentedColormap

p = argparse.ArgumentParser()
p.add_argument("--dir", required=True); p.add_argument("--tag", default="s0_rms1_key7"); p.add_argument("--N", type=int, default=None)
p.add_argument("--out", default="."); p.add_argument("--gfloor", type=float, default=5.0, help="lower limit of the log colour scale for |grad B|"); p.add_argument("--half", type=float, default=0.6, help="half-length of the line cut")
a = p.parse_args(); os.makedirs(a.out, exist_ok=True)
rows = json.load(open(os.path.join(a.dir, f"cold_{a.tag}_ladder.json")))["rows"]
Ns = np.array([r["N"] for r in rows]); N = a.N or int(Ns.max())

ink, ink2, grid_c = "#0b0b0b", "#52514e", "#e6e5e1"; blue, red = "#2a78d6", "#e34948"
div_cmap = LinearSegmentedColormap.from_list("bwr_brand", ["#0d366b", "#2a78d6", "#f0efec", "#e34948", "#7a1f1f"])
seq_cmap = LinearSegmentedColormap.from_list("seq_brand", ["#f0efec", "#9ec1ee", "#2a78d6", "#0d366b", "#06162e"])
plt.rcParams.update({"font.size": 9, "mathtext.fontset": "cm", "axes.labelcolor": ink, "xtick.color": ink2, "ytick.color": ink2,
                     "axes.edgecolor": ink2, "axes.linewidth": 0.6})
def letter(ax, s): ax.set_title(s, loc="left", color=ink, fontsize=9.5)

# ---- ladder ----
fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)
g = np.array([r["maxgrad"] for r in rows])
ax.loglog(Ns, g, "o-", color=blue, lw=1.6, ms=5); ax.loglog(Ns, g[0]*Ns/Ns[0], color=ink2, lw=0.8, ls="--")
ax.annotate(r"$\propto N$", (Ns[-2], g[0]*Ns[-2]/Ns[0]), textcoords="offset points", xytext=(-4, 6), ha="right", color=ink2)
ax.set_ylabel(r"$\max|\nabla\mathbf{B}|$"); letter(ax, "(a)")
ax.set_yticks([30, 40, 60, 100]); ax.set_yticklabels(["30", "40", "60", "100"]); ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
bx.loglog(Ns, [r["tail_max"] for r in rows], "o-", color=red, lw=1.6, ms=5, label="maximum")
bx.loglog(Ns, [r["tail_rms"] for r in rows], "s--", color=blue, lw=1.6, ms=4.5, label="rms")
bx.set_ylabel(r"$|q-\mathsf{T}q|$"); bx.legend(frameon=False, fontsize=8); letter(bx, "(b)")
for x in (ax, bx):
    x.set_xlabel("$N$"); x.set_xticks(Ns); x.set_xticklabels([str(n) for n in Ns]); x.xaxis.set_minor_locator(matplotlib.ticker.NullLocator()); x.grid(True, color=grid_c, lw=0.6)
fig.savefig(os.path.join(a.out, "cold_ladder.pdf")); fig.savefig(os.path.join(a.out, "cold_ladder.png"), dpi=170); plt.close(fig)

# ---- the sharpest sheet at rung N ----
B = np.load(os.path.join(a.dir, f"cold_{a.tag}_N{N}.npz"))["B"]; h = 2*np.pi/N
k1 = np.fft.fftfreq(N)*N; kr = np.fft.rfftfreq(N)*N; kvec = [k1[:, None, None], k1[None, :, None], kr[None, None, :]]
Bh = np.fft.rfftn(B, axes=(1, 2, 3))
G = np.stack([[np.fft.irfftn(1j*kvec[j]*Bh[i], s=(N, N, N), axes=(0, 1, 2)) for j in range(3)] for i in range(3)])   # G[i,j] = d_j B_i
gn = np.sqrt((G**2).sum((0, 1))); i0 = np.unravel_index(gn.argmax(), gn.shape); x0 = np.array(i0)*h
nhat = np.linalg.svd(G[(slice(None), slice(None)) + i0])[2][0]          # direction of fastest variation
Bbar = B.mean(axis=(1, 2, 3)); mB = np.sqrt((B**2).sum(0))
theta = np.degrees(np.arccos(np.clip(np.tensordot(Bbar/np.linalg.norm(Bbar), B, 1)/mB, -1, 1)))
c = int(np.argmin(np.abs(nhat))); u, v = [j for j in range(3) if j != c]     # plane normal to axis c contains most of nhat
print(f"N={N}: max|grad B| = {gn.max():.2f} at x = {np.round(x0, 3)}, sheet normal = {np.round(nhat, 3)}, plane normal to axis {c}; "
      f"max theta = {theta.max():.1f} deg, reversed volume = {100*(theta > 90).mean():.1f}%")

s = np.linspace(-a.half, a.half, 481); X = x0[:, None] + nhat[:, None]*s                 # spectral interpolation along the normal
w = np.where((kr == 0) | (kr == N/2), 1.0, 2.0)
E = [np.exp(1j*np.outer(k1, X[0])), np.exp(1j*np.outer(k1, X[1])), np.exp(1j*np.outer(kr, X[2]))*w[:, None]]
cut = np.stack([np.real(np.einsum("xyp,xp,yp->p", np.tensordot(Bh[i], E[2], 1), E[0], E[1]))/N**3 for i in range(3)])

fig = plt.figure(figsize=(11, 3.4), constrained_layout=True); gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.25])
ext = [-np.pi, np.pi, -np.pi, np.pi]; lab = "xyz"; sl = [slice(None)]*3; sl[c] = i0[c]; sl = tuple(sl)
seg = np.array([nhat*s[0], nhat*s[-1]]); centre = lambda F: np.roll(F, (N//2 - i0[u], N//2 - i0[v]), axis=(0, 1))   # sharpest point at the origin
for n_, (F, cm, lim, cl) in enumerate([(theta[sl], div_cmap, (0, 180), r"$\theta$ (deg)"), (gn[sl], seq_cmap, (0, gn.max()), r"$|\nabla\mathbf{B}|$")]):
    ax = fig.add_subplot(gs[n_]); norm = matplotlib.colors.LogNorm(a.gfloor, lim[1], clip=True) if n_ == 1 else matplotlib.colors.Normalize(*lim)     # log scale for |grad B|
    im = ax.imshow(centre(F).T, origin="lower", extent=ext, cmap=cm, norm=norm, interpolation="nearest")
    ax.plot(seg[:, u], seg[:, v], color=ink if n_ == 0 else red, lw=1.3); ax.set_xlim(-np.pi, np.pi); ax.set_ylim(-np.pi, np.pi)
    ax.set_xlabel(f"${lab[u]}-{lab[u]}_0$"); ax.set_ylabel(f"${lab[v]}-{lab[v]}_0$"); letter(ax, "(%s)" % "ab"[n_])
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03); cb.set_label(cl); cb.outline.set_visible(False)
    if n_ == 0: cb.set_ticks([0, 45, 90, 135, 180])
    else: tk = [v_ for v_ in (1, 2, 5, 10, 20, 50, 100) if a.gfloor <= v_ <= lim[1]]; cb.set_ticks(tk); cb.set_ticklabels([str(v_) for v_ in tk]); cb.minorticks_off()
ax = fig.add_subplot(gs[2])
for i, col in enumerate([blue, red, "#3f8f5f"]): ax.plot(s, cut[i], color=col, lw=1.5, label=f"$B_{lab[i]}$")
ax.plot(s, np.sqrt((cut**2).sum(0)), color=ink, lw=1.0, ls="--", label=r"$|\mathbf{B}|$")
for j in np.arange(-int(a.half/h), int(a.half/h) + 1): ax.axvline(j*h, color=grid_c, lw=0.5, zorder=0)       # grid spacing
ax.set_xlim(s[0], s[-1]); ax.set_xlabel("distance along the sheet normal"); ax.legend(frameon=False, fontsize=8, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0))
letter(ax, "(c)")
fig.savefig(os.path.join(a.out, "cold_sheet.pdf")); fig.savefig(os.path.join(a.out, "cold_sheet.png"), dpi=170); plt.close(fig)
print("saved cold_ladder, cold_sheet in", a.out)
