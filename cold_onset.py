"""Onset figure for the cold-start scan of `cold_ladder.py --scan` (paper 1, Sec. 5).

(a) slope d ln max|grad B| / d ln N between the first and last rung, (b) max|q - Tq| on the last rung, both against the
rms amplitude of the initial fluctuation; one marker per seed, line = median over seeds.

    python cold_onset.py --dir ../data/coldstart_scan_low ../data/coldstart_scan --out ../figures/paper1
"""
import argparse, glob, json, os
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

p = argparse.ArgumentParser(); p.add_argument("--dir", required=True, nargs="+"); p.add_argument("--out", default="."); a = p.parse_args()
ink, ink2, grid_c, blue = "#0b0b0b", "#52514e", "#e6e5e1", "#2a78d6"
plt.rcParams.update({"font.size": 9, "mathtext.fontset": "cm", "axes.labelcolor": ink, "xtick.color": ink2, "ytick.color": ink2,
                     "axes.edgecolor": ink2, "axes.linewidth": 0.6})
data = {}
for f in sorted(sum([glob.glob(os.path.join(d, "scan_*_key*.json")) for d in a.dir], [])):
    for rows in json.load(open(f))["cases"]:
        r0, r1 = rows[0], rows[-1]
        data.setdefault(r0["key"], []).append((r0["rms"], np.log(r1["maxgrad"]/r0["maxgrad"])/np.log(r1["N"]/r0["N"]), r1["tail_max"], r1["min_B"]))
keys = sorted(data); arr = np.array([sorted(data[k]) for k in keys]); rms = arr[0, :, 0]; N0, N1 = rows[0]["N"], rows[-1]["N"]
print("rms      :", rms); print("slope    :\n", np.round(arr[:, :, 1], 2)); print("tail max :\n", np.round(arr[:, :, 2], 2))
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)
for ax, col, lab, let in [(axs[0], 1, rf"$d\ln\,\max|\nabla\mathbf{{B}}|\,/\,d\ln N$  (${N0}\to{N1}$)", "(a)"), (axs[1], 2, rf"$\max|q-\mathsf{{T}}q|$  ($N={N1}$)", "(b)")]:
    ax.plot(rms, np.median(arr[:, :, col], axis=0), color=blue, lw=1.6, zorder=2)
    for k, m in zip(range(len(keys)), "os^D"): ax.plot(rms, arr[k, :, col], m, color=blue, mfc="white", mew=1.1, ms=5, zorder=3)
    ax.set_xlabel(r"$\delta B_{\mathrm{rms}}$"); ax.set_ylabel(lab); ax.set_title(let, loc="left", color=ink, fontsize=9.5)
    ax.grid(True, color=grid_c, lw=0.6)
axs[0].axhline(1, color=ink2, lw=0.8, ls="--"); axs[0].set_ylim(0, 1.08); axs[1].set_yscale("log")
fig.savefig(os.path.join(a.out, "cold_onset.pdf")); fig.savefig(os.path.join(a.out, "cold_onset.png"), dpi=170); print("saved cold_onset in", a.out)
