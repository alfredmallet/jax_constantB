"""Summarize a seed scan: per seed A_Z, Lambda99 peak (and A at peak), maxgrad at A=1.
Usage: python sm22_scan_summary.py DIR [N]   (writes DIR/summary_N{N}.csv)"""
import sys, glob, re, os, numpy as np
from sm22_zray import analyse
d = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 64
rows = []
for f in sorted(glob.glob(f"{d}/snaps_N{N}_s*_th30_k1*.npz"), key=lambda p: int(re.findall(r"_s(\d+)_", p)[0])):
    s = int(re.findall(r"_s(\d+)_", f)[0])
    z = np.load(f); ser = np.load(f.replace("snaps_", "series_"))
    AZ = ser["A"][np.argmax(ser["zfrac"] > 0)] if (ser["zfrac"] > 0).any() else np.nan
    best, Abest, lam_series = 0.0, np.nan, []
    for key in sorted(z.files, key=lambda k: float(k[1:])):
        A = float(key[1:])
        if A < 0.2 or A > 0.9:  # N=64 is grid-limited on sharpening seeds beyond ~0.9
            continue
        r = analyse(z[key])
        L = r["lam_99"] if r else 0.0
        lam_series.append((A, L))
        if L > best:
            best, Abest = L, A
    cross = next((A for A, L in lam_series if L > 0.65), np.nan)
    g1 = np.interp(1.0, ser["A"], ser["maxgrad"]) if ser["A"].max() >= 1.0 else np.nan
    rows.append((s, AZ, best, Abest, cross, g1))
    print(f"seed {s:3d}: A_Z={AZ:.3f} Lam99_peak={best:.2f} @A={Abest:.2f} A(Lam>.65)={cross:.2f} maxgrad(A=1)={g1:.1f}", flush=True)
np.savetxt(os.path.join(d, f"summary_N{N}.csv"), np.array(rows), delimiter=",",
           header="seed,A_Z,lam99_peak,A_peak,A_cross065,maxgrad_A1", comments="")
