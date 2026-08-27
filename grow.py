#!/usr/bin/env python3
"""
grow.py -- CARRIER-FREE continuation: grow a genuinely 3D constant-|B| state
out of the trivial uniform solution, with no arc-polarised carrier anywhere in
the construction.

WHY.  Every quest so far started from the 1D carrier B0(z) and deformed it, so
every conclusion carried an asterisk: is the roughness a property of the
constant-|B| manifold, or of the carrier chart we happened to hang the state
on?  Here the initial state is the uniform field B = Bbar (an exact solution
when |Bbar| = 1), the push is an arbitrary divergence-free seed built from a
vector potential, and the only structure in the problem is the seed and the
path.  Diagnostics never reference a carrier: deflection is measured from the
volume-mean field, "drift" = 1 - |Bbar| records how far the mean sinks below
unity as fluctuation energy is grown, and the k-space inertia ratios lam21 /
lam31 say whether the state is 1D (0, 0), planar (x, 0) or genuinely 3D.

Galerkin-only (strict 2/3 rule) and mu-form Gauss-Newton throughout
(constantB.solver_mu): the residual driven to zero is the exact retained-band
residual, convergence is quadratic, and tail_norm is the honest unresolved
burden.  There is deliberately NO salvage branch: with quadratic Newton, a
residual still above --res-ok after `sweeps` means the step is genuinely bad,
so the step is rejected and d_eps halved.

    python3 grow.py --seed random --key 3 --eps-max 2.0
    python3 grow.py --seed random --key 3 --pin-top 3        # 3D anchored
    python3 grow.py --plot

PINS (v3, --pin-top m).  The seed's m dominant modes are held at the ENERGY
schedule c_j(eps) = eps^2 e_j(seed) -- the amplitude the linearised push would
give them -- by the solver's bordered energy rows, phases left free.  Three
non-coplanar pins exclude 1D and planar states outright, which is what makes a
"genuinely 3D" claim about a grown state provable rather than hopeful.  The
older --freeze-top holds the COEFFICIENTS instead and is deprecated for
growth: a frozen bin cannot help cancel the residual it creates, and the
solver pays for that with a cascade (36^2x72, eps ~ 0.27: maxgrad 2.45 and
Galerkin tail 1.2e-3 frozen, against 0.80 / 1.7e-6 unpinned and 0.87 / 3.8e-6
with energy pins at the slightly larger eps = 0.290).

CSV SCHEMA (v3): a `pin_err` column, max_j |e_j/c_j - 1| (0.0 with no pins),
now trails every row.  Old CSVs therefore do not match; the fresh-state =>
fresh-CSV guard below already refuses to append to one.

PER STEP: push B + de*seed, re-converge, log a CSV row, then adapt -- refine
the grid (ascending x1.5 ladder, zero-pad, re-polish) when the Galerkin tail or
the retained-band edge content gets too big; halve de on a rejected step
(underflow => FOLD CANDIDATE); regrow de after two clean accepts.

--fixed-res (2026-08-26 scheme) grows on the starting grid for the whole
quest: the refine triggers are logged but not enforced and there is no
sharpening stop, so N_grow becomes the single member-selector parameter.
The selected member differs from the trigger-schedule one (measured:
smoother); audit every endpoint with an ascending polish ladder (rule 7),
and never splice fixed-res segments into trigger-schedule CSVs.

RESUMABLE: the state .npz carries B, eps and the full meta (seed spec, smooth,
fix_mean, freeze and pin lists -- the pin TARGETS are not stored, they are
recomputed from the rebuilt seed on whatever grid the run resumes at, because
they are sums and not means), so a resumed run needs no seed CLI flags again
-- they are ignored on resume, and the run continues on whatever grid the state
was left at.  Snapshots grow_epsX.XX.npz are written beside --state.
"""
import argparse
import csv
import glob
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from constantB import zero_pad, save_state, load_state       # enables x64 first
from constantB.spectral import numpy_wavenumbers, numpy_dif
from constantB.solver_mu import MuSolver, noncoplanar
from constantB.seeds_free import (make_seed, top_modes, blob, random_seed,
                                  from_potential)
from constantB.spectra import kspace_inertia

CSV_FIELDS = ["eps", "de", "grid", "res", "cg", "minutes", "maxgrad", "Bbar",
              "maxdefl", "vol_rev", "gal_tail_rms", "gal_tail_max", "edge",
              "fluct_rms", "drift", "lam21", "lam31", "pin_err"]

# Largest relative pin error a step may leave behind.  The bordered rows are
# re-linearised every sweep, so the pin residual converges quadratically and
# lands at ~1e-13; anything above this means the sweep budget ran out, and the
# step is rejected exactly like a missed --res-ok.
_PIN_ERR_MAX = 1e-6


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _ladder(grid, grid_max):
    """Ascending x1.5 (rounded down to even) grid ladder, capped at grid_max.

    ASCENDING is load-bearing (project CLAUDE.md): a descending
    truncate-then-resolve ladder lets the solver hop to a smoother member of
    the branch, so the ladder must start at the state's native resolution.
    The length cap guards the degenerate case of an odd --grid-max, which the
    x1.5-then-even recursion can never hit exactly.
    """
    grids, g = [tuple(grid)], list(grid)
    while tuple(g) != tuple(grid_max) and len(grids) < 16:
        g = [min(int(gi * 1.5) - int(gi * 1.5) % 2, gm)
             for gi, gm in zip(g, grid_max)]
        if tuple(g) == grids[-1]:
            break
        grids.append(tuple(g))
    return grids


def _build_seed(args, S):
    """Build the seed ONCE from the CLI; returns (b, meta_fragment).

    The fragment carries the recipe AND the normalisation constant, which is
    what makes every later rung of the ladder rebuild the same CONTINUUM
    field rather than merely the same shape of field (seeds_free.make_seed).
    """
    if args.seed == "blob":
        return blob(S, w=tuple(args.w), kz=args.kz)
    if args.seed == "random":
        return random_seed(S, kmax=args.kmax, slope=args.slope, key=args.key)
    st = np.load(args.seed_file)
    if "a" not in st.files:
        raise ValueError("--seed-file %s has no 'a' array (found: %s); the "
                         "potential must be stored under the key 'a'"
                         % (args.seed_file, st.files))
    a = np.asarray(st["a"], float)
    if a.shape[1:] != tuple(S.shape):
        a = np.stack([np.asarray(zero_pad(a[i], tuple(S.shape)))
                      for i in range(3)])
    return from_potential(a, S)


def _pin_schedule(S, seed):
    """e_j(seed) for the solver's pins, on THIS solver's grid (SUM units).

    The targets are c_j(eps) = eps^2 e_j(seed): the energy the linearised push
    B = Bbar + eps*seed puts in pin j, so the schedule is exactly "keep the
    seed's own 3D anchor growing at its natural rate" and needs no activation
    threshold -- unlike a coefficient freeze it is satisfiable from eps = 0.

    RECOMPUTED ON EVERY RUNG, never carried across grids: `pinned_energies` is
    a SUM over grid points (e = vol * <|P_j B|^2>, the solver's CG inner
    product), so the same continuum seed has targets a factor vol_new/vol_old
    apart on two rungs of the ladder.  Both sides of e_j/c_j scale with vol,
    so the LOGGED pin_err is grid independent; the targets themselves are not.
    """
    return S.pinned_energies(np.asarray(seed, float))


def _diagnostics(S, B):
    """Per-step carrier-free diagnostics (host numpy).

    Same measures as diagnostics.quest_diagnostics' Galerkin branch, but every
    carrier reference is replaced by the volume-mean field Bbar: deflection is
    from Bbar, the spectral edge monitor is taken on the fluctuation B - Bbar,
    and drift = 1 - |Bbar| replaces the carrier amplitude as the record of how
    far the state has moved off the uniform solution.
    """
    B = np.asarray(B)
    shape = B.shape[1:]
    Bbar = B.mean(axis=(1, 2, 3))
    nb = float(np.linalg.norm(Bbar))
    b = B - Bbar[:, None, None, None]
    nrm = np.sqrt((B ** 2).sum(0))
    cosM = np.clip((B * Bbar[:, None, None, None]).sum(0)
                   / np.maximum(nrm * nb, 1e-300), -1, 1)
    defl = np.degrees(np.arccos(cosM))
    K = numpy_wavenumbers(shape)
    g2 = sum(numpy_dif(B[i], j, K) ** 2 for i in range(3) for j in range(3))
    # Retained-band EDGE content: in Galerkin mode the top-of-grid modes are
    # identically zero, so the honest analogue of the old near-Nyquist monitor
    # is how hard the field presses against its own band edge |k| = N/3.
    bh = np.abs(np.fft.fftn(b, axes=(1, 2, 3))) ** 2
    edge = []
    for ax, N in ((1, shape[0]), (2, shape[1]), (3, shape[2])):
        kc = int(N / 3 - 1e-12) + 1      # top retained mode is floor-strict(N/3)
        E = bh.sum(axis=tuple(i for i in range(4) if i != ax))[:max(kc, 4)]
        edge.append(float(E[-3:].max() / max(E.max(), 1e-300)))
    grms, gmax = S.tail_norm(B)
    _lam, lam21, lam31 = kspace_inertia(B)      # 1D -> (0,0); 3D -> both > 0
    return dict(maxgrad=float(np.sqrt(g2.max())), Bbar=nb,
                maxdefl=float(defl.max()), vol_rev=float((defl > 90).mean()),
                gal_tail_rms=float(grms), gal_tail_max=float(gmax),
                edge=max(edge),
                fluct_rms=float(np.sqrt((b ** 2).sum(0).mean())),
                drift=1.0 - nb, lam21=float(lam21), lam31=float(lam31))


def _write_row(fn, row, fresh):
    with open(fn, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if fresh:
            w.writeheader()
        w.writerow(row)


def _snap_prefix(state):
    """Snapshots are named after the STATE FILE's basename, so two runs
    sharing a directory (e.g. /kaggle/working) never interleave snapshots
    and --plot never mixes them."""
    root = os.path.splitext(os.path.abspath(state))[0]
    return root + "_eps"


def _snap_name(state, eps):
    return f"{_snap_prefix(state)}{eps:.2f}.npz"


# ---------------------------------------------------------------------------
# the continuation
# ---------------------------------------------------------------------------

def run(args):
    """The whole driver, callable programmatically (tests/test_v2.py drives
    it directly).  Returns a process exit code."""
    if args.plot:
        return make_plot(args)

    if os.path.exists(args.state):
        B, eps, meta = load_state(args.state)
        meta["seed_kind"] = str(meta["seed_kind"])
        grid = tuple(B.shape[1:])
        smooth, fix_mean = float(meta["smooth"]), bool(meta["fix_mean"])
        freeze = [tuple(int(v) for v in t)
                  for t in np.asarray(meta["freeze"], int).reshape(-1, 3)]
        freeze_after = (float(meta["freeze_after"])
                        if "freeze_after" in meta else 0.0)
        # Only the pin TRIPLES are restored: e_j(seed) is recomputed below on
        # this rung's grid, because it is a sum, not a mean (_pin_schedule).
        pins = [tuple(int(v) for v in t)
                for t in np.asarray(meta["pins"], int).reshape(-1, 3)] \
            if "pins" in meta else []
        B = np.asarray(B, float)
        print(f"resuming: eps={eps:.3f} grid={grid} seed={meta['seed_kind']} "
              f"smooth={smooth} fix_mean={fix_mean} freeze={freeze} "
              f"pins={pins}")
        print("  (seed/smooth/freeze/pins come from the state; those CLI flags "
              "are ignored on resume)")
        freeze_on = eps >= freeze_after
        S = MuSolver(grid, smooth=smooth, fix_mean=fix_mean,
                     freeze=(freeze if freeze_on else []), pins=pins)
        de_stored = float(meta["de"]) if "de" in meta else 0.03
        if args.de is not None and args.de != de_stored:
            print(f"  --de {args.de:g} overrides the stored step "
                  f"{de_stored:.5f}")
        de0 = args.de if args.de is not None else de_stored
        streak0 = int(meta["streak"]) if "streak" in meta else 0
        fixed_res = bool(args.fixed_res) or \
            ("fixed_res" in meta and bool(np.asarray(meta["fixed_res"])))
    else:
        Bbar0 = np.array(args.Bbar, float)
        nb0 = float(np.linalg.norm(Bbar0))
        if nb0 > 1.0 + 1e-12:
            print(f"ERROR: |Bbar| = {nb0:.6f} > 1 is infeasible -- |B| = 1 "
                  "pointwise forces |mean B| <= 1 (Cauchy-Schwarz).")
            return 2
        if args.fix_mean and nb0 >= 1.0 - 1e-9:
            print(f"ERROR: --fix-mean with |Bbar| = {nb0:.12f} >= 1 - 1e-9 is "
                  "infeasible -- Cauchy-Schwarz: |mean B| = 1")
            print("  together with |B| = 1 pointwise forces B == Bbar "
                  "identically, so the uniform solution is the")
            print("  ONLY state with that mean: there is nothing to grow.  "
                  "Drop --fix-mean, or e.g. --Bbar 0 0 0.9.")
            return 2
        if os.path.exists(args.csv) and os.path.getsize(args.csv) > 0:
            print(f"ERROR: fresh run (no {args.state}) but {args.csv} already "
                  "exists -- appending would silently concatenate two runs "
                  "(project rule: fresh state => fresh CSV). Move it aside or "
                  "pass a new --csv.")
            return 2
        if args.freeze_top > 0 and args.pin_top > 0:
            print("ERROR: --freeze-top and --pin-top both requested.  They "
                  "would select the SAME top modes, and a frozen bin's energy "
                  "cannot be steered (the solver rejects the overlap).")
            print("  Use --pin-top for growth; --freeze-top is kept only to "
                  "reproduce the v1 cascade measurement.")
            return 2
        grid = tuple(args.grid0)
        smooth, fix_mean = float(args.smooth), bool(args.fix_mean)
        de0, streak0 = (args.de if args.de is not None else 0.03), 0
        fixed_res = bool(args.fixed_res)
        seed0, meta = _build_seed(args, MuSolver(grid, smooth=smooth,
                                                 fix_mean=fix_mean))
        freeze_after = max(float(args.freeze_after), 0.0)
        meta.update(Bbar=Bbar0, smooth=smooth, fix_mean=fix_mean,
                    freeze=np.zeros((0, 3), int), freeze_after=freeze_after,
                    pins=np.zeros((0, 3), int), fixed_res=fixed_res)
        freeze, pins = [], []
        if args.pin_top > 0:
            pins = [tuple(int(v) for v in t)
                    for t in top_modes(seed0, args.pin_top)]
            meta["pins"] = np.array(pins, int).reshape(-1, 3)
            nc = bool(noncoplanar(pins))
            print("energy-pinning the seed's top modes: "
                  + ", ".join(str(t) for t in pins))
            print(f"  noncoplanar(pins) = {nc}"
                  + ("   [Lemma: nonzero energy at three non-coplanar "
                     "wavevectors excludes 1D and planar states]" if nc else
                     "   [WARNING: every triple is coplanar -- such a pin set "
                     "cannot by itself force 3D structure]"))
            print("  schedule c_j(eps) = eps^2 e_j(seed), rebuilt on every "
                  "grid rung (SUM units); phases stay free.")
        if args.freeze_top > 0:
            freeze = [tuple(int(v) for v in t)
                      for t in top_modes(seed0, args.freeze_top)]
            meta["freeze"] = np.array(freeze, int).reshape(-1, 3)
            nc = bool(noncoplanar(freeze))
            print("freezing the seed's top modes: "
                  + ", ".join(str(t) for t in freeze)
                  + (" + mean (0,0,0)" if fix_mean else ""))
            print(f"  noncoplanar(freeze) = {nc}"
                  + ("" if nc else "   [WARNING: every triple is coplanar -- "
                     "such a freeze set cannot by itself force 3D structure]"))
            if freeze_after > 0:
                print(f"  (pins activate at eps >= {freeze_after:g}; near the "
                      "uniform start a frozen residual bin is uncancellable)")
        freeze_on = freeze_after <= 0
        S = MuSolver(grid, smooth=smooth, fix_mean=fix_mean,
                     freeze=(freeze if freeze_on else []), pins=pins)
        B = np.zeros((3,) + grid) + Bbar0[:, None, None, None]
        B = np.asarray(S.project(B))          # gn never fixes div: project once
        eps = 0.0
        save_state(args.state, B, eps, meta)
        print(f"initialised: grid {grid}, |Bbar| = {nb0:.6f}, "
              f"seed '{meta['seed_kind']}'")
        if nb0 < 1.0 - 1e-12:
            # |B| = |Bbar| != 1 there, and at a UNIFORM state the k=0 row of
            # the Jacobian vanishes (mean(B.d) = Bbar.mean(d) = 0), so the
            # constant part of the residual is only reachable at second order.
            print("  NOTE: |Bbar| < 1, so the uniform start is OFF the "
                  "manifold and the first solve converges only")
            print("  linearly.  If the first steps reject, raise --sweeps "
                  "BEFORE shrinking --de: a smaller push")
            print("  makes this start HARDER, not easier.")

    seed = np.asarray(make_seed(meta, S), float)
    e_seed = _pin_schedule(S, seed)
    if fixed_res:
        grids = [tuple(B.shape[1:])]
        print(f"fixed-res mode: growing at {grids[0]} for the whole quest -- "
              "gtail/edge are logged, not enforced; no sharpening stop. "
              "Audit the endpoint with an ascending polish ladder (rule 7).")
    else:
        grids = _ladder(B.shape[1:], tuple(args.grid_max))
    de, streak, stop = de0, streak0, ""
    fresh = not os.path.exists(args.csv)
    if not fresh:                          # review m10: schema drift warning
        with open(args.csv) as f:
            hdr = f.readline().strip().split(",")
        if hdr and hdr != CSV_FIELDS:
            print(f"WARNING: {args.csv} header has {len(hdr)} fields, current "
                  f"schema has {len(CSV_FIELDS)} (pin_err added in v3); rows "
                  "will not align -- start a fresh --csv for clean analysis.")
    snap_next = (int(eps / args.snap_de) + 1) * args.snap_de
    t_start = time.time()

    while eps < args.eps_max and time.time() - t_start < args.max_seconds:
        t0 = time.time()
        Bprev = np.array(B)                   # the previous ACCEPTED state
        # Targets for the eps this step is REACHING (traced: no recompile when
        # a rejection halves de and the schedule value changes).
        tgt = (eps + de) ** 2 * e_seed if pins else None
        Btry, res, ci = S.gn(np.asarray(B) + de * seed, sweeps=args.sweeps,
                             cgit=args.cgit, tol=args.res_ok * 1e-2,
                             pin_targets=tgt)
        pin_err = S.pin_error(Btry, tgt, relative=True) if pins else 0.0
        if res > args.res_ok or pin_err > _PIN_ERR_MAX:
            # No salvage: mu-form GN is quadratically convergent, so a residual
            # still above res-ok after `sweeps` means the step is genuinely bad.
            # A missed PIN is equally disqualifying: the anchor would no longer
            # be where the schedule says, and "genuinely 3D" rests on that.
            B, de, streak = Bprev, de * 0.5, 0
            print(f"  [step rejected: res {res:.1e} pin_err {pin_err:.1e}; "
                  f"d_eps -> {de:.5f}]")
            if de < args.de_min:
                stop = "fold"
                break
            continue
        B, eps, streak = np.asarray(Btry), eps + de, streak + 1
        if freeze and not freeze_on and eps >= freeze_after:
            freeze_on = True
            S = MuSolver(B.shape[1:], smooth=smooth, fix_mean=fix_mean,
                         freeze=freeze, pins=pins)
            print(f"   [freeze active from eps={eps:.3f}: "
                  + ", ".join(str(t) for t in freeze) + "]")
        d = _diagnostics(S, B)
        row = dict(eps=round(eps, 4), de=de, grid=str(tuple(B.shape[1:])),
                   res=res, cg=ci, minutes=round((time.time() - t0) / 60, 3),
                   pin_err=pin_err, **d)
        meta["de"], meta["streak"] = de, streak
        save_state(args.state, B, eps, meta)    # state BEFORE the CSV row: a
        _write_row(args.csv, row, fresh)        # kill between the two loses a
        fresh = False                           # row, never duplicates one
        print(f"eps={eps:.3f} grid={tuple(B.shape[1:])} res={res:.1e} "
              f"maxgrad={d['maxgrad']:.2f} drift={d['drift']:.4f} "
              f"defl={d['maxdefl']:.1f} gtail={d['gal_tail_rms']:.1e} "
              f"edge={d['edge']:.1e} lam={d['lam21']:.2f}/{d['lam31']:.2f}"
              + (f" pin={pin_err:.1e}" if pins else ""))
        if eps >= snap_next - 1e-9:
            save_state(_snap_name(args.state, eps), B, eps, meta)
            snap_next += args.snap_de

        # ---- adapt: resolution first, then step size ------------------------
        if (not fixed_res) and (d["gal_tail_rms"] > args.gtail_max
                                or d["edge"] > args.edge_max):
            cur = grids.index(tuple(B.shape[1:])) \
                if tuple(B.shape[1:]) in grids else 0
            if cur + 1 >= len(grids):
                stop = "sharpening"
                break
            old, new = tuple(B.shape[1:]), grids[cur + 1]
            Bf = np.stack([np.asarray(zero_pad(B[i], new)) for i in range(3)])
            S = MuSolver(new, smooth=smooth, fix_mean=fix_mean,
                         freeze=(freeze if freeze_on else []), pins=pins)
            Bf = S.project(Bf)
            # The seed (and with it the pin schedule) is rebuilt BEFORE the
            # polish: e_j(seed) is a grid-dependent sum, so the targets the
            # polish must hold are this rung's, never the previous rung's.
            seed = np.asarray(make_seed(meta, S), float)
            e_seed = _pin_schedule(S, seed)
            tgt = eps ** 2 * e_seed if pins else None
            rin = max(float(np.abs(np.asarray(r)).max()) for r in S.residual(Bf))
            Bp, res2, _ = S.gn(Bf, sweeps=args.sweeps + 4, cgit=args.cgit,
                               tol=args.res_ok * 1e-2, pin_targets=tgt)
            B = np.asarray(Bp)
            pe2 = S.pin_error(B, tgt, relative=True) if pins else 0.0
            print(f"   [refined {old} -> {new}; incoming honest residual "
                  f"{rin:.1e}; polished to {res2:.1e}"
                  + (f"; pin_err {pe2:.1e}]" if pins else "]"))
            if res2 > args.res_ok:
                print(f"   *** WARNING: post-refinement residual {res2:.1e} "
                      f"EXCEEDS --res-ok {args.res_ok:.1e}: the state on the "
                      "finer grid is NOT converged and")
                print("   *** every diagnostic from here on is suspect.  "
                      "Raise --sweeps (not --de) and rerun this rung.")
            if pe2 > _PIN_ERR_MAX:
                print(f"   *** WARNING: post-refinement pin_err {pe2:.1e} "
                      "exceeds the step tolerance: the 3D anchor is off its "
                      "schedule on the new rung.")
            streak = 0                          # algorithm.tex Alg. 1: c <- 0
            meta["de"], meta["streak"] = de, streak
            save_state(args.state, B, eps, meta)
        elif streak >= 2 and de < args.de_max:
            de, streak = min(de * 1.3, args.de_max), 0

    if stop == "fold":
        print("STOP: d_eps underflow -- GN cannot re-converge even for tiny "
              "pushes.  FOLD CANDIDATE (this path")
        print("  may end here).  FIRST check --sweeps (project rule 5b), and "
              "note that a stall at eps = 0 with")
        print("  |Bbar| < 1 is just the off-manifold start.  There is no "
              "carrier to blame: a stall eps that")
        print("  survives every reroute (--grid-max, --smooth, --key, "
              "--pin-top) is a true obstruction.")
    elif stop == "sharpening":
        print("STOP: tails exceed the limit at the largest allowed grid.  "
              "SHARPENING -- the state is leaving")
        print("  the resolvable smoothness class.  Carrier-free, so this is a "
              "property of the seed and the")
        print("  path alone: if the same eps recurs for larger --grid-max it "
              "marks genuine gradient blow-up")
        print("  of THIS path (retry --smooth > 0, another --key, or a "
              "--pin-top set first).")
    print(f"done: eps={eps:.3f} grid={tuple(np.shape(B)[1:])} "
          f"({(time.time() - t_start) / 60:.1f} min)")
    return 0


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def make_plot(args):
    """Spectra ladder overlay over the snapshots + the quest curves."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from constantB.spectra import plot_spectra

    snaps = sorted(glob.glob(_snap_prefix(args.state) + "*.npz"))
    if not snaps:                                 # pre-fix runs used grow_eps*
        snaps = sorted(glob.glob(os.path.join(
            os.path.dirname(os.path.abspath(args.state)), "grow_eps*.npz")))
    if snaps:
        out = args.fig_out.replace(".png", "_spectra.png")
        plot_spectra(snaps, out)
        print(f"wrote {out}  ({len(snaps)} snapshots)")
    else:
        print("no snapshot .npz found beside --state; skipping the spectra figure")

    rows = list(csv.DictReader(open(args.csv)))
    eps = np.array([float(r["eps"]) for r in rows])
    panels = [("maxgrad", r"$\max|\nabla B|_F$", False),
              ("drift", r"$1-|\bar B|$", False),
              ("gal_tail_rms", "Galerkin tail rms", True)]
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
    for a, (key, title, logy) in zip(ax, panels):
        y = np.array([float(r[key]) for r in rows])
        (a.semilogy if logy else a.plot)(eps, np.maximum(y, 1e-300) if logy
                                         else y, lw=1.2)
        a.set_xlabel(r"$\varepsilon$")
        a.set_title(title, fontsize=9)
        a.grid(alpha=0.3)
        a.tick_params(labelsize=8)
    plt.tight_layout()
    plt.savefig(args.fig_out, dpi=200)
    print(f"wrote {args.fig_out}")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--state", default="grow.npz")
    p.add_argument("--csv", default="grow.csv")
    p.add_argument("--grid0", type=int, nargs=3, default=[24, 24, 48])
    p.add_argument("--grid-max", type=int, nargs=3, default=[96, 96, 192])
    p.add_argument("--Bbar", type=float, nargs=3, default=[0.0, 0.0, 1.0])
    p.add_argument("--seed", choices=("blob", "random", "file"), default="blob")
    p.add_argument("--w", type=float, nargs=3, default=[0.8, 0.8, 0.8])
    p.add_argument("--kz", type=int, default=1)
    p.add_argument("--kmax", type=int, default=4)
    p.add_argument("--slope", type=float, default=0.0)
    p.add_argument("--key", type=int, default=0)
    p.add_argument("--seed-file", default="a.npz")
    p.add_argument("--eps-max", type=float, default=3.0)
    p.add_argument("--fixed-res", action="store_true",
                   help="fixed-resolution growth (2026-08-26 scheme): stay on "
                        "the starting grid for the whole quest -- the refine "
                        "triggers (--gtail-max/--edge-max) are logged but not "
                        "enforced, and there is no sharpening stop. N_grow is "
                        "then the single member-selector parameter; audit the "
                        "endpoint with an ascending polish ladder (rule 7). "
                        "Recorded in the state, so resumes stay fixed-res "
                        "without the flag.")
    p.add_argument("--de", type=float, default=None,
                   help="initial step (default 0.03). On RESUME an explicitly "
                        "passed --de overrides the step stored in the state "
                        "(historically it was silently ignored).")
    p.add_argument("--de-min", type=float, default=1e-4)
    p.add_argument("--de-max", type=float, default=0.08)
    p.add_argument("--sweeps", type=int, default=8)
    p.add_argument("--cgit", type=int, default=800)
    p.add_argument("--res-ok", type=float, default=1e-9)
    p.add_argument("--gtail-max", type=float, default=1e-3)
    p.add_argument("--edge-max", type=float, default=1e-5)
    p.add_argument("--smooth", type=float, default=0.0)
    p.add_argument("--fix-mean", action="store_true")
    p.add_argument("--pin-top", type=int, default=0,
                   help="hold the ENERGY of the seed's m dominant modes on the "
                        "schedule c_j(eps) = eps^2 e_j(seed) (bordered energy "
                        "rows; phases free). Three non-coplanar pins exclude "
                        "1D and planar states outright. No activation "
                        "threshold is needed or offered: an energy row is "
                        "satisfiable from eps = 0. Adds the pin_err column.")
    p.add_argument("--freeze-top", type=int, default=0,
                   help="DEPRECATED for growth (use --pin-top): freezes the "
                        "seed's m top COEFFICIENTS. A frozen bin cannot help "
                        "cancel the residual it creates, so the solver pays "
                        "with a cascade -- 36^2x72 at eps ~ 0.27: maxgrad 2.45 "
                        "/ tail 1.2e-3 frozen vs 0.80 / 1.7e-6 unpinned. Kept "
                        "to reproduce that measurement.")
    p.add_argument("--freeze-after", type=float, default=0.2,
                   help="activate the --freeze-top freeze only once eps >= "
                        "this. Near the uniform start B.dB ~ Bbar.dB, so a "
                        "residual bin at a frozen k can ONLY be cancelled by "
                        "dB at that same bin: freezing from eps=0 makes the "
                        "first steps infeasible (observed). 0 freezes from the "
                        "start. Energy pins need no such threshold.")
    p.add_argument("--snap-de", type=float, default=0.25)
    p.add_argument("--max-seconds", type=float, default=1e9)
    p.add_argument("--plot", action="store_true")
    p.add_argument("--fig-out", default="fig_grow.png")
    return p


def main():
    return run(build_parser().parse_args())


if __name__ == "__main__":
    sys.exit(main())
