"""State files: (B, eps, meta) triples in .npz, byte-compatible with the
frozen numpy reference toolkit (and with every state file in the project).
Host numpy on purpose -- jax arrays are converted at this boundary.
"""
import numpy as np

from .seeds import carrier, build_seed


def save_state(fn, B, eps, meta):
    np.savez(fn, B=np.asarray(B), eps=eps, **meta)


def load_state(fn):
    st = np.load(fn, allow_pickle=True)
    meta = {k: st[k] for k in st.files if k not in ("B", "eps")}
    return st["B"], float(st["eps"]), meta


def meta_args(args):
    """Metadata dict from a parsed CLI namespace (A, c, modes, prof)."""
    return dict(A=args.A, c=args.c, modes=np.array(args.modes, float),
                prof=np.array(args.prof, float))


def rebuild_seed(meta, shape):
    """Reconstruct the continuation seed (and carrier) recorded in a state's
    metadata, on the grid `shape`."""
    car = carrier(float(meta["A"]), float(meta["c"]), shape[2])
    modes = [tuple(m) for m in np.atleast_2d(meta["modes"])]
    seed = build_seed(car, modes, shape,
                      prof=tuple(np.array(meta["prof"]).ravel()))
    return seed, car
