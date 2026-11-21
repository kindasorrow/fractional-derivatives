r"""Compact, diff-friendly output for the result files.

Two problems with a plain `json.dump(..., indent=2)` here:

* every array element lands on its own line -- bench.json and analysis.json came
  to ~70-80k lines (2.3 MB each), which no PR diff can show;
* most of that bulk is dense numerical data (error-vs-position curves, 8192 SSM
  rates, kernel curves) that nobody reads as text anyway.

So results are written as a *pair*: a small indented `.json` holding the scalars,
labels and short lists -- the part a reviewer actually reads -- and a sibling
`<name>_arrays.npz` holding every numeric array of length >= `MIN_LEN`.  In the
json each offloaded array becomes `{"__npz__": "<key>", "len": n}`, so the file
still describes its own structure.  `load_split` puts them back together, and
the figures/report code uses it, so nothing downstream has to care.

`dump_compact` (everything on one line, floats to 6 significant digits) is kept
for the small files where splitting is not worth a second artefact.
"""

from __future__ import annotations

import json
import math
import os

import numpy as np

SIG = 6
MIN_LEN = 48          # arrays at least this long go to the .npz


def _round(o):
    if isinstance(o, float):
        if not math.isfinite(o) or o == 0.0:
            return o
        return float(f"{o:.{SIG}g}")
    if isinstance(o, dict):
        return {k: _round(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_round(v) for v in o]
    return o


def dump_compact(obj, fh, sig: int = SIG):
    """Write `obj` to the open text file `fh` on one line, floats to `sig` digits."""
    global SIG
    old, SIG = SIG, sig
    try:
        json.dump(_round(obj), fh, separators=(",", ":"))
        fh.write("\n")
    finally:
        SIG = old


# ---------------------------------------------------------------------------
# json + .npz split
# ---------------------------------------------------------------------------
def npz_path(json_path: str) -> str:
    return os.path.splitext(json_path)[0] + "_arrays.npz"


def _is_numeric_array(o) -> bool:
    return (isinstance(o, (list, tuple)) and len(o) >= MIN_LEN
            and all(isinstance(x, (int, float)) and not isinstance(x, bool)
                    for x in o))


def _as_storage_dtype(o) -> np.ndarray:
    """float32 where it is lossless enough, float64 where it is not.

    These are measured curves and rates and the scalars beside them are already
    rounded to 6 significant digits, so float32 is the natural storage.  But a
    few arrays (e.g. the mode importances of the deep SSM) carry entries around
    1e-46, which float32 would silently flush to zero -- so any array reaching
    below the float32 subnormal range keeps full precision instead.
    """
    a = np.asarray(o, dtype=np.float64)
    nz = np.abs(a[np.nonzero(a)])
    if nz.size and nz.min() < 1e-37:
        return a
    return a.astype(np.float32)


def dump_split(obj, json_path: str, sig: int = SIG, min_len: int = MIN_LEN):
    """Write `obj` as a small indented json plus `<name>_arrays.npz`.

    Returns (json_bytes, npz_bytes).
    """
    global SIG, MIN_LEN
    old_sig, old_min = SIG, MIN_LEN
    SIG, MIN_LEN = sig, min_len
    arrays: dict[str, np.ndarray] = {}

    def walk(o, path: str):
        if isinstance(o, dict):
            return {k: walk(v, f"{path}.{k}" if path else str(k))
                    for k, v in o.items()}
        if _is_numeric_array(o):
            key = path or "root"
            arrays[key] = _as_storage_dtype(o)
            return {"__npz__": key, "len": len(o)}
        if isinstance(o, (list, tuple)):
            return [walk(v, f"{path}.{i}") for i, v in enumerate(o)]
        return _round(o)

    try:
        reduced = walk(obj, "")
    finally:
        SIG, MIN_LEN = old_sig, old_min

    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(reduced, fh, indent=1, ensure_ascii=False)
        fh.write("\n")
    np.savez_compressed(npz_path(json_path), **arrays)
    return os.path.getsize(json_path), os.path.getsize(npz_path(json_path))


def load_split(json_path: str):
    """Inverse of `dump_split`; also reads a plain json written without it."""
    with open(json_path, encoding="utf-8") as fh:
        obj = json.load(fh)
    p = npz_path(json_path)
    if not os.path.exists(p):
        return obj
    with np.load(p) as z:
        store = {k: z[k].tolist() for k in z.files}

    def walk(o):
        if isinstance(o, dict):
            if set(o) == {"__npz__", "len"}:
                return store[o["__npz__"]]
            return {k: walk(v) for k, v in o.items()}
        if isinstance(o, list):
            return [walk(v) for v in o]
        return o

    return walk(obj)
