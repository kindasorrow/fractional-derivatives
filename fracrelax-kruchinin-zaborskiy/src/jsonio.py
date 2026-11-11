r"""Compact, diff-friendly json output for the result files.

`json.dump(..., indent=2)` puts every array element on its own line: bench.json
and analysis.json came to ~70-80k lines (2.3 MB each), which no PR diff can show.
Here floats are rounded to 6 significant digits (far below the 1e-3..1e-2 model
errors, and the noise of a timing measurement) and everything is written on a
single line.  The files stay valid json, just not pretty; for reading use
`results/tables.md` and the figures, or `python -m json.tool`.
"""

from __future__ import annotations

import json
import math

SIG = 6


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
