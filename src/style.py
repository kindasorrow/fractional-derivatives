r"""Shared figure style.

Colours are the validated categorical palette (light mode, surface #fcfcfb):
worst adjacent CVD dE 9.1, normal-vision dE 19.6, all checks PASS.  Three slots
sit below 3:1 contrast on the surface, so every chart also carries direct text
labels / a legend and the numbers are tabulated in the README (relief rule).

Colour follows the *entity*: one hue per model family, variants of a family
share the hue and differ by dash pattern.  References (theory, not a trained
model) are drawn in text ink, never in a series colour.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
INK3 = "#8a8984"
GRID = "#e6e5e1"

SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
        "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

# entity -> (colour, dash, marker, short label)
MODEL_STYLE = {
    "transformer_rope":    (SLOT[0], "-",  "o", "Transformer (RoPE)"),
    "transformer_nope":    (SLOT[0], "--", "s", "Transformer (NoPE)"),
    "transformer_alibi":   (SLOT[0], ":",  "^", "Transformer (ALiBi)"),
    "s4d_deep":            (SLOT[1], "-",  "o", "S4D (deep)"),
    "soe_jiang":           (SLOT[2], "-",  "o", "SOE / Jiang (fixed)"),
    "soe_jiang_table":     (SLOT[2], "--", "s",
                            r"SOE / Jiang (per-$\alpha$ readout)"),
    "soe_jiang_trainband": (SLOT[2], ":",  "^", "SOE / Jiang (train band)"),
    "s4d_lin_log":         (SLOT[3], "-",  "o", "S4D-linear (log init)"),
    "s4d_lin_linear":      (SLOT[4], "-",  "s", "S4D-linear (linear init)"),
    "s4d_lin_narrow":      (SLOT[5], "-",  "^", "S4D-linear (1-scale init)"),
    "s4d_lin_softplus":    (SLOT[6], "-",  "D", "S4D-linear (softplus)"),
    "s4d_lin_log_longtrain":    (SLOT[3], "--", "o",
                                 "S4D-linear, log init, trained at $L$=10000"),
    "s4d_lin_narrow_longtrain": (SLOT[5], "--", "^",
                                 "S4D-lin, 1-scale, trained at $L$=10000"),
}
REFERENCE = dict(color=INK2, ls=(0, (4, 2)), lw=1.6)

# Guard: a non-raw "...\alpha..." silently becomes a BEL character and only
# blows up later inside matplotlib's mathtext parser.  Catch it at import.
for _k, _v in MODEL_STYLE.items():
    assert not any(ord(_c) < 32 for _c in _v[3]), \
        f"control character in the label for {_k!r}: {_v[3]!r} (use a raw string)"


def use_style():
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.labelcolor": INK2,
        "axes.edgecolor": INK3,
        "axes.linewidth": 0.8,
        "axes.titlecolor": INK,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "lines.linewidth": 1.8,
        "lines.markersize": 5.5,
        "lines.solid_capstyle": "round",
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "text.color": INK,
    })


def st(name):
    """(colour, dash, marker, label) for a model name."""
    return MODEL_STYLE.get(name, (INK3, "-", "o", name))


def plot_model(ax, name, x, y, **kw):
    c, ls, mk, lab = st(name)
    return ax.plot(x, y, color=c, ls=ls, marker=mk, label=lab, **kw)


def despine(ax, which=("top", "right")):
    for w in which:
        ax.spines[w].set_visible(False)


def finish(fig, path):
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)
