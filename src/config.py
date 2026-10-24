"""Global experiment configuration (paths + physical / numerical constants)."""

from __future__ import annotations

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results")
CKPT = os.path.join(ROOT, "data", "checkpoints")

for _d in (DATA, RESULTS, CKPT):
    os.makedirs(_d, exist_ok=True)

# ----------------------------------------------------------------------------
# Physics / discretisation
# ----------------------------------------------------------------------------
# Time step is FIXED across all sequence lengths: a longer sequence means a
# longer time horizon T = L*tau, i.e. a longer memory to resolve.  This is what
# makes "train short / test long" a statement about the memory kernel and not
# about a change of units.
TAU = 0.01
LAM = 1.0  # relaxation rate, held fixed so that Jiang nodes <-> S4D poles
           # can be compared for one well-defined operator per alpha.

ALPHAS_TRAIN = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
ALPHAS_HELDOUT = (0.25, 0.55, 0.85)  # unseen orders, same lambda

# ----------------------------------------------------------------------------
# Sequence lengths (the task asks for 1e3 .. 1e4 steps)
# ----------------------------------------------------------------------------
L_TRAIN = 1000
L_LONG = 10000
L_SCALING = (1000, 2000, 4000, 7000, 10000)

# ----------------------------------------------------------------------------
# Model sizes (deliberately small -- the question is inductive bias, not capacity)
# ----------------------------------------------------------------------------
D_MODEL = 64
N_STATES = 64  # SSM states per channel / number of fixed exponentials

SEED = 20261022
