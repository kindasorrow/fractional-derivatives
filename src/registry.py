r"""Model zoo: one place where every trained configuration is defined.

Naming
------
soe_jiang            fixed exponents = Jiang nodes for the *deployment* horizon
                     T = L_LONG*tau (the theory tells you which rates a target
                     horizon needs, so the model is allowed to use them).
soe_jiang_trainband  the same, but the Jiang rule only knows the *training*
                     horizon T = L_TRAIN*tau -- ablation for the band prior.
soe_jiang_table      the same nodes with one free readout vector per training
                     alpha instead of an alpha-MLP (optimisation control).
s4d_lin_*            single-layer linear SSM with learned rates, three
                     initialisations (log / lin / narrow) and two rate
                     parameterisations -- the pole-discovery experiment.
s4d_deep             the headline deep diagonal SSM.
transformer_*        small causal Transformer, rope / nope / alibi.
"""

from __future__ import annotations

from .bands import (ALPHA_REF, MAIN3, S_HI, S_LO_TRAIN, T_LONG, T_TRAIN,
                    jiang_nodes)
from .config import (ALPHAS_TRAIN, DATA, D_MODEL, L_LONG, L_TRAIN,
                     N_STATES, TAU)
from .models import S4D, S4DLinear, SOEFixed, TinyTransformer

def _soe(T):
    s, info = jiang_nodes(T)
    return SOEFixed(s, TAU), dict(n_exp=int(len(s)), jiang=info)


def _soe_table(T):
    """Readout stored as one free vector per training alpha (no alpha-MLP).

    Separates "can the architecture represent the operator" from "can the
    alpha-conditioning MLP be optimised"; it cannot extrapolate to unseen alpha.
    """
    s, info = jiang_nodes(T)
    return (SOEFixed(s, TAU, readout="table", n_alpha=len(ALPHAS_TRAIN)),
            dict(n_exp=int(len(s)), jiang=info, readout="table"))


SPECS = {
    # ---- fixed exponents (Jiang) ----------------------------------------
    "soe_jiang": dict(
        family="soe", build=lambda: _soe(T_LONG),
        label="SOE / Jiang (fixed exponents)", steps=40000),
    "soe_jiang_trainband": dict(
        family="soe", build=lambda: _soe(T_TRAIN),
        label="SOE / Jiang, train-horizon band", steps=40000),
    "soe_jiang_table": dict(
        family="soe", build=lambda: _soe_table(T_LONG),
        label="SOE / Jiang (per-alpha readout)", steps=40000),

    # ---- learned rates, same architecture -------------------------------
    "s4d_lin_log": dict(
        family="s4d_lin",
        build=lambda: (S4DLinear(N_STATES, TAU, S_LO_TRAIN, S_HI, init="log"),
                       dict(init="log", rate_param="log")),
        label="S4D-linear (log init)", steps=40000),
    "s4d_lin_linear": dict(
        family="s4d_lin",
        build=lambda: (S4DLinear(N_STATES, TAU, S_LO_TRAIN, S_HI, init="lin"),
                       dict(init="lin", rate_param="log")),
        label="S4D-linear (linear init)", steps=40000),
    "s4d_lin_narrow": dict(
        family="s4d_lin",
        build=lambda: (S4DLinear(N_STATES, TAU, S_LO_TRAIN, S_HI, init="narrow"),
                       dict(init="narrow", rate_param="log")),
        label="S4D-linear (single-scale init)", steps=40000),
    "s4d_lin_softplus": dict(
        family="s4d_lin",
        build=lambda: (S4DLinear(N_STATES, TAU, S_LO_TRAIN, S_HI, init="log",
                                 rate_param="softplus"),
                       dict(init="log", rate_param="softplus")),
        label="S4D-linear (softplus rates)", steps=40000),

    # ---- control: the same SSM trained ON long sequences ------------------
    # If the training data covers the horizon, does the network find the band
    # Jiang's formula prescribes?  Smaller batch / fewer steps because each
    # step is 10x longer; the model and its budget per sequence are unchanged.
    "s4d_lin_log_longtrain": dict(
        family="s4d_lin",
        build=lambda: (S4DLinear(N_STATES, TAU, S_LO_TRAIN, S_HI, init="log"),
                       dict(init="log", rate_param="log", trained_on="L=10000")),
        label="S4D-linear (trained at L=10000)", steps=20000, train_L=L_LONG),
    "s4d_lin_narrow_longtrain": dict(
        family="s4d_lin",
        build=lambda: (S4DLinear(N_STATES, TAU, S_LO_TRAIN, S_HI, init="narrow"),
                       dict(init="narrow", rate_param="log", trained_on="L=10000")),
        label="S4D-linear (1-scale init, trained at L=10000)",
        steps=20000, train_L=L_LONG),

    # ---- deep SSM --------------------------------------------------------
    "s4d_deep": dict(
        family="s4d",
        build=lambda: (S4D(D_MODEL, N_STATES, 2, TAU, S_LO_TRAIN, S_HI),
                       dict(d_model=D_MODEL, n_states=N_STATES, n_layers=2)),
        label="S4D (deep, 2 blocks)", steps=40000),

    # ---- Transformers ----------------------------------------------------
    "transformer_rope": dict(
        family="transformer",
        build=lambda: (TinyTransformer(D_MODEL, 4, 2, pos="rope"), dict(pos="rope")),
        label="Transformer (RoPE)", steps=40000),
    "transformer_nope": dict(
        family="transformer",
        build=lambda: (TinyTransformer(D_MODEL, 4, 2, pos="nope"), dict(pos="nope")),
        label="Transformer (NoPE)", steps=40000),
    "transformer_alibi": dict(
        family="transformer",
        build=lambda: (TinyTransformer(D_MODEL, 4, 2, pos="alibi"), dict(pos="alibi")),
        label="Transformer (ALiBi)", steps=40000),
}

def build(name: str):
    model, extra = SPECS[name]["build"]()
    return model, extra


def label(name: str) -> str:
    return SPECS[name]["label"]
