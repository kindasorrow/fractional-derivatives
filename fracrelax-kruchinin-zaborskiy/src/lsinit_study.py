r"""Why least-squares initialisation helps on Jiang nodes and destroys S4D.

Review point: at L = 10^4 `s4d_lin_log_lsinit` reaches ~1.2e+2 while the same
model without LS init reaches 5.5e-3.  This module measures *why*, so the
negative result can be stated as a cause rather than an observation.

It reports, for each pole set:

  static  -- the conditioning of the weighted least-squares system, the scale
             of its solution c, how much the modes have to cancel to produce
             the kernel, the fit error in float64, and the error after the
             solution is rounded to the model's float32;
  dynamic -- short training runs from the LS start with the rates learnable
             (the shipped setting) and with the rates frozen, which separates
             "the readout is too large" from "the poles move".

Run:  python -m src.lsinit_study      ->  results/lsinit_study.json
"""

from __future__ import annotations

import os

import numpy as np
import torch

from . import jiang
from .bands import S_HI, S_LO_TRAIN, T_LONG, T_TRAIN, jiang_nodes
from .config import ALPHAS_TRAIN, CKPT, LAM, L_TRAIN, N_STATES, RESULTS, SEED, TAU
from .data import SeqData, path, rel_l2, rel_mse_loss
from .jsonio import dump_split
from .l1 import discrete_kernel
from .lsinit import ls_readout
from .models import S4DLinear, SOEFixed, _init_rates


def bands():
    """The three pole sets in play, each with the horizon its LS fit may use."""
    s_long, _ = jiang_nodes(T_LONG)
    s_train, _ = jiang_nodes(T_TRAIN)
    s_s4d = _init_rates(N_STATES, S_LO_TRAIN, S_HI, "log").numpy().astype(np.float64)
    return {
        "jiang_T100": dict(s=s_long, L_fit=10000,
                           note="Jiang rule for T=100 (soe_jiang*)"),
        "jiang_T10": dict(s=s_train, L_fit=1000,
                          note="Jiang rule for T=10 (soe_jiang_trainband)"),
        "s4d_log_init": dict(s=s_s4d, L_fit=L_TRAIN,
                             note="S4D log-spaced init band [0.1, 100] (s4d_lin_*)"),
    }


# ---------------------------------------------------------------------------
# static: conditioning, scale, cancellation, precision
# ---------------------------------------------------------------------------
def static_probe(s: np.ndarray, L_fit: int, alpha: float, n_log: int = 600):
    """Everything about the LS solution on this pole set at this alpha."""
    g = discrete_kernel(alpha, LAM, TAU, L_fit)
    n_all = np.arange(1, L_fit + 1)
    idx = np.unique(np.round(np.exp(np.linspace(0.0, np.log(L_fit), n_log))).astype(int))
    idx = idx[(idx >= 1) & (idx <= L_fit)]
    target = TAU * g[idx - 1]
    M = jiang.soe_design(s, TAU, n_all[idx - 1])
    wgt = 1.0 / np.abs(target)
    A = M * wgt[:, None]                       # the system actually solved
    c, *_ = np.linalg.lstsq(A, target * wgt, rcond=None)

    sv = np.linalg.svd(A, compute_uv=False)
    k_full = jiang.soe_design(s, TAU, n_all) @ c
    true = TAU * g

    # how much do the modes cancel?  terms summed in absolute value, against
    # the value actually produced, at the lag where it is worst
    terms = np.abs(jiang.soe_design(s, TAU, n_all[idx - 1]) * c[None, :]).sum(1)
    cancel = float(np.max(terms / np.maximum(np.abs(target), 1e-300)))

    # the model stores c in float32 and evaluates the kernel there
    c32 = c.astype(np.float32).astype(np.float64)
    k32 = jiang.soe_design(s, TAU, n_all) @ c32
    rel = lambda k: float(np.linalg.norm(k - true) / np.linalg.norm(true))
    return dict(
        c=c.tolist(), rates=np.asarray(s, float).tolist(),
        alpha=float(alpha), n_exp=int(s.size), L_fit=int(L_fit),
        cond=float(sv[0] / sv[-1]), sv_max=float(sv[0]), sv_min=float(sv[-1]),
        c_abs_max=float(np.abs(c).max()), c_l1=float(np.abs(c).sum()),
        cancellation=cancel,
        kernel_rel_l2_fp64=rel(k_full), kernel_rel_l2_fp32=rel(k32),
    )


# ---------------------------------------------------------------------------
# dynamic: what training does to the LS start
# ---------------------------------------------------------------------------
def _build(kind: str):
    if kind == "jiang_T100":
        s, _ = jiang_nodes(T_LONG)
        return SOEFixed(s, TAU, readout="table", n_alpha=len(ALPHAS_TRAIN)), 10000
    return (S4DLinear(N_STATES, TAU, S_LO_TRAIN, S_HI, init="log",
                      readout="table", n_alpha=len(ALPHAS_TRAIN)), L_TRAIN)


@torch.no_grad()
def _val(model, va, batch=64):
    model.eval()
    e = [rel_l2(model(va.f[i:i + batch], va.alpha[i:i + batch],
                      va.alpha_id[i:i + batch]), va.y[i:i + batch]).cpu()
         for i in range(0, len(va), batch)]
    model.train()
    return float(np.median(torch.cat(e).numpy()))


def dynamic_probe(kind: str, lr_rate: float, steps: int = 2000,
                  lr: float = 3e-3, batch: int = 32, device="cuda",
                  every: int = 250):
    """Train `steps` from the LS start and record what happens."""
    torch.manual_seed(SEED)
    tr = SeqData(path(f"train_L{L_TRAIN}.npz"), device=device)
    va = SeqData(path(f"val_L{L_TRAIN}.npz"), device=device)
    model, L_fit = _build(kind)
    C = ls_readout(model.rates().detach().cpu().numpy(), ALPHAS_TRAIN, L_fit)
    with torch.no_grad():
        model.c.copy_(torch.as_tensor(C, dtype=model.c.dtype))
    model = model.to(device)

    learnable = [p for n, p in model.named_parameters() if n.endswith("raw")]
    other = [p for n, p in model.named_parameters() if not n.endswith("raw")]
    groups = [dict(params=other, lr=lr)]
    if learnable:
        groups.append(dict(params=learnable, lr=lr_rate))
    opt = torch.optim.AdamW(groups, weight_decay=0.0)

    s0 = model.rates().detach().cpu().numpy().copy()
    # gradient norm at the LS start, before any step is taken
    idx = torch.randint(0, len(tr), (batch,), device=device)
    loss0 = rel_mse_loss(model(tr.f[idx], tr.alpha[idx], tr.alpha_id[idx]), tr.y[idx])
    loss0.backward()
    gnorm0 = float(torch.sqrt(sum((p.grad ** 2).sum() for p in model.parameters()
                                  if p.grad is not None)))
    opt.zero_grad(set_to_none=True)

    traj = [dict(step=0, val=_val(model, va), loss=float(loss0.detach()))]
    g = torch.Generator(device=device).manual_seed(SEED)
    for it in range(steps):
        idx = torch.randint(0, len(tr), (batch,), device=device, generator=g)
        loss = rel_mse_loss(model(tr.f[idx], tr.alpha[idx], tr.alpha_id[idx]), tr.y[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (it + 1) % every == 0:
            traj.append(dict(step=it + 1, val=_val(model, va), loss=float(loss)))
    s1 = model.rates().detach().cpu().numpy()
    return dict(kind=kind, lr_rate=lr_rate, steps=steps, grad_norm_at_ls=gnorm0,
                rates_learnable=bool(learnable),
                rate_drift_max=float(np.max(np.abs(np.log10(s1 / s0)))),
                rate_s_min_before=float(s0.min()), rate_s_min_after=float(s1.min()),
                trajectory=traj)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rep = dict(static={}, dynamic=[])

    print("[lsinit] static: conditioning and scale of the LS solution")
    for name, b in bands().items():
        rows = [static_probe(b["s"], b["L_fit"], a) for a in ALPHAS_TRAIN]
        ref = min(rows, key=lambda r: abs(r["alpha"] - 0.5))
        c_ref, s_ref = ref["c"], ref["rates"]
        for r in rows:                      # vectors kept once, not 8 times
            r.pop("c"), r.pop("rates")
        rep["static"][name] = dict(note=b["note"], L_fit=b["L_fit"],
                                   n_exp=int(b["s"].size),
                                   s_min=float(b["s"].min()),
                                   s_max=float(b["s"].max()), per_alpha=rows,
                                   alpha_ref=ref["alpha"], c_ref=c_ref,
                                   rates_ref=s_ref)
        w = max(rows, key=lambda r: r["c_abs_max"])
        print(f"  {name:14s} cond {w['cond']:.2e}  |c|max {w['c_abs_max']:.3e}  "
              f"cancel {w['cancellation']:.2e}  "
              f"fit fp64 {w['kernel_rel_l2_fp64']:.2e} -> fp32 "
              f"{w['kernel_rel_l2_fp32']:.2e}")

    print("[lsinit] dynamic: 2000 steps from the LS start")
    for kind, lr_rate, tagname in (("s4d_log_init", 1e-2, "S4D table-readout diagnostic, rates learnable"),
                                   ("s4d_log_init", 0.0, "S4D table-readout diagnostic, rates frozen"),
                                   ("jiang_T100", 0.0, "Jiang table-readout control, rates fixed")):
        d = dynamic_probe(kind, lr_rate, device=dev)
        d["label"] = tagname
        rep["dynamic"].append(d)
        print(f"  {tagname:42s} val {d['trajectory'][0]['val']:.2e} -> "
              f"{d['trajectory'][-1]['val']:.2e}   |grad| at LS "
              f"{d['grad_norm_at_ls']:.2e}   max rate drift "
              f"{d['rate_drift_max']:.2f} decades")

    out = os.path.join(RESULTS, "lsinit_study.json")
    nb, na = dump_split(rep, out)
    print(f"wrote {out} ({nb / 1024:.0f} kB)")


if __name__ == "__main__":
    main()
