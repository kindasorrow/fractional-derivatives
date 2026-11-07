r"""Training.

    python -m src.train --model s4d_deep
    python -m src.train --all                # every config; 3 seeds for SEEDED
    python -m src.train --all --seeds 1      # primary seed only (faster)

Every model sees exactly the same data (train_L1000.npz), the same number of
optimisation steps, the same batch size and the same scale-free loss, so the
comparison is about inductive bias rather than budget.

Weight decay is OFF for every model: the target map is a *linear* operator
whose optimal readout coefficients are O(1..30), so an L2 pull towards zero
would penalise exactly the structured models.  Over-fitting is instead
controlled by keeping the best-on-validation checkpoint.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch

from .bands import SEEDED
from .config import CKPT, L_TRAIN, RESULTS, SEED, SEEDS, tag
from .data import SeqData, path, rel_l2, rel_mse_loss
from .jsonio import dump_compact
from .lsinit import apply_ls_init
from .models import n_params
from .registry import SPECS, build


def param_groups(model, lr: float, lr_rate: float, wd: float):
    rate, nodecay, decay = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.endswith("raw"):                 # SSM relaxation rates
            rate.append(p)
        elif p.ndim <= 1 or name.endswith(("Bp", "Cp", "c")):
            nodecay.append(p)
        else:
            decay.append(p)
    groups = [dict(params=rate, lr=lr_rate, weight_decay=0.0, tag="rate"),
              dict(params=nodecay, lr=lr, weight_decay=0.0, tag="nodecay"),
              dict(params=decay, lr=lr, weight_decay=wd, tag="decay")]
    return [g for g in groups if g["params"]]


@torch.no_grad()
def evaluate(model, ds: SeqData, batch: int = 64, device="cuda"):
    model.eval()
    errs = []
    for i in range(0, len(ds), batch):
        sl = slice(i, min(i + batch, len(ds)))
        yh = model(ds.f[sl], ds.alpha[sl], ds.alpha_id[sl])
        errs.append(rel_l2(yh, ds.y[sl]).cpu())
    model.train()
    return torch.cat(errs).numpy()


def train_one(name: str, device="cuda", batch: int = 32, lr: float = 3e-3,
              lr_rate: float = 1e-2, wd: float = 0.0, warmup: int = 200,
              steps: int | None = None, seed: int = SEED, log_every: int = 500):
    torch.manual_seed(seed)
    np.random.seed(seed % (2 ** 31))
    spec = SPECS[name]
    steps = steps or spec["steps"]

    train_L = spec.get("train_L", L_TRAIN)
    tr = SeqData(path(f"train_L{train_L}.npz"), device=device)
    va = SeqData(path(f"val_L{train_L}.npz"), device=device)

    model, extra = build(name)
    if spec.get("ls_init"):                  # readout starts at the LS solution
        extra = dict(extra, ls_init=apply_ls_init(model, spec["ls_init"]))
    model = model.to(device)
    npar = n_params(model)
    opt = torch.optim.AdamW(param_groups(model, lr, lr_rate, wd))
    base_lrs = [g["lr"] for g in opt.param_groups]
    sched = lambda it: (it / max(1, warmup) if it < warmup
                        else 0.5 * (1 + math.cos(math.pi * (it - warmup) /
                                                 max(1, steps - warmup))))

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    g = torch.Generator(device=device).manual_seed(seed)
    hist, best = [], dict(val=float("inf"))
    if spec.get("ls_init"):
        # the LS start is itself a candidate checkpoint: if training only makes
        # it worse, the best-on-validation rule must be able to keep it.
        e = evaluate(model, va, device=device)
        hist.append(dict(step=0, loss=float("nan"),
                         val_rel_l2_med=float(np.median(e)),
                         val_rel_l2_mean=float(e.mean())))
        best = dict(val=hist[0]["val_rel_l2_med"], step=0,
                    state={k: v.detach().clone()
                           for k, v in model.state_dict().items()})
        print(f"    step     0  (LS init)  val relL2 med {best['val']:.4e}")
    t0 = time.time()
    for it in range(steps):
        for gi, grp in enumerate(opt.param_groups):
            grp["lr"] = base_lrs[gi] * sched(it)
        idx = torch.randint(0, len(tr), (batch,), device=device, generator=g)
        yh = model(tr.f[idx], tr.alpha[idx], tr.alpha_id[idx])
        loss = rel_mse_loss(yh, tr.y[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if (it + 1) % log_every == 0 or it == steps - 1:
            e = evaluate(model, va, device=device)
            rec = dict(step=it + 1, loss=float(loss.item()),
                       val_rel_l2_med=float(np.median(e)),
                       val_rel_l2_mean=float(e.mean()))
            hist.append(rec)
            if rec["val_rel_l2_med"] < best["val"]:
                best = dict(val=rec["val_rel_l2_med"], step=it + 1,
                            state={k: v.detach().clone()
                                   for k, v in model.state_dict().items()})
            print(f"    step {it + 1:5d}  loss {loss.item():.3e}  "
                  f"val relL2 med {rec['val_rel_l2_med']:.4e}")
    train_time = time.time() - t0
    peak = (torch.cuda.max_memory_allocated() / 2 ** 20) if device == "cuda" else float("nan")

    model.load_state_dict(best["state"])
    out = dict(name=name, tag=tag(name, seed), label=spec["label"],
               family=spec["family"],
               n_params=npar, steps=steps, batch=batch, lr=lr, lr_rate=lr_rate,
               wd=wd, seed=seed, train_L=train_L, train_time_s=train_time,
               train_peak_mem_MiB=peak, best_step=best["step"],
               best_val_rel_l2_med=best["val"], history=hist, extra=extra)
    torch.save(dict(state_dict=best["state"], info=out),
               os.path.join(CKPT, f"{tag(name, seed)}.pt"))
    with open(os.path.join(RESULTS, f"train_{tag(name, seed)}.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"  -> {tag(name, seed)}: {npar} params, best val relL2 {best['val']:.4e} "
          f"@ step {best['step']}, {train_time:.0f}s, peak {peak:.0f} MiB")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--seeds", type=int, default=len(SEEDS),
                    help="seeds per SEEDED model with --all (1 = primary only)")
    ap.add_argument("--seed", type=int, default=None,
                    help="train one explicit seed (with --model)")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if args.all:
        runs = [(nm, SEEDS[0]) for nm in SPECS]
        runs += [(nm, s) for s in SEEDS[1:args.seeds] for nm in SEEDED]
    else:
        seeds = [args.seed] if args.seed is not None else list(SEEDS[:args.seeds])
        runs = [(args.model, s) for s in seeds]
    summary = []
    for nm, sd in runs:
        print(f"[train] {tag(nm, sd)}")
        r = train_one(nm, device=args.device, batch=args.batch,
                      steps=args.steps, seed=sd)
        r.pop("history", None)               # per-run json keeps the history
        summary.append(r)
    with open(os.path.join(RESULTS, "train_summary.json"), "w") as fh:
        dump_compact(summary, fh)


if __name__ == "__main__":
    main()
