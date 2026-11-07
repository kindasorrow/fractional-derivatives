r"""Benchmarks: accuracy / time / memory vs sequence length (tasks 3 and 4).

All neural models are trained ONLY on L = 1000 and evaluated at
L = 1000 .. 10000, which is the "main experiment".

Classical CPU reference solvers live in `src.bench_classical` (separate
process: torch's libiomp and Anaconda's MKL clash); if its json is present it
is merged into the report here.

Besides the length sweep this also reports
  * `seed_runs`: median error per length for every available seed of the SEEDED
    models (the report prints mean +/- sd);
  * `mixed`: error on `mixed_lengths.npz` (random lengths in [1000, 10000], the
    literal task spec), each sequence run at its OWN length, no padding;
  * speed / memory.  NOTE: the ALiBi numbers measured before the chunked
    implementation (`models.alibi_attention`) reflected the full L x L bias
    matrix, not the method.

Accuracy is always relative to the L1 solution (the training target), not to
the continuous equation; see README section 2.

Run:  python -m src.bench_classical && python -m src.bench
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from .bands import SEEDED
from .config import (ALPHAS_TRAIN, CKPT, LAM, L_LONG, L_SCALING, L_TRAIN,
                     N_STATES, RESULTS, SEED, SEEDS, TAU, tag)
from .data import SeqData, path, rel_l2
from .jsonio import dump_compact
from .registry import SPECS, build


# ---------------------------------------------------------------------------
def ckpt_path(name: str, seed: int = SEED) -> str:
    return os.path.join(CKPT, f"{tag(name, seed)}.pt")


def load_model(name: str, device="cuda", seed: int = SEED):
    ck = torch.load(ckpt_path(name, seed), map_location=device,
                    weights_only=False)
    model, _ = build(name)
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval(), ck["info"]


@torch.no_grad()
def accuracy(model, ds: SeqData, device="cuda", batch: int = 8):
    """Per-sequence relative L2 plus the position-resolved RMS error."""
    errs, sq, ysq = [], None, None
    for i in range(0, len(ds), batch):
        sl = slice(i, min(i + batch, len(ds)))
        f = ds.f[sl].to(device)
        y = ds.y[sl].to(device)
        yh = model(f, ds.alpha[sl].to(device), ds.alpha_id[sl].to(device))
        errs.append(rel_l2(yh, y).cpu())
        e2 = ((yh - y) ** 2).sum(0).cpu()
        y2 = (y ** 2).sum(0).cpu()
        sq = e2 if sq is None else sq + e2
        ysq = y2 if ysq is None else ysq + y2
    return torch.cat(errs).numpy(), (sq / len(ds)).sqrt().numpy(), (ysq / len(ds)).sqrt().numpy()


@torch.no_grad()
def accuracy_mixed(model, ds: SeqData, device="cuda"):
    """Relative L2 per sequence, each run at its own length (no padding).

    `mixed_lengths.npz` is zero-padded to 10000 with an explicit `length`; the
    models are causal, but running the padded tensor would still cost the full
    10000 steps and would hide variable-length behaviour, so each sequence is
    cut to its true length.
    """
    errs = []
    for i in range(len(ds)):
        n = int(ds.length[i])
        f = ds.f[i:i + 1, :n].to(device)
        y = ds.y[i:i + 1, :n].to(device)
        yh = model(f, ds.alpha[i:i + 1].to(device), ds.alpha_id[i:i + 1].to(device))
        errs.append(float(rel_l2(yh, y).cpu()))
    return np.asarray(errs), ds.length.astype(int)


@torch.no_grad()
def speed(model, L: int, device="cuda", reps: int = 12, warmup: int = 3,
          batch: int = 1):
    """Wall time and peak memory for one forward pass of `batch` sequences."""
    f = torch.randn(batch, L, device=device)
    a = torch.full((batch,), 0.5, device=device)
    aid = torch.full((batch,), 2, device=device, dtype=torch.long)
    try:
        for _ in range(warmup):
            model(f, a, aid)
        if device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        for _ in range(reps):
            model(f, a, aid)
        if device == "cuda":
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / reps
        peak = (torch.cuda.max_memory_allocated() / 2 ** 20
                if device == "cuda" else float("nan"))
        return dict(time_ms=dt * 1e3, peak_MiB=peak, ok=True,
                    time_ms_per_seq=dt * 1e3 / batch, batch=batch)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return dict(time_ms=None, peak_MiB=None, ok=False, error="OOM",
                    batch=batch)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--models", default=None, help="comma separated subset")
    ap.add_argument("--seeds", type=int, default=len(SEEDS),
                    help="evaluate up to this many seeds of the SEEDED models")
    args = ap.parse_args()
    dev = args.device
    names = args.models.split(",") if args.models else list(SPECS)

    report = dict(device=dev, tau=TAU, lam=LAM, L_train=L_TRAIN,
                  alibi_impl="chunked_q512",     # see models.alibi_attention
                  alphas=list(ALPHAS_TRAIN), models={})

    # ---- per-model accuracy / speed / memory vs length --------------------
    datasets = {L: SeqData(path(f"scale_L{L}.npz")) for L in L_SCALING}
    for nm in names:
        print(f"[bench] {nm}")
        model, info = load_model(nm, dev)
        rec = dict(label=SPECS[nm]["label"], family=SPECS[nm]["family"],
                   n_params=info["n_params"], train_time_s=info["train_time_s"],
                   train_peak_mem_MiB=info["train_peak_mem_MiB"],
                   best_val_rel_l2_med=info["best_val_rel_l2_med"], scaling=[])
        for L in L_SCALING:
            ds = datasets[L]
            b = 8 if L <= 4000 else (4 if L <= 7000 else 2)
            try:
                errs, rms_e, rms_y = accuracy(model, ds, dev, batch=b)
                acc = dict(med=float(np.median(errs)), mean=float(errs.mean()),
                           p90=float(np.percentile(errs, 90)),
                           per_alpha={round(float(a), 3): float(np.median(
                               errs[ds.alpha_id.numpy() == i]))
                               for i, a in enumerate(ds.alphas)})
                pos = dict(rms_err=rms_e[::max(1, L // 400)].tolist(),
                           rms_y=rms_y[::max(1, L // 400)].tolist(),
                           stride=max(1, L // 400))
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                acc, pos = dict(error="OOM"), None
            reps = 12 if L <= 2000 else (8 if L <= 7000 else 5)
            sp = speed(model, L, dev, reps=reps, batch=1)
            sp8 = speed(model, L, dev, reps=reps, batch=8)
            rec["scaling"].append(dict(L=L, acc=acc, speed=sp, speed_b8=sp8,
                                       position=pos))
            print(f"    L={L:6d}  relL2 med "
                  f"{acc.get('med', float('nan')):.4e}  "
                  f"{sp.get('time_ms') if sp['ok'] else 'OOM'}")
        report["models"][nm] = rec
        del model
        torch.cuda.empty_cache()

    # ---- held-out fractional orders at the long length --------------------
    print("[bench] held-out alphas @ L=10000")
    ho = SeqData(path(f"heldout_alpha_L{L_LONG}.npz"))
    for nm in names:
        model, _ = load_model(nm, dev)
        try:
            errs, _, _ = accuracy(model, ho, dev, batch=2)
            report["models"][nm]["heldout_alpha"] = {
                round(float(a), 3): float(np.median(errs[ho.alpha_id.numpy() == i]))
                for i, a in enumerate(ho.alphas)}
            report["models"][nm]["heldout_alpha"]["all_med"] = float(np.median(errs))
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            report["models"][nm]["heldout_alpha"] = dict(error="OOM")
        print(f"    {nm}: {report['models'][nm]['heldout_alpha']}")
        del model
        torch.cuda.empty_cache()

    # ---- per-forcing-family breakdown at the long length ------------------
    print("[bench] per-forcing-family @ L=10000")
    tl = SeqData(path(f"test_L{L_LONG}.npz"))
    kinds = tl.meta["kinds"]
    for nm in names:
        model, _ = load_model(nm, dev)
        try:
            errs, rms_e, rms_y = accuracy(model, tl, dev, batch=2)
            report["models"][nm]["long_test"] = dict(
                med=float(np.median(errs)), mean=float(errs.mean()),
                per_kind={kinds[k]: float(np.median(errs[tl.kind_id == k]))
                          for k in np.unique(tl.kind_id)},
                rms_err=rms_e[::25].tolist(), rms_y=rms_y[::25].tolist(),
                stride=25)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            report["models"][nm]["long_test"] = dict(error="OOM")
        print(f"    {nm}: med {report['models'][nm]['long_test'].get('med')}")
        del model
        torch.cuda.empty_cache()

    # ---- mixed lengths: the literal task spec -----------------------------
    print("[bench] mixed lengths in [1000, 10000]")
    mp = path("mixed_lengths.npz")
    if os.path.exists(mp):
        mx = SeqData(mp)
        bins = [(1000, 3000), (3000, 6000), (6000, 10001)]
        for nm in names:
            model, _ = load_model(nm, dev)
            try:
                errs, lens = accuracy_mixed(model, mx, dev)
                report["models"][nm]["mixed"] = dict(
                    n=int(len(errs)), med=float(np.median(errs)),
                    mean=float(errs.mean()),
                    by_length={f"{lo}-{hi - 1}": float(np.median(
                        errs[(lens >= lo) & (lens < hi)])) for lo, hi in bins})
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                report["models"][nm]["mixed"] = dict(error="OOM")
            print(f"    {nm}: med {report['models'][nm]['mixed'].get('med')}")
            del model
            torch.cuda.empty_cache()
    else:
        print("  (no data/mixed_lengths.npz: run `python -m src.datagen`)")

    # ---- extra seeds of the SEEDED models ----------------------------------
    print("[bench] seeds")
    for nm in [n for n in names if n in SEEDED]:
        runs = {}
        for sd in SEEDS[:args.seeds]:
            if not os.path.exists(ckpt_path(nm, sd)):
                continue
            model, _ = load_model(nm, dev, seed=sd)
            med_by_L = {}
            for L in L_SCALING:
                b = 8 if L <= 4000 else (4 if L <= 7000 else 2)
                try:
                    errs, _, _ = accuracy(model, datasets[L], dev, batch=b)
                    med_by_L[str(L)] = float(np.median(errs))
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    med_by_L[str(L)] = None
            try:
                errs, _, _ = accuracy(model, ho, dev, batch=2)
                ho_med = float(np.median(errs))
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                ho_med = None
            runs[str(sd)] = dict(med_by_L=med_by_L, heldout_all_med=ho_med)
            del model
            torch.cuda.empty_cache()
        report["models"][nm]["seed_runs"] = runs
        print(f"    {nm}: {len(runs)} seed(s)")

    # ---- classical references (produced by src.bench_classical) ----------
    cpath = os.path.join(RESULTS, "bench_classical.json")
    if os.path.exists(cpath):
        with open(cpath) as fh:
            report.update(json.load(fh))
    else:
        print("  (run `python -m src.bench_classical` for the CPU references)")

    with open(os.path.join(RESULTS, "bench.json"), "w") as fh:
        dump_compact(report, fh)
    print("wrote results/bench.json")


if __name__ == "__main__":
    main()
