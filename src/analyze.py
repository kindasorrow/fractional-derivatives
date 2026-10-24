r"""Task 5: what relaxation rates did the SSMs actually learn?

For every SSM we extract
  * the continuous rates s_j (ZOH convention A_j = exp(-s_j tau), so s_j is a
    rate in 1/time and directly comparable with a Jiang node),
  * an importance weight for each mode = the L2 norm of that mode's
    contribution to the model's convolution kernel over the long horizon,
and compare the resulting distribution with
  * the Jiang nodes for the same band,
  * the theoretical prediction that the nodes should be uniform in log s
    (the power-law kernel is scale invariant, so every decade of relaxation
    rate needs the same number of exponentials),
  * the exact spectral density mu(s) of the operator.

Run:  python -m src.analyze && python -m src.analyze_kernels
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch
from scipy import stats

from .config import (ALPHAS_TRAIN, LAM, L_LONG, L_TRAIN, N_STATES, RESULTS,
                     TAU)
from .bench import load_model
from .models import S4D, S4DLinear, SOEFixed
from .bands import S_HI, S_LO_TRAIN, T_LONG, T_TRAIN, jiang_nodes


def _mode_importance(s: np.ndarray, c: np.ndarray, tau: float, L: int):
    """||c_j B_j A_j^n||_2 over n = 0..L-1  (B_j = (1-A_j)/s_j, ZOH)."""
    a = np.exp(-s * tau)
    b = -np.expm1(-s * tau) / s
    a2 = np.clip(a ** 2, 0.0, 1.0 - 1e-16)
    norm = np.sqrt((1.0 - a2 ** L) / (1.0 - a2))
    return np.abs(c) * b * norm


def extract_rates(name: str, device="cuda"):
    """Return dict(rates, importance, init_rates, per_alpha_coeffs)."""
    model, info = load_model(name, device)
    out = dict(name=name, n_params=info["n_params"])
    alphas = np.asarray(ALPHAS_TRAIN, dtype=np.float32)
    at = torch.as_tensor(alphas, device=device)
    aid = torch.arange(len(alphas), device=device)

    if isinstance(model, (SOEFixed, S4DLinear)):
        s = model.rates().detach().cpu().numpy().astype(np.float64)
        with torch.no_grad():
            c = model.coeffs(at, aid).detach().cpu().numpy().astype(np.float64)
        imp = np.stack([_mode_importance(s, c[i], TAU, L_LONG)
                        for i in range(len(alphas))])
        out.update(rates=s.tolist(), coeffs=c.tolist(),
                   importance=imp.mean(0).tolist(),
                   importance_per_alpha=imp.tolist(),
                   alphas=alphas.tolist(), kind="single_layer")
        if hasattr(model, "s_init"):
            out["init_rates"] = model.s_init.detach().cpu().numpy().tolist()
    elif isinstance(model, S4D):
        layers = []
        for li, lyr in enumerate(model.layers):
            ssm = lyr["ssm"]
            s = ssm.rates().detach().cpu().numpy().astype(np.float64)   # (H,N)
            w = (ssm.Cp * ssm.Bp).detach().cpu().numpy().astype(np.float64)
            a = np.exp(-s * TAU)
            b = -np.expm1(-s * TAU) / s
            a2 = np.clip(a ** 2, 0.0, 1.0 - 1e-16)
            imp = np.abs(w) * b * np.sqrt((1.0 - a2 ** L_LONG) / (1.0 - a2))
            layers.append(dict(layer=li, rates=s.ravel().tolist(),
                               importance=imp.ravel().tolist(),
                               init_rates=ssm.s_init.detach().cpu()
                               .numpy().ravel().tolist()))
        out.update(kind="deep", layers=layers,
                   rates=np.concatenate([np.asarray(l["rates"]) for l in layers]).tolist(),
                   importance=np.concatenate([np.asarray(l["importance"]) for l in layers]).tolist(),
                   init_rates=np.concatenate([np.asarray(l["init_rates"]) for l in layers]).tolist())
    else:
        out.update(kind="none")
    del model
    torch.cuda.empty_cache()
    return out


# ---------------------------------------------------------------------------
def log_uniformity_stats(s: np.ndarray, w: np.ndarray | None,
                         band: tuple[float, float]):
    """How close is the (importance-weighted) rate set to uniform in log s?"""
    s = np.asarray(s, dtype=np.float64)
    lo, hi = band
    inside = (s >= lo) & (s <= hi)
    res = dict(n=int(s.size), n_inside_band=int(inside.sum()),
               frac_inside_band=float(inside.mean()),
               s_min=float(s.min()), s_max=float(s.max()),
               decades_spanned=float(np.log10(s.max() / s.min())),
               slowest_time_scale=float(1.0 / s.min()))
    if w is not None:
        w = np.asarray(w, dtype=np.float64)
        keep = w > 0.01 * w.max()          # modes that actually matter
        res["n_significant"] = int(keep.sum())
        if keep.sum() > 2:
            ss = s[keep]
            res["sig_s_min"] = float(ss.min())
            res["sig_s_max"] = float(ss.max())
            res["sig_decades"] = float(np.log10(ss.max() / ss.min()))
            res["sig_per_decade"] = float(keep.sum() /
                                          max(1e-9, np.log10(ss.max() / ss.min())))
            u = (np.log(ss) - np.log(ss.min())) / (np.log(ss.max()) - np.log(ss.min()))
            res["ks_vs_loguniform"] = float(stats.kstest(u, "uniform").statistic)
            res["ks_pvalue"] = float(stats.kstest(u, "uniform").pvalue)
    return res


def qq_against_jiang(s_learned: np.ndarray, s_jiang: np.ndarray, n_q: int = 25):
    q = np.linspace(0.02, 0.98, n_q)
    return dict(q=q.tolist(),
                learned=np.quantile(np.log10(s_learned), q).tolist(),
                jiang=np.quantile(np.log10(s_jiang), q).tolist())


def main(names=None):
    rep = dict(band=dict(tau=TAU, T_train=T_TRAIN, T_long=T_LONG,
                         s_lo_train=S_LO_TRAIN, s_hi=S_HI,
                         s_needed_long=1.0 / T_LONG),
               models={})

    s_jiang_long, info_long = jiang_nodes(T_LONG, N_STATES)
    s_jiang_train, info_train = jiang_nodes(T_TRAIN, N_STATES)
    rep["jiang"] = dict(
        long=dict(s=s_jiang_long.tolist(), info=info_long,
                  per_decade=float(len(s_jiang_long) /
                                   np.log10(s_jiang_long.max() / s_jiang_long.min()))),
        train=dict(s=s_jiang_train.tolist(), info=info_train))

    names = names or ["soe_jiang", "soe_jiang_trainband", "soe_jiang_table",
                      "s4d_lin_log", "s4d_lin_linear", "s4d_lin_narrow",
                      "s4d_lin_softplus", "s4d_lin_log_longtrain",
                      "s4d_lin_narrow_longtrain", "s4d_deep"]
    for nm in names:
        print(f"[analyze] {nm}")
        d = extract_rates(nm)
        s = np.asarray(d["rates"])
        w = np.asarray(d["importance"]) if "importance" in d else None
        d["stats_all"] = log_uniformity_stats(s, w, (s_jiang_long.min(),
                                                     s_jiang_long.max()))
        d["qq_vs_jiang"] = qq_against_jiang(s, s_jiang_long)
        if w is not None:
            keep = w > 0.01 * w.max()
            if keep.sum() > 2:
                d["qq_sig_vs_jiang"] = qq_against_jiang(s[keep], s_jiang_long)
        rep["models"][nm] = d
        st = d["stats_all"]
        print(f"    rates: {st['n']}  span {st['s_min']:.2e}..{st['s_max']:.2e} "
              f"({st['decades_spanned']:.1f} decades), "
              f"significant {st.get('n_significant', '-')}, "
              f"per-decade {st.get('sig_per_decade', float('nan')):.1f}, "
              f"KS vs log-uniform {st.get('ks_vs_loguniform', float('nan')):.3f}")

    with open(os.path.join(RESULTS, "analysis.json"), "w") as fh:
        json.dump(rep, fh, indent=2)
    print("wrote results/analysis.json "
          "(now run `python -m src.analyze_kernels` for the kernel section)")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=None, help="comma separated subset")
    a = ap.parse_args()
    main(a.models.split(",") if a.models else None)
