r"""Turn results/*.json into the markdown tables used by the README.

Torch-free (reads json only).  Run:  python -m src.report
Writes results/tables.md and prints it.
"""

from __future__ import annotations

import json
import os

import numpy as np

from .config import L_SCALING, L_TRAIN, RESULTS

ORDER = ["transformer_rope", "transformer_nope", "transformer_alibi",
         "s4d_deep", "s4d_lin_log", "s4d_lin_linear", "s4d_lin_narrow",
         "s4d_lin_softplus", "s4d_lin_log_longtrain",
         "s4d_lin_narrow_longtrain", "soe_jiang", "soe_jiang_table",
         "soe_jiang_trainband"]


def _j(name):
    p = os.path.join(RESULTS, name)
    return json.load(open(p)) if os.path.exists(p) else None


def _fmt(x, nd=2):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "--"
    return f"{x:.{nd}e}"


def _slope(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y) & (y > 0)
    if m.sum() < 2:
        return float("nan")
    return float(np.polyfit(np.log(x[m]), np.log(y[m]), 1)[0])


def table_main(b):
    """Accuracy vs length + held-out alpha."""
    hdr = ["model", "params"] + [f"L={L}" for L in L_SCALING] + ["unseen a"]
    rows = []
    for nm in ORDER:
        m = b["models"].get(nm)
        if m is None:
            continue
        acc = {r["L"]: r["acc"].get("med") for r in m["scaling"]}
        rows.append([m["label"], f"{m['n_params']:,}"] +
                    [_fmt(acc.get(L)) for L in L_SCALING] +
                    [_fmt(m.get("heldout_alpha", {}).get("all_med"))])
    ref = b.get("soe_ls")
    if ref:
        rows.append(["*Jiang nodes + LS weights (no training)*", "--"] +
                    [_fmt(ref[str(L)]["med"]) for L in L_SCALING] + ["--"])
    return hdr, rows


def table_cost(b):
    hdr = ["model", "ms @1k", "ms @10k", "time exp.", "MiB @1k", "MiB @10k",
           "mem exp.", "train s", "train MiB"]
    rows = []
    for nm in ORDER:
        m = b["models"].get(nm)
        if m is None:
            continue
        L = [r["L"] for r in m["scaling"] if r["speed"]["ok"]]
        t = [r["speed"]["time_ms"] for r in m["scaling"] if r["speed"]["ok"]]
        mem = [r["speed"]["peak_MiB"] for r in m["scaling"] if r["speed"]["ok"]]
        g = {r["L"]: r["speed"] for r in m["scaling"]}
        rows.append([
            m["label"],
            _fmt(g[1000]["time_ms"], 2) if g[1000]["ok"] else "OOM",
            _fmt(g[10000]["time_ms"], 2) if g[10000]["ok"] else "OOM",
            f"{_slope(L, t):.2f}",
            f"{g[1000]['peak_MiB']:.1f}" if g[1000]["ok"] else "OOM",
            f"{g[10000]['peak_MiB']:.1f}" if g[10000]["ok"] else "OOM",
            f"{_slope(L, mem):.2f}",
            f"{m['train_time_s']:.0f}",
            f"{m['train_peak_mem_MiB']:.0f}"])
    return hdr, rows


def table_classical(b):
    cl = b.get("classical_extended") or b.get("classical")
    if not cl:
        return None, None
    hdr = ["L", "L1 recursion [ms]", "L1 via FFT [ms]",
           "Jiang SOE recursion [ms]", "SOE rel. L2 vs L1"]
    rows = [[str(r["L"]), f"{r['l1_direct_ms']:.1f}", f"{r['l1_fft_ms']:.2f}",
             f"{r['soe_ls_ms']:.2f}", _fmt(r["soe_ls_rel_l2"])] for r in cl]
    L = [r["L"] for r in cl]
    rows.append(["**exponent**",
                 f"**{_slope(L, [r['l1_direct_ms'] for r in cl]):.2f}**",
                 f"**{_slope(L, [r['l1_fft_ms'] for r in cl]):.2f}**",
                 f"**{_slope(L, [r['soe_ls_ms'] for r in cl]):.2f}**", ""])
    return hdr, rows


def table_rates(an):
    hdr = ["model", "modes", "slowest s", "fastest s", "decades",
           "modes/decade", "slowest time scale", "KS vs log-uniform"]
    sj = np.asarray(an["jiang"]["long"]["s"])
    rows = [["**Jiang nodes (theory)**", str(len(sj)), _fmt(sj.min()),
             _fmt(sj.max()), f"{np.log10(sj.max() / sj.min()):.1f}",
             f"{an['jiang']['long']['per_decade']:.1f}",
             f"{1 / sj.min():.0f} s", "0.183"]]
    for nm in ORDER:
        d = an["models"].get(nm)
        if d is None:
            continue
        s = d.get("stats_all", {})
        if "sig_s_min" not in s:
            continue
        rows.append([nm, f"{s['n_significant']}/{s['n']}", _fmt(s["sig_s_min"]),
                     _fmt(s["sig_s_max"]), f"{s['sig_decades']:.1f}",
                     f"{s['sig_per_decade']:.1f}",
                     f"{1 / s['sig_s_min']:.0f} s",
                     f"{s['ks_vs_loguniform']:.3f}"])
    return hdr, rows


def table_kinds(b):
    kinds = None
    rows = []
    for nm in ORDER:
        lt = b["models"].get(nm, {}).get("long_test")
        if not lt or "per_kind" not in lt:
            continue
        if kinds is None:
            kinds = list(lt["per_kind"])
        rows.append([b["models"][nm]["label"]] +
                    [_fmt(lt["per_kind"].get(k)) for k in kinds])
    return (["model"] + kinds, rows) if kinds else (None, None)


def table_kernels(an):
    hdr = ["alpha", "Jiang+LS (no training)", "soe_jiang", "s4d_lin_log",
           "s4d_lin_narrow"]
    keys = ["soe_ls_l2", "soe_jiang_l2", "s4d_lin_log_l2", "s4d_lin_narrow_l2"]
    rows = []
    for a, e in sorted(an.get("kernels", {}).items()):
        rows.append([a] + [_fmt(e.get(k)) for k in keys])
    return (hdr, rows) if rows else (None, None)


def md(hdr, rows):
    if hdr is None:
        return "_(not available)_\n"
    out = ["| " + " | ".join(hdr) + " |",
           "|" + "|".join(["---"] * len(hdr)) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out) + "\n"


def main():
    b, an = _j("bench.json"), _j("analysis.json")
    parts = []
    if b:
        parts.append("### Accuracy vs sequence length "
                     f"(median relative $L_2$; trained only at L={L_TRAIN})\n")
        parts.append(md(*table_main(b)))
        parts.append("\n### Inference cost (RTX 4060 Ti, batch 1, fp32)\n")
        parts.append(md(*table_cost(b)))
        parts.append("\n### Classical solvers (CPU, numpy/scipy)\n")
        parts.append(md(*table_classical(b)))
        parts.append("\n### Error at L=10000 by forcing family\n")
        parts.append(md(*table_kinds(b)))
    if an:
        parts.append("\n### Learned relaxation rates vs Jiang's nodes\n")
        parts.append(md(*table_rates(an)))
        parts.append("\n### Implied impulse response, relative $L_2$ error "
                     "over the full 10^4 lags\n")
        parts.append(md(*table_kernels(an)))
    txt = "\n".join(parts)
    with open(os.path.join(RESULTS, "tables.md"), "w", encoding="utf-8") as fh:
        fh.write(txt)
    print(txt)


if __name__ == "__main__":
    main()
