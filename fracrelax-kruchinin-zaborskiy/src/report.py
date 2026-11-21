r"""Turn results/*.json into the markdown tables used by the README.

Torch-free (reads json only).  Run:  python -m src.report
Writes results/tables.md and prints it.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

from .config import L_SCALING, L_TRAIN, RESULTS
from .jsonio import load_split

ORDER = ["transformer_rope", "transformer_nope", "transformer_alibi",
         "s4d_deep", "s4d_lin_log", "s4d_lin_linear", "s4d_lin_narrow",
         "s4d_lin_softplus", "s4d_lin_log_longtrain",
         "s4d_lin_narrow_longtrain", "s4d_lin_log_lsinit", "s4d_lin_log_lsinit_table", "soe_jiang",
         "soe_jiang_table", "soe_jiang_trainband", "soe_jiang_lsinit", "soe_jiang_lsinit_table"]

# What each model is told about the TEST horizon (T = 100 s, L = 10^4), so the
# accuracy table reads unambiguously.  "no" = it only ever saw / was built for
# the training horizon T = 10 s (L = 1000).
KNOWS_T100 = {
    "soe_jiang": "yes: nodes built for T=100",
    "soe_jiang_table": "yes: nodes built for T=100",
    "soe_jiang_lsinit": "yes: nodes + LS weights for T=100",
    "soe_jiang_lsinit_table": "yes: nodes + LS weights for T=100",
    "soe_jiang_trainband": "no: nodes built for T=10",
    "s4d_lin_log_lsinit": "partly: LS weights fitted on L=1000 kernel",
    "s4d_lin_log_lsinit_table": "partly: LS weights fitted on L=1000 kernel",
    "s4d_lin_log_longtrain": "yes: trained on L=10^4",
    "s4d_lin_narrow_longtrain": "yes: trained on L=10^4",
}
KNOWS_DEFAULT = "no"

L1_NOTE = ("> **Reference.** Every error below is relative to the **L1 solution** "
           "(the training target), not to the continuous equation. The L1 scheme "
           "itself deviates from the exact kernel K(t) by +70% at t=tau, 7.6% at "
           "t=0.1, 1.5% at t=0.7 and ~0.1% only by t~12 (`results/validation.json`), "
           "so these numbers say how well a model learns the *L1 operator*. For "
           "statements about the equation, regenerate the data on a 10x finer grid "
           "and decimate.\n")

UNSEEN_ALPHA_NOTE = ("> **Unseen α.** `*_table` models use one independent "
                     "readout vector per training α and define no interpolation "
                     "for unseen α. Their `unseen α` entries are `N/A` and are "
                     "excluded from generalisation comparisons.\n")


def _j(name):
    p = os.path.join(RESULTS, name)
    return load_split(p) if os.path.exists(p) else None


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


def _ms(vals, nd=2):
    """mean +/- sd over seeds (n>=2), or the single value."""
    v = np.asarray([x for x in vals if x is not None], float)
    if v.size == 0:
        return "--"
    if v.size == 1:
        return _fmt(float(v[0]), nd)
    return f"{v.mean():.{nd}e} ± {v.std(ddof=1):.1e}"


def table_main(b):
    """Accuracy vs length + held-out alpha.

    Models with several seeds show mean ± sd over seeds (the `seeds` column says
    how many); the rest are single runs.  `knows T=100?` says whether the model
    was built with / trained on the test horizon.
    """
    hdr = (["model", "knows T=100?", "params", "seeds"] +
           [f"L={L}" for L in L_SCALING] + ["unseen a"])
    rows = []
    for nm in ORDER:
        m = b["models"].get(nm)
        if m is None:
            continue
        runs = m.get("seed_runs") or {}
        acc = {r["L"]: r["acc"].get("med") for r in m["scaling"]}
        if len(runs) >= 2:
            cells = [_ms([r["med_by_L"].get(str(L)) for r in runs.values()])
                     for L in L_SCALING]
            ho = ("N/A" if nm.endswith("_table") else
                  _ms([r["heldout_all_med"] for r in runs.values()]))
        else:
            cells = [_fmt(acc.get(L)) for L in L_SCALING]
            ho = ("N/A" if nm.endswith("_table") else
                  _fmt(m.get("heldout_alpha", {}).get("all_med")))
        rows.append([m["label"], KNOWS_T100.get(nm, KNOWS_DEFAULT),
                     f"{m['n_params']:,}", str(max(1, len(runs)))] + cells + [ho])
    ref = b.get("soe_ls")
    if ref:
        rows.append(["*Jiang nodes + LS weights (no training)*",
                     "yes: nodes + LS weights for T=100", "--", "--"] +
                    [_fmt(ref[str(L)]["med"]) for L in L_SCALING] + ["--"])
    return hdr, rows


def table_mixed(b):
    """Error on mixed_lengths.npz: random lengths in [1000, 10000], no padding."""
    rows, bins = [], None
    for nm in ORDER:
        mx = b["models"].get(nm, {}).get("mixed")
        if not mx or "med" not in mx:
            continue
        if bins is None:
            bins = list(mx["by_length"])
        rows.append([b["models"][nm]["label"], _fmt(mx["med"])] +
                    [_fmt(mx["by_length"].get(k)) for k in bins])
    return (["model", "median"] + [f"len {k}" for k in bins], rows) if bins else (None, None)


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
            m["label"] + (" †" if nm == "transformer_alibi" else ""),
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



def table_lsinit(d):
    """Why LS initialisation works on Jiang nodes and not on the S4D band."""
    hdr = ["pole set", "poles", "max abs c", "cancellation", "kernel err (fp64)",
           "kernel err (fp32)"]
    rows = []
    for key, b in d["static"].items():
        r = max(b["per_alpha"], key=lambda r: r["c_abs_max"])
        rows.append([f"{key} -- {b['note']}", str(b["n_exp"]),
                     _fmt(r["c_abs_max"]), _fmt(r["cancellation"]),
                     _fmt(r["kernel_rel_l2_fp64"]), _fmt(r["kernel_rel_l2_fp32"])])
    return hdr, rows


def table_lsinit_dynamic(d):
    hdr = ["run", "rates", "val at LS start", "val after 2000 steps",
           "grad norm at LS start", "max rate drift"]
    rows = []
    for rec in d["dynamic"]:
        t = rec["trajectory"]
        rows.append([rec["label"],
                     "learnable" if rec["lr_rate"] > 0 else "frozen",
                     _fmt(t[0]["val"]), _fmt(t[-1]["val"]),
                     _fmt(rec["grad_norm_at_ls"]),
                     f"{rec['rate_drift_max']:.2f} decades"])
    return hdr, rows


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
    parts.append(L1_NOTE)
    parts.append(UNSEEN_ALPHA_NOTE)
    if b:
        parts.append("### Accuracy vs sequence length "
                     f"(median relative $L_2$ vs the L1 solution; trained only at L={L_TRAIN})\n")
        parts.append(md(*table_main(b)))
        parts.append("\n`knows T=100?` marks which models were built with, or "
                     "trained on, the test horizon (T = 100 s): `soe_jiang*` get "
                     "their nodes from Jiang's rule for T=100, the `*_longtrain` "
                     "models are trained on L=10^4 sequences. Everything marked "
                     "`no` only knows T=10 s. Rows with seeds > 1 are mean ± sd "
                     "over seeds (n = the `seeds` column).\n")
        mh, mr = table_mixed(b)
        if mh:
            parts.append("\n### Variable-length sequences "
                         "(`mixed_lengths.npz`, lengths uniform in [1000, 10000], "
                         "each run at its own length)\n")
            parts.append(md(mh, mr))
        parts.append("\n### Inference cost (RTX 4060 Ti, batch 1, fp32)\n")
        parts.append(md(*table_cost(b)))
        if b.get("alibi_impl"):
            parts.append("\n† ALiBi bias is built per block of 512 queries "
                         "(`models.alibi_attention`), so memory is O(H·512·L). "
                         "Compute is still O(L²): that part is the method's.\n")
        else:
            parts.append("\n† STALE: these ALiBi time/memory numbers come from "
                         "the original implementation that builds the full "
                         "(1, H, L, L) bias (1.6 GB at L=10^4; SDPA cannot use "
                         "its flash kernel with an arbitrary mask). They "
                         "measure the implementation, not the method. Re-run "
                         "`python -m src.bench` after training to refresh.\n")
        parts.append("\n### Classical solvers (CPU, numpy/scipy)\n")
        parts.append(md(*table_classical(b)))
        parts.append("\n### Error at L=10000 by forcing family\n")
        parts.append(md(*table_kinds(b)))
    ls = _j("lsinit_study.json")
    if ls:
        parts.append("\n### Why LS initialisation works on Jiang nodes and "
                     "not on the S4D band\n")
        parts.append(md(*table_lsinit(ls)))
        parts.append("\nThe worst alpha of the eight is shown. `cancellation` "
                     "is the sum of |term| over the modes divided by the "
                     "kernel value they produce, i.e. how much they must "
                     "cancel.\n")
        parts.append(md(*table_lsinit_dynamic(ls)))
        parts.append("\n**S4D LS-init is a negative result.** Its log-spaced pole "
                     "basis makes the weighted LS fit numerically unstable: "
                     "the largest coefficient is 1.28e+07, the worst cancellation "
                     "factor is 7.34e+08, and kernel error rises from 3.50e-04 "
                     "in float64 to 5.88e-02 in float32. The actual MLP-readout "
                     "run starts with validation error 9.03e+01, reaches "
                     "4.97e+04 by step 500 and 6.57e+05 by step 1000, and its "
                     "best recorded validation error after 40,000 steps is "
                     "4.47e+01. Its L=10000 test error, 1.21e+02, is far above "
                     "the 5.48e-03 of S4D without LS-init. The 2,000-step "
                     "rate-dynamics control uses a per-alpha table readout to "
                     "isolate pole learning: its initial gradient norm is "
                     "5.19e+05, and training with learnable rates raises "
                     "validation error from "
                     "2.74e-02 to 1.25e+03 while freezing rates ends at "
                     "1.04e-02. These results do not support LS-init as an "
                     "improvement for S4D.\n")
    if an:
        parts.append("\n### Learned relaxation rates vs Jiang's nodes\n")
        parts.append(md(*table_rates(an)))
        parts.append("\n### Implied impulse response, relative $L_2$ error "
                     "over the full 10^4 lags\n")
        parts.append(md(*table_kernels(an)))
    txt = "\n".join(parts)
    out = os.path.join(RESULTS, "tables.md")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(txt)
    # The tables carry non-ASCII (superscripts, +/-), and a Windows console --
    # or the pipe `src.pipeline` hands this stage -- is cp1251 here, not UTF-8.
    # The file above is the deliverable; never let echoing it fail the stage.
    try:
        print(txt)
    except UnicodeEncodeError:
        sys.stdout.reconfigure(errors="replace")
        print(txt)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
