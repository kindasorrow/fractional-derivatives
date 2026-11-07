r"""Fast CPU self-checks for the code paths that are easy to get subtly wrong.

    python -m src.selftest

1. every model is strictly causal (perturb the last input, earlier outputs fixed);
2. chunked ALiBi attention == the full (1, H, L, L)-bias reference;
3. the LS-initialised readout really starts at the least-squares solution:
   output error vs the L1 solution is ~1e-5 or better, for the table readout
   AND the alpha-MLP readout, at the training orders.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .bands import jiang_nodes, T_LONG
from .config import ALPHAS_TRAIN, LAM, TAU
from .l1 import l1_solve_fft
from .lsinit import apply_ls_init
from .models import (SOEFixed, S4D, S4DLinear, TinyTransformer, _alibi_bias,
                     alibi_attention)

torch.manual_seed(0)


def check_causal():
    L = 300
    f = torch.randn(2, L)
    a = torch.tensor([0.3, 0.7])
    aid = torch.tensor([1, 5])
    s, _ = jiang_nodes(T_LONG)
    models = {
        "soe": SOEFixed(s, TAU), "s4d_lin": S4DLinear(64, TAU, 0.1, 100.0),
        "s4d": S4D(16, 8, 2, TAU), "tf_rope": TinyTransformer(32, 4, 2, "rope"),
        "tf_nope": TinyTransformer(32, 4, 2, "nope"),
        "tf_alibi": TinyTransformer(32, 4, 2, "alibi"),
    }
    for nm, m in models.items():
        m.eval()
        with torch.no_grad():
            y0 = m(f, a, aid)
            f2 = f.clone()
            f2[:, -1] += 5.0
            y1 = m(f2, a, aid)
        d = (y0[:, :-1] - y1[:, :-1]).abs().max().item()
        assert d < 1e-5, (nm, d)
        print(f"  causal ok: {nm:8s} (max earlier-output change {d:.1e})")


def check_alibi():
    B, H, L, Dh = 2, 4, 700, 16          # L not a multiple of the block size
    q, k, v = (torch.randn(B, H, L, Dh) for _ in range(3))
    ref = F.scaled_dot_product_attention(q, k, v, attn_mask=_alibi_bias(H, L, q.device))
    out = alibi_attention(q, k, v, q_block=128)
    d = (ref - out).abs().max().item()
    assert d < 1e-5, d
    print(f"  alibi chunked == full: max diff {d:.1e}")


def check_lsinit():
    L = 1000
    s, _ = jiang_nodes(T_LONG)
    rng = np.random.default_rng(1)
    f = rng.standard_normal((len(ALPHAS_TRAIN), L))
    y_ref = np.stack([l1_solve_fft(f[i], float(a), LAM, TAU)
                      for i, a in enumerate(ALPHAS_TRAIN)])
    alpha = torch.tensor(ALPHAS_TRAIN, dtype=torch.float32)
    aid = torch.arange(len(ALPHAS_TRAIN))
    for readout in ("table", "mlp"):
        m = SOEFixed(s, TAU, readout=readout, n_alpha=len(ALPHAS_TRAIN))
        apply_ls_init(m, L_fit=10000)
        with torch.no_grad():
            yh = m(torch.as_tensor(f, dtype=torch.float32), alpha, aid).numpy()
        rel = np.linalg.norm(yh - y_ref, axis=1) / np.linalg.norm(y_ref, axis=1)
        assert rel.max() < 1e-4, (readout, rel)
        print(f"  LS init ({readout:5s}): rel L2 vs L1  max {rel.max():.1e}")


if __name__ == "__main__":
    print("[selftest] causality");    check_causal()
    print("[selftest] ALiBi");        check_alibi()
    print("[selftest] LS init");      check_lsinit()
    print("all checks passed")
