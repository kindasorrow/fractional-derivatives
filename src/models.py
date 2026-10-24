r"""The four sequence models that learn f -> y for D^a y = -lam y + f.

All of them are strictly causal and consume (f, alpha); alpha enters as an
extra input channel and, for the deep models, as FiLM modulation per layer.

    SOEFixed      fixed exponents from Jiang's formula, learned readout.
                  y_n = sum_j c_j(a) h_j[n],  h_j[n] = A_j h_j[n-1] + B_j f_n,
                  A_j = exp(-s_j tau) with s_j FROZEN at the Jiang nodes.
                  Exactly the hypothesis class of the true operator.

    S4DLinear     the same single-layer linear SSM, but the rates s_j are
                  LEARNED.  Same budget, same readout, only the pole locations
                  differ -- this is the model used to answer question 5.

    S4D           a deep diagonal SSM (the "S4D / Mamba-class" entry): stack of
                  [pre-norm -> diagonal SSM -> GELU -> pointwise MLP] blocks.

    TinyTransformer  small causal Transformer, pos in {rope, nope, alibi}.

Discretisation convention (shared by SOEFixed / S4DLinear / S4D) is
zero-order hold, so a learned rate s_j is directly comparable with a Jiang
node s_j (same units, same meaning: a relaxation rate in 1/time):

    A = exp(-s tau),    B = (1 - A)/s.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def fft_causal_conv(u: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Causal convolution along the last axis.

    u : (B, C, L) signal      k : (B, C, L) or (C, L) kernel, k[..., 0] is lag 0
    """
    L = u.shape[-1]
    n = 2 * L
    U = torch.fft.rfft(u.float(), n=n)
    K = torch.fft.rfft(k.float(), n=n)
    return torch.fft.irfft(U * K, n=n)[..., :L]


class AlphaEmbed(nn.Module):
    """Small embedding of the fractional order (shared by the deep models)."""

    def __init__(self, dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3, dim), nn.GELU(), nn.Linear(dim, dim))

    def forward(self, alpha: torch.Tensor) -> torch.Tensor:
        a = alpha.view(-1, 1).float()
        return self.net(torch.cat([a, a * a, 1.0 / (1.0 - a + 1e-3)], dim=-1))


class FiLM(nn.Module):
    def __init__(self, emb_dim: int, d_model: int):
        super().__init__()
        self.to_gb = nn.Linear(emb_dim, 2 * d_model)
        nn.init.zeros_(self.to_gb.weight)
        nn.init.zeros_(self.to_gb.bias)

    def forward(self, x: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        g, b = self.to_gb(e).chunk(2, dim=-1)          # (B, d) each
        return x * (1.0 + g[:, :, None]) + b[:, :, None]


# ---------------------------------------------------------------------------
# 1 & 2.  single-layer linear SSM: fixed (Jiang) or learned rates
# ---------------------------------------------------------------------------
class _SOEBase(nn.Module):
    """y_n = sum_j c_j(alpha) * h_j[n] with ZOH first-order modes."""

    def __init__(self, n_exp: int, tau: float, alpha_emb: int = 32,
                 readout: str = "mlp", n_alpha: int = 8):
        super().__init__()
        self.n_exp, self.tau, self.readout = n_exp, tau, readout
        if readout == "mlp":
            self.emb = AlphaEmbed(alpha_emb)
            self.to_c = nn.Linear(alpha_emb, n_exp)
            nn.init.normal_(self.to_c.weight, std=1e-2)
            nn.init.zeros_(self.to_c.bias)
        elif readout == "table":
            self.c = nn.Parameter(torch.zeros(n_alpha, n_exp))
        else:
            raise ValueError(readout)

    def rates(self) -> torch.Tensor:          # (n_exp,) positive
        raise NotImplementedError

    def coeffs(self, alpha, alpha_id):
        if self.readout == "mlp":
            return self.to_c(self.emb(alpha))             # (B, n_exp)
        return self.c[alpha_id.long()]

    def forward(self, f: torch.Tensor, alpha: torch.Tensor,
                alpha_id: torch.Tensor | None = None) -> torch.Tensor:
        B, L = f.shape
        s = self.rates()                                   # (n_exp,)
        st = s * self.tau
        A = torch.exp(-st)
        Bc = -torch.expm1(-st) / s                         # (1-A)/s
        n = torch.arange(L, device=f.device, dtype=torch.float32)
        V = torch.exp(-st[:, None] * n[None, :])            # (n_exp, L), A^n
        c = self.coeffs(alpha, alpha_id) * Bc[None, :]      # (B, n_exp)
        k = c @ V                                           # (B, L) conv kernel
        return fft_causal_conv(f[:, None, :], k[:, None, :])[:, 0, :]


class SOEFixed(_SOEBase):
    """Jiang nodes frozen as the exponents; only the readout is trained."""

    def __init__(self, s_nodes, tau: float, **kw):
        super().__init__(len(s_nodes), tau, **kw)
        self.register_buffer("s_fixed", torch.as_tensor(s_nodes, dtype=torch.float32))

    def rates(self):
        return self.s_fixed


class S4DLinear(_SOEBase):
    """Same architecture with LEARNED rates (the pole-discovery experiment)."""

    def __init__(self, n_exp: int, tau: float, s_lo: float, s_hi: float,
                 init: str = "log", rate_param: str = "log", **kw):
        super().__init__(n_exp, tau, **kw)
        self.rate_param = rate_param
        s0 = _init_rates(n_exp, s_lo, s_hi, init)
        self.register_buffer("s_init", s0.clone())
        if rate_param == "log":
            self.raw = nn.Parameter(torch.log(s0))
        elif rate_param == "softplus":
            self.raw = nn.Parameter(_inv_softplus(s0))
        else:
            raise ValueError(rate_param)

    def rates(self):
        if self.rate_param == "log":
            return torch.exp(self.raw)
        return F.softplus(self.raw) + 1e-8


def _inv_softplus(s: torch.Tensor) -> torch.Tensor:
    """log(exp(s)-1) computed as s + log1p(-exp(-s)) (fp32-safe for s ~ 100)."""
    return s + torch.log1p(-torch.exp(-s))


def _init_rates(n: int, s_lo: float, s_hi: float, init: str) -> torch.Tensor:
    """Rate initialisations.  'log' is the usual S4D(-inv) geometric spread;
    'lin' and 'narrow' are deliberately uninformative controls."""
    if init == "log":
        return torch.exp(torch.linspace(math.log(s_lo), math.log(s_hi), n))
    if init == "lin":
        return torch.linspace(s_hi / n, s_hi, n)
    if init == "narrow":
        return torch.full((n,), 1.0) * torch.exp(0.05 * torch.randn(n))
    raise ValueError(init)


# ---------------------------------------------------------------------------
# 3.  deep diagonal SSM
# ---------------------------------------------------------------------------
class S4DLayer(nn.Module):
    def __init__(self, d_model: int, n_states: int, tau: float,
                 s_lo: float, s_hi: float, init: str = "log",
                 rate_param: str = "log"):
        super().__init__()
        self.tau, self.rate_param = tau, rate_param
        s0 = torch.stack([_init_rates(n_states, s_lo, s_hi, init)
                          for _ in range(d_model)])          # (H, N)
        if rate_param == "log":
            self.raw = nn.Parameter(torch.log(s0))
        else:
            self.raw = nn.Parameter(_inv_softplus(s0))
        self.register_buffer("s_init", s0.clone())
        self.Bp = nn.Parameter(torch.randn(d_model, n_states) / math.sqrt(n_states))
        self.Cp = nn.Parameter(torch.randn(d_model, n_states) / math.sqrt(n_states))
        self.D = nn.Parameter(torch.zeros(d_model))

    def rates(self):
        if self.rate_param == "log":
            return torch.exp(self.raw)
        return F.softplus(self.raw) + 1e-8

    def kernel(self, L: int, chunk: int = 1024) -> torch.Tensor:
        s = self.rates()                                      # (H, N)
        st = s * self.tau
        Bz = -torch.expm1(-st) / s                             # ZOH input gain
        w = self.Cp * self.Bp * Bz                             # (H, N)
        # k[h, l] = sum_n w[h,n] exp(-s[h,n] tau l); chunked over l so that the
        # (H, N, L) Vandermonde is never materialised in full.
        out = []
        for a in range(0, L, chunk):
            n = torch.arange(a, min(a + chunk, L), device=s.device,
                             dtype=torch.float32)
            V = torch.exp(-st[..., None] * n)                  # (H, N, chunk)
            out.append(torch.einsum("hn,hnl->hl", w, V))
        return torch.cat(out, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:        # x: (B, H, L)
        k = self.kernel(x.shape[-1])
        return fft_causal_conv(x, k[None]) + self.D[None, :, None] * x


class S4D(nn.Module):
    def __init__(self, d_model: int = 64, n_states: int = 64, n_layers: int = 2,
                 tau: float = 0.01, s_lo: float = 0.1, s_hi: float = 100.0,
                 init: str = "log", rate_param: str = "log", emb_dim: int = 32):
        super().__init__()
        self.inp = nn.Linear(2, d_model)
        self.emb = AlphaEmbed(emb_dim)
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleDict(dict(
                norm=nn.LayerNorm(d_model),
                film=FiLM(emb_dim, d_model),
                ssm=S4DLayer(d_model, n_states, tau, s_lo, s_hi, init, rate_param),
                mlp=nn.Sequential(nn.Linear(d_model, 2 * d_model), nn.GELU(),
                                  nn.Linear(2 * d_model, d_model)),
            )))
        self.out_norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, 1)

    def forward(self, f, alpha, alpha_id=None):
        B, L = f.shape
        a = alpha.view(-1, 1, 1).expand(B, L, 1).float()
        x = self.inp(torch.stack([f, a[..., 0]], dim=-1))       # (B, L, d)
        e = self.emb(alpha)
        x = x.transpose(1, 2)                                   # (B, d, L)
        for lyr in self.layers:
            h = lyr["norm"](x.transpose(1, 2)).transpose(1, 2)
            h = lyr["film"](h, e)
            h = F.gelu(lyr["ssm"](h))
            h = lyr["mlp"](h.transpose(1, 2)).transpose(1, 2)
            x = x + h
        x = self.out_norm(x.transpose(1, 2))
        return self.out(x)[..., 0]


# ---------------------------------------------------------------------------
# 4.  tiny causal Transformer
# ---------------------------------------------------------------------------
def _rope(q, k, base: float = 10000.0):
    # q, k: (B, H, L, Dh) with Dh even
    B, H, L, Dh = q.shape
    half = Dh // 2
    inv = base ** (-torch.arange(0, half, device=q.device, dtype=torch.float32) / half)
    pos = torch.arange(L, device=q.device, dtype=torch.float32)
    ang = pos[:, None] * inv[None, :]                          # (L, half)
    cos, sin = torch.cos(ang)[None, None], torch.sin(ang)[None, None]

    def rot(x):
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

    return rot(q), rot(k)


def _alibi_bias(n_heads: int, L: int, device) -> torch.Tensor:
    slopes = 2.0 ** (-8.0 * (torch.arange(n_heads, device=device,
                                          dtype=torch.float32) + 1) / n_heads)
    d = torch.arange(L, device=device, dtype=torch.float32)
    dist = d[None, :] - d[:, None]                              # j - i
    bias = slopes[:, None, None] * dist[None]
    causal = torch.triu(torch.full((L, L), float("-inf"), device=device), 1)
    return (bias + causal[None])[None]                          # (1, H, L, L)


class Attn(nn.Module):
    def __init__(self, d_model: int, n_heads: int, pos: str):
        super().__init__()
        self.h, self.dh, self.pos = n_heads, d_model // n_heads, pos
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x):                                       # (B, L, d)
        B, L, d = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(B, L, self.h, self.dh).transpose(1, 2) for t in (q, k, v)]
        if self.pos == "rope":
            q, k = _rope(q, k)
        if self.pos == "alibi":
            o = F.scaled_dot_product_attention(
                q, k, v, attn_mask=_alibi_bias(self.h, L, x.device))
        else:
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(o.transpose(1, 2).reshape(B, L, d))


class TinyTransformer(nn.Module):
    def __init__(self, d_model: int = 64, n_heads: int = 4, n_layers: int = 2,
                 pos: str = "rope", emb_dim: int = 32):
        super().__init__()
        self.inp = nn.Linear(2, d_model)
        self.emb = AlphaEmbed(emb_dim)
        self.blocks = nn.ModuleList()
        for _ in range(n_layers):
            self.blocks.append(nn.ModuleDict(dict(
                n1=nn.LayerNorm(d_model),
                film=FiLM(emb_dim, d_model),
                attn=Attn(d_model, n_heads, pos),
                n2=nn.LayerNorm(d_model),
                mlp=nn.Sequential(nn.Linear(d_model, 2 * d_model), nn.GELU(),
                                  nn.Linear(2 * d_model, d_model)),
            )))
        self.out_norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, 1)

    def forward(self, f, alpha, alpha_id=None):
        B, L = f.shape
        a = alpha.view(-1, 1).expand(B, L).float()
        x = self.inp(torch.stack([f, a], dim=-1))
        e = self.emb(alpha)
        for blk in self.blocks:
            h = blk["film"](blk["n1"](x).transpose(1, 2), e).transpose(1, 2)
            x = x + blk["attn"](h)
            x = x + blk["mlp"](blk["n2"](x))
        return self.out(self.out_norm(x))[..., 0]


# ---------------------------------------------------------------------------
def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
