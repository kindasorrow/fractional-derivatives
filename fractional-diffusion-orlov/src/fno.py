import torch
from torch import nn
from torch.nn import functional as F


class SpectralConv1d(nn.Module):
    def __init__(self, width, modes):
        super().__init__()
        self.modes = modes
        self.weight = nn.Parameter(torch.randn(width, width, modes, dtype=torch.cfloat) / width)

    def forward(self, values):
        spectrum = torch.fft.rfft(values)
        modes = min(self.modes, spectrum.shape[-1])
        transformed = torch.zeros_like(spectrum)
        transformed[:, :, :modes] = torch.einsum(
            'bik,iok->bok', spectrum[:, :, :modes], self.weight[:, :, :modes])
        return torch.fft.irfft(transformed, n=values.shape[-1])


class FourierBlock(nn.Module):
    def __init__(self, width, modes):
        super().__init__()
        self.spectral = SpectralConv1d(width, modes)
        self.pointwise = nn.Conv1d(width, width, 1)

    def forward(self, values):
        return F.gelu(self.spectral(values) + self.pointwise(values))


class FNO1d(nn.Module):
    def __init__(self, width=24, modes=24, depth=4):
        super().__init__()
        if min(width, modes, depth) < 1:
            raise ValueError('width, modes and depth must be positive')
        self.lift = nn.Conv1d(4, width, 1)
        self.blocks = nn.ModuleList([FourierBlock(width, modes) for _ in range(depth)])
        self.project = nn.Sequential(nn.Conv1d(width, width*2, 1), nn.GELU(),
                                     nn.Conv1d(width*2, 1, 1))

    def forward(self, initial, alpha, times):
        if initial.ndim != 2 or alpha.shape != initial.shape[:1] or times.shape != alpha.shape:
            raise ValueError('expected initial [batch,x], alpha and times [batch]')
        if initial.shape[-1] < 3:
            raise ValueError('at least three spatial nodes are required')
        if not torch.isfinite(initial).all():
            raise ValueError('initial values must be finite')
        if not torch.isfinite(alpha).all() or torch.any((alpha <= 0) | (alpha >= 1)):
            raise ValueError('alpha must be finite and strictly between 0 and 1')
        if not torch.isfinite(times).all() or torch.any(times <= 0):
            raise ValueError('times must be finite and strictly positive')
        x = torch.linspace(0, 1, initial.shape[-1], device=initial.device, dtype=initial.dtype)
        features = torch.stack((initial, x.expand_as(initial),
                                ((alpha-.2)/.7)[:, None].expand_as(initial),
                                (torch.log10(times)/2)[:, None].expand_as(initial)), dim=1)
        padding = max(1, initial.shape[-1] // 8)
        values = F.pad(self.lift(features), (0, padding))
        for block in self.blocks:
            values = block(values)
        values = self.project(values[:, :, :-padding])[:, 0]
        boundary = torch.zeros_like(values[:, :1])
        return torch.cat((boundary, values[:, 1:-1], boundary), dim=1)
