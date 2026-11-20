"""Графики для эксперимента."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

COLORS = {"FFT": "C0", "Grünwald–L.": "C1", "Exp. sum": "C2"}


def plot_methods(signals: dict, betas, fit_range, path=None):
    """Одна панель в log-log на каждый beta, три метода + идеальная линия 1/f^beta."""
    fig, axes = plt.subplots(1, len(betas), figsize=(15, 4.3))
    for ax, beta in zip(np.atleast_1d(axes), betas):
        for name, color in COLORS.items():
            f, S = signals[(beta, name)]
            ax.loglog(f, S, color=color, alpha=0.8, lw=1.2, label=name)
        f, S = signals[(beta, "FFT")]
        ref = S[np.argmin(abs(f - 1000))] * (f / 1000) ** (-beta)
        ax.loglog(f, ref, "k--", lw=1, label=rf"идеал $1/f^{{{beta}}}$")
        ax.axvspan(*fit_range, color="gray", alpha=0.08)
        ax.set(title=rf"$\beta={beta}$", xlabel="f, Гц", ylabel="СПМ")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=130)
    return fig


def plot_kellett(f_k, S_k, f_b, S_b, path=None):
    """Фильтр Келлета против нашего подогнанного банка (beta = 1) и отклонение от 1/f в дБ."""
    def dev(f, S):
        ref = S[np.argmin(abs(f - 1000))] * (f / 1000) ** -1.0
        return 10 * np.log10(S / ref)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].loglog(f_k, S_k, label="Келлет")
    ax[0].loglog(f_b, S_b, label="подогнанный банк")
    ax[0].set(xlabel="f, Гц", ylabel="СПМ", title="Спектры")
    ax[1].semilogx(f_k, dev(f_k, S_k), label="Келлет")
    ax[1].semilogx(f_b, dev(f_b, S_b), label="подогнанный банк")
    ax[1].set(xlabel="f, Гц", ylabel="отклонение от 1/f, дБ", ylim=(-6, 6),
              title="Отклонение (оценка Уэлча, зашумлено)")
    for a in ax:
        a.grid(alpha=0.3); a.legend()
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=130)
    return fig


def plot_audio_spectra(spectra: dict, path=None):
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, (f, S) in spectra.items():
        ax.loglog(f, S, label=name)
    ax.set(xlabel="f, Гц", ylabel="СПМ", title="Спектры аудио")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=130)
    return fig