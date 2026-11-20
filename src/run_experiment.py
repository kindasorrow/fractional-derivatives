"""Запуск всего эксперимента: таблица наклонов, графики, wav-файлы.

    python run_experiment.py            # результаты в ./results и ./audio
    python run_experiment.py --show     # также открыть графики
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np

from src.pinknoise import (design_exp_bank, fit_slope, pink_exp, pink_fft,
                       pink_gl, pink_kellett, spectrum, timeit)
from src.pinknoise.analysis import FIT_HI, FIT_LO, FS
from src.pinknoise.audio import save_wav
from src.pinknoise.plotting import plot_audio_spectra, plot_kellett, plot_methods

N = 2 ** 16
BETAS = [0.5, 1.0, 1.5]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--show", action="store_true", help="показать графики")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=Path("."))
    args = ap.parse_args()
    if not args.show:
        matplotlib.use("Agg")

    res_dir, audio_dir = args.out / "results", args.out / "data"
    res_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    white = np.random.default_rng(args.seed).standard_normal(N)

    # ---- 1. три метода x три beta -------------------------------------
    rows, signals = [], {}
    for beta in BETAS:
        a, b = design_exp_bank(beta)          # проектирование фильтра не входит в замер времени
        methods = {
            "FFT": lambda: pink_fft(white, beta),
            "Grünwald–L.": lambda: pink_gl(white, beta),
            "Exp. sum": lambda: pink_exp(white, a, b),
        }
        for name, fn in methods.items():
            x, ms = timeit(fn)
            f, S = spectrum(x)
            signals[(beta, name)] = (f, S)
            rows.append((beta, name, fit_slope(f, S), ms))

    header = f"{'целевой beta':>11} | {'метод':<12} | {'наклон':>7} | {'ошибка':>7} | {'время, мс':>8}"
    lines = [header, "-" * len(header)]
    lines += [f"{b:>11.1f} | {m:<12} | {s:>7.3f} | {s + b:>+7.3f} | {t:>8.2f}" for b, m, s, t in rows]
    table = "\n".join(lines)
    print(table)
    (res_dir / "table.txt").write_text(table + "\n", encoding="utf-8")

    plot_methods(signals, BETAS, (FIT_LO, FIT_HI), res_dir / "spectra.png")

    # ---- 2. Келлет против подогнанного банка (beta = 1) -------------------------------------
    f_k, S_k = spectrum(pink_kellett(white))
    print(f"\nНаклон Келлета: {fit_slope(f_k, S_k):.3f}")
    plot_kellett(f_k, S_k, *signals[(1.0, "Exp. sum")], path=res_dir / "kellett.png")

    # ---- 3. аудио: 5 с белого, розового, броуновского -----------------------------------
    xi = np.random.default_rng(7).standard_normal(5 * FS)
    audio = {"white": pink_fft(xi, 0.0), "pink": pink_fft(xi, 1.0), "brownian": pink_fft(xi, 2.0)}
    spectra = {}
    for name, x in audio.items():
        save_wav(audio_dir / f"{name}.wav", x, FS)
        spectra[name] = spectrum(x)
        print(f"{name:9s} наклон = {fit_slope(*spectra[name]):+.3f}")
    plot_audio_spectra(spectra, res_dir / "audio_spectra.png")

    if args.show:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    main()