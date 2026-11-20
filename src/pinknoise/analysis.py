"""Оценка спектра, подгонка наклона и вспомогательные функции для замера времени."""

from __future__ import annotations

import time
from typing import Callable

import numpy as np
from scipy import signal

FS = 44_100
FIT_LO, FIT_HI = 100.0, 10_000.0   # Гц, диапазон, используемый для подгонки наклона


def spectrum(x: np.ndarray, fs: float = FS, nperseg: int = 4096):
    """Спектральная плотность мощности по Уэлчу (окно Ханна, перекрытие 50 %).

    Бин нулевой частоты отбрасывается, чтобы результат можно было отобразить на логарифмических осях.
    """
    f, S = signal.welch(x, fs=fs, nperseg=nperseg, window="hann", detrend="constant")
    return f[1:], S[1:]


def fit_slope(f: np.ndarray, S: np.ndarray, lo: float = FIT_LO, hi: float = FIT_HI) -> float:
    """Наклон log10(S) от log10(f) на отрезке [lo, hi].

    Сначала спектр пересэмплируется на логарифмически равномерную сетку; без этого
    подгонка была бы смещена в сторону многочисленных близко расположенных высокочастотных бинов.
    Ожидаемый результат: -beta.
    """
    fg = np.geomspace(lo, hi, 200)
    Sg = np.interp(fg, f, S)
    return float(np.polyfit(np.log10(fg), np.log10(Sg), 1)[0])


def timeit(fn: Callable[[], np.ndarray], repeat: int = 5):
    """Возвращает (последний результат, лучшее время выполнения в миллисекундах)."""
    best, out = np.inf, None
    for _ in range(repeat):
        t = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t)
    return out, best * 1e3