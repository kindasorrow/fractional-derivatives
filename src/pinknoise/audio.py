"""Запись шума в wav-файлы."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import wavfile


def to_int16(x: np.ndarray, rms: float = 0.1, peak_limit: float = 0.95) -> np.ndarray:
    """Убрать постоянную составляющую, отмасштабировать до общего RMS (сопоставимая громкость), избежать клиппинга."""
    x = x - x.mean()
    x = x * (rms / x.std())
    peak = np.abs(x).max()
    if peak > peak_limit:
        x = x * (peak_limit / peak)
    return (x * 32767).astype(np.int16)


def save_wav(path: Path, x: np.ndarray, fs: int = 44_100) -> None:
    wavfile.write(path, fs, to_int16(x))