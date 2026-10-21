"""Janelamento das séries temporais e normalização das janelas."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def make_windows(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str = "Falhas",
    timestamp_col: str = "Timestamp",
    window_size: int = 60,
    step: int = 30,
    freq: str | None = "1min",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cria janelas deslizantes de `window_size` amostras, avançando `step` amostras.

    Cada janela recebe o rótulo 1 quando mais da metade das amostras é falha. Se `freq` for
    informado, janelas que atravessam lacunas no tempo (ex.: períodos removidos por estarem
    desligados) são descartadas.

    Returns:
        X: array (n_janelas, window_size, n_features)
        y: rótulo de cada janela
        timestamps: timestamp central de cada janela
    """
    values = df[feature_cols].to_numpy(dtype=np.float32)
    labels = df[label_col].to_numpy()
    ts = df[timestamp_col].to_numpy()

    starts = np.arange(0, len(df) - window_size + 1, step)
    if freq is not None:
        max_span = (pd.Timedelta(freq) * (window_size - 1)).to_timedelta64()
        starts = starts[ts[starts + window_size - 1] - ts[starts] <= max_span]

    idx = starts[:, None] + np.arange(window_size)
    X = values[idx]
    y = (labels[idx].mean(axis=1) > 0.5).astype(int)
    return X, y, ts[starts + window_size // 2]


def temporal_split(
    *arrays: np.ndarray, val_fraction: float = 0.2, gap: int = 0
) -> list[np.ndarray]:
    """Separa as últimas `val_fraction` janelas para validação, respeitando a ordem temporal.

    `gap` janelas são descartadas entre os dois conjuntos para evitar sobreposição.
    Retorna `[a_fit, a_val, b_fit, b_val, ...]`.
    """
    n = len(arrays[0])
    n_val = int(n * val_fraction)
    out = []
    for a in arrays:
        out += [a[: n - n_val - gap], a[n - n_val :]]
    return out


def fit_window_scaler(X: np.ndarray) -> StandardScaler:
    """Ajusta um StandardScaler por variável usando todas as amostras das janelas."""
    return StandardScaler().fit(X.reshape(-1, X.shape[-1]))


def scale_windows(scaler: StandardScaler, X: np.ndarray) -> np.ndarray:
    return scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape).astype(np.float32)
