"""Extração de features estatísticas por janela (para modelos clássicos de ML)."""

import numpy as np
import pandas as pd
from scipy import stats

STAT_NAMES = [
    "mean", "std", "var", "min", "max", "median", "skew", "kurtosis",
    "q25", "q75", "iqr", "amplitude", "energy",
    "autocorr_lag1", "autocorr_lag2", "coeff_var",
]


def extract_window_features(
    X: np.ndarray, feature_names: list[str], index=None
) -> pd.DataFrame:
    """Calcula estatísticas de cada variável em cada janela.

    Args:
        X: janelas com shape (n_janelas, window_size, n_features).
        feature_names: nome de cada variável (última dimensão de X).
        index: índice opcional do DataFrame resultante (ex.: timestamps das janelas).

    Returns:
        DataFrame (n_janelas, n_features * len(STAT_NAMES)) com colunas `<variavel>_<estatistica>`.
        Valores indefinidos (ex.: autocorrelação de série constante) viram 0.
    """
    X = X.astype(np.float64)
    with np.errstate(all="ignore"):
        mean = X.mean(axis=1)
        std = X.std(axis=1, ddof=1)
        q25, q75 = np.percentile(X, [25, 75], axis=1)
        x_min, x_max = X.min(axis=1), X.max(axis=1)
        feats = {
            "mean": mean,
            "std": std,
            "var": X.var(axis=1, ddof=1),
            "min": x_min,
            "max": x_max,
            "median": np.median(X, axis=1),
            "skew": stats.skew(X, axis=1, bias=False),
            "kurtosis": stats.kurtosis(X, axis=1, fisher=True),
            "q25": q25,
            "q75": q75,
            "iqr": q75 - q25,
            "amplitude": x_max - x_min,
            "energy": (X**2).sum(axis=1),
            "autocorr_lag1": _autocorr(X, 1),
            "autocorr_lag2": _autocorr(X, 2),
            "coeff_var": np.where(mean != 0, std / mean, 0.0),
        }

    values = np.stack([feats[name] for name in STAT_NAMES], axis=2)  # (n, n_features, n_stats)
    columns = [f"{var}_{name}" for var in feature_names for name in STAT_NAMES]
    df = pd.DataFrame(values.reshape(len(X), -1), columns=columns, index=index)
    return df.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _autocorr(X: np.ndarray, lag: int) -> np.ndarray:
    """Autocorrelação de Pearson com defasagem `lag`, ao longo do eixo do tempo."""
    a = X[:, lag:] - X[:, lag:].mean(axis=1, keepdims=True)
    b = X[:, :-lag] - X[:, :-lag].mean(axis=1, keepdims=True)
    return (a * b).sum(axis=1) / np.sqrt((a**2).sum(axis=1) * (b**2).sum(axis=1))
