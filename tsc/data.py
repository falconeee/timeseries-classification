"""Carregamento, rotulagem e pré-processamento dos dados brutos."""

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

Period = tuple[str | pd.Timestamp, str | pd.Timestamp]


def load_dataset(
    path: str | Path,
    timestamp_col: str = "Timestamp",
    drop_columns: Iterable[str] = (),
    sep: str = ";",
    decimal: str = ".",
    encoding: str = "utf-8-sig",
    timestamp_format: str | None = None,
) -> pd.DataFrame:
    """Lê um CSV com uma coluna de timestamp e uma coluna por variável (sensor).

    Colunas em `drop_columns` (constantes, irrelevantes, com muitos NaN...) são removidas
    e linhas com valores ausentes são descartadas.
    """
    df = pd.read_csv(path, sep=sep, decimal=decimal, encoding=encoding)
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], format=timestamp_format)
    df = df.drop(columns=list(drop_columns), errors="ignore").dropna()
    return df.sort_values(timestamp_col).reset_index(drop=True)


def label_periods(
    df: pd.DataFrame,
    periods: Iterable[Period],
    timestamp_col: str = "Timestamp",
    label_col: str = "Falhas",
) -> pd.DataFrame:
    """Cria a coluna de rótulo: 1 dentro dos períodos de falha (inclusive), 0 fora."""
    df = df.copy()
    ts = df[timestamp_col]
    label = np.zeros(len(df), dtype=int)
    for start, end in periods:
        label[((ts >= pd.Timestamp(start)) & (ts <= pd.Timestamp(end))).to_numpy()] = 1
    df[label_col] = label
    return df


def remove_off_periods(
    df: pd.DataFrame,
    variable: str,
    threshold: float,
    timestamp_col: str = "Timestamp",
    min_duration: float = 0,
    pre_cut: float = 0,
    post_cut: float = 0,
) -> pd.DataFrame:
    """Remove os períodos em que o equipamento está desligado e os transitórios ao redor.

    O equipamento é considerado desligado quando `variable <= threshold` por pelo menos
    `min_duration` minutos. Além do período desligado, são removidos `pre_cut` minutos antes
    e `post_cut` minutos depois (partida/parada). Trechos ligados com menos de `min_duration`
    minutos entre dois desligamentos também são removidos.
    """
    df = df.sort_values(timestamp_col).reset_index(drop=True)
    ts = df[timestamp_col].to_numpy()
    off = (df[variable] <= threshold).to_numpy()
    if not off.any():
        return df

    min_duration = pd.Timedelta(minutes=min_duration)
    pre_cut = pd.Timedelta(minutes=pre_cut)
    post_cut = pd.Timedelta(minutes=post_cut)
    removed = np.zeros(len(df), dtype=bool)

    # Desligamentos: cada trecho contínuo com off=True vai até a primeira amostra ligada seguinte
    for start_idx, end_idx in _runs(off):
        start = ts[start_idx]
        end = ts[min(end_idx + 1, len(ts) - 1)]
        if end - start >= min_duration:
            removed |= (ts >= start - pre_cut) & (ts <= end + post_cut)

    # Picos curtos de funcionamento entre desligamentos também são descartados
    if min_duration > pd.Timedelta(0):
        for start_idx, end_idx in _runs(~removed):
            if end_idx + 1 < len(ts) and ts[end_idx + 1] - ts[start_idx] < min_duration:
                removed[start_idx : end_idx + 1] = True

    return df.loc[~removed].reset_index(drop=True)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Índices (início, fim) inclusivos de cada trecho contínuo com `mask == True`."""
    padded = np.concatenate([[False], mask, [False]])
    edges = np.flatnonzero(np.diff(padded.astype(int)))
    return list(zip(edges[::2], edges[1::2] - 1))


def select_period(
    df: pd.DataFrame,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    timestamp_col: str = "Timestamp",
) -> pd.DataFrame:
    """Recorta o DataFrame entre `start` e `end` (inclusive)."""
    mask = (df[timestamp_col] >= pd.Timestamp(start)) & (df[timestamp_col] <= pd.Timestamp(end))
    return df.loc[mask].sort_values(timestamp_col).reset_index(drop=True)


def make_synthetic_dataset(
    start: str = "2024-01-01",
    n_days: int = 90,
    n_faults: int = 14,
    n_shutdowns: int = 6,
    seed: int = 42,
) -> tuple[pd.DataFrame, list[Period]]:
    """Gera um dataset sintético de sensores (1 amostra/min) com falhas e paradas.

    Útil para rodar o template sem dados reais. Retorna o DataFrame (sem rótulo) e a lista
    de períodos de falha, no mesmo formato usado para dados reais.

    Durante uma falha a pressão sobe gradualmente, a vazão cai, a temperatura aumenta e a
    vibração fica mais ruidosa. O nível não é afetado (variável irrelevante de propósito).
    """
    rng = np.random.default_rng(seed)
    n = n_days * 24 * 60
    ts = pd.date_range(start, periods=n, freq="1min")
    t = np.arange(n)

    def smooth_noise(scale: float, span: int) -> np.ndarray:
        noise = rng.normal(0, 1, n + span)
        smoothed = np.convolve(noise, np.ones(span) / np.sqrt(span), mode="valid")[:n]
        return scale * smoothed

    daily = np.sin(2 * np.pi * t / (24 * 60))
    load = smooth_noise(0.3, 240)

    # Falhas espalhadas no tempo, com duração entre 2 e 24 horas
    fault = np.zeros(n)
    periods: list[Period] = []
    slots = np.linspace(0, n, n_faults + 1, dtype=int)
    for a, b in zip(slots[:-1], slots[1:]):
        duration = int(rng.integers(2 * 60, 24 * 60))
        begin = int(rng.integers(a, max(a + 1, b - duration)))
        end = min(begin + duration, n - 1)
        fault[begin:end] = np.linspace(0, 1, end - begin)  # degradação progressiva
        periods.append((ts[begin], ts[end - 1]))
    in_fault = fault > 0

    pressure = 3 + 0.3 * load + 0.1 * daily + rng.normal(0, 0.08, n) + 0.25 * fault
    flow = 120 + 10 * load + rng.normal(0, 2.0, n) - 4 * fault
    temperature = 70 + 2 * daily + load + rng.normal(0, 0.4, n) + 0.6 * fault
    vibration = 1 + rng.normal(0, 0.1, n) * np.where(in_fault, 1.6, 1)
    level = 40 + 5 * np.sin(2 * np.pi * t / (6 * 60)) + rng.normal(0, 0.5, n)
    current = 50 + 5 * load + rng.normal(0, 0.5, n)

    # Perturbações normais (não são falhas) para o problema não ficar trivial
    for idx in rng.integers(0, n - 120, size=80):
        duration = int(rng.integers(30, 120))
        pressure[idx : idx + duration] += rng.uniform(0.1, 0.4)
        vibration[idx : idx + duration] += rng.normal(0, 0.15, duration)

    # Paradas do equipamento: corrente ~0
    for idx in rng.integers(0, n - 600, size=n_shutdowns):
        current[idx : idx + int(rng.integers(180, 600))] = rng.normal(0.5, 0.2)

    df = pd.DataFrame(
        {
            "Timestamp": ts,
            "corrente_motor": current,
            "pressao": pressure,
            "vazao": flow,
            "temperatura": temperature,
            "vibracao": vibration,
            "nivel": level,
        }
    )
    return df, periods
