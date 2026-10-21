"""Métricas e gráficos para comparar os classificadores."""

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def classification_metrics(y_true, y_pred, y_score) -> dict[str, float]:
    """Métricas focadas na classe de falha (positiva).

    Em dados muito desbalanceados, métricas "weighted" ou a acurácia simples ficam altas
    mesmo quando o modelo nunca detecta uma falha; por isso usamos a classe positiva,
    a acurácia balanceada e métricas de ranqueamento (ROC AUC e PR AUC).
    """
    has_both = len(np.unique(y_true)) > 1
    return {
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_score) if has_both else math.nan,
        "pr_auc": average_precision_score(y_true, y_score) if has_both else math.nan,
    }


def plot_history(history: dict[str, list[float]], title: str = "", metrics=("loss", "precision", "recall", "auc")):
    """Curvas de treino e validação ao longo das épocas."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for ax, metric in zip(axes.flat, metrics):
        ax.plot(history[metric], label="treino")
        ax.plot(history[f"val_{metric}"], label="validação")
        ax.set_title(metric.capitalize())
        ax.set_xlabel("Época")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    plt.show()


def plot_roc_curves(y_true, scores: dict[str, np.ndarray], title: str = "Curvas ROC"):
    """Sobrepõe as curvas ROC de vários modelos."""
    plt.figure(figsize=(7, 7))
    for name, y_score in scores.items():
        fpr, tpr, _ = roc_curve(y_true, y_score)
        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {roc_auc_score(y_true, y_score):.3f})")
    plt.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
    plt.xlabel("Taxa de falsos positivos")
    plt.ylabel("Taxa de verdadeiros positivos")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.show()


def plot_predictions(timestamps, y_true, y_score, title: str = ""):
    """Probabilidade (ou score) prevista vs. rótulo real ao longo do tempo."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=timestamps, y=y_score, name="Previsto", mode="lines", line=dict(color="black")))
    fig.add_trace(go.Scatter(x=timestamps, y=y_true, name="Real", mode="lines", line=dict(color="red")))
    fig.update_layout(template="plotly_white", hovermode="x unified", title=title)
    fig.show()


def plot_signal(df: pd.DataFrame, variable: str, label_col: str = "Falhas", timestamp_col: str = "Timestamp"):
    """Plota uma variável e, em um segundo eixo, os períodos rotulados como falha."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df[timestamp_col], y=df[variable], name=variable, mode="lines", line=dict(color="black")))
    fig.add_trace(
        go.Scatter(x=df[timestamp_col], y=df[label_col], name=label_col, mode="lines", line=dict(color="red"), yaxis="y2")
    )
    fig.update_layout(
        template="plotly_white",
        hovermode="x unified",
        yaxis2=dict(overlaying="y", side="right", range=[-0.05, 1.05], showgrid=False),
    )
    fig.show()
