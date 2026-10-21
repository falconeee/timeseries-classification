"""Loop de treinamento PyTorch para os modelos de `tsc.models`."""

import copy
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

METRICS = ["loss", "accuracy", "precision", "recall", "auc"]

# No macOS, torch, scikit-learn e XGBoost carregam cópias diferentes do OpenMP no mesmo processo,
# o que trava as operações paralelas do torch na CPU. No Mac o treino roda na GPU (MPS), então
# limitar o torch a uma thread de CPU evita o travamento sem custo relevante.
if sys.platform == "darwin":
    torch.set_num_threads(1)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def binary_focal_loss(
    logits: torch.Tensor, targets: torch.Tensor, gamma: float = 2.0, weights: torch.Tensor | None = None
) -> torch.Tensor:
    """Focal loss binária: dá mais peso aos exemplos difíceis, útil com classes desbalanceadas."""
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    loss = (1 - torch.exp(-bce)) ** gamma * bce
    if weights is not None:
        loss = loss * weights
    return loss.mean()


@dataclass
class TrainResult:
    best_model: nn.Module
    last_model: nn.Module
    best_epoch: int
    history: dict[str, list[float]] = field(default_factory=dict)


def train_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 100,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    gamma: float = 2.0,
    balance_classes: bool = True,
    monitor: str = "val_auc",
    lr_patience: int = 10,
    early_stopping_patience: int | None = None,
    checkpoint_path: str | Path | None = None,
    device: torch.device | None = None,
    seed: int = 42,
    verbose: bool = True,
) -> TrainResult:
    """Treina um classificador binário com AdamW, focal loss e ReduceLROnPlateau.

    O melhor modelo é escolhido pela métrica `monitor` no conjunto de validação
    ("val_auc" é maximizada, "val_loss" é minimizada) e, opcionalmente, salvo em
    `checkpoint_path`. `early_stopping_patience=None` desativa a parada antecipada.
    """
    torch.manual_seed(seed)
    device = device or get_device()
    model = model.to(device)

    class_weights = torch.ones(2)
    if balance_classes:
        classes = np.unique(y_train)
        weights = compute_class_weight("balanced", classes=classes, y=y_train)
        class_weights[torch.as_tensor(classes)] = torch.as_tensor(weights, dtype=torch.float32)
    class_weights = class_weights.to(device)

    loader = DataLoader(
        TensorDataset(torch.as_tensor(X_train, dtype=torch.float32), torch.as_tensor(y_train)),
        batch_size=batch_size,
        shuffle=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, eps=1e-7)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=lr_patience, min_lr=1e-6
    )

    maximize = monitor != "val_loss"
    best_score, best_epoch, best_state = -math.inf if maximize else math.inf, -1, None
    best_val_loss, epochs_without_improvement = math.inf, 0
    history = {name: [] for name in METRICS + [f"val_{m}" for m in METRICS]}

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, train_scores, train_targets = 0.0, [], []
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = binary_focal_loss(logits, yb.float(), gamma, class_weights[yb])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xb)
            train_scores.append(torch.sigmoid(logits).detach().cpu())
            train_targets.append(yb.cpu())

        train_metrics = _metrics(torch.cat(train_targets).numpy(), torch.cat(train_scores).numpy())
        train_metrics["loss"] = total_loss / len(loader.dataset)

        val_logits = _predict_logits(model, X_val, device=device)
        val_metrics = _metrics(y_val, torch.sigmoid(val_logits).numpy())
        val_metrics["loss"] = binary_focal_loss(val_logits, torch.as_tensor(y_val, dtype=torch.float32), gamma).item()

        for name in METRICS:
            history[name].append(train_metrics[name])
            history[f"val_{name}"].append(val_metrics[name])
        scheduler.step(val_metrics["loss"])

        score = val_metrics[monitor.removeprefix("val_")]
        improved = not math.isnan(score) and (score > best_score if maximize else score < best_score)
        if improved:
            best_score, best_epoch = score, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if checkpoint_path is not None:
                Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save(best_state, checkpoint_path)

        if verbose:
            print(
                f"Época {epoch:3d}/{epochs} | "
                + " | ".join(f"{k} {v[-1]:.4f}" for k, v in history.items() if k.startswith("val_"))
                + (" *" if improved else "")
            )

        if early_stopping_patience is not None:
            if val_metrics["loss"] < best_val_loss:
                best_val_loss, epochs_without_improvement = val_metrics["loss"], 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= early_stopping_patience:
                    if verbose:
                        print(f"Early stopping na época {epoch}.")
                    break

    best_model = copy.deepcopy(model)
    if best_state is None:
        print(f"Aviso: '{monitor}' nunca foi calculável; usando o modelo da última época.")
        best_epoch = epoch
    else:
        best_model.load_state_dict(best_state)
    return TrainResult(best_model=best_model, last_model=model, best_epoch=best_epoch, history=history)


def predict_proba(
    model: nn.Module, X: np.ndarray, batch_size: int = 512, device: torch.device | None = None
) -> np.ndarray:
    """Probabilidade de falha para cada janela de X."""
    return torch.sigmoid(_predict_logits(model, X, batch_size, device)).numpy()


@torch.no_grad()
def _predict_logits(
    model: nn.Module, X: np.ndarray, batch_size: int = 512, device: torch.device | None = None
) -> torch.Tensor:
    device = device or next(model.parameters()).device
    model.eval()
    X = torch.as_tensor(X, dtype=torch.float32)
    return torch.cat([model(X[i : i + batch_size].to(device)).cpu() for i in range(0, len(X), batch_size)])


def _metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    y_pred = (y_score >= 0.5).astype(int)
    return {
        "accuracy": float((y_pred == y_true).mean()),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "auc": roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else math.nan,
    }
