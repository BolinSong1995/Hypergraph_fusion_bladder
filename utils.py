from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int = 1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def cox_ph_loss(
    risk: torch.Tensor,
    time: torch.Tensor,
    event: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Negative Cox partial log-likelihood.

    Higher model output indicates higher risk.
    """
    order = torch.argsort(time, descending=True)
    risk = risk[order]
    event = event[order]

    log_cumulative_risk = torch.logcumsumexp(risk, dim=0)
    partial_log_likelihood = (risk - log_cumulative_risk) * event
    return -partial_log_likelihood.sum() / (event.sum() + eps)


def harrell_c_index(risk, time, event) -> float:
    """Harrell's concordance index.

    Higher risk score corresponds to shorter survival.
    Prediction ties receive 0.5 credit.
    """
    risk = np.asarray(risk, dtype=float)
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)

    comparable = 0.0
    concordant = 0.0

    for i in range(len(time)):
        for j in range(i + 1, len(time)):
            if time[i] < time[j] and event[i] == 1:
                comparable += 1
                if risk[i] > risk[j]:
                    concordant += 1
                elif risk[i] == risk[j]:
                    concordant += 0.5

            elif time[j] < time[i] and event[j] == 1:
                comparable += 1
                if risk[j] > risk[i]:
                    concordant += 1
                elif risk[i] == risk[j]:
                    concordant += 0.5

    return np.nan if comparable == 0 else concordant / comparable


def load_checkpoint(path, model, device="cpu"):
    """Load either a wrapped checkpoint or a plain model state_dict."""
    ckpt = torch.load(path, map_location=device)
    state = (
        ckpt["model_state_dict"]
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt
        else ckpt
    )
    model.load_state_dict(state, strict=True)
    return ckpt


def save_checkpoint(
    path,
    model,
    epoch,
    best_val_cindex,
    seed,
    model_config,
    train_config,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": int(epoch),
            "best_internal_val_cindex": float(best_val_cindex),
            "seed": int(seed),
            "architecture": "CT-Fusion hypergraph",
            "model_config": vars(model_config),
            "train_config": vars(train_config),
        },
        path,
    )
