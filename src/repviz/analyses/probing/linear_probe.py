"""Linear probe, k-NN classifier, and few-shot evaluation."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearProbe(nn.Module):
    """Single linear layer probe on frozen features."""

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def train_linear_probe(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    num_classes: int,
    epochs: int = 100,
    lr: float = 0.01,
    batch_size: int = 256,
    weight_decay: float = 0.0,
    device: str = "cpu",
) -> tuple[LinearProbe, list[float]]:
    """Train a linear probe with SGD.

    Args:
        train_features: (N, D) frozen features.
        train_labels: (N,) integer labels.
        num_classes: Number of classes.
        epochs: Training epochs.
        lr: Learning rate.
        batch_size: Mini-batch size.
        weight_decay: L2 regularization.
        device: Device to train on.

    Returns:
        Trained probe and list of per-epoch losses.
    """
    D = train_features.shape[1]
    probe = LinearProbe(D, num_classes).to(device)
    optimizer = torch.optim.SGD(probe.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)

    features = train_features.float().to(device)
    labels = train_labels.long().to(device)
    N = features.shape[0]

    losses = []
    for epoch in range(epochs):
        perm = torch.randperm(N, device=device)
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, N, batch_size):
            idx = perm[i:i + batch_size]
            logits = probe(features[idx])
            loss = F.cross_entropy(logits, labels[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        losses.append(epoch_loss / max(n_batches, 1))

    return probe, losses


@torch.no_grad()
def evaluate_linear_probe(
    probe: LinearProbe,
    features: torch.Tensor,
    labels: torch.Tensor,
    device: str = "cpu",
) -> float:
    """Evaluate linear probe accuracy.

    Returns:
        Top-1 accuracy in [0, 1].
    """
    probe.eval()
    features = features.float().to(device)
    labels = labels.long().to(device)
    logits = probe(features)
    preds = logits.argmax(dim=1)
    return float((preds == labels).float().mean())


def few_shot_prototype(
    support_features: torch.Tensor,
    support_labels: torch.Tensor,
    query_features: torch.Tensor,
    query_labels: torch.Tensor,
) -> float:
    """Few-shot classification via prototype networks.

    Computes class prototypes from support set, classifies query by nearest prototype.

    Args:
        support_features: (N_support, D) support features.
        support_labels: (N_support,) support labels.
        query_features: (N_query, D) query features.
        query_labels: (N_query,) query labels.

    Returns:
        Top-1 accuracy on query set.
    """
    support_features = support_features.float()
    query_features = query_features.float()

    classes = support_labels.unique()
    prototypes = []
    for c in classes:
        mask = support_labels == c
        prototypes.append(support_features[mask].mean(dim=0))
    prototypes = torch.stack(prototypes)  # (C, D)

    # Classify by nearest prototype (cosine)
    prototypes = F.normalize(prototypes, dim=-1)
    query_norm = F.normalize(query_features, dim=-1)
    sim = query_norm @ prototypes.T  # (N_query, C)
    pred_idx = sim.argmax(dim=1)
    pred_labels = classes[pred_idx]

    return float((pred_labels == query_labels).float().mean())
