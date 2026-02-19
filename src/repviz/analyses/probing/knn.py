"""k-NN classification on frozen features."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def knn_classify(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_features: torch.Tensor,
    k: int = 20,
    temperature: float = 0.07,
) -> torch.Tensor:
    """k-NN classification using cosine similarity.

    Args:
        train_features: (N_train, D) normalized features.
        train_labels: (N_train,) integer labels.
        test_features: (N_test, D) normalized features.
        k: Number of neighbors.
        temperature: Softmax temperature for weighted voting.

    Returns:
        (N_test,) predicted labels.
    """
    train_features = F.normalize(train_features.float(), dim=-1)
    test_features = F.normalize(test_features.float(), dim=-1)

    num_classes = int(train_labels.max().item()) + 1

    # Compute similarities in chunks to save memory
    chunk_size = 256
    all_preds = []

    for i in range(0, test_features.shape[0], chunk_size):
        chunk = test_features[i:i + chunk_size]
        sim = chunk @ train_features.T  # (chunk, N_train)

        topk_sim, topk_idx = sim.topk(k, dim=1)  # (chunk, k)
        topk_labels = train_labels[topk_idx]  # (chunk, k)

        # Weighted voting
        weights = (topk_sim / temperature).softmax(dim=1)  # (chunk, k)

        # Accumulate votes per class
        votes = torch.zeros(chunk.shape[0], num_classes, device=chunk.device)
        votes.scatter_add_(1, topk_labels.long(), weights)

        preds = votes.argmax(dim=1)
        all_preds.append(preds)

    return torch.cat(all_preds)


def knn_accuracy(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    k: int = 20,
    temperature: float = 0.07,
) -> float:
    """Compute k-NN classification accuracy.

    Returns:
        Top-1 accuracy in [0, 1].
    """
    preds = knn_classify(train_features, train_labels, test_features, k, temperature)
    return float((preds == test_labels).float().mean())
