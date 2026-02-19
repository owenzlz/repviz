#!/usr/bin/env python3
"""Comprehensive test script for all repviz analysis modules."""

from __future__ import annotations

import os
import sys
import time
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

results: dict[str, str] = {}

def run_test(name: str, fn):
    """Run a test function and record pass/fail."""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    t0 = time.time()
    try:
        fn()
        dt = time.time() - t0
        results[name] = f"PASS ({dt:.1f}s)"
        print(f"  ✅ PASS ({dt:.1f}s)")
    except Exception as e:
        dt = time.time() - t0
        results[name] = f"FAIL: {e}"
        print(f"  ❌ FAIL ({dt:.1f}s): {e}")
        traceback.print_exc()
    plt.close("all")


# ─── Setup: Load model and images ───────────────────────────────────

print("Loading DINOv2 ViT-S/14...", flush=True)
from repviz.models import load_dinov2
backbone = load_dinov2("dinov2_vits14", device="cpu")
print("Model loaded.", flush=True)

print("Downloading demo images...", flush=True)
from repviz.utils.data import make_dinov2_transform, inverse_normalize, URLImageDataset
from io import BytesIO
from PIL import Image
from urllib.request import urlopen
import sys; sys.stdout.flush()

DEMO_URLS = [
    "http://images.cocodataset.org/val2017/000000039769.jpg",
    "http://images.cocodataset.org/val2017/000000397133.jpg",
    "http://images.cocodataset.org/val2017/000000037777.jpg",
]

transform = make_dinov2_transform(224)
images_pil = []
images_tensor = []
for i, url in enumerate(DEMO_URLS):
    print(f"  Downloading image {i+1}/{len(DEMO_URLS)}...", flush=True)
    resp = urlopen(url, timeout=30)
    img = Image.open(BytesIO(resp.read())).convert("RGB")
    images_pil.append(img)
    images_tensor.append(transform(img))
print("Images downloaded.", flush=True)

batch = torch.stack(images_tensor)
print(f"Batch shape: {batch.shape}")

# Extract features using get_intermediate_layers (gives patch tokens + CLS)
print("Extracting features...", flush=True)
output = backbone.extract_with_attention(batch)

patch_features = output.patch_tokens  # (B, N, D)
cls_features = output.cls_token       # (B, D)
layer_features = output.intermediate_features
layer_features_full = layer_features

B, N, D = patch_features.shape
h = w = int(N ** 0.5)
print(f"Patches: {N} ({h}x{w}), dim={D}, layers={len(layer_features)}")

# ─── Phase 1: Geometry ───────────────────────────────────────────────

def test_pca():
    from repviz.analyses.geometry import compute_pca, pca_feature_map, batch_pca_feature_maps
    proj, pca = compute_pca(patch_features[0], n_components=3)
    assert proj.shape == (N, 3)
    fm, _ = pca_feature_map(patch_features[0], h, w)
    assert fm.shape == (h, w, 3)
    fms, _ = batch_pca_feature_maps(patch_features, h, w)
    assert fms.shape == (B, h, w, 3)
    from repviz.visualization import plot_pca_rgb
    img_np = inverse_normalize(batch[0])
    plot_pca_rgb(img_np, fms[0], save_path=OUTPUT_DIR / "pca_rgb.png")

def test_svd_spectrum():
    from repviz.analyses.geometry import singular_value_spectrum, effective_rank
    s = singular_value_spectrum(patch_features[0])
    assert len(s) > 0
    er = effective_rank(patch_features[0])
    assert 1.0 < er < D
    from repviz.visualization import plot_singular_values
    plot_singular_values(s, er, save_path=OUTPUT_DIR / "svd_spectrum.png")

def test_isotropy():
    from repviz.analyses.geometry import average_cosine_similarity, isotropy_score, feature_norms
    acs = average_cosine_similarity(patch_features[0])
    assert -1.0 <= acs <= 1.0
    iso = isotropy_score(patch_features[0])
    assert 0.0 <= iso <= 1.0
    norms = feature_norms(patch_features[0])
    assert norms.shape == (N,)

def test_intrinsic_dim():
    from repviz.analyses.geometry import two_nn_id, mle_id
    # Use a subset for speed
    feats = patch_features.reshape(-1, D)[:200]
    id_2nn = two_nn_id(feats)
    id_mle = mle_id(feats, k=5)
    print(f"  Two-NN ID: {id_2nn:.1f}, MLE ID: {id_mle:.1f}")
    assert not np.isnan(id_2nn)
    assert not np.isnan(id_mle)

run_test("Geometry: PCA", test_pca)
run_test("Geometry: SVD Spectrum", test_svd_spectrum)
run_test("Geometry: Isotropy", test_isotropy)
run_test("Geometry: Intrinsic Dimensionality", test_intrinsic_dim)

# ─── Phase 1: Dense ──────────────────────────────────────────────────

def test_cosine_sim():
    from repviz.analyses.dense import cosine_similarity_map, cross_image_similarity, patch_correspondence, patch_clustering
    sim = cosine_similarity_map(patch_features[0], query_idx=N//2, h=h, w=w)
    assert sim.shape == (h, w)
    xsim = cross_image_similarity(patch_features[0], patch_features[1], N//2, h, w)
    assert xsim.shape == (h, w)
    idx, scores = patch_correspondence(patch_features[0], patch_features[1])
    assert idx.shape == (N,)
    seg = patch_clustering(patch_features[0], n_clusters=5, h=h, w=w)
    assert seg.shape == (h, w)
    from repviz.visualization import plot_cosine_sim_map
    img_np = inverse_normalize(batch[0])
    plot_cosine_sim_map(img_np, sim, save_path=OUTPUT_DIR / "cosine_sim.png")

run_test("Dense: Cosine Similarity & Clustering", test_cosine_sim)

# ─── Phase 1: Attention ──────────────────────────────────────────────

def test_attention():
    from repviz.analyses.attention import attention_rollout, attention_distance, attention_entropy, head_similarity_matrix
    attn_maps = backbone.extract_attention_maps(batch[:1])
    assert len(attn_maps) > 0
    rollout = attention_rollout(attn_maps)
    first_key = sorted(attn_maps.keys())[0]
    attn0 = attn_maps[first_key]
    # attn0 is (B, H, N, N) where N = 1 + h*w (CLS + patches)
    dist = attention_distance(attn0, h=h, w=w)
    ent = attention_entropy(attn0)
    assert len(ent) > 0
    sim = head_similarity_matrix(attn0)
    assert sim.shape[0] == sim.shape[1]

run_test("Attention: Maps & Analysis", test_attention)

# ─── Phase 1: CKA ────────────────────────────────────────────────────

def test_cka():
    from repviz.analyses.layerwise import linear_cka, layerwise_cka_matrix
    score = linear_cka(patch_features[0], patch_features[1])
    assert 0 <= score <= 1.0 + 1e-6
    # Use a subset of layers for speed
    subset = {k: v for i, (k, v) in enumerate(sorted(layer_features.items())) if i % 4 == 0}
    matrix, names = layerwise_cka_matrix(subset)
    assert matrix.shape[0] == len(names)
    from repviz.visualization import plot_cka_matrix
    plot_cka_matrix(matrix, names, save_path=OUTPUT_DIR / "cka_matrix.png")

run_test("Layerwise: CKA", test_cka)

# ─── Phase 2: Progression ────────────────────────────────────────────

def test_progression():
    from repviz.analyses.layerwise import feature_norm_progression, effective_rank_progression, consecutive_layer_similarity
    norms = feature_norm_progression(layer_features_full)
    assert len(norms) > 0
    print(f"  Norm range: {min(norms.values()):.2f} - {max(norms.values()):.2f}")
    ranks = effective_rank_progression(layer_features_full)
    assert len(ranks) > 0
    print(f"  Rank range: {min(ranks.values()):.1f} - {max(ranks.values()):.1f}")
    # Use subset for speed
    subset = {k: v for i, (k, v) in enumerate(sorted(layer_features_full.items())) if i % 4 == 0}
    sim = consecutive_layer_similarity(subset, method="procrustes", max_samples=100)
    assert len(sim) > 0
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    layer_names = sorted(norms.keys())
    axes[0].plot([norms[n] for n in layer_names], 'o-')
    axes[0].set_title("Feature Norm Progression")
    axes[0].set_xlabel("Layer")
    layer_names_r = sorted(ranks.keys())
    axes[1].plot([ranks[n] for n in layer_names_r], 's-')
    axes[1].set_title("Effective Rank Progression")
    axes[1].set_xlabel("Layer")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "progression.png", dpi=100)

run_test("Layerwise: Progression", test_progression)

# ─── Phase 2: Intrinsic Dim per Layer ────────────────────────────────

def test_id_per_layer():
    from repviz.analyses.geometry import id_per_layer
    subset = {k: v for i, (k, v) in enumerate(sorted(layer_features_full.items())) if i % 4 == 0}
    ids = id_per_layer(subset, method="two_nn", max_samples=200)
    assert len(ids) > 0
    print(f"  ID per layer: {ids}")

run_test("Geometry: ID per Layer", test_id_per_layer)

# ─── Phase 3: Probing ────────────────────────────────────────────────

def test_linear_probe():
    from repviz.analyses.probing import train_linear_probe, evaluate_linear_probe, few_shot_prototype
    # Create synthetic labels (3 images = 3 classes)
    feats = cls_features  # (3, D)
    labels = torch.arange(3)
    probe, losses = train_linear_probe(feats, labels, num_classes=3, epochs=50, lr=0.1)
    acc = evaluate_linear_probe(probe, feats, labels)
    print(f"  Linear probe train acc: {acc:.2f}")
    assert acc > 0.0  # Should memorize 3 examples

def test_knn():
    from repviz.analyses.probing import knn_classify, knn_accuracy
    # Synthetic: use patch features as "train", cls as "test"-ish
    train_f = patch_features.reshape(-1, D)[:100]
    train_l = torch.randint(0, 5, (100,))
    test_f = patch_features.reshape(-1, D)[100:120]
    test_l = torch.randint(0, 5, (20,))
    preds = knn_classify(train_f, train_l, test_f, k=5)
    assert preds.shape == (20,)
    acc = knn_accuracy(train_f, train_l, test_f, test_l, k=5)
    print(f"  k-NN acc (random labels): {acc:.2f}")

def test_few_shot():
    from repviz.analyses.probing import few_shot_prototype
    # 3 images, 3 classes: support=first 2 patches per image, query=rest
    support_f = torch.cat([patch_features[i, :2] for i in range(min(B,3))])  # (2*B, D)
    support_l = torch.tensor([i for i in range(min(B,3)) for _ in range(2)])
    query_f = torch.cat([patch_features[i, 2:5] for i in range(min(B,3))])   # (3*B, D)
    query_l = torch.tensor([i for i in range(min(B,3)) for _ in range(3)])
    acc = few_shot_prototype(support_f, support_l, query_f, query_l)
    print(f"  Few-shot acc: {acc:.2f}")

run_test("Probing: Linear Probe", test_linear_probe)
run_test("Probing: k-NN", test_knn)
run_test("Probing: Few-shot Prototype", test_few_shot)

# ─── Phase 4: Robustness ─────────────────────────────────────────────

def test_augmentation_invariance():
    from repviz.analyses.robustness import augmentation_invariance_suite
    scores = augmentation_invariance_suite(backbone, batch)
    assert len(scores) > 0
    print(f"  Invariance scores: {scores}")
    fig, ax = plt.subplots(figsize=(8, 4))
    names = list(scores.keys())
    vals = [scores[n] for n in names]
    ax.barh(names, vals)
    ax.set_xlim(0, 1)
    ax.set_title("Augmentation Invariance (Cosine Similarity)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "augmentation_invariance.png", dpi=100)

def test_resolution_sensitivity():
    from repviz.analyses.robustness import resolution_sensitivity
    # Use only small resolutions for speed
    scores = resolution_sensitivity(backbone, images_pil, resolutions=[224, 518])
    assert len(scores) > 0
    print(f"  Resolution scores: {scores}")

run_test("Robustness: Augmentation Invariance", test_augmentation_invariance)
run_test("Robustness: Resolution Sensitivity", test_resolution_sensitivity)

# ─── Phase 4: Cross-model ────────────────────────────────────────────

def test_cross_model():
    from repviz.analyses.cross_model import cross_model_cka, cross_model_cka_matrix, cross_model_mutual_knn
    # Simulate two "models" with different feature subsets
    feats_a = cls_features
    feats_b = cls_features + torch.randn_like(cls_features) * 0.1
    cka = cross_model_cka(feats_a, feats_b)
    print(f"  Cross-model CKA (near-identical): {cka:.3f}")
    assert 0 <= cka <= 1.0 + 1e-6
    # Matrix
    model_feats = {"model_a": feats_a, "model_b": feats_b}
    matrix, names = cross_model_cka_matrix(model_feats)
    assert matrix.shape == (2, 2)
    # Mutual kNN
    mknn = cross_model_mutual_knn(feats_a, feats_b, k=2)
    print(f"  Mutual k-NN: {mknn:.3f}")

run_test("Cross-model: CKA & Mutual kNN", test_cross_model)

# ─── Phase 4: Neurons ────────────────────────────────────────────────

def test_neurons():
    from repviz.analyses.neurons import dead_neuron_fraction, channel_redundancy_summary, neuron_selectivity_index, topk_activating_patches
    flat_feats = patch_features.reshape(-1, D)
    dead = dead_neuron_fraction(flat_feats)
    print(f"  Dead neuron fraction: {dead:.4f}")
    assert 0 <= dead <= 1
    redundancy = channel_redundancy_summary(flat_feats)
    print(f"  Channel redundancy: {redundancy}")
    # Selectivity with synthetic labels
    labels = torch.randint(0, 3, (flat_feats.shape[0],))
    sel = neuron_selectivity_index(flat_feats, labels)
    assert sel.shape == (D,)
    # Top-k
    idx, vals = topk_activating_patches(flat_feats, dim_idx=0, k=5)
    assert len(idx) == 5

run_test("Neurons: Analysis", test_neurons)

# ─── Phase 4: Weights ────────────────────────────────────────────────

def test_weights():
    from repviz.analyses.weights import weight_distributions, weight_effective_rank, weight_spectral_analysis
    dists = weight_distributions(backbone.model)
    assert len(dists) > 0
    print(f"  Weight layers: {len(dists)}")
    ranks = weight_effective_rank(backbone.model)
    assert len(ranks) > 0
    # Full spectral analysis (subset for speed)
    spectral = weight_spectral_analysis(backbone.model)
    assert len(spectral) > 0
    # Plot one distribution
    first_name = list(dists.keys())[0]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(dists[first_name], bins=100, density=True)
    ax.set_title(f"Weight Distribution: {first_name}")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "weight_dist.png", dpi=100)
    # Plot spectral summary
    fig, ax = plt.subplots(figsize=(10, 4))
    names = sorted(ranks.keys())[:20]  # First 20 layers
    ax.bar(range(len(names)), [ranks[n] for n in names])
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.split(".")[-2] + "." + n.split(".")[-1] if "." in n else n for n in names], rotation=90, fontsize=6)
    ax.set_title("Weight Matrix Effective Rank")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "weight_ranks.png", dpi=100)

run_test("Weights: Spectral Analysis", test_weights)

# ─── Summary ──────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("  SUMMARY")
print("=" * 60)
passed = sum(1 for v in results.values() if v.startswith("PASS"))
failed = sum(1 for v in results.values() if v.startswith("FAIL"))
for name, result in results.items():
    icon = "✅" if result.startswith("PASS") else "❌"
    print(f"  {icon} {name}: {result}")
print(f"\n  Total: {passed} passed, {failed} failed out of {len(results)}")
print(f"  Outputs saved to: {OUTPUT_DIR}")

if failed > 0:
    sys.exit(1)
