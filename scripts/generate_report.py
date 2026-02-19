#!/usr/bin/env python3
"""Generate a comprehensive, publication-quality HTML report for DINOv2 S and B models."""

from __future__ import annotations

import base64
import gc
import io
import os
import sys
import time
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import seaborn as sns
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

PROJECT_DIR = Path(__file__).parent.parent
ASSETS_DIR = PROJECT_DIR / "report_assets"
ASSETS_DIR.mkdir(exist_ok=True)

# ── Globals ──
figures = {}  # name -> {"path": Path, "b64": str, "caption": str, "explanation": str}
metrics = {}  # metric_name -> {"S": val, "B": val}
section_html = {}  # section_name -> html string

DEMO_URLS = [
    ("cats", "http://images.cocodataset.org/val2017/000000039769.jpg"),
    ("kitchen", "http://images.cocodataset.org/val2017/000000397133.jpg"),
    ("tennis", "http://images.cocodataset.org/val2017/000000037777.jpg"),
    ("elephant", "http://images.cocodataset.org/val2017/000000252219.jpg"),
    ("donut", "http://images.cocodataset.org/val2017/000000087038.jpg"),
    ("giraffe", "http://images.cocodataset.org/val2017/000000174482.jpg"),
    ("bus", "http://images.cocodataset.org/val2017/000000403385.jpg"),
    ("person", "http://images.cocodataset.org/val2017/000000006471.jpg"),
]

MODEL_NAMES = {
    "S": "dinov2_vits14",
    "B": "dinov2_vitb14",
    "L": "dinov2_vitl14",
}

TAGS = list(MODEL_NAMES.keys())
TAG_COLORS = {"S": "#2196F3", "B": "#009688", "L": "#E91E63"}
TAG_COLORS_LIGHT = {"S": "#64B5F6", "B": "#4DB6AC", "L": "#F48FB1"}


def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def save_fig(fig, name):
    path = ASSETS_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    b64 = fig_to_b64(fig)
    plt.close(fig)
    return path, b64


def add_figure(name, fig, caption, explanation):
    path, b64 = save_fig(fig, name)
    figures[name] = {"path": str(path), "b64": b64, "caption": caption, "explanation": explanation}


def p(msg):
    print(msg, flush=True)


# ── Download images ──
p("Downloading images...")
from repviz.utils.data import make_dinov2_transform, inverse_normalize
from PIL import Image
from urllib.request import urlopen
from io import BytesIO

transform = make_dinov2_transform(518)
images_pil = {}
images_tensor = {}
for label, url in DEMO_URLS:
    p(f"  {label}...")
    try:
        resp = urlopen(url, timeout=30)
        img = Image.open(BytesIO(resp.read())).convert("RGB")
        images_pil[label] = img
        images_tensor[label] = transform(img)
    except Exception as e:
        p(f"  WARN: Failed to download {label}: {e}")

image_labels = list(images_pil.keys())
p(f"Downloaded {len(image_labels)} images: {image_labels}")


# ── Load models one at a time, run analyses ──
from repviz.models import load_dinov2

model_data = {}  # "S"/"B" -> dict of extracted data

for tag, model_name in MODEL_NAMES.items():
    p(f"\n{'='*60}")
    p(f"Loading {model_name} ({tag})...")
    p(f"{'='*60}")
    backbone = load_dinov2(model_name, device="cpu")
    
    data = {
        "backbone": backbone,
        "model_name": model_name,
        "embed_dim": backbone.embed_dim,
        "num_layers": backbone.num_layers,
        "patch_tokens": {},
        "cls_tokens": {},
        "intermediate": {},
        "attention_maps": {},
    }
    
    # Extract features image by image
    for label in image_labels:
        p(f"  Extracting features for {label}...")
        tensor = images_tensor[label].unsqueeze(0)
        output = backbone.extract_with_attention(tensor)
        data["patch_tokens"][label] = output.patch_tokens.detach().cpu()
        data["cls_tokens"][label] = output.cls_token.detach().cpu()
        data["intermediate"][label] = {k: v.detach().cpu() for k, v in output.intermediate_features.items()}
        data["attention_maps"][label] = {k: v.detach().cpu() for k, v in output.attention_maps.items()}
    
    model_data[tag] = data
    p(f"  embed_dim={backbone.embed_dim}, num_layers={backbone.num_layers}")

# Compute spatial dims from first image
first_label = image_labels[0]
N = int(model_data["S"]["patch_tokens"][first_label].shape[1])
h = w = int(round(N ** 0.5))
assert h * w == N, f"Non-square grid: {h}x{w} != {N}"
p(f"Patch grid: {h}x{w} = {N} patches")


# ══════════════════════════════════════════════════════════════════════
#  SECTION 1: PCA Feature Maps
# ══════════════════════════════════════════════════════════════════════
p("\n[1/11] PCA Feature Maps...")
try:
    from repviz.analyses.geometry import batch_pca_feature_maps, pca_feature_map
    
    for tag in TAGS:
        all_patches = torch.cat([model_data[tag]["patch_tokens"][l] for l in image_labels], dim=0)
        pca_maps, _ = batch_pca_feature_maps(all_patches, h, w)
        
        n_imgs = len(image_labels)
        fig, axes = plt.subplots(2, n_imgs, figsize=(3*n_imgs, 6))
        for i, label in enumerate(image_labels):
            img_np = inverse_normalize(images_tensor[label])
            axes[0, i].imshow(img_np)
            axes[0, i].set_title(label, fontsize=9)
            axes[0, i].axis("off")
            axes[1, i].imshow(pca_maps[i])
            axes[1, i].axis("off")
        axes[0, 0].set_ylabel("Original", fontsize=11)
        axes[1, 0].set_ylabel("PCA RGB", fontsize=11)
        fig.suptitle(f"DINOv2-{tag}: PCA Feature Maps", fontsize=14, fontweight="bold")
        fig.tight_layout()
        add_figure(f"pca_grid_{tag}", fig,
            f"PCA Feature Maps — DINOv2-{tag}",
            f"The top 3 principal components of DINOv2-{tag} patch features are mapped to RGB channels. "
            f"Semantically similar regions (e.g., same object parts) share colors, revealing that DINOv2 learns "
            f"part-aware representations without any segmentation supervision. Notice how object boundaries "
            f"emerge clearly in the PCA space.")
    
    p("  Done.")
except Exception as e:
    p(f"  FAILED: {e}")
    traceback.print_exc()

# ══════════════════════════════════════════════════════════════════════
#  SECTION 2: SVD Spectrum & Effective Rank
# ══════════════════════════════════════════════════════════════════════
p("\n[2/11] SVD Spectrum & Effective Rank...")
try:
    from repviz.analyses.geometry import singular_value_spectrum, effective_rank
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    er_vals = {}
    for tag in TAGS:
        all_patches = torch.cat([model_data[tag]["patch_tokens"][l] for l in image_labels], dim=0)
        all_flat = all_patches.reshape(-1, model_data[tag]["embed_dim"])
        sv = singular_value_spectrum(all_flat[:2000])
        er = effective_rank(all_flat[:2000])
        er_vals[tag] = er
        metrics[f"Effective Rank"] = metrics.get("Effective Rank", {})
        metrics["Effective Rank"][tag] = f"{er:.1f}"
        axes[0].plot(sv / sv[0], linewidth=2, label=f"DINOv2-{tag} (eff.rank={er:.0f})")
    
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Component Index")
    axes[0].set_ylabel("Normalized Singular Value")
    axes[0].set_title("Singular Value Spectrum")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    tags = list(er_vals.keys())
    axes[1].bar(tags, [er_vals[t] for t in tags], color=[TAG_COLORS[t] for t in tags], width=0.5)
    axes[1].set_ylabel("Effective Rank")
    axes[1].set_title("Effective Rank Comparison")
    for i, t in enumerate(tags):
        axes[1].text(i, er_vals[t] + 0.5, f"{er_vals[t]:.1f}", ha="center", fontweight="bold")
    
    fig.tight_layout()
    add_figure("svd_spectrum", fig,
        "SVD Spectrum & Effective Rank",
        "The singular value spectrum shows how feature variance distributes across dimensions. "
        "A slower decay indicates more dimensions are actively used. Effective rank (exponential entropy "
        "of normalized singular values) quantifies this: higher means more uniformly distributed variance. "
        "DINOv2-B typically uses its larger embedding space more efficiently than DINOv2-S.")
    
    p("  Done.")
except Exception as e:
    p(f"  FAILED: {e}")
    traceback.print_exc()

# ══════════════════════════════════════════════════════════════════════
#  SECTION 3: Feature Geometry
# ══════════════════════════════════════════════════════════════════════
p("\n[3/11] Feature Geometry...")
try:
    from repviz.analyses.geometry import (
        average_cosine_similarity, isotropy_score, feature_norms,
        cosine_similarity_distribution, two_nn_id, mle_id
    )
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    for tag in TAGS:
        all_patches = torch.cat([model_data[tag]["patch_tokens"][l] for l in image_labels], dim=0)
        all_flat = all_patches.reshape(-1, model_data[tag]["embed_dim"])
        subset = all_flat[:2000]
        
        acs = average_cosine_similarity(subset)
        iso = isotropy_score(subset)
        norms = feature_norms(subset)
        cos_dist = cosine_similarity_distribution(subset, n_pairs=5000)
        id_2nn = two_nn_id(subset[:500])
        id_mle = mle_id(subset[:500], k=5)
        
        metrics["Avg Cosine Sim"] = metrics.get("Avg Cosine Sim", {})
        metrics["Avg Cosine Sim"][tag] = f"{acs:.3f}"
        metrics["Isotropy"] = metrics.get("Isotropy", {})
        metrics["Isotropy"][tag] = f"{iso:.3f}"
        metrics["Mean Feature Norm"] = metrics.get("Mean Feature Norm", {})
        metrics["Mean Feature Norm"][tag] = f"{norms.mean():.2f}"
        metrics["Intrinsic Dim (2-NN)"] = metrics.get("Intrinsic Dim (2-NN)", {})
        metrics["Intrinsic Dim (2-NN)"][tag] = f"{id_2nn:.1f}"
        metrics["Intrinsic Dim (MLE)"] = metrics.get("Intrinsic Dim (MLE)", {})
        metrics["Intrinsic Dim (MLE)"][tag] = f"{id_mle:.1f}"
        
        color = TAG_COLORS[tag]
        alpha = 0.6
        
        cos_dist_np = cos_dist if isinstance(cos_dist, np.ndarray) else cos_dist.numpy()
        norms_np = norms if isinstance(norms, np.ndarray) else norms.numpy()
        axes[0, 0].hist(cos_dist_np, bins=80, alpha=alpha, label=f"DINOv2-{tag}", color=color, density=True)
        axes[0, 1].hist(norms_np, bins=60, alpha=alpha, label=f"DINOv2-{tag}", color=color, density=True)
    
    axes[0, 0].set_title("Cosine Similarity Distribution")
    axes[0, 0].set_xlabel("Cosine Similarity")
    axes[0, 0].legend()
    
    axes[0, 1].set_title("Feature Norm Distribution")
    axes[0, 1].set_xlabel("L2 Norm")
    axes[0, 1].legend()
    
    # Isotropy bar
    iso_vals = {t: float(metrics["Isotropy"][t]) for t in TAGS}
    axes[1, 0].bar(TAGS, [iso_vals[t] for t in TAGS], color=[TAG_COLORS[t] for t in TAGS], width=0.5)
    axes[1, 0].set_title("Isotropy Score")
    axes[1, 0].set_ylim(0, 1)
    for i, t in enumerate(TAGS):
        axes[1, 0].text(i, iso_vals[t] + 0.02, f"{iso_vals[t]:.3f}", ha="center", fontweight="bold")
    
    # Intrinsic dim bar
    id_vals_2nn = {t: float(metrics["Intrinsic Dim (2-NN)"][t]) for t in TAGS}
    id_vals_mle = {t: float(metrics["Intrinsic Dim (MLE)"][t]) for t in TAGS}
    x = np.arange(len(TAGS))
    bw = 0.3
    axes[1, 1].bar(x - bw/2, [id_vals_2nn[t] for t in TAGS], bw, label="2-NN", color=[TAG_COLORS[t] for t in TAGS])
    axes[1, 1].bar(x + bw/2, [id_vals_mle[t] for t in TAGS], bw, label="MLE", color=[TAG_COLORS_LIGHT[t] for t in TAGS])
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(TAGS)
    axes[1, 1].set_title("Intrinsic Dimensionality")
    axes[1, 1].legend()
    
    fig.suptitle("Feature Geometry Comparison", fontsize=14, fontweight="bold")
    fig.tight_layout()
    add_figure("feature_geometry", fig,
        "Feature Geometry: Isotropy, Cosine Similarity, Norms, and Intrinsic Dimensionality",
        "These plots characterize the geometric structure of learned representations. "
        "Cosine similarity distribution shows how uniformly features spread in the embedding space — "
        "a narrow peak near 0 indicates isotropic features. Feature norms reveal magnitude consistency. "
        "Isotropy score (0=anisotropic, 1=perfectly isotropic) and intrinsic dimensionality "
        "(estimated via 2-NN and MLE methods) measure how many effective dimensions the features occupy.")
    
    p("  Done.")
except Exception as e:
    p(f"  FAILED: {e}")
    traceback.print_exc()

# ══════════════════════════════════════════════════════════════════════
#  SECTION 4: Attention Maps
# ══════════════════════════════════════════════════════════════════════
p("\n[4/11] Attention Maps...")
try:
    from repviz.analyses.attention import attention_rollout, attention_distance, attention_entropy
    
    select_labels = image_labels[:4]
    
    for tag in TAGS:
        fig, axes = plt.subplots(2, len(select_labels), figsize=(4*len(select_labels), 8))
        for i, label in enumerate(select_labels):
            attn_maps = model_data[tag]["attention_maps"][label]
            rollout = attention_rollout(attn_maps)
            # rollout shape may be (1, 1+N) or (1, N) — handle both
            if rollout.shape[-1] == 1 + h*w:
                attn_to_patches = rollout[0, 1:]
            else:
                attn_to_patches = rollout[0]
            # Trim or pad to match h*w
            attn_to_patches = attn_to_patches[:int(h*w)]
            attn_map = attn_to_patches.reshape(int(h), int(w))
            if isinstance(attn_map, torch.Tensor):
                attn_map = attn_map.numpy()
            attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
            
            img_np = inverse_normalize(images_tensor[label])
            axes[0, i].imshow(img_np)
            axes[0, i].set_title(label, fontsize=10)
            axes[0, i].axis("off")
            axes[1, i].imshow(attn_map, cmap="inferno")
            axes[1, i].axis("off")
        
        axes[0, 0].set_ylabel("Original", fontsize=11)
        axes[1, 0].set_ylabel("Attention Rollout", fontsize=11)
        fig.suptitle(f"DINOv2-{tag}: Attention Rollout (CLS → patches)", fontsize=14, fontweight="bold")
        fig.tight_layout()
        add_figure(f"attention_rollout_{tag}", fig,
            f"Attention Rollout — DINOv2-{tag}",
            f"Attention rollout aggregates attention weights across all layers to show where the [CLS] token "
            f"attends in the image. Bright regions indicate high attention. DINOv2 models attend strongly to "
            f"salient objects and their boundaries, demonstrating emergent object-awareness.")
    
    # Attention distance and entropy per layer
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for tag in TAGS:
        attn_maps = model_data[tag]["attention_maps"][first_label]
        layer_keys = sorted(attn_maps.keys())
        distances = []
        entropies = []
        for k in layer_keys:
            a = attn_maps[k]  # (1, H, N_full, N_full)
            dist = attention_distance(a, h=int(h), w=int(w))
            ent = attention_entropy(a)
            distances.append(dist.mean().item())
            entropies.append(ent.mean().item())
        
        color = TAG_COLORS[tag]
        axes[0].plot(distances, 'o-', label=f"DINOv2-{tag}", color=color)
        axes[1].plot(entropies, 'o-', label=f"DINOv2-{tag}", color=color)
    
    axes[0].set_title("Mean Attention Distance per Layer")
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("Distance (patches)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_title("Mean Attention Entropy per Layer")
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("Entropy (nats)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    fig.tight_layout()
    add_figure("attention_dist_entropy", fig,
        "Attention Distance and Entropy Across Layers",
        "Attention distance measures how far each head attends spatially — higher values indicate global attention. "
        "Attention entropy quantifies how spread out the attention distribution is. Early layers tend to have local, "
        "low-entropy attention while deeper layers develop longer-range, higher-entropy patterns.")
    
    p("  Done.")
except Exception as e:
    p(f"  FAILED: {e}")
    traceback.print_exc()

# ══════════════════════════════════════════════════════════════════════
#  SECTION 5: Layer-wise CKA
# ══════════════════════════════════════════════════════════════════════
p("\n[5/11] Layer-wise CKA...")
try:
    from repviz.analyses.layerwise import layerwise_cka_matrix, feature_norm_progression, effective_rank_progression
    
    for tag in TAGS:
        intermediate = model_data[tag]["intermediate"][first_label]
        # Use every 2nd layer for speed
        keys = sorted(intermediate.keys())
        subset = {k: intermediate[k][:, 1:].reshape(-1, model_data[tag]["embed_dim"])[:500] for k in keys[::2]}
        
        matrix, names = layerwise_cka_matrix(subset)
        
        fig, ax = plt.subplots(figsize=(8, 7))
        sns.heatmap(matrix, xticklabels=names, yticklabels=names, cmap="magma", vmin=0, vmax=1, square=True, ax=ax)
        ax.set_title(f"DINOv2-{tag}: Layer-wise CKA Similarity", fontsize=13, fontweight="bold")
        plt.xticks(rotation=45, ha="right", fontsize=7)
        plt.yticks(rotation=0, fontsize=7)
        fig.tight_layout()
        add_figure(f"cka_matrix_{tag}", fig,
            f"CKA Similarity Matrix — DINOv2-{tag}",
            f"Centered Kernel Alignment (CKA) measures representational similarity between layers. "
            f"Block-diagonal structure indicates groups of layers that compute similar representations. "
            f"Off-diagonal bright regions show which distant layers share similar representations.")
    
    # Norm and rank progression
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for tag in TAGS:
        intermediate = model_data[tag]["intermediate"][first_label]
        norms = feature_norm_progression(intermediate)
        ranks = effective_rank_progression(intermediate)
        
        color = TAG_COLORS[tag]
        layer_names = sorted(norms.keys())
        axes[0].plot([norms[n] for n in layer_names], 'o-', label=f"DINOv2-{tag}", color=color)
        layer_names_r = sorted(ranks.keys())
        axes[1].plot([ranks[n] for n in layer_names_r], 'o-', label=f"DINOv2-{tag}", color=color)
    
    axes[0].set_title("Feature Norm Progression")
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("Mean Norm")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_title("Effective Rank Progression")
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("Effective Rank")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    fig.tight_layout()
    add_figure("norm_rank_progression", fig,
        "Feature Norm and Effective Rank Across Layers",
        "Feature norms typically grow through the network as representations become more refined. "
        "Effective rank shows how the dimensionality of representations evolves — it often peaks in "
        "middle layers and may decrease in final layers as features become more task-specific.")
    
    p("  Done.")
except Exception as e:
    p(f"  FAILED: {e}")
    traceback.print_exc()

# ══════════════════════════════════════════════════════════════════════
#  SECTION 6: Dense Features
# ══════════════════════════════════════════════════════════════════════
p("\n[6/11] Dense Features...")
try:
    from repviz.analyses.dense import cosine_similarity_map, patch_clustering
    
    select_labels = image_labels[:4]
    
    for tag in TAGS:
        fig, axes = plt.subplots(3, len(select_labels), figsize=(4*len(select_labels), 12))
        for i, label in enumerate(select_labels):
            patches = model_data[tag]["patch_tokens"][label][0]  # (N, D)
            img_np = inverse_normalize(images_tensor[label])
            
            # Cosine sim from center patch
            center_idx = N // 2
            sim = cosine_similarity_map(patches, query_idx=center_idx, h=int(h), w=int(w))
            
            # Clustering
            clusters = patch_clustering(patches, n_clusters=6, h=int(h), w=int(w))
            
            axes[0, i].imshow(img_np)
            axes[0, i].set_title(label, fontsize=10)
            axes[0, i].axis("off")
            
            sim_np = sim.numpy() if isinstance(sim, torch.Tensor) else sim
            im = axes[1, i].imshow(sim_np, cmap="RdBu_r", vmin=-1, vmax=1)
            axes[1, i].plot(w//2, h//2, 'kx', markersize=10, markeredgewidth=2)
            axes[1, i].axis("off")
            
            cmap = plt.cm.get_cmap("tab20", 6)
            clusters_np = clusters.numpy() if isinstance(clusters, torch.Tensor) else clusters
            axes[2, i].imshow(clusters_np, cmap=cmap, interpolation="nearest")
            axes[2, i].axis("off")
        
        axes[0, 0].set_ylabel("Original", fontsize=11)
        axes[1, 0].set_ylabel("Cosine Sim", fontsize=11)
        axes[2, 0].set_ylabel("Clustering", fontsize=11)
        fig.suptitle(f"DINOv2-{tag}: Dense Feature Analysis", fontsize=14, fontweight="bold")
        fig.tight_layout()
        add_figure(f"dense_features_{tag}", fig,
            f"Dense Features — DINOv2-{tag}",
            f"Cosine similarity maps show which patches have similar representations to the center patch (marked ×). "
            f"K-means clustering of patch features produces unsupervised segmentations that closely follow object boundaries. "
            f"This demonstrates DINOv2's strong spatial awareness and semantic grouping capabilities.")
    
    p("  Done.")
except Exception as e:
    p(f"  FAILED: {e}")
    traceback.print_exc()

# ══════════════════════════════════════════════════════════════════════
#  SECTION 7: Robustness
# ══════════════════════════════════════════════════════════════════════
p("\n[7/11] Robustness...")
try:
    from repviz.analyses.robustness import augmentation_invariance_suite, resolution_sensitivity
    
    # Augmentation invariance
    fig, ax = plt.subplots(figsize=(10, 5))
    bw = 0.8 / len(TAGS)
    aug_results = {}
    
    for idx, tag in enumerate(TAGS):
        backbone = model_data[tag]["backbone"]
        batch = torch.stack([images_tensor[l] for l in image_labels[:4]])
        scores = augmentation_invariance_suite(backbone, batch)
        aug_results[tag] = scores
        
        names = list(scores.keys())
        vals = [scores[n] for n in names]
        x = np.arange(len(names))
        color = TAG_COLORS[tag]
        offset = (idx - (len(TAGS)-1)/2) * bw
        ax.bar(x + offset, vals, bw, label=f"DINOv2-{tag}", color=color)
    
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("Augmentation Invariance")
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    add_figure("augmentation_invariance", fig,
        "Augmentation Invariance",
        "Each bar shows the average cosine similarity between features of original and augmented images. "
        "Higher values indicate greater robustness to that transformation. DINOv2 models are generally very "
        "robust to geometric and photometric augmentations thanks to self-supervised training with heavy augmentation.")
    
    # Resolution sensitivity
    fig, ax = plt.subplots(figsize=(8, 5))
    for tag in TAGS:
        backbone = model_data[tag]["backbone"]
        pil_list = [images_pil[l] for l in image_labels[:3]]
        resolutions = [224, 336, 448, 518]
        scores = resolution_sensitivity(backbone, pil_list, resolutions=resolutions)
        
        color = TAG_COLORS[tag]
        res_keys = sorted(scores.keys())
        ax.plot(res_keys, [scores[r] for r in res_keys], 'o-', label=f"DINOv2-{tag}", color=color)
    
    ax.set_xlabel("Resolution")
    ax.set_ylabel("CLS Cosine Similarity to Max Resolution")
    ax.set_title("Resolution Sensitivity")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    add_figure("resolution_sensitivity", fig,
        "Resolution Sensitivity",
        "This plot shows how CLS token representations change as input resolution varies. "
        "Values closer to 1.0 indicate resolution-invariant features. DINOv2 uses a ViT backbone "
        "with flexible positional embeddings, making it relatively robust across resolutions.")
    
    p("  Done.")
except Exception as e:
    p(f"  FAILED: {e}")
    traceback.print_exc()

# ══════════════════════════════════════════════════════════════════════
#  SECTION 8: Weight Analysis
# ══════════════════════════════════════════════════════════════════════
p("\n[8/11] Weight Analysis...")
try:
    from repviz.analyses.weights import weight_distributions, weight_effective_rank, weight_spectral_analysis
    
    for tag in TAGS:
        backbone = model_data[tag]["backbone"]
        
        dists = weight_distributions(backbone.model)
        ranks = weight_effective_rank(backbone.model)
        
        # Weight distribution histogram (sample a few layers)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        layer_names = sorted(dists.keys())
        sample_layers = [layer_names[0], layer_names[len(layer_names)//2], layer_names[-1]]
        for i, ln in enumerate(sample_layers):
            axes[i].hist(dists[ln], bins=100, density=True, color=TAG_COLORS[tag], alpha=0.8)
            short_name = ln.split(".")[-2] + "." + ln.split(".")[-1] if "." in ln else ln
            axes[i].set_title(short_name, fontsize=9)
            axes[i].set_xlabel("Weight Value")
        fig.suptitle(f"DINOv2-{tag}: Weight Distributions (Early/Mid/Late)", fontsize=13, fontweight="bold")
        fig.tight_layout()
        add_figure(f"weight_dists_{tag}", fig,
            f"Weight Distributions — DINOv2-{tag}",
            f"Weight distributions for early, middle, and late layers in the network. "
            f"Well-trained networks typically show near-Gaussian distributions. "
            f"Changes in spread across layers can indicate varying levels of specialization.")
        
        # Effective rank per layer
        fig, ax = plt.subplots(figsize=(12, 4))
        rnames = sorted(ranks.keys())[:30]
        ax.bar(range(len(rnames)), [ranks[n] for n in rnames], color=TAG_COLORS[tag])
        ax.set_xticks(range(len(rnames)))
        ax.set_xticklabels([n.split(".")[-2]+"."+n.split(".")[-1] if "." in n else n for n in rnames], rotation=90, fontsize=6)
        ax.set_title(f"DINOv2-{tag}: Weight Matrix Effective Rank")
        ax.set_ylabel("Effective Rank")
        fig.tight_layout()
        add_figure(f"weight_ranks_{tag}", fig,
            f"Weight Effective Rank — DINOv2-{tag}",
            f"Effective rank of weight matrices measures how many dimensions each layer actively uses. "
            f"Low effective rank suggests the layer could be compressed without much information loss. "
            f"Attention projection layers often have different rank profiles than MLP layers.")
    
    p("  Done.")
except Exception as e:
    p(f"  FAILED: {e}")
    traceback.print_exc()

# ══════════════════════════════════════════════════════════════════════
#  SECTION 9: Cross-Model Comparison
# ══════════════════════════════════════════════════════════════════════
p("\n[9/11] Cross-Model Comparison...")
try:
    from repviz.analyses.cross_model import cross_model_cka, cross_model_mutual_knn
    from repviz.analyses.layerwise import linear_cka
    
    # CKA between S and B layer representations
    # Generate cross-model CKA for all pairs
    import itertools
    tag_pairs = list(itertools.combinations(TAGS, 2))
    
    n_pairs = len(tag_pairs)
    fig, axes = plt.subplots(1, n_pairs, figsize=(7*n_pairs, 6))
    if n_pairs == 1:
        axes = [axes]
    
    for pi, (t1, t2) in enumerate(tag_pairs):
        inter_1 = model_data[t1]["intermediate"][first_label]
        inter_2 = model_data[t2]["intermediate"][first_label]
        keys_1 = sorted(inter_1.keys())
        keys_2 = sorted(inter_2.keys())
        step_1 = max(1, len(keys_1) // 8)
        step_2 = max(1, len(keys_2) // 8)
        sub_1 = keys_1[::step_1]
        sub_2 = keys_2[::step_2]
        
        cka_mat = np.zeros((len(sub_1), len(sub_2)))
        for i, k1 in enumerate(sub_1):
            for j, k2 in enumerate(sub_2):
                f1 = inter_1[k1][:, 1:].reshape(-1, model_data[t1]["embed_dim"])[:300]
                f2 = inter_2[k2][:, 1:].reshape(-1, model_data[t2]["embed_dim"])[:300]
                cka_mat[i, j] = linear_cka(f1, f2)
        
        sns.heatmap(cka_mat, xticklabels=sub_2, yticklabels=sub_1, cmap="magma", vmin=0, vmax=1, square=True, ax=axes[pi])
        axes[pi].set_xlabel(f"DINOv2-{t2} layers")
        axes[pi].set_ylabel(f"DINOv2-{t1} layers")
        axes[pi].set_title(f"{t1} vs {t2}", fontsize=12, fontweight="bold")
        axes[pi].tick_params(labelsize=6)
    
    fig.suptitle("Cross-Model CKA Similarity", fontsize=14, fontweight="bold")
    fig.tight_layout()
    add_figure("cross_model_cka", fig,
        "Cross-Model CKA Between All Model Pairs",
        "These heatmaps show representational similarity between layers of different-sized models. "
        "A diagonal pattern indicates that corresponding layers learn similar features despite different "
        "model capacities. Off-diagonal bright regions reveal where one model's representations map to "
        "different depths in the other.")
    
    # Mutual k-NN on CLS tokens for all pairs
    metrics["Mutual k-NN (CLS)"] = {}
    for t1, t2 in tag_pairs:
        cls_1 = torch.cat([model_data[t1]["cls_tokens"][l] for l in image_labels], dim=0)
        cls_2 = torch.cat([model_data[t2]["cls_tokens"][l] for l in image_labels], dim=0)
        mknn = cross_model_mutual_knn(cls_1, cls_2, k=min(3, len(image_labels)-1))
        metrics["Mutual k-NN (CLS)"][f"{t1}↔{t2}"] = f"{mknn:.3f}"
        p(f"  Mutual k-NN ({t1}↔{t2}): {mknn:.3f}")
    
    p("  Done.")
except Exception as e:
    p(f"  FAILED: {e}")
    traceback.print_exc()

# ══════════════════════════════════════════════════════════════════════
#  SECTION 10: Neuron Analysis
# ══════════════════════════════════════════════════════════════════════
p("\n[10/11] Neuron Analysis...")
try:
    from repviz.analyses.neurons import dead_neuron_fraction, channel_redundancy_summary
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    dead_vals = {}
    redundancy_vals = {}
    
    for tag in TAGS:
        all_patches = torch.cat([model_data[tag]["patch_tokens"][l] for l in image_labels], dim=0)
        flat = all_patches.reshape(-1, model_data[tag]["embed_dim"])
        
        dead = dead_neuron_fraction(flat)
        dead_vals[tag] = dead
        metrics["Dead Neuron Fraction"] = metrics.get("Dead Neuron Fraction", {})
        metrics["Dead Neuron Fraction"][tag] = f"{dead:.4f}"
        
        redundancy = channel_redundancy_summary(flat)
        redundancy_vals[tag] = redundancy
        if "mean_abs_correlation" in redundancy:
            metrics["Mean Channel Correlation"] = metrics.get("Mean Channel Correlation", {})
            metrics["Mean Channel Correlation"][tag] = f"{redundancy['mean_abs_correlation']:.3f}"
        if "highly_correlated_fraction" in redundancy:
            metrics["Highly Correlated Channels"] = metrics.get("Highly Correlated Channels", {})
            metrics["Highly Correlated Channels"][tag] = f"{redundancy['highly_correlated_fraction']:.3f}"
    
    # Dead neurons bar
    axes[0].bar(TAGS, [dead_vals[t] for t in TAGS], color=[TAG_COLORS[t] for t in TAGS], width=0.5)
    axes[0].set_title("Dead Neuron Fraction")
    axes[0].set_ylabel("Fraction")
    for i, t in enumerate(TAGS):
        axes[0].text(i, dead_vals[t] + 0.001, f"{dead_vals[t]:.4f}", ha="center", fontweight="bold")
    
    # Redundancy bars
    r_metrics = list(redundancy_vals["S"].keys())
    x = np.arange(len(r_metrics))
    bw = 0.8 / len(TAGS)
    for idx, t in enumerate(TAGS):
        offset = (idx - (len(TAGS)-1)/2) * bw
        axes[1].bar(x + offset, [redundancy_vals[t][m] for m in r_metrics], bw, label=f"DINOv2-{t}", color=TAG_COLORS[t])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([m.replace("_", "\n") for m in r_metrics], fontsize=8)
    axes[1].set_title("Channel Redundancy Metrics")
    axes[1].legend()
    
    fig.tight_layout()
    add_figure("neuron_analysis", fig,
        "Neuron Analysis: Dead Neurons and Channel Redundancy",
        "Dead neuron fraction measures channels that are always zero across all inputs — lower is better. "
        "Channel redundancy quantifies how correlated different feature channels are. High redundancy "
        "suggests the model has capacity for compression without losing information.")
    
    p("  Done.")
except Exception as e:
    p(f"  FAILED: {e}")
    traceback.print_exc()


# ══════════════════════════════════════════════════════════════════════
#  SECTION 11: Build HTML Report
# ══════════════════════════════════════════════════════════════════════
p("\n[11/11] Building HTML report...")

# Collect parameters
model_params = {}
for tag in TAGS:
    p_count = sum(p.numel() for p in model_data[tag]["backbone"].model.parameters())
    model_params[tag] = p_count
metrics["Parameters"] = {t: f"{model_params[t]/1e6:.1f}M" for t in TAGS}
metrics["Embedding Dim"] = {t: str(model_data[t]["embed_dim"]) for t in TAGS}
metrics["Num Layers"] = {t: str(model_data[t]["num_layers"]) for t in TAGS}

sections = [
    ("pca", "PCA Feature Maps", [f for f in figures if f.startswith("pca_")]),
    ("svd", "SVD Spectrum & Effective Rank", ["svd_spectrum"]),
    ("geometry", "Feature Geometry", ["feature_geometry"]),
    ("attention", "Attention Maps", [f for f in figures if f.startswith("attention_")]),
    ("cka", "Layer-wise CKA", [f for f in figures if f.startswith("cka_")] + ["norm_rank_progression"]),
    ("dense", "Dense Features", [f for f in figures if f.startswith("dense_")]),
    ("robustness", "Robustness", ["augmentation_invariance", "resolution_sensitivity"]),
    ("weights", "Weight Analysis", [f for f in figures if f.startswith("weight_")]),
    ("crossmodel", "Cross-Model Comparison", ["cross_model_cka"]),
    ("neurons", "Neuron Analysis", ["neuron_analysis"]),
]

# Build metrics table HTML
metrics_html = '<table class="metrics-table"><thead><tr><th>Metric</th>'
# Get all unique model tags
all_tags = set()
for v in metrics.values():
    all_tags.update(v.keys())
all_tags = sorted(all_tags)
for t in all_tags:
    metrics_html += f'<th>DINOv2-{t}</th>'
metrics_html += '</tr></thead><tbody>'
for metric_name, vals in metrics.items():
    metrics_html += f'<tr><td><strong>{metric_name}</strong></td>'
    for t in all_tags:
        metrics_html += f'<td>{vals.get(t, "—")}</td>'
    metrics_html += '</tr>'
metrics_html += '</tbody></table>'

# Build nav
nav_html = '<nav class="sidebar">\n<h2>RepViz Report</h2>\n<ul>\n'
for sid, sname, _ in sections:
    nav_html += f'<li><a href="#{sid}">{sname}</a></li>\n'
nav_html += '<li><a href="#summary">Summary Table</a></li>\n'
nav_html += '</ul>\n<button onclick="toggleTheme()" class="theme-btn">🌗 Toggle Theme</button>\n</nav>'

# Build content
content_html = ''
for sid, sname, fig_names in sections:
    content_html += f'<section id="{sid}" class="card">\n<h2>{sname}</h2>\n'
    for fn in fig_names:
        if fn in figures:
            f = figures[fn]
            content_html += f'''<div class="figure">
<img src="data:image/png;base64,{f["b64"]}" alt="{f["caption"]}" loading="lazy">
<div class="caption"><strong>{f["caption"]}</strong></div>
<p class="explanation">{f["explanation"]}</p>
</div>\n'''
    content_html += '</section>\n'

content_html += f'<section id="summary" class="card">\n<h2>Summary: Key Metrics Comparison</h2>\n{metrics_html}\n</section>\n'

html = f'''<!DOCTYPE html>
<html lang="en" data-theme="light">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>RepViz: DINOv2 Representation Analysis Report</title>
<style>
:root {{
    --bg: #f8f9fa; --card-bg: #ffffff; --text: #212529; --text-muted: #6c757d;
    --accent: #0d6efd; --accent2: #0891b2; --border: #dee2e6; --shadow: rgba(0,0,0,0.08);
    --sidebar-bg: #1a1a2e; --sidebar-text: #e0e0e0;
}}
[data-theme="dark"] {{
    --bg: #0d1117; --card-bg: #161b22; --text: #c9d1d9; --text-muted: #8b949e;
    --accent: #58a6ff; --accent2: #39d2c0; --border: #30363d; --shadow: rgba(0,0,0,0.3);
    --sidebar-bg: #010409; --sidebar-text: #c9d1d9;
}}
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: var(--bg); color: var(--text); line-height: 1.6; }}
.sidebar {{
    position: fixed; top: 0; left: 0; width: 240px; height: 100vh;
    background: var(--sidebar-bg); color: var(--sidebar-text); padding: 24px 16px;
    overflow-y: auto; z-index: 100;
}}
.sidebar h2 {{ font-size: 18px; margin-bottom: 20px; color: var(--accent2); }}
.sidebar ul {{ list-style: none; }}
.sidebar li {{ margin-bottom: 8px; }}
.sidebar a {{ color: var(--sidebar-text); text-decoration: none; font-size: 14px; padding: 6px 10px;
    display: block; border-radius: 6px; transition: background 0.2s; }}
.sidebar a:hover {{ background: rgba(255,255,255,0.1); }}
.theme-btn {{ margin-top: 20px; padding: 8px 16px; border: 1px solid var(--sidebar-text);
    background: transparent; color: var(--sidebar-text); border-radius: 6px; cursor: pointer; font-size: 13px; }}
.main {{ margin-left: 260px; padding: 32px 40px; max-width: 1200px; }}
.card {{ background: var(--card-bg); border-radius: 12px; padding: 32px; margin-bottom: 32px;
    box-shadow: 0 2px 12px var(--shadow); border: 1px solid var(--border); }}
.card h2 {{ font-size: 22px; margin-bottom: 20px; color: var(--accent); border-bottom: 2px solid var(--accent2);
    padding-bottom: 8px; }}
.figure {{ margin: 24px 0; text-align: center; }}
.figure img {{ max-width: 100%; border-radius: 8px; box-shadow: 0 2px 8px var(--shadow); }}
.caption {{ font-weight: 600; font-size: 14px; margin-top: 12px; color: var(--text); }}
.explanation {{ font-size: 13px; color: var(--text-muted); max-width: 800px; margin: 8px auto 0; text-align: left; }}
.metrics-table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
.metrics-table th, .metrics-table td {{ padding: 10px 16px; text-align: left; border-bottom: 1px solid var(--border); }}
.metrics-table th {{ background: var(--accent); color: white; }}
.metrics-table tr:nth-child(even) {{ background: rgba(0,0,0,0.03); }}
[data-theme="dark"] .metrics-table tr:nth-child(even) {{ background: rgba(255,255,255,0.03); }}
h1 {{ font-size: 28px; margin-bottom: 8px; }}
.subtitle {{ color: var(--text-muted); margin-bottom: 32px; font-size: 15px; }}
@media (max-width: 768px) {{
    .sidebar {{ display: none; }}
    .main {{ margin-left: 0; padding: 16px; }}
}}
</style>
</head>
<body>
{nav_html}
<div class="main">
<h1>🔬 RepViz: DINOv2 Representation Analysis</h1>
<p class="subtitle">Comprehensive analysis of {' and '.join(f'DINOv2-{t} ({model_params[t]/1e6:.0f}M params)' for t in TAGS)} across {len(image_labels)} diverse images.</p>
{content_html}
</div>
<script>
function toggleTheme() {{
    const html = document.documentElement;
    html.setAttribute('data-theme', html.getAttribute('data-theme') === 'dark' ? 'light' : 'dark');
}}
// Smooth scroll
document.querySelectorAll('.sidebar a').forEach(a => {{
    a.addEventListener('click', e => {{
        e.preventDefault();
        document.querySelector(a.getAttribute('href')).scrollIntoView({{behavior: 'smooth'}});
    }});
}});
</script>
</body>
</html>'''

report_path = PROJECT_DIR / "report.html"
with open(report_path, "w") as f:
    f.write(html)

p(f"\n✅ Report saved to {report_path}")
p(f"   {len(figures)} figures generated")
p(f"   {len(metrics)} metrics collected")
p(f"   File size: {report_path.stat().st_size / 1024 / 1024:.1f} MB")
