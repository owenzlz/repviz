# repviz — Foundation Model Representation Analyzer

## Project Overview

A comprehensive toolkit for analyzing and visualizing learned representations in pretrained vision foundation models. Uses DINOv3 as the primary subject but designed to be model-agnostic.

**Why DINOv3?**
- Dense per-patch features (not just CLS token)
- Multiple model scales (ViT-S/16 → ViT-7B/16, plus ConvNeXt variants)
- Self-supervised (no label bias in learned features)
- Both ViT and ConvNeXt architectures available
- "Gram anchoring" innovation for feature quality — interesting to study
- Available via PyTorch Hub and HuggingFace Transformers (≥4.56.0)

**DINOv3 paper:** arXiv:2508.10104 (Meta FAIR, Aug 2025)
**DINOv3 repo:** https://github.com/facebookresearch/dinov3

---

## 10 Analysis Categories (~25 concrete analyses)

### 1. Feature Space Geometry
What the representation space "looks like" globally.

| Analysis | What it reveals | Method |
|----------|----------------|--------|
| PCA of patch features | Dominant structure; DINOv2/v3's famous semantic PCA→RGB maps | Top-k PCA on all patch tokens, visualize as RGB |
| Singular value spectrum | Effective rank, dimensional collapse, how many dims are "used" | SVD of feature matrix, plot singular values, effective rank = exp(entropy of normalized singular values) |
| Intrinsic dimensionality | True manifold dimension vs ambient dimension | Two-NN estimator (Facco et al. 2017) or MLE-based ID per layer |
| Feature isotropy | Uniform on hypersphere or clustered? | Avg cosine sim between random pairs; isotropy score (Mu et al. 2018) |
| Representation topology | Persistent homological features (loops, voids) | TDA with Ripser — persistence diagrams on sampled feature clouds |

### 2. Per-Layer / Depth Analysis
How representation evolves through the network.

| Analysis | Method |
|----------|--------|
| Layer-wise CKA (Centered Kernel Alignment) | CKA similarity matrix between all layers — which layers learn similar/different things |
| Layer-wise linear probing | Linear classifiers on each layer's output → accuracy curve |
| Representation similarity across layers | Procrustes distance, CCA, mutual k-NN between consecutive layers |
| Feature norm progression | L2 norm of patch/CLS tokens per layer — residual stream dynamics |
| Rank evolution | Effective rank at each layer — does rank increase or collapse? |

### 3. Attention Analysis (ViT-specific)
What the self-attention heads learn.

| Analysis | Method |
|----------|--------|
| Attention map visualization | Per-head, per-layer attention maps overlaid on images |
| Attention head clustering | Cluster heads by attention patterns (local/global, object/texture) |
| Attention distance | Mean attention distance per head per layer — receptive field growth |
| Attention entropy | Per-head entropy — sharp (focused) vs diffuse |
| Register token analysis | DINOv3 uses registers — what do they attend to? artifact analysis |

### 4. Semantic Content Probing
What information is encoded in the features.

| Analysis | Method |
|----------|--------|
| Linear probing suite | ImageNet, Places365, DTD (texture), CUB (fine-grained) |
| Dense linear probing | Segmentation (ADE20K), depth (NYUv2) on patch features |
| Property probing | Probes for color, shape, texture, count, spatial relations |
| k-NN classification | Nearest neighbor in feature space on ImageNet — no training |
| Information plane | Mutual information between features and input/labels per layer |

### 5. Spatial / Dense Feature Analysis
Exploiting per-patch ViT features.

| Analysis | Method |
|----------|--------|
| Cosine similarity maps | Query patch → cosine sim to all patches (same/other images) |
| Object discovery | Self-similarity threshold → unsupervised segmentation |
| Patch feature clustering | K-means on patches → segmentation masks |
| Cross-image correspondence | NN patches across pairs — evaluate on SPair-71k / PF-PASCAL |
| Spatial frequency analysis | FFT of feature maps per layer — low vs high frequency content |

### 6. Robustness & Invariance Testing
What transformations the representation is robust to.

| Analysis | Method |
|----------|--------|
| Augmentation invariance | CKA/cosine sim under crops, rotations, color jitter, blur |
| Domain shift sensitivity | Features on ImageNet vs Sketch/R/A/V2 — representation shift |
| Resolution sensitivity | Multiple resolutions → feature stability |
| Adversarial robustness | PGD attacks on linear probes |
| Occlusion sensitivity | Mask patches, measure feature change |

### 7. Cross-Model Comparison
How DINOv3 compares to other foundations.

| Analysis | Method |
|----------|--------|
| CKA across models | DINOv3 vs DINOv2 vs CLIP vs MAE vs SigLIP |
| Platonic representation | Following Huh et al. 2024 — mutual k-NN + kernel alignment to LLM features |
| Relative representation | Angle-based relative reps (Moschella et al. 2023) — isomorphism? |
| Transfer learning gap | Same probe across models → where each excels |

### 8. Neuron / Channel-Level Analysis
Individual feature dimensions.

| Analysis | Method |
|----------|--------|
| Feature visualization | Activation maximization per channel — what maximally activates each? |
| Dead neuron analysis | Fraction of features never activated across dataset |
| Neuron selectivity | Class selectivity index per feature dimension |
| Channel redundancy | Correlation matrix — how redundant is the representation? |
| Top-k activating images | Per dim, find top-k patches — manual inspect what each encodes |

### 9. Weight Space Analysis
Analyzing the model parameters themselves.

| Analysis | Method |
|----------|--------|
| Weight distribution | Histograms per layer — Gaussian? Heavy-tailed? |
| Weight matrix rank | Effective rank per weight matrix — which layers are low-rank? |
| Weight spectral analysis | SVD of weights — power-law (Martin & Mahoney) |
| Gram matrix analysis | Weight Gram matrices — implicit regularization structure |

### 10. Downstream Task Probing Suite
Comprehensive feature quality benchmark.

| Task | Dataset | Method |
|------|---------|--------|
| Classification | ImageNet-1K | Linear, k-NN, few-shot |
| Segmentation | ADE20K | Linear head on patches |
| Depth estimation | NYUv2 | Linear head on patches |
| Object detection | COCO | ViTDet-style probe |
| Retrieval | Oxford/Paris | Global feature retrieval |
| Correspondence | SPair-71k | Patch NN matching |
| Few-shot | Meta-Dataset | Prototype network, frozen features |

---

## Implementation Phases

### Phase 1 — Core infra + visual bang (Week 1)
- Model loading abstraction (DINOv3 via torch.hub / HuggingFace)
- Feature extractor with intermediate layer hooks
- PCA visualization (iconic DINOv2/v3 RGB maps)
- Cosine similarity maps
- Attention map visualization
- Basic k-NN classification on ImageNet

### Phase 2 — Geometric & layer analysis (Week 2)
- SVD spectrum + effective rank
- Layer-wise CKA matrix
- Feature norm / rank progression through layers
- Attention distance + entropy
- Intrinsic dimensionality estimation

### Phase 3 — Probing & dense analysis (Week 3)
- Full linear probing suite (classification + dense)
- Patch clustering → unsupervised segmentation
- Cross-image correspondence evaluation
- Object discovery

### Phase 4 — Robustness, cross-model, deep dives (Week 4)
- Augmentation invariance tests
- Domain shift analysis
- Cross-model CKA (DINOv3 vs DINOv2 vs CLIP)
- Platonic representation analysis
- Weight spectral analysis
- Neuron-level analysis

### Phase 5 — Dashboard & report generation
- Interactive Gradio/Streamlit app
- Auto-generated HTML report with all figures
- Comparative tables across models/scales

---

## Design Principles

1. **Model-agnostic** — Every analysis takes a standardized `FeatureExtractor`. Swap DINOv3 for any model.
2. **Layer-aware** — Always support analysis at any layer, not just final output.
3. **Scale-aware** — Run across model scales (ViT-S → 7B) to study scaling behavior.
4. **Reproducible** — Seeded, config-driven, cached intermediate features.
5. **Visual-first** — Every analysis produces a publication-quality figure.

---

## Key References

- **DINOv3** — arXiv:2508.10104 — The primary subject model
- **DINOv2** — arXiv:2304.07193 — Predecessor, introduced PCA feature viz
- **Platonic Representation Hypothesis** — arXiv:2405.07987 — Cross-model convergence
- **Battle of the Backbones** — arXiv:2310.19909 — Comprehensive backbone benchmarking
- **A Cookbook of Self-Supervised Learning** — arXiv:2304.12210 — SSL analysis methodology
- **LiFT** — arXiv:2403.14625 — Dense ViT feature analysis & enhancement
- **Vision Transformers Need Registers** — arXiv:2309.16588 — Attention artifact analysis
- **CKA** — Kornblith et al. 2019 — Similarity of Neural Network Representations Revisited
- **Heavy-Tailed Self-Regularization** — Martin & Mahoney — Weight matrix spectral analysis
- **Two-NN Intrinsic Dimensionality** — Facco et al. 2017

---

## Repo Structure

```
repviz/
├── README.md
├── PLAN.md                     # This file
├── pyproject.toml
├── configs/
│   ├── models/
│   │   ├── dinov3.yaml
│   │   ├── dinov2.yaml
│   │   └── clip.yaml
│   └── analyses/
│       ├── full_suite.yaml
│       └── quick.yaml
├── src/
│   └── repviz/
│       ├── __init__.py
│       ├── models/             # Model loading abstraction
│       │   ├── __init__.py
│       │   ├── base.py         # Abstract backbone interface
│       │   ├── dinov3.py
│       │   ├── dinov2.py
│       │   └── clip.py
│       ├── extractors/         # Feature extraction with hooks
│       │   ├── __init__.py
│       │   ├── feature_extractor.py
│       │   └── attention_extractor.py
│       ├── analyses/           # Each analysis module
│       │   ├── __init__.py
│       │   ├── geometry/       # PCA, SVD, intrinsic dim, isotropy
│       │   ├── layerwise/      # CKA, probing, rank evolution
│       │   ├── attention/      # Attention maps, heads, distance
│       │   ├── probing/        # Linear probing, k-NN
│       │   ├── dense/          # Cosine sim, segmentation, correspondence
│       │   ├── robustness/     # Augmentation, domain shift, adversarial
│       │   ├── cross_model/    # CKA, platonic, relative rep
│       │   ├── neurons/        # Feature viz, selectivity, dead neurons
│       │   └── weights/        # Weight distributions, spectral analysis
│       ├── visualization/      # Plotting utilities
│       │   ├── __init__.py
│       │   ├── feature_viz.py
│       │   ├── attention_viz.py
│       │   ├── interactive.py  # Gradio/Streamlit dashboard
│       │   └── report.py       # Auto-generate HTML report
│       └── utils/
│           ├── __init__.py
│           ├── metrics.py      # CKA, CCA, mutual kNN
│           ├── data.py         # Dataset loading
│           └── hooks.py        # PyTorch hook utilities
├── notebooks/
│   ├── 01_quickstart.ipynb
│   ├── 02_pca_visualization.ipynb
│   └── ...
├── scripts/
│   ├── run_full_analysis.py
│   ├── extract_features.py
│   └── generate_report.py
└── tests/
```
