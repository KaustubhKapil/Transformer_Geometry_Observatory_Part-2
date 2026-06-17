# TGO-II: Representation Geometry Observatory

TGO-II is the second observatory in the **Transformer Geometry Observatory (TGO)** program.

While TGO-I studied the **spectral geometry** of Vision Transformer representations, TGO-II focuses on **representation similarity** and **local manifold complexity**. The goal is to determine whether the layer clustering observed in TGO-I reflects genuine representational redundancy, progressive specialization, or both.

## Research Objective

TGO-II is designed around one central question:

> Can representational complexity increase while adjacent Transformer layers remain highly similar?

To study this, TGO-II measures:

- **CKA (Centered Kernel Alignment)**  
- **SVCCA (Singular Vector Canonical Correlation Analysis)**  
- **TwoNN Intrinsic Dimension**

These observables are used to examine:

- similarity between layers
- similarity across epochs
- representational redundancy
- evolution of local intrinsic dimension
- transformation zones inside the network

---

## What TGO-II Does

TGO-II runs a Vision Transformer through training and periodically analyzes internal activations to answer the following questions:

- Are clustered layers actually redundant?
- Do layers span similar subspaces?
- Does representational complexity increase while similarity remains high?
- How does intrinsic dimensionality evolve during training?
- When do transformation zones emerge?

---

## Observables

### 1. CKA
CKA is used to measure similarity between representations across:

- layers
- epochs
- checkpoints

CKA heatmaps are used to identify:

- highly similar layer groups
- redundancy candidates
- transformation zones

### 2. SVCCA
SVCCA is used to verify whether two layers span similar subspaces.

It complements CKA by testing whether similarity persists when viewed from a subspace-based perspective.

### 3. TwoNN Intrinsic Dimension
TwoNN is used to estimate the local intrinsic dimension of representations.

This helps determine whether the representation manifold becomes:

- richer
- more complex
- more specialized
- more redundant

as training progresses.

---

## Key Outputs

TGO-II generates:

- layer-wise CKA heatmaps
- epoch-wise CKA summaries
- SVCCA similarity matrices
- TwoNN intrinsic dimension curves
- summary JSON files
- checkpoint logs
- visualization figures

---

## Checkpointing Policy

TGO-II does **not** save a checkpoint after every epoch.

Instead, it saves:

- `last.pth` → overwritten every epoch
- `best.pth` → updated whenever validation accuracy improves

This keeps storage compact while preserving the most useful model states.

---

## Training Summaries

During training, the framework generates adequate summaries to track progress, including:

- per-epoch JSON summaries
- periodic visual summaries
- similarity plots at configured intervals
- intrinsic dimension evolution plots

The exact frequency is controlled through the configuration file.

---

## Example Research Questions

TGO-II is built to investigate questions such as:

1. Do clustered layers perform redundant computations?
2. Do adjacent layers occupy similar representational subspaces?
3. Does token/feature complexity increase while layer similarity stays high?
4. Where do the major transformation zones appear in the network?
5. How does intrinsic dimension change across depth and epochs?

---

## Expected Interpretation

The combination of these observables allows TGO-II to distinguish between competing explanations:

### World A: Specialization
- CKA decreases
- SVCCA decreases
- TwoNN intrinsic dimension increases

This suggests layers are becoming more distinct and specialized.

### World B: Redundancy
- CKA remains high
- SVCCA remains high
- TwoNN intrinsic dimension increases

This suggests representational complexity increases without strong inter-layer specialization.

---

## Relation to TGO-I

TGO-I found:

- layer clustering
- rank expansion
- reduced anisotropy
- increased entropy
- possible token diversification or semantic expansion

TGO-II is the next step.

It asks whether those clustered layers are truly redundant or merely geometrically similar while still performing distinct roles.

---

## Folder Structure

A typical run produces outputs such as:

```text
results/
├── checkpoints/
│   ├── last.pth
│   └── best.pth
├── summaries/
│   ├── epoch_001.json
│   ├── epoch_002.json
│   └── ...
├── figures/
│   ├── cka_heatmap.png
│   ├── svcca_heatmap.png
│   ├── twonn_intrinsic_dimension.png
│   └── ...
└── logs/
    └── training.log
