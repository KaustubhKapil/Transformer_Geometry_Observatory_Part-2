# TGO-II: Representaion Similarity Observatory

This repository implements a **Representation Geometry Observatory** for ViT-Small/16 on ImageNet-100. It is the second part of the **Transformer Geometry Observatory (TGO)** framework.

While TGO-I explored the spectral geometry of Vision Transformers through covariance structure, eigenspectra, rank evolution, and dimensional utilization, TGO-II focuses on **representation similarity** and **intrinsic dimensionality**.

Here we analyze:

- CKA (Centered Kernel Alignment)
- SVCCA (Singular Vector Canonical Correlation Analysis)
- TwoNN Intrinsic Dimension
- Token Covariance

These observables help us understand:

- Whether adjacent Transformer layers perform similar computations
- Whether the layer clustering observed in TGO-I corresponds to genuine representational redundancy
- How representational complexity evolves throughout training
- Whether the network develops distinct transformation zones
- Whether increasing complexity is accompanied by specialization or redundancy

The primary research question of TGO-II is:

> Can representational complexity increase while adjacent Transformer layers remain highly similar?

---

## Research Motivation

TGO-I revealed two notable observations:

1. Multiple Transformer layers exhibited nearly identical spectral trajectories across Effective Rank, Stable Rank, Participation Ratio, Spectral Entropy, and Spectral Anisotropy.
2. Representational complexity increased consistently throughout training.

These observations motivate an important question:

> Are the observed layer clusters genuinely redundant, or do they perform distinct computations despite exhibiting similar spectral behavior?

TGO-II is designed to answer this question through direct representation similarity analysis.



## Files

- `tgo_v2/main.py` — entry point
- `tgo_v2/trainer.py` — training and observatory pipeline
- `tgo_v2/dataset.py` — dataset construction and fixed subset selection
- `tgo_v2/hooks.py` — ViT activation capture
- `tgo_v2/cka.py` — CKA computation
- `tgo_v2/svcca.py` — SVCCA computation
- `tgo_v2/twonn.py` — TwoNN intrinsic dimension estimation
- `tgo_v2/observatory.py` — representation observatory engine
- `tgo_v2/visualization.py` — plotting helpers



## Run

```bash
python -m tgo_v2.main --config configs/vit_small_imagenet100.yaml
```

Optional resume:

```bash
python -m tgo_v2.main \
    --config configs/vit_small_imagenet100.yaml \
    --resume results_tgo_v2/checkpoints/last.pth
```

---

## Expected Data Layout

```text
/path/to/imagenet100/train/<class_name>/*.JPEG
/path/to/imagenet100/val/<class_name>/*.JPEG
```

---

## Outputs

```text
results_tgo_v2/
├── checkpoints/
│   ├── best.pth
│   └── last.pth
│
├── summaries/
│   ├── epoch_001.json
│   ├── epoch_002.json
│   └── ...
│
├── cka/
│   ├── cka_epoch_010.png
│   ├── cka_epoch_050.png
│   └── ...
│
├── svcca/
│   ├── svcca_epoch_010.png
│   ├── svcca_epoch_050.png
│   └── ...
│
├── intrinsic_dimension/
│   ├── twonn_epoch_010.png
│   ├── twonn_epoch_050.png
│   └── ...
│
└── logs/
```



## Checkpointing

Unlike TGO-I, checkpoints are not stored every epoch.

Only two checkpoints are maintained:

```text
best.pth
last.pth
```

- `last.pth` is overwritten every epoch.
- `best.pth` is updated whenever validation accuracy improves.

This significantly reduces storage overhead during long training runs.


## Notes

- This is the second of six Transformer Geometry Observatory analyses.
- Analysis metrics are computed on a fixed validation subset.
- CKA measures representation similarity.
- SVCCA measures subspace similarity.
- TwoNN estimates local intrinsic dimensionality.
- TGO-II directly follows the findings of TGO-I.
- The observatory is designed to investigate representational redundancy, specialization, and transformation zones in Vision Transformers.

## Relation to TGO-I

TGO-I focused on:

- Covariance Geometry
- Eigenvalue Geometry
- Effective Rank
- Stable Rank
- Participation Ratio
- Spectral Entropy
- Spectral Flatness
- Spectral Anisotropy

TGO-II moves beyond spectral observables and directly studies the similarity of learned representations.

The findings of TGO-II will determine whether the clustered layers observed in TGO-I correspond to:

- Redundant computations
- Specialized computations
- Distinct representational stages

## Results

TGO-II yeilded decisive results:

- SVCCA and CKA decrease over training epochs for mean and interlayer evaluations 
- TwoNN-ID increase over the epochs

### SVCCA and CKA
It was observed that there was a significant drop in both mean and interlayer evaluations displayed an sudden fall in their respective ratio values and then stabilized in a gradual slope as the epochs progressed. This implies that the interlayer similarity decreases over time. This clearly disproves on of my prior hypothesis from TGO-I linking layer clustering to redundancy and potential removal. To add to this, it is also visible that token covarance matrix shows mild token similarities persisttant even at the end of the training schedule which might show that token diversification might not have been the dominant factor affecting rank explosion as hypothesized in TGO-I. This leaves us only with a few possibilites with the argument of **semantic exploration** becoming the major centre of experiments of the future TGOs.

### TwoNN-ID
This metric analysis brought a more exciting prospect. There was a clear rise in intrinsic dimensions or degrees of freedom over the training schedule. this leads us towards a new justification for the rise in rank of feature covariance matrix from TGO-I. **The representation/semantic manifold expands to higher degrees of freedom allowing more directions to be explored for cross-freature relations to develop**.

### Important Note
There is one striking similarity between trends observed in TGO-I and TGO-II, Block4 and Block5 transition i.e. Layer 4 and Layer 5 off 12 layers of ViT-Small/16 show a change in representation. Effective ranks and Anisotropic structures start to cluster after the 4th layer. This trend of representational variation can be seen through the adjacent layer CKA as well, layers 3 and 4 are similar but 4 and 5 show sudden mismatches. This is persistant through out the remaining results as well.
