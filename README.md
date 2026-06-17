# TGO-Part-2

This repository implements a Representation Geometry Observatory for ViT-Small/16 on ImageNet-100. It is the second part of the Transformer Geometry Observatory (TGO) framework.

While TGO-I explored the spectral geometry of Vision Transformers through covariance structure, eigenspectra, and dimensional utilization, TGO-II focuses on **representation similarity** and **intrinsic dimensionality**.

Here we analyze:

- CKA (Centered Kernel Alignment)
- SVCCA (Singular Vector Canonical Correlation Analysis)
- TwoNN Intrinsic Dimension

These observables help us understand:

- Whether adjacent Transformer layers perform similar computations
- Whether the layer clustering observed in TGO-I corresponds to genuine representational redundancy
- How representational complexity evolves throughout training
- Whether the network develops distinct transformation zones
- Whether increasing complexity is accompanied by specialization or redundancy

The primary research question of TGO-II is:

> Can representational complexity increase while adjacent Transformer layers remain highly similar?

---

## Files

- `tgo_v2/main.py` — entry point
- `tgo_v2/trainer.py` — training + observatory pipeline
- `tgo_v2/dataset.py` — dataset construction and fixed subset selection
- `tgo_v2/hooks.py` — ViT forward-hook activation capture
- `tgo_v2/cka.py` — CKA computation
- `tgo_v2/svcca.py` — SVCCA computation
- `tgo_v2/twonn.py` — TwoNN intrinsic dimension estimation
- `tgo_v2/observatory.py` — observatory orchestration
- `tgo_v2/visualization.py` — plotting helpers

---

## Run

```bash
python -m tgo_v2.main --config configs/vit_small_imagenet100.yaml
