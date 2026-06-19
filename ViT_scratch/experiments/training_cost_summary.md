# Training Cost Summary: Training Time / Inference Cost / Memory Consumption

> Extracted from `ViT_scratch/experiments/*/training.log`
> Generated: 2026-06-09

---

## Experiment 1: Image Classification / NYUv2 Accuracy / Default Learning Rate

**Common settings:** NYUv2 Accuracy, 30 epochs, learning rate = 0.001

| # | Fusion Method | α | β | Training Time | Training Time (h:m) | GPU Used (MB) | GPU Reserved (MB) | CPU Mem Used (GB) | CPU Mem Total (GB) | Log Directory |
|:-:|:---|:---:|:---:|---:|:---:|---:|---:|---:|---:|:---|
| 1 | Late-Fusion | 0.0 | 0.0 | 7,917.69 s | 2h 12m | 19.05 | 60 | 11.12 | 15.82 | `NYU_latefusion_lr1e-3` |
| 2 | Share-Fusion | 0.0 | 0.5 | 16,305.87 s | 4h 32m | 19.05 | 60 | 13.11 | 15.82 | `NYU_sharefusion_a0.0_b0.5_lr1e-3` |
| 3 | Share-Fusion | 0.5 | 0.0 | 7,818.84 s | 2h 10m | 19.05 | 60 | 9.72 | 15.82 | `NYU_sharefusion_a0.5_b0.0_lr1e-3` |
| 4 | Share-Fusion | 0.5 | 0.5 | 14,739.99 s | 4h 06m | 19.05 | 60 | 12.30 | 15.82 | `NYU_sharefusion_a0.5_b0.5_lr1e-3` |
| 5 | Share-Fusion | 0.25 | 0.25 | 13,841.06 s | 3h 51m | 19.05 | 60 | 11.43 | 15.82 | `NYU_sharefusion_a0.25_b0.25_lr1e-3` |
| 6 | Share-Fusion | learn | learn | — *(revise: 13,205.90 s / 3h 40m)* | — | 19.07 | 60–62 | 11.91 | 15.82 | `NYU_sharefusion_alearn_blearn_lr1e-3` |
| 7 | AR-Fusion | learn | learn | — *(incomplete: stopped at epoch 15)* | — | 19.07 | 60–64 | 13.67 | 15.82 | `NYU_ARfusion_alearn_blearn_lr1e-3` |

**Notes:**
- #6: Time not recorded in original log. Revised run (`_lr1e-3_revise`) recorded 13,205.90 s.
- #7: Training stopped at epoch 15 (out of 30). No Time recorded.

---

## Experiment 2: Image Classification / NYUv2 Accuracy / Adjusted Learning Rate

**Common settings:** NYUv2 Accuracy, 30 epochs, lr = 0.01 (epochs 0–9) → 0.001 (epochs 10–29)

| # | Fusion Method | α | β | Training Time | Training Time (h:m) | GPU Used (MB) | GPU Reserved (MB) | CPU Mem Used (GB) | CPU Mem Total (GB) | Log Directory |
|:-:|:---|:---:|:---:|---:|:---:|---:|---:|---:|---:|:---|
| 1 | Late-Fusion | 0.0 | 0.0 | — | — | 19.05 | 60 | 11.70 | 15.82 | `NYU_latefusion` |
| 2 | Share-Fusion | 0.0 | 0.5 | — | — | 19.05 | 60 | 12.78 | 15.82 | `NYU_sharefusion_a0.0_b0.5` |
| 3 | Share-Fusion | 0.5 | 0.0 | — | — | 19.05 | 60 | 12.35 | 15.82 | `NYU_sharefusion_a0.5_b0.0` |
| 4 | Share-Fusion | 0.5 | 0.5 | — | — | 19.05 | 60 | 13.74 | 15.82 | `NYU_sharefusion_a0.5_b0.5` |
| 5 | Share-Fusion | 0.25 | 0.25 | — | — | 19.05 | 60 | 12.59 | 15.82 | `NYU_sharefusion_a0.25_b0.25` |
| 6 | Share-Fusion | learn | learn | 16,970.83 s | 4h 43m | 19.06–19.07 | 56–64 | 11.00 | 15.82 | `NYU_sharefusion_alearn_blearn` |
| 7 | AR-Fusion | learn | learn | 26,203.61 s | 7h 17m | 19.07 | 60 | (variable) | 15.82 | `NYU_ARfusion_alearn_blearn` |

**Notes:**
- #1–#5: Time not recorded (older log format without timer).
- #6: Log contains two runs (first aborted at epoch 8; second completed all 30 epochs). Time reflects the second run only.
- #7: Longest training time across all experiments.

---

## Experiment 3: Object Detection / COCO RGB-D / DETR / AP Metrics

**Common settings:** COCO RGB-D, DETR, RGB encoder frozen, Transformer encoder frozen, fine-tune depth embedding and depth encoder

| # | Fusion Method | α | β | Training Time | GPU Memory | CPU Memory | Status |
|:-:|:---|:---:|:---:|:---:|:---:|:---:|:---|
| 1 | Late-Fusion | 0.0 | 0.0 | — | — | — | **Log not found** |
| 2 | Share-Fusion | 0.0 | 0.5 | — | — | — | **Log not found** |
| 3 | Share-Fusion | 0.5 | 0.0 | — | — | — | **Log not found** |
| 4 | Share-Fusion | 0.5 | 0.5 | — | — | — | **Log not found** |
| 5 | Share-Fusion | 0.25 | 0.25 | — | — | — | **Log not found** |
| 6 | Share-Fusion | learn | learn | — | — | — | **Log not found** |
| 7 | AR-Fusion | learn | learn | — | — | — | **Log not found** |

> No COCO RGB-D / DETR experiment directories or training.log files exist in the repository.

---

## Experiment 4: Attention Map Consistency / Default Learning Rate

**Common settings:** Image Classification, default learning rate (lr = 0.001)
**Metrics:** WRGBD Rf / WRGBD top-80% / NYUv2 Rf / NYUv2 top-80% / CMI

> Same training runs as Experiment 1. Training time and memory are identical.

| # | Fusion Method | α | β | Training Time | Training Time (h:m) | GPU Used (MB) | GPU Reserved (MB) | CPU Mem Used (GB) | Reference Log |
|:-:|:---|:---:|:---:|---:|:---:|---:|---:|---:|:---|
| 1 | Late-Fusion | 0.0 | 0.0 | 7,917.69 s | 2h 12m | 19.05 | 60 | 11.12 | `NYU_latefusion_lr1e-3` |
| 2 | Share-Fusion | 0.0 | 0.5 | 16,305.87 s | 4h 32m | 19.05 | 60 | 13.11 | `NYU_sharefusion_a0.0_b0.5_lr1e-3` |
| 3 | Share-Fusion | 0.5 | 0.0 | 7,818.84 s | 2h 10m | 19.05 | 60 | 9.72 | `NYU_sharefusion_a0.5_b0.0_lr1e-3` |
| 4 | Share-Fusion | 0.5 | 0.5 | 14,739.99 s | 4h 06m | 19.05 | 60 | 12.30 | `NYU_sharefusion_a0.5_b0.5_lr1e-3` |
| 5 | Share-Fusion | 0.25 | 0.25 | 13,841.06 s | 3h 51m | 19.05 | 60 | 11.43 | `NYU_sharefusion_a0.25_b0.25_lr1e-3` |
| 6 | Share-Fusion | learn | learn | — *(revise: 13,205.90 s / 3h 40m)* | — | 19.07 | 60–62 | 11.91 | `NYU_sharefusion_alearn_blearn_lr1e-3` |
| 7 | AR-Fusion | learn | learn | — *(incomplete)* | — | 19.07 | 60–64 | 13.67 | `NYU_ARfusion_alearn_blearn_lr1e-3` |

---

## Experiment 5: Attention Map Consistency / Adjusted Learning Rate

**Common settings:** Image Classification, adjusted learning rate (lr = 0.01 → 0.001)
**Metrics:** Rf / top-Attn80%

> Same training runs as Experiment 2. Training time and memory are identical.

| # | Fusion Method | α | β | Training Time | Training Time (h:m) | GPU Used (MB) | GPU Reserved (MB) | CPU Mem Used (GB) | Reference Log |
|:-:|:---|:---:|:---:|---:|:---:|---:|---:|---:|:---|
| 1 | Late-Fusion | 0.0 | 0.0 | — | — | 19.05 | 60 | 11.70 | `NYU_latefusion` |
| 2 | Share-Fusion | 0.0 | 0.5 | — | — | 19.05 | 60 | 12.78 | `NYU_sharefusion_a0.0_b0.5` |
| 3 | Share-Fusion | 0.5 | 0.0 | — | — | 19.05 | 60 | 12.35 | `NYU_sharefusion_a0.5_b0.0` |
| 4 | Share-Fusion | 0.5 | 0.5 | — | — | 19.05 | 60 | 13.74 | `NYU_sharefusion_a0.5_b0.5` |
| 5 | Share-Fusion | 0.25 | 0.25 | — | — | 19.05 | 60 | 12.59 | `NYU_sharefusion_a0.25_b0.25` |
| 6 | Share-Fusion | learn | learn | 16,970.83 s | 4h 43m | 19.06–19.07 | 56–64 | 11.00 | `NYU_sharefusion_alearn_blearn` |
| 7 | AR-Fusion | learn | learn | 26,203.61 s | 7h 17m | 19.07 | 60 | (variable) | `NYU_ARfusion_alearn_blearn` |

---

## Cross-Experiment Summary

### Training Time Comparison (available data only)

| Fusion Method | α | β | Exp1 (lr=0.001) | Exp2 (adjusted lr) |
|:---|:---:|:---:|---:|---:|
| Late-Fusion | 0.0 | 0.0 | 7,917.69 s (2h 12m) | — |
| Share-Fusion | 0.0 | 0.5 | 16,305.87 s (4h 32m) | — |
| Share-Fusion | 0.5 | 0.0 | 7,818.84 s (2h 10m) | — |
| Share-Fusion | 0.5 | 0.5 | 14,739.99 s (4h 06m) | — |
| Share-Fusion | 0.25 | 0.25 | 13,841.06 s (3h 51m) | — |
| Share-Fusion | learn | learn | *13,205.90 s (3h 40m)** | 16,970.83 s (4h 43m) |
| AR-Fusion | learn | learn | *(incomplete)* | 26,203.61 s (7h 17m) |

*\* revise version*

### GPU Memory Consumption

All fusion methods show nearly identical GPU memory usage:

| Phase | GPU Used (MB) | GPU Reserved (MB) |
|:---|---:|---:|
| Before validation | 19.05–19.07 | 56–64 |
| Test batch 0 | 19.30–19.32 | 56–64 |
| Test batch 100+ | 19.74–19.76 | 56–64 |

> No significant difference in GPU memory across fusion methods.

### Inference Cost

Inference time per batch is **not recorded** in the training logs. Only GPU/CPU memory snapshots during validation batches (0, 100, 200, 300) are available.

### Data Gaps

| Gap | Affected Conditions |
|:---|:---|
| Training Time not recorded | Exp1 #6, #7; Exp2 #1–#5 |
| Training incomplete | Exp1 #7 (AR-Fusion, stopped at epoch 15) |
| No experiment logs exist | Exp3 (COCO RGB-D / DETR), all 7 conditions |
