# Inference Cost Summary: Time / Inference Cost / Memory Consumption

> Extracted from `ViT_scratch/experiments/*/inference.log`
> Generated: 2026-06-09

---

## Common Settings

| Parameter | Value |
|:---|:---|
| Device | cuda |
| Batch size | 16 |
| Seed | 42 |
| Test samples | 5,069 |
| Total batches | 317 |

---

## Experiment 1: Default Learning Rate (lr = 0.001)

| # | Fusion Method | α | β | Total Time (s) | Per-sample (ms) | Per-batch (ms) | Throughput (samples/s) | GPU Peak Alloc (MB) | GPU Peak Rsv (MB) | CPU Used (GB) | CPU Total (GB) | Log Directory |
|:-:|:---|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|
| 1 | Late-Fusion | 0.0 | 0.0 | **427.74** | **84.38** | **1,349.32** | **11.85** | 133.17 | 168 | 14.75 | 15.82 | `NYU_latefusion_lr1e-3` |
| 2 | Share-Fusion | 0.0 | 0.5 | 59.15 | 11.67 | 186.60 | 85.69 | 135.27 | 168 | 14.92 | 15.82 | `NYU_sharefusion_a0.0_b0.5_lr1e-3` |
| 3 | Share-Fusion | 0.5 | 0.0 | 39.35 | 7.76 | 124.12 | 128.83 | 135.27 | 168 | 14.97 | 15.82 | `NYU_sharefusion_a0.5_b0.0_lr1e-3` |
| 4 | Share-Fusion | 0.5 | 0.5 | 39.28 | 7.75 | 123.92 | 129.04 | 135.27 | 168 | 14.98 | 15.82 | `NYU_sharefusion_a0.5_b0.5_lr1e-3` |
| 5 | Share-Fusion | 0.25 | 0.25 | 39.39 | 7.77 | 124.27 | 128.67 | 135.27 | 168 | 14.92 | 15.82 | `NYU_sharefusion_a0.25_b0.25_lr1e-3` |
| 6 | Share-Fusion | learn | learn | 40.02 | 7.89 | 126.24 | 126.67 | 135.27 | 168 | 14.92 | 15.82 | `NYU_sharefusion_alearn_blearn_lr1e-3_revise` |
| 7 | AR-Fusion | learn | learn | 39.67 | 7.83 | 125.14 | 127.78 | 135.27 | 168 | 14.95 | 15.82 | `NYU_ARfusion_alearn_blearn_lr1e-3` |

**Notes:**
- **#1 (Late-Fusion) is a clear outlier**: 427.74 s — roughly 10× slower than all other conditions (~39 s). The GPU allocation before inference was only 1.37 MB (vs 12.12 MB for others), suggesting the model was not pre-loaded to GPU; the first batch took 128.0 ms (vs ~8–10 ms typical). This is likely a cold-start or model-loading issue, not a true inference cost difference.
- **#2 (Share-Fusion α=0.0 β=0.5)** is also ~1.5× slower (59.15 s). The first batch took only 13.2 ms, but cumulative time at batch 100 was 32.15 s (vs ~12 s typical), suggesting transient system load.
- #7 (AR-Fusion) uses `model_16.pt` (training stopped at epoch 15), not `model_final.pt`.

---

## Experiment 2: Adjusted Learning Rate (lr = 0.01 → 0.001)

| # | Fusion Method | α | β | Total Time (s) | Per-sample (ms) | Per-batch (ms) | Throughput (samples/s) | GPU Peak Alloc (MB) | GPU Peak Rsv (MB) | CPU Used (GB) | CPU Total (GB) | Log Directory |
|:-:|:---|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|
| 1 | Late-Fusion | 0.0 | 0.0 | 39.18 | 7.73 | 123.61 | 129.37 | 135.27 | 168 | 14.92 | 15.82 | `NYU_latefusion` |
| 2 | Share-Fusion | 0.0 | 0.5 | 39.71 | 7.83 | 125.27 | 127.65 | 135.27 | 168 | 14.92 | 15.82 | `NYU_sharefusion_a0.0_b0.5` |
| 3 | Share-Fusion | 0.5 | 0.0 | 39.59 | 7.81 | 124.88 | 128.05 | 135.27 | 168 | 14.96 | 15.82 | `NYU_sharefusion_a0.5_b0.0` |
| 4 | Share-Fusion | 0.5 | 0.5 | 39.51 | 7.80 | 124.65 | 128.28 | 135.27 | 168 | 14.99 | 15.82 | `NYU_sharefusion_a0.5_b0.5` |
| 5 | Share-Fusion | 0.25 | 0.25 | 39.82 | 7.86 | 125.62 | 127.29 | 135.27 | 168 | 14.95 | 15.82 | `NYU_sharefusion_a0.25_b0.25` |
| 6 | Share-Fusion | learn | learn | 40.15 | 7.92 | 126.66 | 126.25 | 135.27 | 168 | 14.93 | 15.82 | `NYU_sharefusion_alearn_blearn` |
| 7 | AR-Fusion | learn | learn | 39.88 | 7.87 | 125.80 | 127.11 | 135.27 | 168 | 14.94 | 15.82 | `NYU_ARfusion_alearn_blearn` |

**Notes:**
- All conditions show consistent inference time (~39–40 s). No outliers.

---

## Cross-Experiment Summary

### Inference Time Comparison

| Fusion Method | α | β | Exp1 (lr=0.001) | Exp2 (adjusted lr) |
|:---|:---:|:---:|---:|---:|
| Late-Fusion | 0.0 | 0.0 | *427.74 s (outlier)* | 39.18 s |
| Share-Fusion | 0.0 | 0.5 | *59.15 s (elevated)* | 39.71 s |
| Share-Fusion | 0.5 | 0.0 | 39.35 s | 39.59 s |
| Share-Fusion | 0.5 | 0.5 | 39.28 s | 39.51 s |
| Share-Fusion | 0.25 | 0.25 | 39.39 s | 39.82 s |
| Share-Fusion | learn | learn | 40.02 s | 40.15 s |
| AR-Fusion | learn | learn | 39.67 s | 39.88 s |

### Throughput Comparison (samples/s)

| Fusion Method | α | β | Exp1 (lr=0.001) | Exp2 (adjusted lr) |
|:---|:---:|:---:|---:|---:|
| Late-Fusion | 0.0 | 0.0 | *11.85* | 129.37 |
| Share-Fusion | 0.0 | 0.5 | *85.69* | 127.65 |
| Share-Fusion | 0.5 | 0.0 | 128.83 | 128.05 |
| Share-Fusion | 0.5 | 0.5 | 129.04 | 128.28 |
| Share-Fusion | 0.25 | 0.25 | 128.67 | 127.29 |
| Share-Fusion | learn | learn | 126.67 | 126.25 |
| AR-Fusion | learn | learn | 127.78 | 127.11 |

### GPU Memory Consumption (During Inference)

All fusion methods show identical GPU memory growth pattern:

| Phase | GPU Allocated (MB) | GPU Reserved (MB) |
|:---|---:|---:|
| Before inference | 12.12–12.13 | 24 |
| Batch 0 | 21.01 | 48 |
| Batch 100 | 59.10 | 88 |
| Batch 200 | 97.18–97.19 | 128 |
| Batch 300 (peak) | 135.27 | 168 |
| After inference | 132.78–132.79 | 152–156 |

> Exception: Exp1 #1 (Late-Fusion lr=0.001) started at 1.37 MB allocated / 2 MB reserved, suggesting model was not pre-loaded.

### CPU Memory Consumption

| Metric | Range |
|:---|---:|
| CPU Used (after inference) | 14.75–14.99 GB |
| CPU Total | 15.82 GB |
| CPU Available | 0.83–1.07 GB |

> No significant difference in CPU memory across fusion methods or experiments.

### Key Findings

1. **Inference time is effectively identical across all fusion methods** (~39–40 s for 5,069 samples), once outlier runs are excluded.
2. **Learnable α/β adds negligible overhead** — Share-Fusion (learn) and AR-Fusion (learn) are within 1 s of fixed-parameter variants.
3. **GPU memory profile is uniform**: all methods peak at ~135 MB allocated / 168 MB reserved regardless of fusion type or α/β values.
4. **Exp1 #1 and #2 are outliers** due to apparent system/loading issues, not architectural differences.
