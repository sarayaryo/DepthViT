# DepthViT

RGB-D マルチモーダル Vision Transformer による画像分類の実験フレームワーク。

---

## Setup

```bash
pip install -r requirements.txt
cd ViT_scratch
```

---

## Arguments

### train.py

| Argument | Type | Default | Description |
|:---|:---:|:---:|:---|
| `--exp_name` | str | `vit-with-10-epochs` | 実験名（ログ・モデル保存先のフォルダ名） |
| `--method` | int | `0` | モデル: `0`=ViT (RGB only), `1`=Early-Fusion, `2`=Late-Fusion |
| `--dataset_type` | int | `1` | データセット: `0`=W-RGBD, `1`=NYUv2, `2`=TinyImageNet, `3`=SUN RGB-D |
| `--dataset` | str | `rod_sample` | `../data/` 配下のデータセットフォルダ名 |
| `--dataset_path` | str | `None` | データセットの絶対パス（指定時は `--dataset` を上書き） |
| `--batch_size` | int | `4` | バッチサイズ |
| `--epochs` | int | `3` | エポック数 |
| `--lr` | float | `1e-4` | 学習率 |
| `--alpha` | float | `0.5` | Share-Fusion の alpha |
| `--beta` | float | `0.5` | Share-Fusion の beta（MI loss 重み） |
| `--topk` | float | `0.1` | Attention map の precision@k |
| `--max_data_size` | int | `100000` | データ件数の上限 |
| `--save_model_every` | int | `10` | N エポックごとにモデル保存 |
| `--weight_decay` | float | `0.02` | Optimizer の weight decay |
| `--device` | str | `cuda` | `cuda` or `cpu` |

### config (train.py 内)

| Key | Description |
|:---|:---|
| `use_method1` | `True`: Share-Fusion / Late-Fusion, `False`: 無効 |
| `use_method3` | `True`: Agreement-Refined Fusion |
| `learnable_alpha_beta` | `True`: alpha/beta を学習可能パラメータにする |

---

## Training

### Quick test

```bash
python train.py --method 2 --dataset rod_sample --batch_size 4 --epochs 3 --max_data_size 10
```

### W-RGBD (dataset_type=0)

```bash
python train.py --method 2 --dataset_type 0 --exp_name WRGBD_sharefusion \
  --alpha 0.5 --beta 0.5 \
  --dataset rgbd-dataset-10k --batch_size 16 --epochs 30 --lr 1e-3
```

### NYUv2 (dataset_type=1)

Late-Fusion (alpha=0, beta=0):
```bash
python train.py --method 2 --dataset_type 1 --exp_name NYU_latefusion_lr1e-3 \
  --alpha 0.0 --beta 0.0 \
  --dataset nyu_data/nyu2 --batch_size 16 --epochs 30 --lr 1e-3
```

Share-Fusion (alpha=0.5, beta=0.5):
```bash
python train.py --method 2 --dataset_type 1 --exp_name NYU_sharefusion_a0.5_b0.5_lr1e-3 \
  --alpha 0.5 --beta 0.5 \
  --dataset nyu_data/nyu2 --batch_size 16 --epochs 30 --lr 1e-3
```

Share-Fusion (learnable alpha/beta): config 内で `"learnable_alpha_beta": True` に設定。

AR-Fusion: config 内で `"use_method3": True` に設定。

### TinyImageNet (dataset_type=2)

```bash
python train.py --method 2 --dataset_type 2 --exp_name TinyImageNet_sharefusion \
  --alpha 0.5 --beta 0.5 \
  --dataset rgbd_tinyimagenet --batch_size 16 --epochs 30 --lr 1e-3
```

### SUN RGB-D (dataset_type=3)

SUN RGB-D は `../data/` 配下にないため、`--dataset_path` で絶対パスを指定する。

Late-Fusion:
```bash
python train.py --method 2 --dataset_type 3 --exp_name SUNRGBD_latefusion_lr1e-3 \
  --alpha 0.0 --beta 0.0 \
  --dataset_path "S:\SUNRGBD\SUNRGBD" --batch_size 16 --epochs 30 --lr 1e-3
```

Share-Fusion (alpha=0.5, beta=0.5):
```bash
python train.py --method 2 --dataset_type 3 --exp_name SUNRGBD_sharefusion_a0.5_b0.5_lr1e-3 \
  --alpha 0.5 --beta 0.5 \
  --dataset_path "S:\SUNRGBD\SUNRGBD" --batch_size 16 --epochs 30 --lr 1e-3
```

Share-Fusion (alpha=0.0, beta=0.5):
```bash
python train.py --method 2 --dataset_type 3 --exp_name SUNRGBD_sharefusion_a0.0_b0.5_lr1e-3 \
  --alpha 0.0 --beta 0.5 \
  --dataset_path "S:\SUNRGBD\SUNRGBD" --batch_size 16 --epochs 30 --lr 1e-3
```

Share-Fusion (alpha=0.5, beta=0.0):
```bash
python train.py --method 2 --dataset_type 3 --exp_name SUNRGBD_sharefusion_a0.5_b0.0_lr1e-3 \
  --alpha 0.5 --beta 0.0 \
  --dataset_path "S:\SUNRGBD\SUNRGBD" --batch_size 16 --epochs 30 --lr 1e-3
```

Share-Fusion (alpha=0.25, beta=0.25):
```bash
python train.py --method 2 --dataset_type 3 --exp_name SUNRGBD_sharefusion_a0.25_b0.25_lr1e-3 \
  --alpha 0.25 --beta 0.25 \
  --dataset_path "S:\SUNRGBD\SUNRGBD" --batch_size 16 --epochs 30 --lr 1e-3
```

---

## Inference (test_for_inferencelog.py)

学習済みモデルの推論性能（時間・メモリ・精度）を測定する。

```bash
# 全14条件、フルデータ
python test_for_inferencelog.py

# サンプル200件で1条件だけ試す（動作確認）
python test_for_inferencelog.py --max_samples 200 --only NYU_latefusion_lr1e-3

# サンプル200件で default lr 7条件を回す
python test_for_inferencelog.py --max_samples 200 --group default
```

出力先: `experiments/<exp_name>/inference.log`

記録内容:
- 実験メタ情報（名前、lr、checkpoint、デバイス、サンプル数）
- 推論前後の CPU/GPU メモリ
- 100 バッチごとのメモリスナップショットと経過時間
- 総時間 / サンプル単位時間 (ms) / バッチ単位時間 (ms) / スループット (samples/s)
- Accuracy / Loss / CMI

---

## Datasets

| dataset_type | Dataset | Path | Image Count | Categories |
|:---:|:---|:---|---:|---:|
| 0 | W-RGBD | `../data/rgbd-dataset-10k/` | ~10,000 | 51 |
| 1 | NYUv2 | `../data/nyu_data/nyu2/` | ~75,000 | 27 |
| 2 | TinyImageNet | `../data/rgbd_tinyimagenet/` | - | - |
| 3 | SUN RGB-D | `S:\SUNRGBD\SUNRGBD\` | 10,335 | 45 |

SUN RGB-D の depth 画像はビットシフトエンコード（16bit PNG）されており、データローダー内で自動的にデコードされる。詳細は `data/data_shape.md` を参照。

---

## Directory Structure

```
DepthViT/
  data/
    nyu_data/nyu2/          # NYUv2
    rgbd-dataset-10k/       # W-RGBD (mini)
    rgbd_tinyimagenet/      # TinyImageNet
    data_shape.md           # Dataset structure comparison
  ViT_scratch/
    train.py                # Training script
    data.py                 # Dataset / DataLoader definitions
    test.py                 # Evaluation
    test_for_inferencelog.py # Inference benchmarking
    experiments/            # Experiment logs and saved models
      training_cost_summary.md
      inference_cost_summary.md
      NYU_latefusion_lr1e-3/
      NYU_sharefusion_*/
      ...
```
