# Experiment Log

---

## 2026-06-17: SUN RGB-D 精度改善の検討

### 背景

SUN RGB-D 19カテゴリ版の分類精度が、NYUv2と比較して著しく低い。

| データセット | ベスト精度 | クラス数 | データ量（推定） | タスクの性質 |
|---|---|---|---|---|
| NYUv2 | 99.4% | 25 | ~40,000枚 | 物体レベル分類（desk, bed, chair 等） |
| SUN RGB-D 19cat | 41.8% | 19 | ~7,600枚 | シーンレベル分類（bedroom, office, kitchen 等） |

NYUv2は物体の粗い形状で判別可能だが、SUN RGB-Dのシーン分類は部屋全体の空間構造・家具配置の把握が必要であり、本質的に難易度が高い。

### 現状の問題

SUN RGB-D 19cat Late-Fusion lr=1e-3（最良条件）の学習曲線:

| epoch | Train loss | Valid loss | Gap | Valid Acc |
|---|---|---|---|---|
| 0 | 2.64 | 2.59 | -0.05 | 0.193 |
| 5 | 2.20 | 2.26 | +0.06 | 0.294 |
| 10 | 1.82 | 2.02 | +0.20 | 0.377 |
| 15 | 1.71 | 1.98 | +0.27 | 0.391 |
| 20 | 1.65 | 1.98 | +0.33 | 0.386 |
| 25 | 1.61 | 1.93 | +0.32 | 0.411 |
| Test | 1.57 | 1.89 | +0.32 | 0.418 |

- epoch 10以降 valid loss が横ばい、train loss のみ低下 → 過学習
- epoch を増やしても改善見込みなし

### 原因分析

1. **入力解像度が低すぎる（32x32）** — シーン分類に必要な空間情報が失われている
2. **正則化がない** — dropout が全て 0.0 のため、少ないデータで過学習しやすい
3. **データ量の差** — NYUv2 の約 1/5 のデータ量で、より難しいタスクを解いている

### 変更内容

#### 1. Dropout 追加

変更ファイル: `train.py` config（177-178行目）

| パラメータ | 変更前 | 変更後 |
|---|---|---|
| `hidden_dropout_prob` | 0.0 | 0.1 |
| `attention_probs_dropout_prob` | 0.0 | 0.1 |

vit.py 側は config の値を参照しているため変更不要。

#### 2. 入力解像度の拡大

変更ファイル: `train.py` config（180行目）+ transform（696-697行目）

| パラメータ | 変更前 | 変更後 |
|---|---|---|
| `image_size` | 32 | 64 |
| `transforms.Resize` | (32, 32) | (64, 64) |

patch_size=4 のままだとパッチ数が 64→256 に増加（メモリ・計算量 約4倍）。
メモリが厳しい場合は patch_size=8 にすればパッチ数は 64 のまま。

#### 3. スケジューラの解除

変更ファイル: `train.py`（756行目付近）

MultiStepLR(milestones=[10], gamma=0.1) を無効化。lr=1e-3 を固定で使用。

### 補足: learnable_alpha_beta バグの修正（同日）

`train.py` config の `learnable_alpha_beta` がデフォルト True にハードコードされていたため、`--alpha 0.0 --beta 0.0`（Late-Fusion）を指定しても無視され、sigmoid(0)=0.5 からの学習可能パラメータとして動作していた。

- `config["learnable_alpha_beta"]` のデフォルトを False に変更
- argparse に `--learnable_alpha_beta` フラグを追加
- 非19cat版の SUNRGBD 実験はこのバグの影響を受けていた（早期停止・低精度）

---

## 2026-06-19: SUN RGB-D 精度改善 第2弾 — epochs 60 + LR スケジュール比較

### 背景

前回の変更（dropout=0.1, image_size=64, scheduler OFF）を適用した結果:

| 条件 | Test Acc | Valid Loss (ep25) | Train-Valid Gap |
|---|---|---|---|
| Late-Fusion (旧 32x32, scheduler ON) | 41.8% | 1.93 | +0.32 |
| Late-Fusion (新 64x64, scheduler OFF) | 42.6% | 1.89 | +0.30 |
| Share-Fusion (旧 32x32, scheduler ON) | 37.8% | 1.99 | +0.32 |
| **Share-Fusion (新 64x64, scheduler OFF)** | **43.0%** | **1.81** | **+0.22** |

Share-Fusion 新設定では epoch 20→25 で valid loss が 1.964→1.805 と大幅に改善しており、**30 epoch 時点で学習が飽和していない**。epochs を増やし、LR スケジュールを最適化することで更なる改善を狙う。

### 変更内容

#### 1. epochs: 30 → 60

30 epoch で valid loss がまだ下降トレンドにあるため、倍に増やして飽和点を探る。

#### 2. patience: 3 → 5

変更ファイル: `train.py` 行771

validation は 5 epoch ごとに実行される。patience=3 だと 15 epoch 連続で改善なしで早期停止。60 epoch 訓練では tight すぎるため、patience=5（25 epoch 分の許容）に緩和。

#### 3. `--scheduler` 引数の追加

変更ファイル: `train.py`（argparse + scheduler 生成ロジック）

LR スケジュールをコマンドラインで切り替え可能にした。

| 値 | 挙動 |
|---|---|
| `--scheduler none` | スケジューラなし。lr=1e-3 を全 epoch 固定（デフォルト） |
| `--scheduler cosine` | `CosineAnnealingLR(T_max=epochs, eta_min=1e-5)` |

#### CosineAnnealingLR の lr 変化（60 epochs, lr_init=1e-3, eta_min=1e-5）

```
lr = eta_min + 0.5 * (lr_init - eta_min) * (1 + cos(π * epoch / T_max))
```

| Epoch | lr（概算） |
|---|---|
| 0 | 1.000e-3 |
| 5 | 9.66e-4 |
| 10 | 8.70e-4 |
| 15 | 7.50e-4 |
| 20 | 5.00e-4 |
| 25 | 3.47e-4 |
| 30 | 5.00e-4 |
| 35 | 2.50e-4 |
| 40 | 1.31e-4 |
| 45 | 5.05e-5 |
| 50 | 1.42e-5 |
| 55 | 1.05e-5 |
| 60 | 1.00e-5 |

序盤は lr ≈ 1e-3 を維持し、後半に向けて滑らかに減衰する。旧 MultiStepLR(milestones=[10]) が epoch 10 で lr を 1e-3→1e-4 に急降下させていたのと対照的に、学習を途中で止めない。

### 実験条件（4条件）

| # | Fusion | LR スケジュール | exp_name |
|---|---|---|---|
| 1 | Late-Fusion (α=0.0, β=0.0) | lr=1e-3 固定 | `SUNRGBD_19cat_latefusion_lr1e-3_ep60` |
| 2 | Late-Fusion (α=0.0, β=0.0) | Cosine (1e-3→1e-5) | `SUNRGBD_19cat_latefusion_lr1e-3_cosine_ep60` |
| 3 | Share-Fusion (α=0.5, β=0.5) | lr=1e-3 固定 | `SUNRGBD_19cat_sharefusion_a0.5_b0.5_lr1e-3_ep60` |
| 4 | Share-Fusion (α=0.5, β=0.5) | Cosine (1e-3→1e-5) | `SUNRGBD_19cat_sharefusion_a0.5_b0.5_lr1e-3_cosine_ep60` |

共通設定: image_size=64, dropout=0.1, batch_size=16, weight_decay=0.02, patch_size=4

所要時間見積もり: 約2時間/条件 × 4 = 約8時間

### 結果

| # | 条件 | Best Valid Loss (epoch) | Best Valid Acc | Test Acc | Train-Valid Gap (best) |
|---|---|---|---|---|---|
| 1 | Late-Fusion 固定 | 1.902 (ep45) | 44.7% | 43.3% | +0.58 |
| 2 | Late-Fusion cosine | 1.965 (ep35) | 41.7% | 44.5% | +0.60 |
| 3 | **Share-Fusion 固定** | **1.815 (ep45)** | **45.8%** | **49.6%** | +0.51 |
| 4 | Share-Fusion cosine | 1.891 (ep40) | 45.1% | 45.7% | +0.68 |

※ Test Acc は最終 epoch（ep59）のモデルで評価。ベストチェックポイントでの評価ではない（後日修正済み）。

#### 最良条件の学習曲線（Share-Fusion lr=1e-3 固定）

| Epoch | Train Loss | Valid Loss | Gap | Valid Acc |
|---|---|---|---|---|
| 0 | 2.658 | 2.572 | -0.09 | 20.8% |
| 10 | 2.051 | 2.121 | -0.07 | 32.6% |
| 20 | 1.773 | 1.960 | +0.19 | 41.1% |
| 25 | 1.645 | 1.864 | +0.22 | 43.6% |
| 30 | 1.534 | 1.895 | +0.34 | 42.3% |
| 40 | 1.377 | 1.856 | +0.48 | 43.8% |
| 45 | 1.307 | **1.815** | +0.51 | **45.8%** |
| 50 | 1.252 | 1.908 | +0.66 | 45.3% |
| 55 | 1.201 | 1.858 | +0.66 | 46.4% |
| Test | — | 1.817 | — | 49.6% |

### 分析

#### lr 固定 > Cosine 減衰

両 Fusion とも lr=1e-3 固定が cosine より好成績。Cosine は epoch 25 で lr=6.3e-4、epoch 35 で lr=3.8e-4 と、まだ学習可能な段階で lr が下がりすぎた。このモデル規模・データ量では、lr=1e-3 を長く維持する方が有効。

→ **Cosine スケジューラは不採用**

#### Share-Fusion > Late-Fusion

全条件で Share-Fusion (α=0.5, β=0.5) が Late-Fusion を上回った。RGB と Depth の固定重み付き融合が、独立エンコード後の結合より効果的。

#### epochs 延長の効果

30 epoch (前回ベスト 43.0%) → 60 epoch (49.6%) で +6.6% の改善。Best valid loss は epoch 45 で記録されており、30 epoch では学習途中で打ち切られていたことが確認された。

#### 過学習の進行

epoch 45 以降、Train-Valid Gap が 0.5→0.66 に拡大。Train loss は下がり続けるが Valid loss は横ばい〜微増。dropout=0.1 だけでは長期学習での過学習抑制が不十分。

#### Test Acc と Valid Acc の乖離

Share-Fusion 固定の Test Acc (49.6%) が Best Valid Acc (45.8%) より高い。テストは最終 epoch のモデルで評価されており、ベストチェックポイントでの評価ではなかったため、正確な比較ができていなかった。

---

## 2026-06-19: コード修正 — テスト時にベストモデルをロード

### 問題

テストフェーズは最終 epoch のモデル状態で実行されていたため、ベストチェックポイント（valid loss 最小時）の性能を正確に評価できていなかった。

### 変更内容

`train.py` の `train()` メソッド:

1. `best_epoch` 変数を追加し、valid loss が改善した際の epoch を記録
2. テストフェーズの直前で `model_{best_epoch}.pt` をロードしてからテストを実行
3. ロードした epoch をログに記録

---

## 2026-06-19: SUN RGB-D 精度改善 第3弾 — lr=1e-2→1e-3 ステップ + 100 epochs

### 背景

第2弾で lr=1e-3 固定 60 epoch が最良（49.6%）だったが、以下の課題がある:

- epoch 45 で best valid loss を記録後、過学習が進行
- 序盤の学習速度をさらに上げることで、より早く良い表現を獲得できる可能性
- 100 epoch に延長することで、lr=1e-3 フェーズでの学習時間を十分に確保

### 変更内容

#### 1. `--scheduler step` の追加

変更ファイル: `train.py`（argparse + scheduler 生成ロジック）

| 値 | 挙動 |
|---|---|
| `--scheduler step` | `MultiStepLR(milestones=[10], gamma=0.1)` — epoch 10 で lr を 1/10 に |

`--lr 1e-2 --scheduler step` で使用:

| Epoch | lr |
|---|---|
| 0-9 | 1e-2 |
| 10-99 | 1e-3 |

序盤 10 epoch を高い lr=1e-2 で素早く粗い特徴を獲得し、残り 90 epoch を lr=1e-3 で精密に最適化する。

#### 2. epochs: 60 → 100

100 epoch に延長。lr=1e-3 フェーズが 90 epoch 分あるため、前回（60 epoch で lr=1e-3 が 60 epoch 分）より十分な学習時間を確保。

### 実験条件（2条件）

| # | Fusion | LR スケジュール | exp_name |
|---|---|---|---|
| 1 | Late-Fusion (α=0.0, β=0.0) | 1e-2→1e-3 step | `SUNRGBD_19cat_latefusion_lr1e-2_step_ep100` |
| 2 | Share-Fusion (α=0.5, β=0.5) | 1e-2→1e-3 step | `SUNRGBD_19cat_sharefusion_a0.5_b0.5_lr1e-2_step_ep100` |

共通設定: image_size=64, dropout=0.1, batch_size=16, weight_decay=0.02, patch_size=4, patience=5

所要時間見積もり: 約3.3時間/条件 × 2 = 約6.5時間

### 結果

| # | 条件 | Best Valid Loss (epoch) | Best Valid Acc | Test Acc | Train-Valid Gap (best) | 早期停止 |
|---|---|---|---|---|---|---|
| 1 | **Late-Fusion step** | **1.677 (ep45)** | **50.4%** | **49.3%** | +0.50 | ep70 |
| 2 | Share-Fusion step | 1.796 (ep45) | 46.2% | 43.4% | +0.45 | ep70 |

※ ベストチェックポイント（ep46 保存）をロードしてテスト評価。

#### Late-Fusion（最良条件）の学習曲線

| Epoch | lr | Train Loss | Valid Loss | Gap | Valid Acc |
|---|---|---|---|---|---|
| 0 | 1e-2 | 2.789 | 2.716 | -0.07 | 16.2% |
| 5 | 1e-2 | 2.306 | 2.278 | +0.03 | 30.1% |
| 10 | 1e-3 | 1.910 | 1.955 | -0.05 | 39.2% |
| 15 | 1e-3 | 1.697 | 1.849 | -0.15 | 42.3% |
| 20 | 1e-3 | 1.589 | 1.778 | -0.19 | 44.5% |
| 25 | 1e-3 | 1.496 | 1.741 | -0.25 | 46.7% |
| 30 | 1e-3 | 1.394 | 1.685 | -0.29 | 47.7% |
| 35 | 1e-3 | 1.323 | 1.708 | +0.39 | 48.7% |
| 40 | 1e-3 | 1.236 | 1.703 | +0.47 | 48.1% |
| 45 | 1e-3 | 1.180 | **1.677** | +0.50 | **50.4%** |
| 50 | 1e-3 | 1.110 | 1.697 | +0.59 | 49.5% |
| 55 | 1e-3 | 1.059 | 1.773 | +0.71 | 49.7% |
| 60 | 1e-3 | 0.998 | 1.781 | +0.78 | 50.7% |
| 65 | 1e-3 | 0.932 | 1.828 | +0.90 | 49.5% |
| 70 | 1e-3 | 0.887 | 1.862 | +0.98 | 50.5% |
| Test (best) | — | — | 1.714 | — | 49.3% |

#### Share-Fusion の学習曲線

| Epoch | lr | Train Loss | Valid Loss | Gap | Valid Acc |
|---|---|---|---|---|---|
| 0 | 1e-2 | 2.885 | 2.696 | -0.19 | 15.1% |
| 5 | 1e-2 | 2.425 | 2.330 | +0.10 | 29.3% |
| 10 | 1e-3 | 2.056 | 2.055 | 0.00 | 34.7% |
| 15 | 1e-3 | 1.879 | 1.971 | -0.09 | 37.3% |
| 20 | 1e-3 | 1.762 | 1.918 | -0.16 | 39.5% |
| 25 | 1e-3 | 1.665 | 1.847 | -0.18 | 41.9% |
| 30 | 1e-3 | 1.580 | 1.807 | -0.23 | 43.1% |
| 40 | 1e-3 | 1.406 | 1.805 | +0.40 | 44.1% |
| 45 | 1e-3 | 1.349 | **1.796** | +0.45 | **46.2%** |
| 50 | 1e-3 | 1.278 | 1.802 | +0.52 | 46.7% |
| 55 | 1e-3 | 1.201 | 1.811 | +0.61 | 45.9% |
| 60 | 1e-3 | 1.129 | 1.911 | +0.78 | 44.1% |
| 65 | 1e-3 | 1.083 | 1.864 | +0.78 | 47.0% |
| 70 | 1e-3 | 1.035 | 1.878 | +0.84 | 45.1% |
| Test (best) | — | — | 1.812 | — | 43.4% |

### 分析

#### Late-Fusion が逆転して過去最高精度を達成

第2弾では Share-Fusion > Late-Fusion だったが、step scheduler（lr=1e-2→1e-3）では完全に逆転。

| 実験 | Late-Fusion Best Valid Acc | Share-Fusion Best Valid Acc |
|---|---|---|
| 第2弾 (lr=1e-3 固定 60ep) | 44.7% | 45.8% |
| 第3弾 (lr=1e-2→1e-3 step) | **50.4% (+5.7%)** | 46.2% (+0.4%) |

Late-Fusion は +5.7% の大幅改善に対し、Share-Fusion は +0.4% でほぼ横ばい。lr=1e-2 の高速学習は RGB/Depth を独立にエンコードする Late-Fusion に効くが、α=0.5/β=0.5 固定混合の Share-Fusion は高い lr で混合特徴の学習が不安定になった可能性がある。

#### lr=1e-2 warmup の効果

ep10 時点の Valid Acc を比較:
- 第2弾 (lr=1e-3 固定): Late 32.6%, Share 32.6%
- 第3弾 (lr=1e-2→1e-3): Late **39.2%** (+6.6%), Share 34.7% (+2.1%)

Late-Fusion は序盤の高速学習の恩恵を大きく受けたが、Share-Fusion への効果は限定的。

#### 過学習パターン

両条件とも ep45 で best valid loss を記録後、Train loss のみ低下して Gap 拡大:
- Late: Gap 0.50 (ep45) → 0.98 (ep70)
- Share: Gap 0.45 (ep45) → 0.84 (ep70)

100 epoch を使い切れず ep70 で早期停止。dropout=0.1 では長期学習の過学習を抑えきれていない。

#### ベストモデルロードの検証

Test Acc と Best Valid Acc の差が小さく整合的:
- Late: Valid 50.4% → Test 49.3% (差 -1.1%)
- Share: Valid 46.2% → Test 43.4% (差 -2.8%)

第2弾で Valid 45.8% vs Test 49.6% と乖離していた問題が解消。

---

## 2026-06-20: SUN RGB-D 精度改善 第4弾 — lr=5e-3→1e-3 + dropout=0.2 + learnable α/β

### 背景

第3弾の結果から、以下の課題が明確になった:

1. **lr=1e-2 は Share-Fusion に高すぎる** — Late-Fusion には +5.7% の効果があったが、Share-Fusion には +0.4% しか改善せず。固定比率混合と高 lr の相性が悪い。
2. **dropout=0.1 では過学習を抑えきれない** — 両条件とも ep45 でベストを記録後、Gap が急拡大して ep70 で早期停止。100 epoch を活かしきれていない。
3. **α/β 固定が Share-Fusion のボトルネック** — 全層で α=0.5/β=0.5 固定は、層ごとに最適な RGB/Depth 混合比率を学習できない制約になっている可能性。

### 変更内容

#### 1. lr: 1e-2 → 5e-3（warmup 強度の緩和）

`MultiStepLR(milestones=[10], gamma=0.2)` に変更（5e-3 × 0.2 = 1e-3）。

| Epoch | lr |
|---|---|
| 0-9 | 5e-3 |
| 10-99 | 1e-3 |

lr=1e-2 が Share-Fusion に過剰だったため、半分の 5e-3 に緩和。序盤の加速効果を維持しつつ、Share-Fusion の安定性を確保する。

#### 2. dropout: 0.1 → 0.2

変更ファイル: `train.py` config（177-178行目）

| パラメータ | 変更前 | 変更後 |
|---|---|---|
| `hidden_dropout_prob` | 0.1 | 0.2 |
| `attention_probs_dropout_prob` | 0.1 | 0.2 |

ep45 以降の過学習拡大を抑制し、100 epoch をより有効に使うことを狙う。

#### 3. learnable α/β 条件の追加

`--learnable_alpha_beta` フラグで α/β を学習可能パラメータに。初期値 sigmoid(0.5) から各層が最適な混合比率を学習する。固定比率混合の制約を外すことで、浅い層は RGB 寄り・深い層は Depth 寄りなど、層ごとの適応が期待できる。

### 実験条件（3条件）

| # | Fusion | α/β | LR スケジュール | exp_name |
|---|---|---|---|---|
| 1 | Late-Fusion | 0.0/0.0 固定 | 5e-3→1e-3 step | `SUNRGBD_19cat_latefusion_lr5e-3_step_do0.2_ep100` |
| 2 | Share-Fusion | 0.5/0.5 固定 | 5e-3→1e-3 step | `SUNRGBD_19cat_sharefusion_a0.5_b0.5_lr5e-3_step_do0.2_ep100` |
| 3 | Share-Fusion | 0.5/0.5 学習 | 5e-3→1e-3 step | `SUNRGBD_19cat_sharefusion_a0.5_b0.5_learnable_lr5e-3_step_do0.2_ep100` |

共通設定: image_size=64, dropout=0.2, batch_size=16, weight_decay=0.02, patch_size=4, patience=5

所要時間見積もり: 約3.3時間/条件 × 3 = 約10時間

### 検証ポイント

1. **lr=5e-3 は Share-Fusion に適切か** — 第3弾の lr=1e-2 では +0.4% だった Share-Fusion が、緩和した lr でどこまで改善するか
2. **dropout=0.2 の過学習抑制効果** — ep45 以降も valid loss が改善し続け、早期停止が後方にずれるか
3. **learnable α/β の効果** — 固定版と比較して、層ごとの適応的混合が精度向上に寄与するか
4. **Late-Fusion への dropout=0.2 の影響** — 第3弾で 50.4% だった Late-Fusion が、dropout 強化と lr 緩和でどう変わるか

### 全実験の精度推移

| 実験 | 条件 | Best Valid Acc | Test Acc |
|---|---|---|---|
| 第1弾 (32→64, dropout追加, 30ep) | Late-Fusion | — | 42.6% |
| 第1弾 (32→64, dropout追加, 30ep) | Share-Fusion | — | 43.0% |
| 第2弾 (lr=1e-3 固定 60ep) | Late-Fusion | 44.7% | 43.3%* |
| 第2弾 (lr=1e-3 固定 60ep) | Share-Fusion | 45.8% | 49.6%* |
| 第3弾 (lr=1e-2→1e-3 step 100ep) | **Late-Fusion** | **50.4%** | **49.3%** |
| 第3弾 (lr=1e-2→1e-3 step 100ep) | Share-Fusion | 46.2% | 43.4% |
| 第4弾 (lr=5e-3→1e-3, do=0.2, 100ep) | Late-Fusion | — | — |
| 第4弾 (lr=5e-3→1e-3, do=0.2, 100ep) | Share-Fusion 固定 | — | — |
| 第4弾 (lr=5e-3→1e-3, do=0.2, 100ep) | Share-Fusion 学習 | — | — |

*第2弾の Test Acc は最終 epoch モデルでの評価（ベストモデルロード未実装時）

### 結果

（実験完了後に記入）

---
