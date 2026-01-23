
from data import load_datapath_NYU, load_datapath_WRGBD, load_datapath_TinyImageNet, get_dataloader, getlabels_NYU
from utils import load_experiment

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image
import numpy as np
import os

class ConditionalCLUB(nn.Module):
    def __init__(self, x_dim, y_dim, num_classes, hidden_size=64):
        super().__init__()
        self.class_emb = nn.Embedding(num_classes, 16)
        input_dim = x_dim + 16
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU()
        )
        self.mu_layer = nn.Linear(hidden_size, y_dim)
        self.logvar_layer = nn.Linear(hidden_size, y_dim)

    def get_mu_logvar(self, x, labels):
        c = self.class_emb(labels)
        inputs = torch.cat([x, c], dim=1)
        hidden = self.net(inputs)
        return self.mu_layer(hidden), self.logvar_layer(hidden)

    def log_likelihood(self, y, mu, logvar):
        return -0.5 * (torch.sum((y - mu) ** 2 / torch.exp(logvar), dim=1) + torch.sum(logvar, dim=1))

    def learning_loss(self, x, y, labels):
        mu, logvar = self.get_mu_logvar(x, labels)
        return -self.log_likelihood(y, mu, logvar).mean()

    def calculate_cmi(self, x, y, labels):
        self.eval()
        with torch.no_grad():
            mu, logvar = self.get_mu_logvar(x, labels)
            positive_score = self.log_likelihood(y, mu, logvar)
            cmi_values = []
            for i in range(x.size(0)):
                mask = (labels == labels[i])
                if mask.sum() <= 1: continue
                y_same_class = y[mask]
                x_expanded = x[i].unsqueeze(0).repeat(y_same_class.size(0), 1)
                c_expanded = labels[i].unsqueeze(0).repeat(y_same_class.size(0))
                mu_neg, logvar_neg = self.get_mu_logvar(x_expanded, c_expanded)
                negative_scores = self.log_likelihood(y_same_class, mu_neg, logvar_neg)
                cmi_values.append(positive_score[i] - negative_scores.mean())
            return torch.stack(cmi_values).mean().item() if cmi_values else 0.0

# ==========================================
# 2. Attention Map のベクトル化関数
# ==========================================

def save_club_model(model, path, config_dict):
    """
    モデルの重みと、初期化に必要な設定(config)をまとめて保存します。
    """
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config_dict
    }, path)
    print(f"Model saved to: {path}")

def load_club_model(path, device):
    """
    保存されたファイルから設定を読み込み、モデルを再構築して重みをロードします。
    """
    checkpoint = torch.load(path, map_location=device)
    config = checkpoint['config']
    
    # 保存された設定を使ってモデルを初期化
    model = ConditionalCLUB(
        x_dim=config['x_dim'],
        y_dim=config['y_dim'],
        num_classes=config['num_classes'],
        hidden_size=config['hidden_size']
    )
    
    # 重みをロード
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval() # 読み込み後はデフォルトで評価モードへ
    
    print(f"Model loaded from: {path}")
    return model

# ==========================================
# 3. 特徴量抽出 (ユーザーコードを統合)
# ==========================================
def extract_attention_features(image_paths_all, depth_paths_all, labels_all, model, device, batch_size=12):
    
    model.to(device)
    model.eval()
    
    f_r_list = []
    f_d_list = []
    labels_list = []
    
    total_images = len(image_paths_all)
    
    # バッチごとに処理を行うループ
    for start_idx in range(0, total_images, batch_size):
        end_idx = min(start_idx + batch_size, total_images)
        indices = list(range(start_idx, end_idx))
        
        # パスから画像をロード
        raw_images = [np.asarray(Image.open(image_paths_all[i]).convert("RGB")) for i in indices]
        raw_depths = [np.asarray(Image.open(depth_paths_all[i])) for i in indices]
        
        current_labels = [labels_all[i] for i in indices]
        
        transform1 = transforms.Compose([
            transforms.Resize((32, 32)), 
            transforms.ToTensor()
        ])
        depth_transform = transform1
        test_transform = transform1

        images = torch.stack([test_transform(Image.fromarray(img)) for img in raw_images]).to(device)
        depths = torch.stack([depth_transform(Image.fromarray(dpt)) for dpt in raw_depths]).to(device)

        with torch.no_grad():
            # モデル推論
            logits, _, _, attention_maps_img, attention_maps_dpt = model(images, depths, attentions_choice=True)
            
            if isinstance(attention_maps_img, list):
                # 最終層だけ使う、あるいは平均する
                 f_r_seq = torch.stack(attention_maps_img, dim=0).mean(dim=0).mean(dim=1) # [B, N, N]
                 f_d_seq = torch.stack(attention_maps_dpt, dim=0).mean(dim=0).mean(dim=1) # [B, N, N]
            else:
                 f_r_seq = attention_maps_img.mean(dim=1) # [B, N, N]
                 f_d_seq = attention_maps_dpt.mean(dim=1) # [B, N, N]

        # --- Patch-wise データの整形 ---
        # [Batch, N, Dim] -> [Batch * N, Dim]
        b, n, d = f_r_seq.shape
        
        f_r_flat = f_r_seq.reshape(-1, d).cpu()
        f_d_flat = f_d_seq.reshape(-1, d).cpu()
        
        # ラベルもパッチ数分だけ複製
        # [Batch] -> [Batch, N] -> [Batch * N]
        labels_seq = torch.tensor(current_labels).unsqueeze(1).repeat(1, n).reshape(-1)
        
        f_r_list.append(f_r_flat)
        f_d_list.append(f_d_flat)
        labels_list.append(labels_seq)

    # 結合
    f_r_all = torch.cat(f_r_list, dim=0).to(device)
    f_d_all = torch.cat(f_d_list, dim=0).to(device)
    labels_all = torch.cat(labels_list, dim=0).to(device)
    
    return f_r_all, f_d_all, labels_all

# ==========================================
# 4. メイン実行部
# ==========================================
def main(image_paths, depth_paths, labels_mapping, model):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # ラベルデータの準備（ユーザーコードの `labels` がリストや配列であることを想定）
    # getlabels_NYUなどで全データのラベルを取得しておいてください
    # ここでは仮に image_paths と対応する labels_all があるとします
    labels_all, _ = getlabels_NYU(image_paths) # ここは既存関数を使用
    
    print(f"Total samples: {len(image_paths)}")
    
    # 1. 特徴量抽出 (全データに対してループ)
    print("Extracting Attention Maps...")
    f_r, f_d, labels = extract_attention_features(
        image_paths, depth_paths, labels_all, model, device, batch_size=32
    )
    
    print(f"Extracted Features Shape: {f_r.shape}") # [N, 65] になっているはず

    # 2. CMI推定器の準備
    x_dim = f_r.shape[1]
    y_dim = f_d.shape[1]
    # ラベルを0始まりの連番に変換（Embedding用）
    unique_labels = torch.unique(labels)
    label_map = {old.item(): new for new, old in enumerate(unique_labels)}
    mapped_labels = torch.tensor([label_map[l.item()] for l in labels], device=device)
    num_classes = len(unique_labels)

    cmi_estimator = ConditionalCLUB(x_dim, y_dim, num_classes).to(device)
    optimizer = optim.Adam(cmi_estimator.parameters(), lr=1e-3)

    # 3. 学習 (Training)
    print("Training CMI Estimator...")
    cmi_estimator.train()
    dataset_size = f_r.size(0)
    train_batch = 128
    epochs = 100

    for epoch in range(epochs):
        perm = torch.randperm(dataset_size)
        for i in range(0, dataset_size, train_batch):
            idx = perm[i : i + train_batch]
            optimizer.zero_grad()
            loss = cmi_estimator.learning_loss(f_r[idx], f_d[idx], mapped_labels[idx])
            loss.backward()
            optimizer.step()
            
    # 学習が終わったら保存
    save_path = "cmi_estimator_checkpoint.pth"
    
    # 再構築に必要な情報を記録
    config_dict = {
        'x_dim': x_dim,
        'y_dim': y_dim,
        'num_classes': num_classes,
        'hidden_size': 64 # __init__のデフォルト値または指定値
    }
    
    save_club_model(cmi_estimator, save_path, config_dict)

    # 4. 推論 (Inference)
    print("Calculating CMI...")
    final_cmi = cmi_estimator.calculate_cmi(f_r, f_d, mapped_labels)
    return final_cmi
    
    print("="*40)
    print(f"Attention Map CMI (CLS token focus): {final_cmi:.6f}")
    print("="*40)
    
    # 解釈
    if final_cmi < 0.5:
        print("判定: 低い -> RGBとDepthは「どこを見るか」に関して役割分担している可能性が高い")
    else:
        print("判定: 高い -> RGBとDepthは同じ場所に注目しており、視点の重複(冗長)がある")