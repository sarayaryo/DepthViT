import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sys

# 既存のモジュールをインポート（パスが通っている前提）
try:
    from train import Trainer
    from data import load_datapath_NYU, load_datapath_WRGBD, load_datapath_TinyImageNet, get_dataloader
    from utils import load_experiment
except ImportError:
    pass

# ==========================================
# 1. Conditional CLUB モデル定義
# ==========================================
class ConditionalCLUB(nn.Module):
    def __init__(self, x_dim, y_dim, num_classes, hidden_size=64):
        super().__init__()
        # クラスラベルを埋め込む層（例: 10クラス -> 16次元）
        self.class_emb = nn.Embedding(num_classes, 16)
        
        # ネットワークへの入力は 特徴量X + クラス埋め込み
        input_dim = x_dim + 16
        
        # p(y|x, c) の分布（平均と分散）を予測するネットワーク
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.mu_layer = nn.Linear(hidden_size, y_dim)
        self.logvar_layer = nn.Linear(hidden_size, y_dim)

    def get_mu_logvar(self, x, labels):
        c = self.class_emb(labels)       # [Batch, 16]
        inputs = torch.cat([x, c], dim=1) 
        hidden = self.net(inputs)
        return self.mu_layer(hidden), self.logvar_layer(hidden)

    def log_likelihood(self, y, mu, logvar):
        # ガウス分布 N(mu, var) における y の対数尤度
        return -0.5 * (torch.sum((y - mu) ** 2 / torch.exp(logvar), dim=1) + torch.sum(logvar, dim=1))

    def learning_loss(self, x, y, labels):
        """ 推定器の学習用ロス（対数尤度の最大化＝負の対数尤度の最小化） """
        mu, logvar = self.get_mu_logvar(x, labels)
        ll = self.log_likelihood(y, mu, logvar)
        return -ll.mean()

    def calculate_cmi(self, x, y, labels):
        """ CMIの計算: I(x; y | c) """
        self.eval()
        with torch.no_grad():
            mu, logvar = self.get_mu_logvar(x, labels)
            positive_score = self.log_likelihood(y, mu, logvar)

            # 条件付きネガティブサンプリング（バッチ処理用簡易版）
            # 「同じラベルを持つ他のサンプル」と比較する
            cmi_values = []
            batch_size = x.size(0)
            
            for i in range(batch_size):
                # バッチ内で自分と同じラベルを持つデータを探す
                mask = (labels == labels[i])
                if mask.sum() <= 1: continue # 相手がいなければスキップ
                
                y_same_class = y[mask]
                
                # x[i] を相手の数だけ複製してペアを作る
                x_expanded = x[i].unsqueeze(0).repeat(y_same_class.size(0), 1)
                c_expanded = labels[i].unsqueeze(0).repeat(y_same_class.size(0))
                
                mu_neg, logvar_neg = self.get_mu_logvar(x_expanded, c_expanded)
                negative_scores = self.log_likelihood(y_same_class, mu_neg, logvar_neg)
                
                # CMI = Log P(y|x,c) - E[Log P(y_other|x,c)]
                cmi_values.append(positive_score[i] - negative_scores.mean())

            if not cmi_values: return 0.0
            return torch.stack(cmi_values).mean().item()

# ==========================================
# 2. attention_data からの特徴量抽出
# ==========================================
def process_attention_data_forCMI(attention_data, device):
    """
    trainer.evaluate の出力である辞書リストから f_r, f_d ベクトルを抽出・作成する
    """
    f_r_list = []
    f_d_list = []
    labels_list = []

    print("Processing attention data...")
    for item in attention_data:
        # item の構造: 
        # {'attention_img': (4, 65, 65), 'attention_dpt': (4, 65, 65), 'label': int, ...}
        
        att_img = torch.tensor(item['attention_img'], dtype=torch.float32).to(device)
        att_dpt = torch.tensor(item['attention_dpt'], dtype=torch.float32).to(device)
        label = torch.tensor(item['label'], dtype=torch.long).to(device)

        # ベクトル化戦略:
        # 1. Head方向(dim 0)の平均: [4, 65, 65] -> [65, 65]
        # 2. CLSトークン行(index 0)を取得: [65, 65] -> [65]
        # これにより「CLSトークンが画像のどのパッチに注目したか」という分布ベクトルになります
        
        vec_r = att_img.mean(dim=0)[0, :] # Shape: [65]
        vec_d = att_dpt.mean(dim=0)[0, :] # Shape: [65]

        f_r_list.append(vec_r)
        f_d_list.append(vec_d)
        labels_list.append(label)

    f_r_all = torch.stack(f_r_list)
    f_d_all = torch.stack(f_d_list)
    labels_all = torch.stack(labels_list)
    
    return f_r_all, f_d_all, labels_all

# ==========================================
# 3. 学習と保存を行う関数
# ==========================================
def train_and_save_cmi_model(f_r, f_d, labels, save_path, epochs=500, lr=1e-4):
    device = f_r.device
    x_dim = f_r.shape[1]
    y_dim = f_d.shape[1]
    num_classes = labels.max().item() + 1
    
    # Conditional CLUBの初期化
    cmi_model = ConditionalCLUB(x_dim, y_dim, num_classes, hidden_size=64).to(device)
    optimizer = optim.Adam(cmi_model.parameters(), lr=lr)

    print(f"Start training CMI Model... (Dim: {x_dim} -> {y_dim}, Classes: {num_classes})")
    cmi_model.train()
    
    # バッチ学習（学習の安定化のため）
    dataset_size = f_r.size(0)
    batch_size = 64
    
    for epoch in range(epochs):
        perm = torch.randperm(dataset_size)
        epoch_loss = 0
        
        for i in range(0, dataset_size, batch_size):
            idx = perm[i : i + batch_size]
            batch_x = f_r[idx]
            batch_y = f_d[idx]
            batch_c = labels[idx]

            optimizer.zero_grad()
            loss = cmi_model.learning_loss(batch_x, batch_y, batch_c)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        if epoch % 50 == 0:
            print(f"Epoch {epoch}: Loss = {-1 * epoch_loss / (dataset_size/batch_size):.4f}")

    # 保存ロジック: 重みだけでなく、モデル再構築に必要な設定(config)も保存する
    save_dict = {
        'model_state_dict': cmi_model.state_dict(),
        'config': {
            'x_dim': x_dim,
            'y_dim': y_dim,
            'num_classes': num_classes,
            'hidden_size': 64
        }
    }
    torch.save(save_dict, save_path)
    print(f"CMI model saved to: {save_path}")
    return cmi_model

# ==========================================
# 4. メイン実行部
# ==========================================
def main():
    # デバイス設定
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. データと実験設定のロード
    base_path = r'../data/nyu_data/nyu2' # linux
    image_paths, depth_paths, labels = load_datapath_NYU(base_path)

    # data_limit = 200
    # image_paths = image_paths[:data_limit]
    # depth_paths = depth_paths[:data_limit]
    # labels = labels[:data_limit]
    
    # Trainerの初期化（loss_fnなどはダミーでも良いが推論に必要なら定義）
    from torch import nn
    loss_fn = nn.CrossEntropyLoss()
    
    name = "NYU"
    experiment_name = name + "_sharefusion_alearn_blearn"
    config, model, _, _, _, label_mapping = load_experiment(
        experiment_name,
        checkpoint_name="model_final.pt",
        depth=True,
        map_location=device
        )
    model = model.to(device)

    trainer = Trainer(
        model=model,
        optimizer=None,
        loss_fn=loss_fn,
        method=2, 
        exp_name= experiment_name + "_infer",
        device=device
    )

    # 2. 推論を実行して Attention Map を収集
    # evaluate用のデータローダーを作成
    from torchvision import transforms
    transform1 = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
    _, test_loader, _, _, _ = get_dataloader(image_paths, depth_paths, 16, transform1, 1) # 1=NYU
    
    print("Running inference to collect attention maps...")
    # infer_mode=True を指定して attention_data_final を取得
    accuracy, CLS_token, avg_loss, attention_data_final, wrong_images, correct_images = trainer.evaluate(
        test_loader, attentions_choice=True, infer_mode=True
    )
    
    # 3. 特徴量の抽出・整形
    f_r, f_d, labels_tensor = process_attention_data_forCMI(attention_data_final, device)
    
    # 4. CMI推定器の学習と保存
    save_dir = r"experiments\vclub_model"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, experiment_name + "_CMI_dim_65.pth")
    
    train_and_save_cmi_model(f_r, f_d, labels_tensor, save_path, epochs=500, lr=1e-4)

if __name__ == "__main__":
    main()